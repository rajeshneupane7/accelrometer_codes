import pandas as pd
import numpy as np
import os
import tempfile
import zipfile
from typing import Union

class AccelPipeline:
    def __init__(self, df: pd.DataFrame = None, data_dir: str = None, 
                 calc_odba: bool = True, calc_vedba: bool = True,
                 max_accel_clip: float = 5.0):
        """
        Initialize the pipeline.
        
        Args:
            df (pd.DataFrame): Directly pass a dataframe.
            data_dir (str): Path to data.
            calc_odba (bool): Calculate ODBA.
            calc_vedba (bool): Calculate VeDBA.
            max_accel_clip (float): Clip acceleration values to remove noise spikes.
        """
        if df is not None:
            self.df = df.copy()
        elif data_dir is not None:
            self.df = self._load_data(data_dir)
        else:
            raise ValueError("Either 'df' or 'data_dir' must be provided.")
            
        self.calc_odba = calc_odba
        self.calc_vedba = calc_vedba
        self.max_accel_clip = max_accel_clip
        
        # Ensure Timestamp format
        if 'local_ts' in self.df.columns:
            self.df['local_ts'] = pd.to_datetime(self.df['local_ts'])
        
        # Sort by Subject THEN Time
        self.df = self.df.sort_values(by=['subject', 'local_ts']).reset_index(drop=True)
        
        # FIX: Remove duplicates to ensure unique index for rolling/transfrom operations
        print("Removing duplicate timestamps...")
        initial_len = len(self.df)
        self.df = self.df.drop_duplicates(subset=['subject', 'local_ts'], keep='first')
        print(f"Removed {initial_len - len(self.df)} duplicate rows.")

    def _load_data(self, data_dir: str) -> pd.DataFrame:
        """Loads data from ZIPs or CSVs found in data_dir."""
        all_dfs = []
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Directory {data_dir} not found.")

        for fname in os.listdir(data_dir):
            fpath = os.path.join(data_dir, fname)
            
            if fname.lower().endswith('.zip'):
                try:
                    with zipfile.ZipFile(fpath, "r") as zf:
                        with tempfile.TemporaryDirectory() as temp_dir:
                            zf.extractall(temp_dir)
                            for root, dirs, files in os.walk(temp_dir):
                                for d in dirs:
                                    if d.startswith('Processed'):
                                        second_path = os.path.join(root, d)
                                        for excel_file in os.listdir(second_path):
                                            if excel_file.lower().endswith(('.xls', '.xlsx')):
                                                all_dfs.append(pd.read_excel(os.path.join(second_path, excel_file)))
                except Exception as e:
                    print(f"Error processing ZIP {fpath}: {e}")

            elif fname.lower().endswith('.csv'):
                try:
                    all_dfs.append(pd.read_csv(fpath))
                except Exception as e:
                    print(f"Error processing CSV {fpath}: {e}")

        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    def clip_noise(self):
        """Removes sensor noise spikes by clipping acceleration values."""
        print(f"--- Clipping noise to +/- {self.max_accel_clip}g ---")
        cols = ['x_g', 'y_g', 'z_g']
        for c in cols:
            if c in self.df.columns:
                self.df[c] = self.df[c].clip(lower=-self.max_accel_clip, upper=self.max_accel_clip)
        return self.df

    def convert_to_gravity(self):
        """Converts raw accel to g-units, Magnitude, and ENMO."""
        print("--- Converting to Gravity Units & ENMO ---")
        scale = 16384.0 
        
        cols_to_check = ['x', 'y', 'z']
        if all(c in self.df.columns for c in cols_to_check):
            if self.df[cols_to_check].max().max() > 10: 
                self.df['x_g'] = self.df['x'] / scale
                self.df['y_g'] = self.df['y'] / scale
                self.df['z_g'] = self.df['z'] / scale
            else:
                self.df['x_g'] = self.df['x']
                self.df['y_g'] = self.df['y']
                self.df['z_g'] = self.df['z']
        
        self.df['mag'] = np.sqrt(self.df['x_g']**2 + self.df['y_g']**2 + self.df['z_g']**2)
        self.df['enmo'] = np.maximum(self.df['mag'] - 1, 0)
        return self.df

    def _get_dynamic_component(self, window_seconds=1):
        """
        Subtracts static gravity.
        Note: Works on irregularly sampled data by using the timestamp index.
        """
        temp_df = self.df.set_index('local_ts').sort_index()
        cols = ['x_g', 'y_g', 'z_g']
        
        # Rolling Mean (Static Gravity) - relies on timestamp
        static_component = temp_df.groupby('subject')[cols].rolling(f'{window_seconds}s', min_periods=1).mean()
        
        static_reset = static_component.reset_index()
        merged = pd.merge(self.df, static_reset, on=['subject', 'local_ts'], suffixes=('', '_static'))
        
        dynamic_df = pd.DataFrame()
        dynamic_df['x_d'] = merged['x_g'] - merged['x_g_static']
        dynamic_df['y_d'] = merged['y_g'] - merged['y_g_static']
        dynamic_df['z_d'] = merged['z_g'] - merged['z_g_static']
        
        return dynamic_df.fillna(0)

    def calc_dynamic_features(self):
        """Calculates ODBA, VeDBA, and ZCR (Zero Crossing Rate)."""
        dyn = self._get_dynamic_component()
        
        if self.calc_odba:
            self.df['odba'] = dyn['x_d'].abs() + dyn['y_d'].abs() + dyn['z_d'].abs()
        if self.calc_vedba:
            self.df['vedba'] = np.sqrt(dyn['x_d']**2 + dyn['y_d']**2 + dyn['z_d']**2)
            
        # Calculate Zero Crossing Rate (ZCR)
        print("--- Calculating ZCR (Zero Crossing Rate) ---")
        
        def calc_zcr(series):
            # Simple approximation for pandas rolling
            return (np.diff(np.sign(series)) != 0).sum()

        # Calculate ZCR on the magnitude signal over 1-second windows
        temp_df = self.df.set_index('local_ts')
        self.df['zcr'] = temp_df.groupby('subject')['mag'].transform(
            lambda x: x.rolling('1s', min_periods=1).apply(calc_zcr, raw=False)
        ).fillna(0)
        
        return self.df

    def resample_and_label(self, df: pd.DataFrame, interval_seconds=10, coherence_threshold=0.7):
        print(f"--- Resampling ({interval_seconds}s) with Threshold: {coherence_threshold} ---")
        
        def label_aggregator(x):
            if x.empty: return np.nan
            counts = x.value_counts(normalize=True)
            
            if coherence_threshold is None:
                return counts.index[0]
            
            if counts.iloc[0] >= coherence_threshold:
                return counts.index[0]
            return np.nan 

        agg_dict = {
            'x_g': ['mean', 'std', 'min', 'max'],
            'y_g': ['mean', 'std', 'min', 'max'],
            'z_g': ['mean', 'std', 'min', 'max'],
            'mag': ['mean', 'std'],
            'zcr': ['mean'], # Add ZCR to features
            'behavioral_category': label_aggregator
        }
        
        if 'odba' in df.columns:
            agg_dict['odba'] = ['mean', 'std']
        if 'vedba' in df.columns:
            agg_dict['vedba'] = ['mean', 'std']

        resampled = (
            df.set_index('local_ts')
            .groupby('subject')
            .resample(f'{interval_seconds}s')
            .agg(agg_dict)
        )
        
        resampled.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in resampled.columns]
        resampled = resampled.rename(columns={'behavioral_category_label_aggregator': 'behavioral_category'})
        
        final_df = resampled.dropna(subset=['behavioral_category']).reset_index()
        print(f"Generated {len(final_df)} labeled windows.")
        return final_df