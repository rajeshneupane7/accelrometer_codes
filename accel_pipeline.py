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
        
        # Remove duplicates to ensure clean merging
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
        """Calculates ODBA and VeDBA. ZCR is now calculated during resampling."""
        dyn = self._get_dynamic_component()
        
        if self.calc_odba:
            self.df['odba'] = dyn['x_d'].abs() + dyn['y_d'].abs() + dyn['z_d'].abs()
        if self.calc_vedba:
            self.df['vedba'] = np.sqrt(dyn['x_d']**2 + dyn['y_d']**2 + dyn['z_d']**2)
            
        # ZCR removed from here to improve performance
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

        # --- OPTIMIZED ZCR FUNCTION ---
        def calculate_zcr(series):
            """
            Calculates Zero Crossing Rate for the entire window.
            Fast numpy implementation.
            """
            if len(series) < 2:
                return 0
            # Subtract mean to detect crossings relative to the signal's center
            mean_val = np.mean(series)
            signed_diff = np.sign(series - mean_val)
            # Count where sign changes
            return (np.diff(signed_diff) != 0).sum()

        agg_dict = {
            'x_g': ['mean', 'std', 'min', 'max'],
            'y_g': ['mean', 'std', 'min', 'max'],
            'z_g': ['mean', 'std', 'min', 'max'],
            'mag': ['mean', 'std'],
            'zcr': calculate_zcr, # Add ZCR calculation here (Fast)
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
        
        # Flatten columns
        # If agg returns a name (function name), use it. If None (string), use string.
        new_cols = []
        for col in resampled.columns:
            if col[1] == '':
                new_cols.append(col[0])
            elif col[1] == '<lambda>': # Python < 3.9 sometimes names lambdas this way
                new_cols.append(f"{col[0]}_func")
            else:
                new_cols.append(f"{col[0]}_{col[1]}")
                
        resampled.columns = new_cols
        resampled = resampled.rename(columns={'behavioral_category_label_aggregator': 'behavioral_category'})
        
        final_df = resampled.dropna(subset=['behavioral_category']).reset_index()
        print(f"Generated {len(final_df)} labeled windows.")
        return final_df