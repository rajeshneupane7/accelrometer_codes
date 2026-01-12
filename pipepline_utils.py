import pandas as pd
import numpy as np
import os
import tempfile
import zipfile

def load_all_zips(data_dir):
    all_dfs = []
    if not os.path.isdir(data_dir):
        return pd.DataFrame()
    for fname in os.listdir(data_dir):
        if not fname.lower().endswith('.zip'):
            continue
        zip_path = os.path.join(data_dir, fname)
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(temp_dir)
                    for root, dirs, files in os.walk(temp_dir):
                        for d in dirs:
                            if d.startswith('Processed'):
                                second_path = os.path.join(root, d)
                                for excel_file in os.listdir(second_path):
                                    if excel_file.lower().endswith(('.xls', '.xlsx')):
                                        all_dfs.append(pd.read_excel(os.path.join(second_path, excel_file)))
            except Exception as e:
                print(f"Error processing {zip_path}: {e}")
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. DATA PROCESSING PIPELINE
# ==========================================
class AccelPipeline:
    def __init__(self, df):
        self.df = df.copy()
        
        # Ensure Timestamp format
        self.df['local_ts'] = pd.to_datetime(self.df['local_ts'])
        
        # Sort by Subject THEN Time to ensure rolling window correctness
        self.df = self.df.sort_values(by=['subject', 'local_ts']).reset_index(drop=True)
        
    def convert_to_gravity(self):
        print("--- Converting to Gravity Units & Calculating ENMO ---")
        scale = 16384.0
        
        self.df['x_g'] = self.df['x'] / scale
        self.df['y_g'] = self.df['y'] / scale
        self.df['z_g'] = self.df['z'] / scale
        
        # Magnitude
        self.df['mag'] = np.sqrt(self.df['x_g']**2 + self.df['y_g']**2 + self.df['z_g']**2)
        
        # ENMO: max(mag - 1, 0)
        self.df['enmo'] = np.maximum(self.df['mag'] - 1, 0)
        return self.df

    def _get_dynamic_component(self, window_seconds=1):
        # Create temp df indexed by time
        temp_df = self.df.set_index('local_ts').sort_index()
        cols = ['x_g', 'y_g', 'z_g']
        
        # Group by subject -> Rolling Mean (Static Gravity)
        static_component = temp_df.groupby('subject')[cols].rolling(f'{window_seconds}s').mean()
        
        # Merge static values back to main dataframe
        static_reset = static_component.reset_index()
        merged = pd.merge(self.df, static_reset, on=['subject', 'local_ts'], suffixes=('', '_static'))
        
        # Dynamic = Raw - Static
        dynamic_df = pd.DataFrame()
        dynamic_df['x_d'] = merged['x_g'] - merged['x_g_static']
        dynamic_df['y_d'] = merged['y_g'] - merged['y_g_static']
        dynamic_df['z_d'] = merged['z_g'] - merged['z_g_static']
        
        return dynamic_df.fillna(0)

    def calc_odba(self):
        print("--- Calculating ODBA ---")
        dyn = self._get_dynamic_component()
        self.df['odba'] = dyn['x_d'].abs() + dyn['y_d'].abs() + dyn['z_d'].abs()
        return self.df

    def calc_vedba(self):
        print("--- Calculating VeDBA ---")
        dyn = self._get_dynamic_component()
        self.df['vedba'] = np.sqrt(dyn['x_d']**2 + dyn['y_d']**2 + dyn['z_d']**2)
        return self.df

    def resample_and_label(self, interval_seconds=10, coherence_threshold=0.7):
        """
        Resamples data into windows. 
        ASSIGN LABELS based on threshold:
        If > 70% of the raw samples in the window are 'Grazing', the window is 'Grazing'.
        Otherwise, the window is discarded (Ambiguous).
        """
        print(f"--- Resampling ({interval_seconds}s) with Threshold {coherence_threshold*100}% ---")
        
        # Custom Aggregator for Labels
        def threshold_labeler(x):
            if x.empty: return np.nan
            counts = x.value_counts(normalize=True)
            # Check if the most frequent label crosses the threshold
            if counts.iloc[0] >= coherence_threshold:
                return counts.index[0]
            return np.nan # Drop this window (too messy/transitioning)

        # Feature Aggregators
        agg_dict = {
            'x_g': ['mean', 'std', 'min', 'max'],
            'y_g': ['mean', 'std', 'min', 'max'],
            'z_g': ['mean', 'std', 'min', 'max'],
            'mag': ['mean', 'std'],      
            #'enmo': ['mean', 'max'],
            #'vebda':['mean', 'std'], 
            'odba': ['mean', 'std'],    
            'behavioral_category': threshold_labeler # <--- LOGIC APPLIED HERE
        }

        resampled = (
            self.df.set_index('local_ts')
            .groupby('subject')
            .resample(f'{interval_seconds}s')
            .agg(agg_dict)
        )
        
        # Flatten columns
        resampled.columns = [f"{c[0]}_{c[1]}" if c[1] else c[0] for c in resampled.columns]
        resampled = resampled.rename(columns={'behavioral_category_threshold_labeler': 'behavioral_category'})
        
        # DROP windows that failed the threshold check
        final_df = resampled.dropna(subset=['behavioral_category']).reset_index()
        print(final_df.columns)
        print(f"Generated {len(final_df)} labeled windows.")
        return final_df
