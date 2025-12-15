# feature_pipeline.py
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

class AdaptiveAccelPipeline:
    def __init__(self, df):
        self.df = df.copy()
        self.df = self.df.sort_values(["subject", "local_ts"])

    def apply_adaptive_filter(self, target_cutoff_hz=5.0):
        def process_subject(g):
            diffs = g["local_ts"].diff().dt.total_seconds()
            diffs = diffs[diffs > 0]
            if len(diffs) < 10:
                return g

            fs = 1 / diffs.median()
            nyq = fs * 0.5

            if fs <= target_cutoff_hz * 2:
                cutoff = nyq * 0.9
                if cutoff < 1:
                    return g
            else:
                cutoff = target_cutoff_hz

            try:
                b, a = butter(4, cutoff / nyq, btype="low")
                for c in ["x", "y", "z"]:
                    g[c] = filtfilt(b, a, g[c])
            except Exception:
                pass

            return g

        self.df = self.df.groupby("subject", group_keys=False).apply(process_subject)
        return self.df

    def compute_physics_features(self, scale=16384.0):
        self.df["x_g"] = self.df["x"] / scale
        self.df["y_g"] = self.df["y"] / scale
        self.df["z_g"] = self.df["z"] / scale
        self.df["mag"] = np.sqrt(
            self.df["x_g"]**2 + self.df["y_g"]**2 + self.df["z_g"]**2
        )
        self.df["enmo"] = np.maximum(self.df["mag"] - 1, 0)

        self.df["odba"] = (
            (self.df["x_g"] - self.df["x_g"].mean()).abs() +
            (self.df["y_g"] - self.df["y_g"].mean()).abs() +
            (self.df["z_g"] - self.df["z_g"].mean()).abs()
        )
        return self.df

    def resample_and_label(self, interval_seconds=10, coherence_threshold=0.7):
        def labeler(x):
            if x.empty:
                return np.nan
            freq = x.value_counts(normalize=True)
            return freq.index[0] if freq.iloc[0] >= coherence_threshold else np.nan

        agg = {
            "x_g": ["mean", "std"],
            "y_g": ["mean", "std"],
            "z_g": ["mean", "std"],
            "mag": ["mean", "std"],
            "enmo": ["mean", "max"],
            "odba": ["mean", "std"],
            "behavioral_category": labeler
        }

        out = (
            self.df.set_index("local_ts")
            .groupby("subject")
            .resample(f"{interval_seconds}s")
            .agg(agg)
        )

        out.columns = ["_".join(c) for c in out.columns]
        out = out.rename(columns={"behavioral_category_labeler": "label"})
        return out.dropna(subset=["label"]).reset_index()

    @staticmethod
    def make_sequences(df, feature_cols, target_col, time_steps):
        X, y = [], []
        for _, g in df.groupby("subject"):
            g = g.sort_values("local_ts")
            feats = g[feature_cols].values
            labels = g[target_col].values
            for i in range(len(g) - time_steps):
                X.append(feats[i:i+time_steps])
                y.append(labels[i+time_steps])
        return np.array(X), np.array(y)
