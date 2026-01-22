import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import optuna
import os

from models import LSTMClassifier, BiLSTMClassifier, CNN1DClassifier, TransformerClassifier
from accel_pipeline import AccelPipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)

class WindowedTimeSeriesDataset(Dataset):
    def __init__(self, X, y, window_size, stride=1):
        self.X = X
        self.y = y
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        return (len(self.X) - self.window_size) // self.stride + 1

    def __getitem__(self, idx):
        i = idx * self.stride
        if i + self.window_size > len(self.X):
            i = len(self.X) - self.window_size
        return (
            torch.tensor(self.X[i:i+self.window_size], dtype=torch.float32),
            torch.tensor(self.y[i+self.window_size-1], dtype=torch.long)
        )

class ActivityExperimentLibrary:
    def __init__(self, pipeline: AccelPipeline, device='cuda'):
        self.pipeline = pipeline
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.label_encoder = LabelEncoder()
        self.scaler = None

    def prepare_data(self, df: pd.DataFrame, feature_cols: list, is_training=True):
        valid_cols = [c for c in feature_cols if c in df.columns]
        X_raw = df[valid_cols].values
        
        if is_training:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X_raw)
            y = self.label_encoder.fit_transform(df['behavioral_category'])
            self.n_classes = len(self.label_encoder.classes_)
        else:
            if self.scaler is None:
                raise RuntimeError("Scaler not fitted.")
            X = self.scaler.transform(X_raw)
            
            unique_labels = df['behavioral_category'].unique()
            known_labels = self.label_encoder.classes_
            mask = df['behavioral_category'].isin(known_labels)
            df_filtered = df[mask]
            X = X[mask]
            
            if len(df_filtered) == 0:
                 return np.array([]), np.array([])
                 
            y = self.label_encoder.transform(df_filtered['behavioral_category'])
            
        return X, y

    def get_model(self, model_name, trial_params):
        n_features = trial_params['n_features']
        n_classes = trial_params['n_classes']
        
        if model_name == "LSTM":
            return LSTMClassifier(n_features, trial_params['hidden_dim'], 
                                 trial_params['n_layers'], n_classes, trial_params['dropout'])
        elif model_name == "BiLSTM":
            return BiLSTMClassifier(n_features, trial_params['hidden_dim'], 
                                   trial_params['n_layers'], n_classes, trial_params['dropout'])
        elif model_name == "CNN":
            return CNN1DClassifier(n_features, trial_params['n_filters'], 
                                  trial_params['kernel_size'], n_classes)
        elif model_name == "Transformer":
            return TransformerClassifier(n_features, trial_params['n_heads'], 
                                       trial_params['n_layers'], n_classes, 
                                       dropout=trial_params['dropout'])
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def get_hyperparams(self, trial, model_name, n_features, n_classes):
        params = {
            'lr': trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical("batch_size", [32, 64, 128]),
            'n_features': n_features,
            'n_classes': n_classes
        }

        if model_name in ["LSTM", "BiLSTM"]:
            params.update({
                'hidden_dim': trial.suggest_int("hidden_dim", 32, 128),
                'n_layers': trial.suggest_int("n_layers", 1, 3),
                'dropout': trial.suggest_float("dropout", 0.0, 0.5)
            })
        elif model_name == "CNN":
            params.update({
                'n_filters': trial.suggest_int("n_filters", 16, 64),
                'kernel_size': trial.suggest_int("kernel_size", 3, 7)
            })
        elif model_name == "Transformer":
            params.update({
                'n_heads': trial.suggest_categorical("n_heads", [2, 4, 8]),
                'n_layers': trial.suggest_int("n_layers", 1, 3),
                'dropout': trial.suggest_float("dropout", 0.0, 0.3)
            })
        return params

    def train_and_evaluate_fold(self, train_df_raw, test_df_raw, feature_cols, window_size, model_name, n_trials, interval, threshold):
        # 1. RESAMPLE
        train_df = self.pipeline.resample_and_label(
            train_df_raw, 
            interval_seconds=interval, 
            coherence_threshold=threshold
        )
        
        test_df = self.pipeline.resample_and_label(
            test_df_raw, 
            interval_seconds=interval, 
            coherence_threshold=None 
        )

        if train_df.empty or test_df.empty:
            return None, None, None, None

        n_test_classes = test_df['behavioral_category'].nunique()

        # 2. PREPARE DATA
        X_train, y_train = self.prepare_data(train_df, feature_cols, is_training=True)
        X_test, y_test = self.prepare_data(test_df, feature_cols, is_training=False)
        
        if len(X_test) == 0:
            return None, None, None, None

        # 3. VALIDATION SPLIT
        val_split_idx = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:val_split_idx], X_train[val_split_idx:]
        y_tr, y_val = y_train[:val_split_idx], y_train[val_split_idx:]

        # 4. OPTUNA OBJECTIVE
        def objective(trial):
            params = self.get_hyperparams(trial, model_name, X_tr.shape[1], self.n_classes)
            model = self.get_model(model_name, params).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
            loss_fn = nn.CrossEntropyLoss()
            
            train_ds = WindowedTimeSeriesDataset(X_tr, y_tr, window_size)
            val_ds = WindowedTimeSeriesDataset(X_val, y_val, window_size)
            train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
            
            model.train()
            for _ in range(5): 
                for xb, yb in train_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    loss = loss_fn(model(xb), yb)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for xb, yb in DataLoader(val_ds, batch_size=64):
                    xb = xb.to(self.device)
                    preds = model(xb).argmax(1).cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(yb.numpy())
            
            return f1_score(val_true, val_preds, average='weighted', zero_division=0)

        # 5. RUN OPTUNA
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params

        # 6. FINAL TRAINING
        final_params = best_params.copy()
        final_params['n_features'] = X_train.shape[1]
        final_params['n_classes'] = self.n_classes

        model = self.get_model(model_name, final_params).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=final_params['lr'])
        loss_fn = nn.CrossEntropyLoss()

        train_ds_full = WindowedTimeSeriesDataset(X_train, y_train, window_size)
        train_loader_full = DataLoader(train_ds_full, batch_size=final_params['batch_size'], shuffle=True)

        model.train()
        for _ in range(20):
            for xb, yb in train_loader_full:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()

        # 7. EVALUATE
        test_ds = WindowedTimeSeriesDataset(X_test, y_test, window_size)
        test_loader = DataLoader(test_ds, batch_size=64)
        
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(self.device)
                preds = model(xb).argmax(1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(yb.numpy())

        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        acc = accuracy_score(y_true, y_pred)
        
        return f1, acc, best_params, n_test_classes

    def run_loso_experiment(self, resample_interval=30, window_size=10, 
                            thresholds=[0.6], models_to_test=["LSTM", "CNN", "Transformer", "BiLSTM"], 
                            n_trials=5, output_dir="results"):
        os.makedirs(output_dir, exist_ok=True)
        
        # PREPROCESSING (Clipping Noise)
        self.pipeline.convert_to_gravity()
        self.pipeline.clip_noise() # NEW: Handle spikes
        self.pipeline.calc_dynamic_features()
        df_raw = self.pipeline.df 

        print("df_raw columns:", df_raw.columns.tolist())
        subjects = df_raw['subject'].unique()
        all_results = []

        # Generate Feature Columns (Updated to include ZCR)
        feature_cols = []
        for ax in ['x_g', 'y_g', 'z_g']:
            for stat in ['mean', 'std', 'min', 'max']:
                feature_cols.append(f"{ax}_{stat}")
        for stat in ['mean', 'std']:
            feature_cols.append(f"mag_{stat}")
        
        feature_cols.append("zcr_mean") # NEW: Add ZCR
        
        if self.pipeline.calc_odba:
            for stat in ['mean', 'std']:
                feature_cols.append(f"odba_{stat}")
        if self.pipeline.calc_vedba:
            for stat in ['mean', 'std']:
                feature_cols.append(f"vedba_{stat}")

        print(f"Starting LOSO Experiment with {len(subjects)} subjects.")
        print(f"Features to use: {feature_cols}\n")

        for test_subject in subjects:
            print(f"--- Testing on Subject: {test_subject} ---")
            
            train_df_raw = df_raw[df_raw['subject'] != test_subject].copy()
            test_df_raw = df_raw[df_raw['subject'] == test_subject].copy()
            
            if train_df_raw.empty or test_df_raw.empty:
                continue

            for model_name in models_to_test:
                try:
                    f1, acc, params, n_classes = self.train_and_evaluate_fold(
                        train_df_raw, test_df_raw, 
                        feature_cols, window_size, model_name, n_trials, 
                        interval=resample_interval, 
                        threshold=thresholds[0] 
                    )
                    
                    if f1 is not None:
                        result = {
                            "Test_Subject": test_subject,
                            "Model": model_name,
                            "Resample": resample_interval,
                            "Window": window_size,
                            "Test_F1": f1,
                            "Test_Acc": acc,
                            "N_Test_Classes": n_classes,
                            "Best_Params": str(params)
                        }
                        all_results.append(result)
                        print(f"Subject {test_subject} | {model_name} | F1: {f1:.4f}")
                    
                except Exception as e:
                    print(f"Error training {model_name} on subject {test_subject}: {e}")

        results_df = pd.DataFrame(all_results)
        
        if not results_df.empty:
            save_path = os.path.join(output_dir, "loso_results.csv")
            results_df.to_csv(save_path, index=False)
            print(f"\n✅ Experiment Complete. Results saved to {save_path}")
            np.save(os.path.join(output_dir, "classes.npy"), self.label_encoder.classes_)
        else:
            print("\n⚠ No results generated.")

        return results_df