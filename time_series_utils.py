import pandas as pd
import numpy as np
import optuna
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# Reduce Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================
# 1. WINDOWED DATASET
# =====================================================
class WindowedTimeSeriesDataset(Dataset):
    def __init__(self, X, y, window_size, stride=1):
        self.X = X
        self.y = y
        self.window_size = window_size
        self.stride = stride

    def __len__(self):
        return (len(self.X) - self.window_size) // self.stride

    def __getitem__(self, idx):
        i = idx * self.stride
        return (
            torch.tensor(self.X[i:i+self.window_size], dtype=torch.float32),
            torch.tensor(self.y[i+self.window_size-1], dtype=torch.long)
        )

# =====================================================
# 2. MODELS
# =====================================================
class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim, n_layers, n_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class CNN1DClassifier(nn.Module):
    def __init__(self, n_features, n_filters, kernel_size, n_classes):
        super().__init__()
        self.conv = nn.Conv1d(n_features, n_filters, kernel_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(n_filters, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# =====================================================
# 3. DEEP ACTIVITY MODELER (ANALOGOUS TO ActivityModeler)
# =====================================================
class DeepActivityModeler:
    def __init__(self, data, target_col="behavioral_category", device=None):
        self.data = data
        self.target_col = target_col
        self.le = LabelEncoder()
        self.results_log = []
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found.")

    def prepare_data(self, feature_cols):
        X = self.data[feature_cols].values
        y = self.le.fit_transform(self.data[self.target_col])
        n_classes = len(np.unique(y))
        return X, y, n_classes

    def get_optimizer_objective(
        self,
        trial,
        model_name,
        X,
        y,
        n_classes,
        window_size
    ):
        # Shared hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        if model_name == "LSTM":
            hidden_dim = trial.suggest_int("hidden_dim", 32, 128)
            n_layers = trial.suggest_int("n_layers", 1, 3)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)

            model = LSTMClassifier(
                n_features=X.shape[1],
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                n_classes=n_classes,
                dropout=dropout
            )

        elif model_name == "CNN":
            n_filters = trial.suggest_int("n_filters", 16, 64)
            kernel_size = trial.suggest_int("kernel_size", 3, 7)

            model = CNN1DClassifier(
                n_features=X.shape[1],
                n_filters=n_filters,
                kernel_size=kernel_size,
                n_classes=n_classes
            )

        model.to(self.device)

        dataset = WindowedTimeSeriesDataset(X, y, window_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        # Short training for Optuna
        model.train()
        for _ in range(5):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()

        # Evaluation
        model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                preds = model(xb).argmax(1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(yb.numpy())

        return f1_score(y_true, y_pred, average="weighted")

    def run_optuna_experiments(self, window_size, n_trials=20):
        base_features = [
            ("Raw Accel", ['x_g', 'y_g', 'z_g']),
            ("ODBA", ['odba']),
            ("VeDBA", ['vedba']),
            ("Magnitude", ['mag'])
        ]

        experiments = {}
        curr_cols, curr_names = [], []

        for name, cols in base_features:
            curr_cols += cols
            curr_names.append(name)
            experiments[f"Seq: {' + '.join(curr_names)}"] = curr_cols.copy()

        model_names = ["LSTM", "CNN"]

        for feat_name, cols in experiments.items():
            if not set(cols).issubset(self.data.columns):
                continue

            X, y, n_classes = self.prepare_data(cols)

            for model_name in model_names:
                print(f"🔍 {model_name} | {feat_name} | window={window_size}")

                study = optuna.create_study(direction="maximize")
                study.optimize(
                    lambda t: self.get_optimizer_objective(
                        t, model_name, X, y, n_classes, window_size
                    ),
                    n_trials=n_trials
                )

                self.results_log.append({
                    "Model": model_name,
                    "Feature_Set": feat_name,
                    "Window": window_size,
                    "Best_F1": round(study.best_value, 4),
                    "Best_Params": str(study.best_params),
                    "Features_Used": str(cols)
                })

        return pd.DataFrame(self.results_log).sort_values(
            "Best_F1", ascending=False
        )

