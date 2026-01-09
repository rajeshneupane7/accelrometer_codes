import pandas as pd
import numpy as np
import optuna
import xgboost as xgb  # Import XGBoost
import ast
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Suppress Optuna's massive log output
optuna.logging.set_verbosity(optuna.logging.WARNING)

class ActivityModeler:
    def __init__(self, data, target_col='behavioral_category'):
        self.data = data
        self.target_col = target_col
        self.le = LabelEncoder()
        self.results_log = []
        
        if self.target_col not in self.data.columns:
            raise ValueError(f"Target column '{self.target_col}' not found.")
            
        counts = self.data[self.target_col].value_counts().to_dict()
        self.target_distribution_str = str(counts)

    def prepare_data(self, feature_cols):
        X = self.data[feature_cols]
        y = self.data[self.target_col]
        
        # XGBoost requires labels to be integers starting from 0
        y_encoded = self.le.fit_transform(y)
        
        # Stratify to maintain class balance
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
            )
        except ValueError:
            # Fallback if a class has too few samples
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42
            )
        return X_train, X_test, y_train, y_test

    def get_optimizer_objective(self, trial, algo_name, X, y):
        """
        Defines the hyperparameter search space for each algorithm.
        Returns the cross-validation score to maximize.
        """
        # 1. XGBoost (New)
        if algo_name == "XGBoost":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'eval_metric': 'mlogloss',
                'random_state': 42
            }
            model = xgb.XGBClassifier(**params)

        # 2. Random Forest
        elif algo_name == "RandomForest":
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'random_state': 42
            }
            model = RandomForestClassifier(**params)

        # 3. SVM
        elif algo_name == "SVM (RBF)":
            params = {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'kernel': 'rbf',
                'random_state': 42
            }
            model = SVC(**params)

        # 4. KNN
        elif algo_name == "KNN":
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 20),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'p': trial.suggest_categorical('p', [1, 2])
            }
            model = KNeighborsClassifier(**params)

        # 5. Logistic Regression
        elif algo_name == "LogisticReg":
            params = {
                'C': trial.suggest_float('C', 0.01, 100.0, log=True),
                'solver': 'lbfgs',
                'max_iter': 2000,
                'random_state': 42
            }
            model = LogisticRegression(**params)
            
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

        # Cross-Validation
        scores = cross_val_score(model, X, y, cv=3, scoring='f1_weighted')
        return scores.mean()

    def tune_and_evaluate(self, algo_name, X_train, X_test, y_train, y_test, n_trials=10):
        """
        Runs Optuna optimization, then trains the best model.
        """
        # Scaling is crucial for SVM/KNN/LogReg (XGBoost/RF handle unscaled fine, but scaling doesn't hurt)
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # --- 1. OPTUNA OPTIMIZATION ---
        def objective_wrapper(trial):
            return self.get_optimizer_objective(trial, algo_name, X_train_s, y_train)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective_wrapper, n_trials=n_trials)
        
        best_params = study.best_params

        # --- 2. RE-TRAIN BEST MODEL ---
        if algo_name == "XGBoost":
            # Must ensure fixed params are passed again
            model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        elif algo_name == "RandomForest":
            model = RandomForestClassifier(**best_params, random_state=42)
        elif algo_name == "SVM (RBF)":
            model = SVC(**best_params, kernel='rbf', random_state=42)
        elif algo_name == "KNN":
            model = KNeighborsClassifier(**best_params)
        elif algo_name == "LogisticReg":
            model = LogisticRegression(**best_params, solver='lbfgs', max_iter=2000, random_state=42)

        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        # --- 3. CALCULATE METRICS ---
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        return acc, prec, rec, f1, best_params

    def run_optuna_experiments(self, n_trials=10):
        """
        Main loop: Features -> Algorithms -> Optuna Tuning
        """
        base_features = [
            ("Raw Accel", ['x_g', 'y_g', 'z_g']),
            ("ODBA", ['odba']),
            ("VeDBA", ['vedba']),
            ("Magnitude", ['mag'])
        ]
        
        experiments_to_run = {}
        # Individual
        for name, cols in base_features:
            experiments_to_run[f"Indiv: {name}"] = cols
            
        # Cumulative
        curr_cols = []
        curr_names = []
        for name, cols in base_features:
            curr_cols = curr_cols + cols
            curr_names.append(name)
            seq_name = f"Seq: {' + '.join(curr_names)}"
            experiments_to_run[seq_name] = curr_cols

        # ADDED XGBOOST HERE
        #algo_names = ["XGBoost", "RandomForest", "SVM (RBF)", "KNN", "LogisticReg"]
        algo_names=['LogisticReg']
        
        print(f"Starting Optuna Tuning on {len(self.data)} rows.")
        print(f"Optimizing {len(experiments_to_run)} Feature Sets x {len(algo_names)} Models...")
        print("-" * 60)

        for feat_name, cols in experiments_to_run.items():
            if not set(cols).issubset(self.data.columns):
                print(f"⚠️ Skipping {feat_name} (Missing columns)")
                continue

            X_train, X_test, y_train, y_test = self.prepare_data(cols)

            for algo in algo_names:
                print(f" Tuning {algo} | Features: {feat_name}...")
                
                acc, prec, rec, f1, best_params = self.tune_and_evaluate(
                    algo, X_train, X_test, y_train, y_test, n_trials=n_trials
                )
                
                self.results_log.append({
                    'Algorithm': algo,
                    'Feature_Set': feat_name,
                    'Accuracy': round(acc, 4),
                    'Precision': round(prec, 4),
                    'Recall': round(rec, 4),
                    'F1_Score': round(f1, 4),
                    'Best_Params': str(best_params),
                    'Features_Used': str(cols)
                })

        return pd.DataFrame(self.results_log).sort_values(by='F1_Score', ascending=False)


    

# timeseries_models.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim, n_classes):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])

def train_lstm(X, y, n_classes, epochs=20, batch_size=32):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMClassifier(X.shape[2], 64, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X).argmax(1).numpy()

    return {
        "Accuracy": accuracy_score(y.numpy(), preds),
        "F1": f1_score(y.numpy(), preds, average="weighted")
    }
