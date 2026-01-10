import pandas as pd
from pipepline_utils import load_all_zips, AccelPipeline
from sklearn.pipeline import  Pipeline

from modeling_utils  import ActivityModeler
data = load_all_zips('/home/rajesh/work/acclerometer_project/zip_data')
pipeline = AccelPipeline(data)
pipeline.convert_to_gravity()
pipeline.calc_odba()
pipeline.calc_vedba()


def run_timeseries_pipeline(pipeline, time_steps):
    import ast
    import torch
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader

    # 🔽 IMPORT from your single-file deep modeler
    from time_series_utils import (
        DeepActivityModeler,
        WindowedTimeSeriesDataset,
        LSTMClassifier,
        CNN1DClassifier
    )

    temp_results = []
    resampled_map = {}

    # ======================================================
    # 1. TRAIN ACROSS TIME WINDOWS
    # ======================================================
    for t in time_steps:
        print(f"\n⏱ Running time window = {t}s")

        df_ready = pipeline.resample_data(interval_seconds=t)

        if df_ready is None or df_ready.empty:
            continue
        if 'behavioral_category' not in df_ready.columns:
            continue

        # Optional subsampling (same as classical)
        df_ready = df_ready.sample(frac=0.01, random_state=42)

        modeler = DeepActivityModeler(
            df_ready,
            target_col='behavioral_category'
        )

        results_df = modeler.run_optuna_experiments(
            window_size=t,
            n_trials=20
        )

        if results_df is None or results_df.empty:
            continue

        results_df = results_df.copy()
        results_df["time"] = t

        temp_results.append(results_df)
        resampled_map[t] = df_ready

    if not temp_results:
        raise RuntimeError("No valid time-series models trained.")

    final_df = pd.concat(temp_results, ignore_index=True)

    # ======================================================
    # 2. SELECT BEST MODEL (GLOBAL)
    # ======================================================
    best_row = final_df.sort_values("Best_F1", ascending=False).iloc[0]

    best_time = best_row["time"]
    best_model_name = best_row["Model"]
    best_features = ast.literal_eval(best_row["Features_Used"])
    best_params = ast.literal_eval(best_row["Best_Params"])

    df_best = resampled_map[best_time]

    # ======================================================
    # 3. PREPARE FULL DATA
    # ======================================================
    X = df_best[best_features].values
    le = LabelEncoder()
    y = le.fit_transform(df_best["behavioral_category"])
    n_classes = len(np.unique(y))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = WindowedTimeSeriesDataset(
        X, y, window_size=best_time
    )

    loader = DataLoader(
        dataset,
        batch_size=best_params.get("batch_size", 64),
        shuffle=True
    )

    # ======================================================
    # 4. REBUILD BEST MODEL
    # ======================================================
    if best_model_name == "LSTM":
        model = LSTMClassifier(
            n_features=X.shape[1],
            hidden_dim=best_params["hidden_dim"],
            n_layers=best_params["n_layers"],
            n_classes=n_classes,
            dropout=best_params["dropout"]
        )

    elif best_model_name == "CNN":
        model = CNN1DClassifier(
            n_features=X.shape[1],
            n_filters=best_params["n_filters"],
            kernel_size=best_params["kernel_size"],
            n_classes=n_classes
        )

    else:
        raise ValueError(f"Unknown model {best_model_name}")

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params["lr"]
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    # ======================================================
    # 5. FINAL TRAINING (FULL DATA)
    # ======================================================
    model.train()
    for _ in range(20):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    return {
        "model": model,
        "label_encoder": le,
        "time": best_time,
        "model_type": best_model_name,
        "features": best_features,
        "best_params": best_params,
        "f1_score": best_row["Best_F1"]
    }
time_steps = [10, 15, 20]

model_dict = run_timeseries_pipeline(
    pipeline=pipeline,
    time_steps=time_steps
)

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from time_series_utils import WindowedTimeSeriesDataset
from torch.utils.data import DataLoader

# Load test data
test_df = pd.read_excel(
    "/home/rajesh/work/acclerometer_project/data/30July25/Processed Files/"
    "processed 500_AED4_30July25_700_900.xlsx"
)

test_pipeline = AccelPipeline(test_df)
test_pipeline.convert_to_gravity()
test_pipeline.calc_odba()
test_pipeline.calc_vedba()

data = test_pipeline.resample_data(
    interval_seconds=model_dict["time"]
)

X_test = data[model_dict["features"]].values
y_true = data["behavioral_category"].values

dataset = WindowedTimeSeriesDataset(
    X_test,
    model_dict["label_encoder"].transform(y_true),
    window_size=model_dict["time"]
)

loader = DataLoader(dataset, batch_size=64, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model_dict["model"]
model.eval()

y_pred = []

with torch.no_grad():
    for xb, _ in loader:
        xb = xb.to(device)
        preds = model(xb).argmax(1).cpu().numpy()
        y_pred.extend(preds)

y_pred = model_dict["label_encoder"].inverse_transform(y_pred)

data = data.iloc[model_dict["time"]:]  # align with windows
data["predicted"] = y_pred

accuracy  = accuracy_score(data["behavioral_category"], y_pred)
precision = precision_score(data["behavioral_category"], y_pred, average="weighted")
recall    = recall_score(data["behavioral_category"], y_pred, average="weighted")
f1        = f1_score(data["behavioral_category"], y_pred, average="weighted")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1       :", f1)