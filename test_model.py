import argparse
import os
import ast
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader

# Import utility classes from your existing files
from pipepline_utils import load_all_zips, AccelPipeline, load_test_data
from time_series_utils import (
    WindowedTimeSeriesDataset, 
    LSTMClassifier, 
    BiLSTMClassifier,  # <--- ADD THIS
    GRUClassifier,      # <--- ADD THIS
    CNN1DClassifier
)

def find_model_file(directory):
    """Helper to find the model file whether it ends in .pt or .pth"""
    for filename in ["best_model.pth", "best_model.pt"]:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            print(f"🔍 Found model at: {path}")
            return path
    return None

def run_inference(data_dir, output_dir):
    # 1. Load Training Configuration
    results_path = os.path.join(output_dir, "results.csv")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found at {results_path}. Cannot determine model config.")

    results_df = pd.read_csv(results_path)
    
    # Find the row with the best F1 score
    best_row = results_df.loc[results_df['Best_F1'].idxmax()]

    print("="*60)
    print("Best Configuration Loaded:")
    print(f"Model Type:    {best_row['Model']}")
    print(f"Window Size:   {best_row['window_size']}")
    print(f"Resample Int:  {best_row['resample_interval']}s")
    print(f"Threshold:     {best_row['threshold']}")
    print(f"Best F1:       {best_row['Best_F1']}")
    print("="*60)

    # Extract hyperparameters
    best_interval = int(best_row['resample_interval'])
    best_window = int(best_row['window_size'])
    best_model_name = best_row['Model']
    best_features = ast.literal_eval(best_row['Features_Used'])
    best_params = ast.literal_eval(best_row['Best_Params'])
    best_threshold = float(best_row['threshold'])

    # 2. Load Artifacts
    le_path = os.path.join(output_dir, "label_classes.npy")
    le = LabelEncoder()
    le.classes_ = np.load(le_path, allow_pickle=True)
    n_classes = len(le.classes_)
    
    # FIND MODEL (Supports both .pt and .pth)
    model_path = find_model_file(output_dir)
    if not model_path:
        raise FileNotFoundError("Could not find best_model.pt or best_model.pth in the output directory.")

    # 3. Preprocess New Data
    print(f"\n📥 Loading data from {data_dir}...")
    raw_data = load_test_data(data_dir)

    if raw_data.empty:
        print("❌ No data found in the provided zip files.")
        return

    pipeline = AccelPipeline(raw_data)
    pipeline.convert_to_gravity()
    pipeline.calc_odba()
    pipeline.calc_vedba() 

    print(f"⏱ Resampling data (interval={best_interval}s, threshold={best_threshold})...")
    df_processed = pipeline.resample_and_label(
        interval_seconds=best_interval, 
        coherence_threshold=best_threshold
    )

    if df_processed.empty:
        print("❌ Dataframe is empty after resampling.")
        return

    # 4. Prepare Tensors
    missing_cols = [c for c in best_features if c not in df_processed.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in processed data: {missing_cols}")

    X_test = df_processed[best_features].values

    # Handle Ground Truth
    has_ground_truth = 'behavioral_category' in df_processed.columns
    
    if has_ground_truth:
        known_mask = df_processed['behavioral_category'].isin(le.classes_)
        if not known_mask.all():
            print(f"⚠ Warning: Dropping {sum(~known_mask)} rows with unknown labels.")
            df_processed = df_processed[known_mask]
            X_test = df_processed[best_features].values
        y_true = le.transform(df_processed['behavioral_category'])
    else:
        y_true = np.zeros(len(X_test))

    # Create Windowed Dataset
    test_dataset = WindowedTimeSeriesDataset(X_test, y_true, window_size=best_window)
    batch_size = best_params.get("batch_size", 64)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 5. Reconstruct Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")

    # 5. Reconstruct Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")

    if best_model_name == "LSTM":
        model = LSTMClassifier(
            n_features=X_test.shape[1],
            hidden_dim=best_params["hidden_dim"],
            n_layers=best_params["n_layers"],
            n_classes=n_classes,
            dropout=best_params["dropout"]
        )
        
    elif best_model_name == "BiLSTM": # <--- ADD THIS BLOCK
        model = BiLSTMClassifier(
            n_features=X_test.shape[1],
            hidden_dim=best_params["hidden_dim"],
            n_layers=best_params["n_layers"],
            n_classes=n_classes,
            dropout=best_params["dropout"]
        )

    elif best_model_name == "GRU": # <--- ADD THIS BLOCK
        model = GRUClassifier(
            n_features=X_test.shape[1],
            hidden_dim=best_params["hidden_dim"],
            n_layers=best_params["n_layers"],
            n_classes=n_classes,
            dropout=best_params["dropout"]
        )
        
    elif best_model_name == "CNN":
        model = CNN1DClassifier(
            n_features=X_test.shape[1],
            n_filters=best_params["n_filters"],
            kernel_size=best_params["kernel_size"],
            n_classes=n_classes
        )
    else:
        raise ValueError(f"Unknown model architecture: {best_model_name}")

    # LOAD WEIGHTS (Works for .pth and .pt)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    # LOAD WEIGHTS (Works for .pth and .pt)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 6. Inference Loop
    print("\n🚀 Starting Inference...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    # 7. Results
    pred_labels = le.inverse_transform(all_preds)
    
    if has_ground_truth:
        true_labels = le.inverse_transform(all_labels)
        acc = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average='weighted')
        print(f"\n✅ Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")
        print(classification_report(true_labels, pred_labels))
        results_output = pd.DataFrame({'True_Label': true_labels, 'Predicted_Label': pred_labels})
    else:
        print(f"\n✅ Generated {len(pred_labels)} predictions.")
        results_output = pd.DataFrame({'Predicted_Label': pred_labels})

    out_file = os.path.join(output_dir, "test_predictions.csv")
    results_output.to_csv(out_file, index=False)
    print(f"💾 Saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test model on .pth or .pt files")
    parser.add_argument("--data_dir", required=True, help="Path to NEW zip files")
    parser.add_argument("--output_dir", required=True, help="Path to folder with results.csv and model file")
    args = parser.parse_args()
    
    run_inference(args.data_dir, args.output_dir)