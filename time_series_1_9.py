import argparse

import os

import ast

import pandas as pd

import numpy as np

import torch



from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from torch.utils.data import DataLoader



#from pipeline_utils import load_all_zips, AccelPipeline

from pipeline_utils  import load_all_zips, AccelPipeline
from time_series_utils import (

    DeepActivityModeler,
    WindowedTimeSeriesDataset,
    LSTMClassifier,
    CNN1DClassifier,
    BiLSTMClassifier,  # <--- ADD THIS
    GRUClassifier

)



# ======================================================

# TIME SERIES TRAINING PIPELINE

# ======================================================

def run_timeseries_pipeline(pipeline, thresholds, resample_intervals,window_sizes, n_trials):

    temp_results = []

    resampled_map = {}



    for t in resample_intervals:
        for thresh in thresholds:

            print(f"\n⏱ Running resample = {t}s")



            df_ready = pipeline.resample_and_label(interval_seconds=t, coherence_threshold=thresh)



            if df_ready is None or df_ready.empty:

                continue

            if "behavioral_category" not in df_ready.columns:

                continue





            for window in window_sizes:
                modeler = DeepActivityModeler(df_ready, target_col="behavioral_category")



                results_df = modeler.run_optuna_experiments(

                window_size=window,

                n_trials=n_trials

                    )



                if results_df is None or results_df.empty:

                    continue



                results_df["resample_interval"] = t
                results_df["window_size"] = window
                results_df['threshold']= thresh
                resampled_map[(t, window)]= df_ready
                temp_results.append(results_df)


            if not temp_results:

                raise RuntimeError("No valid time-series models trained.")



    final_df = pd.concat(temp_results, ignore_index=True)



    best_row = final_df.sort_values("Best_F1", ascending=False).iloc[0]



    best_interval = best_row["resample_interval"]
    best_window = best_row["window_size"]
    best_model = best_row["Model"]
    best_features = ast.literal_eval(best_row["Features_Used"])
    best_params = ast.literal_eval(best_row["Best_Params"])



    df_best = resampled_map[(best_interval, best_window)]



    X = df_best[best_features].values

    le = LabelEncoder()

    y = le.fit_transform(df_best["behavioral_category"])

    n_classes = len(np.unique(y))



    device = "cuda" if torch.cuda.is_available() else "cpu"



    dataset = WindowedTimeSeriesDataset(X, y, window_size=best_window)

    loader = DataLoader(

        dataset,

        batch_size=best_params.get("batch_size", 64),

        shuffle=True

    )



    if best_model == "LSTM":

        model = LSTMClassifier(

            n_features=X.shape[1],

            hidden_dim=best_params["hidden_dim"],

            n_layers=best_params["n_layers"],

            n_classes=n_classes,

            dropout=best_params["dropout"]

        )

    elif best_model == "CNN":

        model = CNN1DClassifier(

            n_features=X.shape[1],

            n_filters=best_params["n_filters"],

            kernel_size=best_params["kernel_size"],

            n_classes=n_classes

        )
    elif best_model=='GRU':
        model = GRUClassifier(
            n_features=X.shape[1],
            hidden_dim=best_params["hidden_dim"],
            n_layers=best_params["n_layers"],
            n_classes=n_classes,
            dropout=best_params["dropout"]
        )

    elif best_model=='BiLSTM':
        model = BiLSTMClassifier(
            n_features=X.shape[1],
            hidden_dim=best_params["hidden_dim"],
            n_layers=best_params["n_layers"],
            n_classes=n_classes,
            dropout=best_params["dropout"]
        )
    else:

        raise ValueError(f"Unknown model {best_model}")



    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])

    loss_fn = torch.nn.CrossEntropyLoss()



    model.train()

    for _ in range(20):

        for xb, yb in loader:

            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()

            loss = loss_fn(model(xb), yb)

            loss.backward()

            optimizer.step()



    return model, le, final_df





# ======================================================

# MAIN (SLURM ENTRY POINT)

# ======================================================

def main(args):

    os.makedirs(args.output_dir, exist_ok=True)



    print("📥 Loading data...")

    data = load_all_zips(args.data_dir)



    pipeline = AccelPipeline(data)

    pipeline.convert_to_gravity()

    pipeline.calc_odba()

    pipeline.calc_vedba()



    resample_intervals = [
        int(x) for x in args.resample_intervals.split(",")
    ]
    window_sizes = [
        int(x) for x in args.windows.split(",")
    ]

    thresholds=[
        float(x) for x in args.thresholds.split(",")
    ]

    model, le, results_df = run_timeseries_pipeline(
        pipeline=pipeline,
        resample_intervals=resample_intervals,
        window_sizes=window_sizes,
        n_trials=args.n_trials,
        thresholds=thresholds)


    results_path = os.path.join(args.output_dir, "results.csv")

    results_df.to_csv(results_path, index=False)



    torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))

    np.save(os.path.join(args.output_dir, "label_classes.npy"), le.classes_)



    print("✅ Training complete")

    print(f"📁 Results saved to {args.output_dir}")





if __name__ == "__main__":

    parser = argparse.ArgumentParser()



    parser.add_argument("--data_dir", required=True, help="Directory with zip files")

    parser.add_argument("--resample_intervals", default="20")

    parser.add_argument("--windows", default="25", help="Comma-separated windows")

    parser.add_argument("--n_trials", type=int, default=20, help="Optuna trials")

    parser.add_argument("--output_dir", default="results", help="Output directory")

    parser.add_argument("--thresholds", default="0.8", help="purity to select the windows")



    args = parser.parse_args()

    main(args)

