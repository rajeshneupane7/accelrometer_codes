import argparse
import os
import sys
import pandas as pd

# Import the library components
from accel_pipeline import AccelPipeline
from experiment_engine import ActivityExperimentLibrary

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Accelerometer Time Series Experiments with LOSO Validation."
    )

    # Data Arguments
    parser.add_argument(
        "--data_dir", 
        type=str, 
        required=True, 
        help="Directory containing zip files or csv files."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="experiment_results", 
        help="Directory to save results and models."
    )

    # Feature Arguments
    parser.add_argument(
        "--skip_odba", 
        action="store_true", 
        help="If set, skip calculating ODBA features."
    )
    parser.add_argument(
        "--skip_vedba", 
        action="store_true", 
        help="If set, skip calculating VeDBA features."
    )

    # Pipeline Arguments
    parser.add_argument(
        "--resample_interval", 
        type=int, 
        default=30, 
        help="Resampling interval in seconds (e.g., 30s windows)."
    )
    parser.add_argument(
        "--window_size", 
        type=int, 
        default=10, 
        help="Window size (in number of resampled steps) for the deep learning models."
    )
    parser.add_argument(
        "--train_threshold", 
        type=float, 
        default=0.7, 
        help="Coherence threshold (0.0 to 1.0) for filtering training data. "
             "Testing data is always raw (unthresholded)."
    )

    # Model & Experiment Arguments
    parser.add_argument(
        "--models", 
        type=str, 
        default="LSTM,BiLSTM,CNN,Transformer",
        help="Comma-separated list of models to test. "
             "Options: LSTM, BiLSTM, CNN, Transformer."
    )
    parser.add_argument(
        "--n_trials", 
        type=int, 
        default=10, 
        help="Number of Optuna trials for hyperparameter tuning per fold."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda", 
        help="Device to use (cuda or cpu)."
    )

    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Initialize Pipeline
    print("Initializing Pipeline...")
    pipeline = AccelPipeline(
        data_dir=args.data_dir,
        calc_odba=not args.skip_odba,
        calc_vedba=not args.skip_vedba
    )

    # 2. Initialize Experiment Engine
    print("Initializing Experiment Engine...")
    experimenter = ActivityExperimentLibrary(pipeline, device=args.device)

    # 3. Parse Models List
    models_list = [m.strip() for m in args.models.split(",")]
    valid_models = ["LSTM", "BiLSTM", "CNN", "Transformer"]
    
    for m in models_list:
        if m not in valid_models:
            print(f"Warning: Unknown model '{m}'. Valid options are {valid_models}")
            sys.exit(1)

    # 4. Run Experiment
    print(f"\n{'='*50}")
    print(f"Starting LOSO Experiment:")
    print(f"  Models:        {models_list}")
    print(f"  Resample:      {args.resample_interval}s")
    print(f"  Window Size:   {args.window_size}")
    print(f"  Train Thresh:  {args.train_threshold}")
    print(f"  Test Thresh:   None (Raw)")
    print(f"  Features:      ODBA={not args.skip_odba}, VeDBA={not args.skip_vedba}")
    print(f"{'='*50}\n")

    results_df = experimenter.run_loso_experiment(
        resample_interval=args.resample_interval,
        window_size=args.window_size,
        thresholds=[args.train_threshold], # Only applies to training
        models_to_test=models_list,
        n_trials=args.n_trials,
        output_dir=args.output_dir
    )

    # 5. Print Summary
    if results_df is not None and not results_df.empty:
        print("\n" + "="*50)
        print("FINAL SUMMARY")
        print("="*50)
        summary = results_df.groupby("Model")[["Test_F1", "Test_Acc"]].agg(['mean', 'std'])
        print(summary)
        
        # Save summary to a text file
        summary_path = os.path.join(args.output_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write("FINAL SUMMARY\n")
            f.write("="*50 + "\n")
            f.write(summary.to_string())
        print(f"\nSummary saved to {summary_path}")
    else:
        print("No results generated. Please check your data and logs.")

if __name__ == "__main__":
    main()