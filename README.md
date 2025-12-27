# accelrometer_codes

This repository contains code and resources for processing, analyzing, and extracting features from accelerometer data. The main focus is on building pipelines to import, clean, and process raw accelerometer files and develop time-series machine learning models for subsequent analysis.

## Repository Structure

- **data_io.py**  
  Scripts for reading, importing, and preparing accelerometer datasets.

- **feature_pipeline.py**  
  Main feature engineering pipeline, with utilities/functions for extracting features from raw accelerometer data.

- **timeseries_models.py**  
  Contains models and helper code for working with time-series data, such as deep learning or statistical approaches.

- **processing.ipynb**  
  Jupyter Notebook for data exploration, preprocessing, and visualizations of the accelerometer data.

- **processing_with_torch.ipynb**  
  Shows how to use PyTorch for deep learning-based processing of accelerometer data.

- **processing_with_ts.ipynb**  
  Notebook focused on time-series data processing and experiments.

- **run_experiments.ipynb**  
  Contains scripts for running various machine learning experiments on the processed data.

- **initial_results.csv, final_combined_results.csv, pytorch_results.csv, timeseries_results.csv**  
  Output and result files from different stages and experiments.

- **__pycache__/**  
  Python cache files (auto-generated, not hand-edited).

## Getting Started

Clone the repo:
```bash
git clone https://github.com/rajeshneupane7/accelrometer_codes.git
cd accelrometer_codes
```

Explore the Jupyter notebooks for step-by-step workflows. Python scripts provide modularized functions and pipelines.

## Requirements

- Python 3.x
- Jupyter Notebook
- numpy, pandas, scikit-learn
- torch (for deep learning with PyTorch)
- Additional requirements will vary depending on the notebook (see imports in individual `.ipynb` files).

## Usage

1. Prepare your accelerometer data files.
2. Use the scripts and notebooks to process and extract features.
3. Experiment with different models using the provided pipelines.
4. Consult the CSV files for experiment results, or run your own.

## License

This repository does not yet specify an open-source license.

---

*Repository maintained by [rajeshneupane7](https://github.com/rajeshneupane7).*
