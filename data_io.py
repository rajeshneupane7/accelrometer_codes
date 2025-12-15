# data_io.py
import os
import zipfile
import tempfile
import pandas as pd

def load_accelerometer_data(main_path):
    """
    Extracts ZIP files, finds Processed folders, merges Excel files.
    """
    all_files = []

    if not os.path.exists(main_path):
        raise FileNotFoundError(f"Path not found: {main_path}")

    for zip_file in os.listdir(main_path):
        if not zip_file.endswith(".zip"):
            continue

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                with zipfile.ZipFile(os.path.join(main_path, zip_file), "r") as zf:
                    zf.extractall(temp_dir)

                for root, dirs, _ in os.walk(temp_dir):
                    for d in dirs:
                        if d.startswith("Processed"):
                            proc_path = os.path.join(root, d)
                            for f in os.listdir(proc_path):
                                if f.endswith((".xls", ".xlsx")):
                                    all_files.append(
                                        pd.read_excel(os.path.join(proc_path, f))
                                    )
            except Exception as e:
                print(f"ZIP error ({zip_file}): {e}")

    if not all_files:
        return pd.DataFrame()

    df = pd.concat(all_files, ignore_index=True)
    df["local_ts"] = pd.to_datetime(df["local_ts"])
    return df
