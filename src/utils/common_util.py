import os
import pandas as pd
import pickle
import logging
from typing import Dict

from solver.config import PROCESSED_DATA_DIR, GENERATED_DATA_DIR

def save_pickle(obj, file_path: str):
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

def load_pickle(file_path: str):      
    # Load the label scaler
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def save_results(results: dict, output_dir: str = PROCESSED_DATA_DIR):
    """Save optimization results to CSV files.

    Args:
        results (dict): The dictionary containing optimization results.
        output_dir (str): Directory to save the CSV files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for key, value in results.items():
        output_path = os.path.join(output_dir, f"{key}.csv")
        if value.ndim > 2:
            value = value.reshape(value.shape[0], -1)
        pd.DataFrame(value).to_csv(output_path, index=False)
        logging.info(f"Saved {key} to {output_path}")

def save_dataset(dataset: dict, filename: str):
    """Save the dataset to a file using pickle.

    Args:
        dataset (dict): The dataset containing X_data and y_data.
        filename (str): The filename to save the dataset to.
    """
    file_path = os.path.join(GENERATED_DATA_DIR, filename)
    save_pickle(dataset, file_path)
    logging.info(f"Dataset saved to {file_path}")
