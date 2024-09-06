# src/utils/test_util.py

import os
import numpy as np
import pandas as pd

# Define base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_dataset(args):
    """
    Load the dataset and extract the specified number of test scenarios.

    Args:
        args: An argument object containing the following attributes:
            - data_path (str): Path to the CSV file relative to BASE_DIR.
            - num_test_scenarios (int): Number of test scenarios to extract.

    Returns:
        np.ndarray: A 1D array containing the test scenarios.
    """
    # Construct the full path to the dataset
    data_file_path = os.path.join(BASE_DIR, args.data_path)

    # Load the dataset from the CSV file
    data = pd.read_csv(data_file_path).values.flatten()

    # Validate and extract the test scenarios
    num_train_scenarios = 731
    start_idx = num_train_scenarios
    end_idx = start_idx + args.num_test_scenarios

    return data[start_idx:end_idx]

def cal_metric(actual, prediction):
    """
    Calculates the element-wise and overall Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE)
    between two arrays.

    Parameters:
        actual (np.ndarray): The actual values.
        prediction (np.ndarray): The predicted values.

    Returns:
        tuple: A tuple containing:
            - mae_array (np.ndarray): An array of the element-wise Mean Absolute Errors.
            - mape_array (np.ndarray): An array of the element-wise Mean Absolute Percentage Errors (expressed as percentages).
            - overall_mae (float): The overall Mean Absolute Error across all elements.
            - overall_mape (float): The overall Mean Absolute Percentage Error across all elements (expressed as a percentage).
    """
    # Ensure the inputs are NumPy arrays
    actual = np.array(actual)
    prediction = np.array(prediction)

    # Calculate element-wise MAE
    mae_array = np.abs(actual - prediction)

    # Calculate element-wise MAPE, handling division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_array = np.abs((actual - prediction) / actual) * 100
        mape_array[np.isinf(mape_array)] = np.nan  # Replace inf values with NaN if division by zero occurs

    # Calculate overall MAE and MAPE
    overall_mae = np.nanmean(mae_array)
    overall_mape = np.nanmean(mape_array)

    return {
        'mae_array': mae_array,
        'mape_array': mape_array,
        'overall_mae': overall_mae,
        'overall_mape': overall_mape
        }