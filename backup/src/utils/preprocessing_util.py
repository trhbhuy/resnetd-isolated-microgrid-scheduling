import numpy as np
import torch

from .common_utils import save_pickle, load_pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

def scaling_data(data: np.ndarray, method: str ='MinMax') -> tuple:
    """
    Scale the data using the specified method.

    Args:
        data (np.ndarray): The data to scale.
        method (str): The scaling method to use. Options are 'MinMax' or 'Standard'.

    Returns:
        tuple: (scaled_data, scaler) where scaled_data is the transformed data and scaler is the scaler object.
    """
    if method == 'MinMax':
        scaler = MinMaxScaler()
    elif method == 'Standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Unsupported scaling method")

    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def split_data(data: np.ndarray, labels: np.ndarray, split_type: str = 'random', test_size: float = 0.2, random_state: int = 42, n_splits: int = 5) -> tuple:
    """
    Split the data into train and validation sets.

    Args:
        data (torch.Tensor or np.ndarray): The feature data to split.
        labels (torch.Tensor or np.ndarray): The labels corresponding to the data.
        split_type (str): The type of split to use ('random', 'kfold', 'stratified').
        test_size (float): The proportion of the dataset to include in the test split (only for 'random').
        random_state (int): Random seed for reproducibility.
        n_splits (int): Number of splits for cross-validation (only for 'kfold' or 'stratified').

    Returns:
        If split_type is 'random':
            tuple: data_train, data_val, label_train, label_val
        If split_type is 'kfold' or 'stratified':
            generator: yields (train_idx, val_idx) for each fold
    """
    if split_type == 'random':
        # Random train/test split
        data_train, data_val, label_train, label_val = train_test_split(data, labels, test_size=test_size, random_state=random_state)
    
    elif split_type == 'kfold':
        # KFold cross-validation split
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, val_idx in kf.split(data):
            data_train, data_val = data[train_idx], data[val_idx]
            label_train, label_val = labels[train_idx], labels[val_idx]
            break  # Only use the first split

    elif split_type == 'stratified':
        # Stratified KFold cross-validation split (useful for classification tasks)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, val_idx in skf.split(data, labels):
            data_train, data_val = data[train_idx], data[val_idx]
            label_train, label_val = labels[train_idx], labels[val_idx]
            break  # Only use the first split

    else:
        raise ValueError(f"Unsupported split_type: {split_type}. Choose from 'random', 'kfold', or 'stratified'.")

    return data_train, data_val, label_train, label_val

def save_scaler(scaler, file_path: str):
    """
    Save a scaler object to a file using pickle.

    Parameters:
    scaler (object): The scaler object to be saved.
    file_path (str): The file path where the scaler should be saved.
    """
    save_pickle(scaler, file_path)

def data_loader(file_path: str) -> dict:
    """Load the dataset from a pickle file (expand in future with options)."""
    dataset = load_pickle(file_path)
    return dataset

def save_data(train_data: dict, val_data: dict, train_path: str, val_path: str):
    """Save the train and validation sets to .pt files."""
    torch.save(train_data, train_path)
    torch.save(val_data, val_path)