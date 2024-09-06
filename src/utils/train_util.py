import os
import torch
import numpy as np
from torch.utils.data import TensorDataset

class MeanMeter:
    """Computes and stores the mean of values."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets the meter by clearing all stored values."""
        self.values = []

    def update(self, val):
        """Adds a new value to the list."""
        self.values.append(val)

    def compute(self):
        """Computes the mean of all stored values."""
        if not self.values:
            return 0.0  # Safeguard against division by zero
        return np.mean(self.values)

def data_loading(arg, is_train=True):
    """
    Load dataset for training or validation.

    Args:
        args (argparse.Namespace): Arguments containing data directory information.
        is_train (bool): Whether to load training data or validation data.

    Returns:
        TensorDataset: A dataset containing input features and labels.
    """
    # Set data path
    data_dir = os.path.join(arg.data_dir, arg.rid)
    if is_train:
        data_path = os.path.join(data_dir, 'train.pt')
    else:
        data_path = os.path.join(data_dir, 'val.pt')
        
    # Load data
    data = torch.load(data_path)
    X_data = data['data_seq']
    y_data = data['label']

    return TensorDataset(X_data, y_data)    

def save_model(model, optimizer, opt, epoch, save_file):
    """
    Save the model's state dictionary and optimizer's state dictionary to a file.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        opt (argparse.Namespace): Arguments used for training configuration.
        epoch (int): The current epoch number.
        save_file (str): The path where the model should be saved.
    """
    print('==> Saving model...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    print(f'Model saved successfully at {save_file}')
    del state  # Explicitly delete to free up memory
