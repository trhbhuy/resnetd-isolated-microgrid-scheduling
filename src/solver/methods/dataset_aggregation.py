import numpy as np
from typing import Dict

def dataset_aggregation(records: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Process the loaded results into a single dataset for model training.

    Args:
        results (dict): Dictionary containing the optimization results.

    Returns:
        dict: A dictionary containing X_data (features) and y_data (target) with y_data having shape (..., 1).
    """
    # Combine data_seq features
    data_seq = np.vstack([records['time_step'], records['p_net'], records['soc_ess_prev']]).T

    # Define the target variable label
    label = records['p_ess']

    # Create the dataset dictionary
    dataset = {
        'data_seq': data_seq,
        'label': label
    }

    return dataset