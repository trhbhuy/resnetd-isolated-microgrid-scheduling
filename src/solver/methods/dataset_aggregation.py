import numpy as np
from typing import Dict, List

import numpy as np
from typing import Dict, List

def dataset_aggregation(records: Dict[str, np.ndarray], feature_keys: List[str], label_keys: List[str]) -> Dict[str, np.ndarray]:
    """Process the loaded results into a single dataset for model training.

    Args:
        records (dict): Dictionary containing the optimization results.
        feature_keys (list): List of keys to be used as features.
        label_keys (str or list): Key or list of keys to be used as the label(s).

    Returns:
        dict: A dictionary containing X_data (features) and y_data (target) with y_data having shape (..., 1).
    """
    # Combine the selected features
    data_seq = np.vstack([records[key].ravel() for key in feature_keys]).T

    # Define the target variable label
    label = np.vstack([records[key].ravel() for key in label_keys]).T

    # Create the dataset dictionary
    dataset = {
        'data_seq': data_seq,
        'label': label
    }

    return dataset