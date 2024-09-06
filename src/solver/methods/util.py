import numpy as np
from typing import Dict

def reshape_data(data: Dict[str, np.ndarray], num_scenarios: int) -> Dict[str, np.ndarray]:
    """Reshape specific data for easier scenario-based processing.

    Args:
        data (dict): Dictionary containing raw data.
        num_scenarios (int): Number of scenarios to reshape.

    Returns:
        dict: Reshaped data for scenario-based analysis.
    """
    reshaped_data = {}
    for key, value in data.items():
        reshaped_data[key] = value.reshape((num_scenarios, -1))
    return reshaped_data
