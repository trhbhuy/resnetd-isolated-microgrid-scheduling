# src/optimization/util.py

import os
import numpy as np
from typing import Dict, Callable, Tuple, Optional

from ..config import GENERATED_DATA_DIR
from ..utils.file_util import load_pickle
from ..utils.numeric_util import get_boundary_tolerance, get_deviation

def scaler_loader():
    """Load the data and label scalers from the generated files."""

    # Paths to the scaler files
    data_scaler_path = os.path.join(GENERATED_DATA_DIR, 'data_scaler.pkl')
    label_scaler_path = os.path.join(GENERATED_DATA_DIR, 'label_scaler.pkl')

    # Load the data scaler
    data_scaler = load_pickle(data_scaler_path)

    # Load the label scaler
    label_scaler = load_pickle(label_scaler_path)
            
    return data_scaler, label_scaler

def check_boundary_constraint(value, lower_limit, upper_limit):
    """
    Calculate the boundary violation if the value is outside the specified limits.

    Args:
        value (float): The value to check.
        lower_limit (float): The minimum allowable value.
        upper_limit (float): The maximum allowable value.

    Returns:
        float: The calculated boundary violation if the value is out of bounds, otherwise 0.0.
    """
    return get_boundary_tolerance(value, lower_limit, upper_limit)

def check_ramp_constraint(new_value, prior_value, ramp_rate):
    """
    Check if the new value violates the ramp rate constraint based on the prior value.

    Args:
        new_value (float): The value to check.
        prior_value (float): The value from the prior timestep.
        ramp_rate (float): The maximum allowable change between successive values.

    Returns:
        float: The calculated boundary violation if the ramp rate is exceeded, otherwise 0.0.
    """
    # Calculate the permissible range for the new value based on the ramp rate
    min_allowed_value = prior_value - ramp_rate
    max_allowed_value = prior_value + ramp_rate

    # Check if the new value is within the allowable range
    tolerance = get_boundary_tolerance(new_value, min_allowed_value, max_allowed_value)

    return tolerance

def check_setpoint(value, setpoint, tolerance_threshold=1e-5):
    """
    Calculate the deviation of a value from its setpoint.

    Args:
        value (float): The current value to check.
        setpoint (float): The target setpoint value.
        tolerance_threshold (float, optional): An allowable tolerance range. Defaults to 0.

    Returns:
        float: The deviation from the setpoint. Returns 0 if within the tolerance threshold.
    """
    return get_deviation(value, setpoint, tolerance_threshold)

def generate_pla_points(lb: float, ub: float, func: Callable[[float], float], npts: int = 101) -> Tuple[np.ndarray, np.ndarray]:
    """Generate piecewise linear approximation (PLA) points for a given function."""
    ptu = np.linspace(lb, ub, npts)
    ptf = np.array([func(u) for u in ptu])
    return ptu, ptf

def calculate_F_deg(p_deg: float, w1: float, w2: float, w3: float) -> float:
    """Calculate the fuel consumption for the Diesel Engine Generator (DEG)."""
    return w3 * p_deg**2 + w2 * p_deg + w1

def calculate_F_ess(p_ess: float) -> float:
    """Calculate the operation cost for the Energy Storage System (ESS)."""
    return p_ess**2

def extract_results(variable, inner_set: np.ndarray, outer_set: Optional[np.ndarray] = None) -> np.ndarray:
    """Extract results from a given variable for the specified sets."""
    if outer_set is not None:
        return np.array([[variable[ii, tt].X for tt in inner_set] for ii in outer_set])
    return np.array([variable[tt].X for tt in inner_set])
