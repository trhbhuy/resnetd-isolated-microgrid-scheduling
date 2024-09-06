# src/optimization/run_scenario.py

import logging
import numpy as np
from typing import Dict
from .feature_engineering import feature_engineering
from .util import reshape_data

def run_scenario(microgrid: object, data: Dict[str, np.ndarray], num_scenarios: int) -> Dict[str, np.ndarray]:
    """Process multiple scenarios and store results in a dictionary.

    Args:
        microgrid (Microgrid): The microgrid object to perform optimization.
        data (dict): Solar, wind, and load data.
        num_scenarios (int): Number of scenarios to process.

    Returns:
        dict: A dictionary containing results for all scenarios.
    """
    # Initialize dictionary to store results
    records = {key: [] for key in [
        'ObjVal', 'p_deg', 'u_deg', 'p_pv', 'p_wg', 'p_ess_ch', 'p_ess_dch',
        'u_ess_ch', 'u_ess_dch', 'soc_ess', 'p_ls_1', 'p_ls_2', 'p_it', 
        'q_it', 'l_P_it', 'l_Q_it', 'v_it'
    ]}

    # Define number of time steps
    T_num = microgrid.T_num

    # Process data and record results for all scenarios
    for scn in range(num_scenarios):
        start_idx = scn * T_num
        end_idx = start_idx + T_num

        # Perform optimization for the current scenario
        results = microgrid.optim(
            data['p_pv_max'][start_idx:end_idx],
            data['p_wg_max'][start_idx:end_idx],
            data['p_if'][start_idx:end_idx],
            data['p_fl_1'][start_idx:end_idx],
            data['p_fl_2'][start_idx:end_idx]
        )

        # Append each result to the corresponding list
        for key in records.keys():
            records[key].append(results[key])

        # Log progress
        logging.info(f"Scenario {scn}: Optimal ObjVal = {results['ObjVal']:.4f}")

    # Convert lists to arrays for final storage
    records = {key: np.array(records[key]) for key in records.keys()}

    # Reshape input data and store in the dictionary
    scenario_data = reshape_data(data, num_scenarios)
    records.update(scenario_data)

    # Calculate and add auxiliary variables
    records = feature_engineering(microgrid, records)

    return records