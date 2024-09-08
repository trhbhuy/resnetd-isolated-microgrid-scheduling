import numpy as np
from typing import Dict

def feature_engineering(microgrid: object, records: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculate and add auxiliary variables such as time_step, net load, and previous SOC to the records.

    Args:
        microgrid (Microgrid): An instance of the Microgrid class containing relevant parameters.
        records (Dict[str, np.ndarray]): A dictionary containing optimization results.

    Returns:
        Dict[str, np.ndarray]: Updated records with additional auxiliary variables.
    """
    # Extract relevant parameters
    T_num = microgrid.T_num
    num_scenarios = len(records['ObjVal'])
    soc_ess_setpoint = microgrid.ess.soc_ess_setpoint

    # Create the time_step array (repeated for each scenario)
    records['time_step'] = np.tile(np.arange(T_num), num_scenarios)

    # Calculate Net load
    records['p_net'] = (records['p_if'] + records['p_fl_1'] + records['p_fl_2'] - (records['p_pv_max'] + records['p_wg_max'])).ravel()

    # Calculate the SOC of ESS at the previous time step
    soc_ess_flattened = records['soc_ess'].ravel()
    records['soc_ess_prev'] = np.roll(soc_ess_flattened, shift=1)
    records['soc_ess_prev'][0] = soc_ess_setpoint

    # Calculate ESS charging/discharging power
    records['p_ess'] = (records['p_ess_ch'] - records['p_ess_dch']).reshape(-1, 1)

    return records