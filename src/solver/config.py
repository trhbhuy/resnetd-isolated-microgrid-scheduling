import os
import pandas as pd
import numpy as np

# Define base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
GENERATED_DATA_DIR = os.path.join(DATA_DIR, 'generated')

# Paths to specific data files
NETWORK_DATA_FILE = os.path.join(RAW_DATA_DIR, 'case6.xlsx')

if not os.path.exists(NETWORK_DATA_FILE):
    raise FileNotFoundError(f"The data file {NETWORK_DATA_FILE} was not found.")

# Load data from Excel
bus_data = pd.read_excel(NETWORK_DATA_FILE, sheet_name='bus').values
branch_data = pd.read_excel(NETWORK_DATA_FILE, sheet_name='branch').values

# Base values and general parameters
BASE_MVA = 1e3
B_NUM = len(bus_data)
BR_NUM = len(branch_data)
B_SET = np.arange(B_NUM)

# Time settings
T_NUM = 24  # 24 hours
T_SET = np.arange(T_NUM)
DELTA_T = 24 / T_NUM  # Time step in hours

# Voltage limits
V_MIN = 0.95
V_MAX = 1.05

# Branch (line) data
f = branch_data[:, 0].astype('int')  # From bus
t = branch_data[:, 1].astype('int')  # To bus
BRANCH_NUM = len(f)
r = branch_data[:, 2]  # Resistance
x = branch_data[:, 3]  # Reactance

# Define branch connections
BRANCH_IJ = list(zip(f, t))
BRANCH_JI = list(zip(t, f))
BRANCH_IJ_ALL = BRANCH_IJ + BRANCH_JI

# Resistance and reactance dictionaries
R_IJ = dict(zip(BRANCH_IJ, r))
X_IJ = dict(zip(BRANCH_IJ, x))

# Maximum line current dictionary
I_IJ_MAX = branch_data[:, 13]
I_MAX = dict(zip(BRANCH_IJ, I_IJ_MAX))

# Node connections
NINSERT_SET = {node: branch_data[branch_data[:, 0] == node][:, 1].astype('int').tolist() for node in B_SET}
NOUT_SET = {node: branch_data[branch_data[:, 1] == node][:, 0].astype('int').tolist() for node in B_SET}
N_ALL_SET = {node: branch_data[branch_data[:, 0] == node][:, 1].astype('int').tolist() + 
                   branch_data[branch_data[:, 1] == node][:, 0].astype('int').tolist() for node in B_SET}

# Location of microgrid components
IF_NODE = 0  # Interface node
FL_NODE = 1  # Flexible load node
ESS_NODE = 2  # Energy storage system node
PV_NODE = 3  # Solar PV node
WG_NODE = 4  # Wind generator node
DEG_NODE = 5  # Diesel engine generator node

# Solar PV parameters
P_PV_RATE = 150  # Rated power of PV (kW)
N_PV = 0.167  # Efficiency factor for PV
PHI_PV = 0.24  # Loss factor for PV

# Wind generator parameters
P_WG_RATE = 150  # Rated power of WG (kW)
N_WG = 0.88  # Efficiency factor for WG
PHI_WG = 0.19  # Loss factor for WG

# Diesel engine generator (DEG) parameters
P_DEG_MAX = 600  # Maximum power output (kW)
P_DEG_MIN = 20  # Minimum power output (kW)
R_DEG = 200  # Ramp rate (kW/h)
W1_DEG = 1.3  # Cost function coefficient
W2_DEG = 0.0304  # Cost function coefficient
W3_DEG = 0.00104  # Cost function coefficient

# Energy storage system (ESS) parameters
P_ESS_CH_MAX = 100  # Maximum charging power (kW)
P_ESS_DCH_MAX = 100  # Maximum discharging power (kW)
ESS_DOD = 0.8  # Depth of discharge
SOC_ESS_MAX = 200  # Maximum state of charge (kWh)
N_ESS_CH = 0.98  # Charging efficiency
N_ESS_DCH = 0.98  # Discharging efficiency
SOC_ESS_MIN = (1 - ESS_DOD) * SOC_ESS_MAX  # Minimum state of charge considering DoD
SOC_ESS_SETPOINT = SOC_ESS_MAX  # Reference state of charge for ESS
PHI_ESS = 1e-6  # Loss factor for ESS
SOC_ESS_THRESHOLD = SOC_ESS_SETPOINT - N_ESS_CH * P_ESS_CH_MAX
PENALTY_COEFFICIENT = 100  # Factor to penalize bad actions


# Flexible load (FL) parameters
LS_SETTING = 0.7  # Load shedding setting
PHI_LS_1 = 0.45  # Loss factor 1 for load shedding
PHI_LS_2 = 0.50  # Loss factor 2 for load shedding

def create_directories():
    """
    Ensure that all necessary directories exist. If not, create them.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(GENERATED_DATA_DIR, exist_ok=True)

def print_config():
    """
    Utility function to print the current configuration settings.
    Useful for debugging and verification.
    """
    print("Microgrid Configuration Settings:")
    print(f"Base MVA: {BASE_MVA}")
    print(f"Number of buses: {B_NUM}")
    print(f"Number of branches: {BR_NUM}")
    print(f"Time horizon: {T_NUM} hours")
    print(f"Time step: {DELTA_T} hours")
    print(f"Voltage limits: {V_MIN} - {V_MAX} p.u.")
    print(f"Max SOC for ESS: {SOC_ESS_MAX} kWh")
    print(f"Min SOC for ESS: {SOC_ESS_MIN} kWh")
    # Add more configuration details as needed

# Automatically create directories when the module is imported
create_directories()

if __name__ == "__main__":
    print_config()