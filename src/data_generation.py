import time
import logging
from solver.platform.microgrid import Microgrid
from solver.methods.data_loader import load_data
from solver.methods.run_scenario import run_scenario
from solver.methods.dataset_aggregation import dataset_aggregation

from utils.common_utils import save_results, save_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Timing the execution
    start_time = time.time()

    # Load data
    data = load_data()

    # Initialize Microgrid
    microgrid = Microgrid()

    # Define the number of scenarios
    num_scenarios = len(data['p_pv_max']) // microgrid.T_num

    # Step 1: Process scenarios and store results
    results = run_scenario(microgrid, data, num_scenarios)

    # Save the results to CSV files
    save_results(results)

    # Step 2: Process the dataset to obtain training dataset
    dataset = dataset_aggregation(results)

    # Log progress
    logging.info(f"data_seq shape: {dataset['data_seq'].shape}, label shape: {dataset['label'].shape}")

    # Save the dataset
    save_dataset(dataset, "dataset.pkl")

    # Print elapsed time
    elapsed_time = time.time() - start_time
    logging.info(f"Total elapsed time: {elapsed_time:.2f} seconds")

#python3 src/data_generation.py
if __name__ == "__main__":
    main()