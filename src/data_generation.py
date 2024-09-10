import time
import logging
from solver.platform.microgrid import Microgrid
from solver.methods.data_loader import load_data
from solver.methods.optimization import run_optim
from solver.methods.dataset_aggregation import dataset_aggregation
from utils.common_util import save_results, save_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_optimization(microgrid, data):
    """Run scenarios and aggregate the results."""
    # Run scenarios
    logging.info("Running scenarios...")
    results = run_optim(microgrid, data)

    # Save results
    logging.info("Saving scenario results...")
    save_results(results)

    return results

def process_datasets(results):
    """Aggregate and save datasets."""
    # Aggregate datasets
    logging.info("Aggregating datasets...")
    dataset = dataset_aggregation(results, feature_keys=['time_step', 'p_net', 'soc_ess_prev'], label_keys=['p_ess'])

    # Log dataset shapes
    logging.info(f"Dataset shapes: data_seq: {dataset['data_seq'].shape}, label: {dataset['label'].shape}")

    # Save datasets
    logging.info("Saving aggregated datasets...")
    save_dataset(dataset, "dataset.pkl")

def main():
    """Main entry point for the data generation process."""
    # Timing the execution
    start_time = time.time()

    try:
        # Step 1: Load data
        logging.info("Loading data...")
        data = load_data()
        if data is None:
            raise ValueError("Failed to load data.")

        # Step 2: Initialize the microgrid platform
        logging.info("Initializing Microgrid...")
        microgrid = Microgrid()

        # Step 3: Process optimization process
        results = process_optimization(microgrid, data)

        # Step 4: Process and save datasets
        process_datasets(results)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Log the total elapsed time
        elapsed_time = time.time() - start_time
        logging.info(f"Total elapsed time: {elapsed_time:.2f} seconds")

#python3 src/data_generation.py
# Entry point
if __name__ == "__main__":
    main()