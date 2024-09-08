import os
import argparse
import torch
import logging
from utils.preprocessing_util import scaling_data, split_data, save_scaler, data_loader, save_data

# Define base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    """Parse command-line arguments for data preprocessing."""
    parser = argparse.ArgumentParser(description='Arguments for data preprocessing')

    # Base directories
    parser.add_argument('--base_dir', type=str, default=BASE_DIR, help='Base directory for the project')
    parser.add_argument('--data_dir', type=str, default='data', help='Relative path to the data directory')
    parser.add_argument('--generated_dir', type=str, default='generated', help='Subdirectory for generated data')

    # Dataset and processing arguments
    parser.add_argument('--dataset_name', type=str, default='dataset.pkl', help='Filename for the dataset')
    parser.add_argument('--sub_dirs', type=str, nargs='+', default='', help='List of subdirectories for specific data types')
    parser.add_argument('--num_train_samples', type=int, default=17544, help='Number of training samples (default: 2 years)')
    parser.add_argument('--scaler_type', type=str, default='MinMax', help='Type of scaler to use (MinMax or Standard)')
    parser.add_argument('--split_type', type=str, default='random', help='Split type: random, kfold, or stratified')
    parser.add_argument('--validation_split_size', type=float, default=0.2, help='Fraction of data for validation split')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')

    return parser.parse_args()

def set_paths(args):
    """Set up paths for data loading, saving, and directories."""
    # Directory for generated data
    GENERATED_DIR = os.path.join(args.base_dir, args.data_dir, args.generated_dir, args.sub_dir)

    # Ensure the directory exists
    os.makedirs(GENERATED_DIR, exist_ok=True)

    # Paths for data loading and saving
    args.data_path = os.path.join(GENERATED_DIR, args.dataset_name)
    args.train_save_path = os.path.join(GENERATED_DIR, 'train.pt')
    args.val_save_path = os.path.join(GENERATED_DIR, 'val.pt')
    args.data_scaler_save_path = os.path.join(GENERATED_DIR, 'data_scaler.pkl')
    args.label_scaler_save_path = os.path.join(GENERATED_DIR, 'label_scaler.pkl')

def preprocess_data(args):
    """Preprocess the data including loading, scaling, and saving."""
    # Step 1: Load the dataset
    logging.info(f"Loading dataset from {args.data_path}")
    dataset = data_loader(args.data_path)
    data_seq, label = dataset['data_seq'], dataset['label']

    # Step 2: Split the dataset into train and test (first two years as train, last year as test)
    logging.info("Splitting dataset into train and test sets")
    data_train, data_test = data_seq[:args.num_train_samples], data_seq[args.num_train_samples:]
    label_train, label_test = label[:args.num_train_samples], label[args.num_train_samples:]

    # Step 3: Scale the data
    logging.info("Scaling training data")
    data_train_scaled, data_scaler = scaling_data(data_train, args.scaler_type)
    label_train_scaled, label_scaler = scaling_data(label_train, args.scaler_type)

    # Save the scalers
    logging.info(f"Saving scalers")
    save_scaler(data_scaler, args.data_scaler_save_path)
    save_scaler(label_scaler, args.label_scaler_save_path)

    # Step 4: Convert scaled data to torch.tensor
    logging.info("Converting scaled data to torch tensors")
    data_train_tensor = torch.tensor(data_train_scaled, dtype=torch.float32)
    label_train_tensor = torch.tensor(label_train_scaled, dtype=torch.float32)

    # Step 5: Split data into train and validation
    logging.info("Splitting data into train and validation sets")
    data_train, data_val, label_train, label_val = split_data(data_train_tensor, label_train_tensor, args.split_type, args.validation_split_size, args.random_seed)

    # Step 6: Save train and validation sets
    logging.info(f"Saving train data and validation data")
    train_data = {'data_seq': data_train, 'label': label_train}
    val_data = {'data_seq': data_val, 'label': label_val}
    save_data(train_data, val_data, args.train_save_path, args.val_save_path)

def main():
    """Main entry point for preprocessing."""
    # Parse arguments
    args = parse_args()

    # Preprocess multiple datasets based on subdirectories
    for subdir in args.sub_dirs:
        logging.info(f"Processing dataset for {subdir}")
        args.sub_dir = subdir

        # Set paths for loading and saving
        set_paths(args)

        # Preprocess the dataset
        preprocess_data(args)

# python3 src/preprocessing.py
# Entry point for the script
if __name__ == '__main__':
    main()