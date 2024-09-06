# src/preprocessing.py

import os
import argparse
import torch
import logging
from utils.preprocessing_util import scaling_data, split_data, save_scaler, data_loader, save_data

# Define base paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
GENERATED_DIR = os.path.join(DATA_DIR, 'generated')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser('Arguments for testing')
    parser.add_argument('--data_path', type=str, default='dataset.pkl', help='Path to the dataset file')
    parser.add_argument('--num_train_samples', type=int, default=17544, help='Number of training samples (default 2 years)')
    parser.add_argument('--scaler_type', type=str, default='MinMax', help='Type of scaler to use (MinMax or Standard)')
    parser.add_argument('--split_type', type=str, default='random', help='Type of split to use (random, kfold, stratified)')
    parser.add_argument('--validation_split_size', type=float, default=0.2, help='Fraction of the training data for validation')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set folder
    args.dataset_path = os.path.join(GENERATED_DIR, args.data_path)
    args.train_save_path = os.path.join(GENERATED_DIR, 'train.pt')
    args.val_save_path = os.path.join(GENERATED_DIR, 'val.pt')
    args.data_scaler_save_path = os.path.join(GENERATED_DIR, 'data_scaler.pkl')
    args.label_scaler_save_path = os.path.join(GENERATED_DIR, 'label_scaler.pkl')
    
    return args

def preprocess(args):
    # Step 1: Load the dataset
    logging.info("Loading the dataset...")
    dataset = data_loader(args.dataset_path)
    data_seq, label = dataset['data_seq'], dataset['label']

    # Step 2: Split the dataset into train and test (first two years as train, last year as test)
    logging.info("Splitting the dataset into train and test...")
    data_train, data_test = data_seq[:args.num_train_samples], data_seq[args.num_train_samples:]
    label_train, label_test = label[:args.num_train_samples], label[args.num_train_samples:]

    # Step 3: Scale the data
    logging.info("Scaling the training data...")
    data_train_scaled, data_scaler = scaling_data(data_train, args.scaler_type)
    label_train_scaled, label_scaler = scaling_data(label_train, args.scaler_type)

    # Save the scalers
    logging.info("Saving data and label scalers...")
    save_scaler(data_scaler, args.data_scaler_save_path)
    save_scaler(label_scaler, args.label_scaler_save_path)

    # Step 4: Convert scaled data to torch.tensor
    logging.info("Converting scaled data to torch tensors...")
    data_train_tensor = torch.tensor(data_train_scaled, dtype=torch.float32)
    label_train_tensor = torch.tensor(label_train_scaled, dtype=torch.float32)

    # Step 5: Split data into train and validation
    logging.info("Splitting data into train and validation sets...")
    data_train, data_val, label_train, label_val = split_data(data_train_tensor, label_train_tensor, args.split_type, args.validation_split_size, args.random_seed)

    # Step 6: Save train and validation sets
    logging.info("Saving train and validation sets...")
    train_data = {'data_seq': data_train, 'label': label_train}
    val_data = {'data_seq': data_val, 'label': label_val}
    save_data(train_data, val_data, args.train_save_path, args.val_save_path)

    logging.info("Preprocessing completed successfully.")

# python3 src/preprocessing.py
if __name__ == "__main__":
    args = parse_args()
    preprocess(args)