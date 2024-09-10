import pickle

def save_pickle(dataset: dict, file_path: str):
    with open(file_path, 'wb') as file:
        pickle.dump(dataset, file)

def load_pickle(file_path: str):
    # Load the label scaler
    with open(file_path, 'rb') as f:
        return pickle.load(f)