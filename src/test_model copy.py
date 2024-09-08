import os
import re
import argparse
import logging
import numpy as np
import torch

from solver.platform.test_env import MicrogridEnv
from networks.resnetd import ResNetD
from utils.test_util import cal_metric, load_dataset

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    """
    Parse command-line arguments for the testing script.
    """
    parser = argparse.ArgumentParser('Arguments for testing')
    parser.add_argument('--env', type=str, default='microgrid', help='Environment to be used for testing')
    parser.add_argument('--num_test_scenarios', type=int, default=90, help='Number of test scenarios')
    parser.add_argument('--data_path', type=str, default='data/processed/ObjVal.csv', help='Path to the test dataset')
    parser.add_argument('--pretrained_model', type=str, default='resnetd', help='Pretrained model to be loaded')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for the model')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs used for training')
    parser.add_argument('--ckpt', type=str, default='best', help='Checkpoint to load (e.g., final, highest, or specific epoch)')

    args = parser.parse_args()

    return args

def load_env(args):
    """
    Load the specified environment for testing.
    """
    if args.env == 'microgrid':
        return MicrogridEnv()
    else:
        raise ValueError(f"Unsupported environment: {args.env}")

def find_best_ckpt(pretrained_path):
    """
    Find the checkpoint file with the highest epoch number in the specified path.
    """
    ckpt_files = [f for f in os.listdir(pretrained_path) if re.match(r'ckpt_epoch_\d+\.pth', f)]
    
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {pretrained_path}")

    # Extract epoch numbers from filenames
    epoch_numbers = [int(re.findall(r'\d+', ckpt)[0]) for ckpt in ckpt_files]
    best_epoch = max(epoch_numbers)

    # Find the file with the highest epoch
    best_ckpt = f"ckpt_epoch_{best_epoch}.pth"
    logging.info(f"Best checkpoint found: {best_ckpt}")
    
    return best_ckpt

def load_models_weights(args, model):
    """
    Load the pretrained model weights from the specified checkpoint.
    """
    logging.info(f"Loading weights for model: {model.__class__.__name__}")

    # Define the folder path for the pretrained model
    args.model_name = f'{args.pretrained_model}_lr{args.learning_rate}_bs{args.batch_size}_{args.epochs}epochs'
    args.pretrained_path = os.path.join(BASE_DIR, 'models', args.sub_dir, args.model_name)

    # Determine the checkpoint to load: final, highest, or a specific epoch
    if args.ckpt == 'last':
        model_file = os.path.join(args.pretrained_path, 'last.pth')
    elif args.ckpt == 'best':
        model_file = os.path.join(args.pretrained_path, find_best_ckpt(args.pretrained_path))
    else:
        model_file = os.path.join(args.pretrained_path, f'ckpt_epoch_{args.ckpt}.pth')

    ckpt = torch.load(model_file)
    model.load_state_dict(state_dict=ckpt['model'])

    return model

def load_model(args, verbose=False, is_cuda=False):
    """
    Initialize and load the model with pretrained weights.

    Args:
        args: Parsed command-line arguments.
        verbose (bool): Whether to print status messages.
        is_cuda (bool): Whether to use CUDA.

    Returns:
        model: The loaded model.
    """
    if args.pretrained_model == 'resnetd':
        model = ResNetD(input_shape=3, num_classes=1)
        model = load_models_weights(args, model, verbose)
        
    if is_cuda:
        model = model.cuda()

    return model

def inference(args, model, verbose=False):
    """
    Perform inference using the provided model in the specified environment.

    Args:
        args: Configuration arguments containing environment and model details.
        model: The pre-trained model used for inference.
        verbose (bool): If True, prints detailed information during inference.

    Returns:
        tuple: Aggregated rewards and episode information as numpy arrays.
    """
    model.eval()

    # Initialize the environment
    env = load_env(args)

    # Containers for aggregated rewards and episode information
    aggregated_rewards = []
    episode_info = []

    # Evaluate the model for each day
    for scenario_idx in range(env.num_scenarios):
        state, info = env.reset(scenario_idx)
        total_reward = 0

        while True:
            # Convert the state to a PyTorch tensor
            state_torch = torch.tensor(state, dtype=torch.float32)
            
            # Predict the action using the model
            with torch.no_grad():
                action = model(state_torch).numpy()

            # Step the environment with the predicted action
            next_state, reward, terminated, _, info = env.step(action)
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward
            episode_info.append(info)
            
            # Break if the episode has terminated
            if terminated:
                break
        
        aggregated_rewards.append(total_reward)

    return np.array(aggregated_rewards), np.array(episode_info)

def evaluate(args, model, best_rewards, verbose=False):
    """
    Evaluate the model using the provided dataset and calculate evaluation metrics.

    Args:
        args: Parsed command-line arguments.
        model: The trained model to be evaluated.
        best_rewards: True rewards for comparison.
        verbose (bool): Whether to print status messages.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    # Perform inference with the model
    predicted_rewards, inference_info = inference(args, model, verbose)
    
    # Calculate evaluation metrics (e.g., MAE, MAPE) based on the true values and predictions
    evaluation_metrics = cal_metric(best_rewards, predicted_rewards)
    
    # Optionally, print the evaluation results
    if verbose:
        print(f"Overall MAE: {evaluation_metrics['overall_mae']:.4f}, "
              f"Overall MAPE: {evaluation_metrics['overall_mape']:.4f}%")
    
    return evaluation_metrics, inference_info

def test(args, verbose=True, is_cuda=False):
    """
    Test the model by evaluating its predictions against the actual rewards.

    Args:
        args: Parsed command-line arguments.
        verbose (bool): Whether to print status messages.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    # Load the actual rewards (ground truth) from the dataset
    best_rewards = load_dataset(args)

    # Load the pre-trained model with the specified configuration
    model = load_model(args, verbose=verbose, is_cuda=False)

    # Evaluate the model's predictions against the actual rewards
    evaluation_metrics, inference_info = evaluate(args, model, best_rewards, verbose)

    return evaluation_metrics, inference_info

# python3 src/test_model.py
if __name__ == '__main__':
    args = parse_args()
    test(args, verbose=True, is_cuda=False)