import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from networks.resnetd import ResNetD
from utils.train_util import MeanMeter, data_loading, save_model

NUM_FEATURES = 3
NUM_CLASSES = 1

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def parse_option():
    """
    Parse command-line arguments for the training script.
    """
    parser = argparse.ArgumentParser('Argument for training')
    parser.add_argument('--data_dir', type=str, default='data/generated', help='Directory of data for training')
    parser.add_argument('--model', type=str, default='resnetd', help='Model type')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--gpu_device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--lr_decay_epochs', type=int, default=50, help='LR decay epochs')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='LR decay rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--use_early_stop', action='store_true', default=True, help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--rid', type=str, default='', help='Run ID')

    opt = parser.parse_args()

    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu_device)
    
    # Create model save folder
    opt.model_name = f'{opt.model}_lr{opt.learning_rate}_bs{opt.batch_size}_{opt.epochs}epochs'
    opt.save_folder = os.path.join(BASE_DIR, 'models', opt.model_name)
    
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder, exist_ok=True)

    return opt

def set_loader(opt):
    """
    Set up data loaders for training and validation.
    
    Args:
        args (Namespace): Parsed command-line arguments.
    
    Returns:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
    """
    train_dataset = data_loading(opt, is_train=True)
    val_dataset = data_loading(opt, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=False)
    
    return train_loader, val_loader


def set_model(opt):
    """
    Initialize the model and criterion.
    
    Args:
        args (Namespace): Parsed command-line arguments.
    
    Returns:
        model (torch.nn.Module): Initialized model.
        criterion (torch.nn.Module): Loss function.
    """
    if opt.model == 'resnetd':
        model = ResNetD(input_shape=NUM_FEATURES, num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Model {opt.model} is not supported.")

    criterion = nn.MSELoss()
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    
    return model, criterion

optimize_dict = {
    'SGD' : optim.SGD,
    'RMSprop': optim.RMSprop,
    'Adam': optim.Adam
}

def set_optimizer(opt, model, optim_choice='Adam'):
    """
    Set up the optimizer.
    
    Args:
        args (Namespace): Parsed command-line arguments.
        model (torch.nn.Module): Model to optimize.
        optim_choice (str): Optimizer type (e.g., 'Adam').
    
    Returns:
        optimizer (torch.optim.Optimizer): Initialized optimizer.
    """
    optimizer_cls = optimize_dict[optim_choice]
    if optim_choice == 'Adam':
        optimizer = optimizer_cls(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=opt.learning_rate, momentum=opt.momentum, weight_decay=opt.weight_decay)
    
    return optimizer


def train_model(train_loader, model, criterion, optimizer, epoch, opt, step):
    """
    Train the model for one epoch.
    
    Args:
        train_loader (DataLoader): DataLoader for training data.
        model (torch.nn.Module): Model to train.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        epoch (int): Current epoch.
        args (Namespace): Parsed command-line arguments.
        step (int): Global training step.
    
    Returns:
        step (int): Updated training step.
        avg_train_loss (float): Average training loss for the epoch.
    """
    model.train()
    losses = MeanMeter()
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.update(loss.item())
        step += 1
    
    avg_train_loss = losses.compute()
    return step, avg_train_loss


def validate_model(valid_loader, model, criterion):
    """
    Validate the model on validation data.
    
    Args:
        val_loader (DataLoader): DataLoader for validation data.
        model (torch.nn.Module): Model to validate.
        criterion (torch.nn.Module): Loss function.
    
    Returns:
        avg_valid_loss (float): Average validation loss.
    """
    model.eval()
    losses = MeanMeter()
    
    with torch.no_grad():
        for inputs, targets in valid_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.update(loss.item())
    
    avg_valid_loss = losses.compute()
    return avg_valid_loss


def main():
    """
    Main function to train the model.
    """
    opt = parse_option()

    train_loader, val_loader = set_loader(opt)
    model, criterion = set_model(opt)
    optimizer = set_optimizer(opt, model, optim_choice='Adam')
    
    step = 0
    best_val_loss = float('inf')
    patience_counter = 0

    # Define the learning rate scheduler
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0 if epoch < opt.lr_decay_epochs else torch.exp(torch.tensor(-0.01)))
    
    for epoch in range(1, opt.epochs + 1):
        step, train_loss = train_model(train_loader, model, criterion, optimizer, epoch, opt, step)
        val_loss = validate_model(val_loader, model, criterion)
        
        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        # Update learning rate
        scheduler.step()

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = f'ckpt_epoch_{epoch}.pth'
            save_file = os.path.join(opt.save_folder, ckpt)
            save_model(model, optimizer, opt, epoch, save_file)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if opt.use_early_stop and patience_counter >= opt.patience:
            print(f"Early stopping at epoch {epoch}. Best Validation Loss: {best_val_loss:.4f}")
            break
    
    # Save the last model
    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

# python3 src/train_model.py --data_dir data/generated --model resnetd --batch_size 48 --epochs 200 --gpu_device 0 --learning_rate 0.005 --lr_decay_epochs 50 --use_early_stop --patience 30
# Entry point
if __name__ == '__main__':
    main()