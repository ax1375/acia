"""
main.py - Main training script
"""
import torch
from torch.utils.data import DataLoader, ConcatDataset
from anticausal import ColoredMNIST
from opt import CausalRepresentationNetwork, CausalOptimizer, train_model
from util import visualize_results, training_callback

def cmnist():
    # Set random seed
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets
    train_env1 = ColoredMNIST('e1', train=True)
    train_env2 = ColoredMNIST('e2', train=True)
    test_env1 = ColoredMNIST('e1', train=False)
    test_env2 = ColoredMNIST('e2', train=False)

    # Create dataloaders
    train_loader = DataLoader(
        ConcatDataset([train_env1, train_env2]),
        batch_size=32,
        shuffle=True
    )
    test_loader = DataLoader(
        ConcatDataset([test_env1, test_env2]),
        batch_size=32,
        shuffle=False
    )

    # Initialize model and train
    model = CausalRepresentationNetwork().to(device)
    # Train with callback
    history = train_model(train_loader, test_loader, model,
                          n_epochs=10, callback=training_callback)
    return history


if __name__ == "__main__":
    cmnist()