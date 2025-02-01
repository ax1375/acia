"""
main.py - Main training script
"""
import torch
from torch.utils.data import DataLoader, ConcatDataset
from anticausal import ColoredMNIST, RotatedMNIST, BallAgentDataset, BallAgentEnvironment, Camelyon17Dataset
from opt import CausalRepresentationNetwork, rtrain_model, ctrain_model, BallCausalModel, ball_train_model, CamelyonModel, CamelyonTrainer, cvisualize_representations
from util import training_callback
import os
import pandas as pd

def cmnist():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_env1 = ColoredMNIST('e1', train=True)
    train_env2 = ColoredMNIST('e2', train=True)
    test_env1 = ColoredMNIST('e1', train=False)
    test_env2 = ColoredMNIST('e2', train=False)
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
    model = CausalRepresentationNetwork().to(device)
    history = ctrain_model(train_loader, test_loader, model, n_epochs=10, callback=training_callback)
    return history


def rmnist():
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rmnist = RotatedMNIST()
    # Create training set with 15° and 75° rotations
    train_env1 = rmnist['15']
    train_env2 = rmnist['75']

    # Create test set with other rotations
    test_env1 = rmnist['30']
    test_env2 = rmnist['45']
    test_env3 = rmnist['60']
    train_loader = DataLoader(
        ConcatDataset([train_env1, train_env2]),
        batch_size=32,
        shuffle=True,
        pin_memory=True
    )
    test_loader = DataLoader(
        ConcatDataset([test_env1, test_env2, test_env3]),
        batch_size=32,
        shuffle=False,
        pin_memory=True
    )
    model = CausalRepresentationNetwork().to(device)
    history, metrics_tracker = rtrain_model(train_loader, test_loader, model, n_epochs=20)
    print(history, metrics_tracker)
    return history, metrics_tracker


ROOT_DIR = "C:/University/Spring2025/Research/Session9/baseline/data/camelyon17_v1.0/patches"
METADATA_PATH = "C:/University/Spring2025/Research/Session9/baseline/data/camelyon17_v1.0/metadata.csv"
def camelyon17(root_dir, metadata_path):
    train_datasets = []
    for hospital_id in range(3):
        dataset = pd.read_csv(metadata_path)
        hospital_data = dataset[dataset['center'] == hospital_id]
        tumor = hospital_data[hospital_data['tumor'] == 1].head(50)
        non_tumor = hospital_data[hospital_data['tumor'] == 0].head(50)
        balanced_data = pd.concat([tumor, non_tumor])
        train_datasets.append(
            Camelyon17Dataset(root_dir, metadata_path, hospital_id, balanced_data.index)
        )

    # Test on hospitals 3,4 (unseen during training)
    test_datasets = []
    for hospital_id in range(3, 5):  # 3,4
        dataset = pd.read_csv(metadata_path)
        hospital_data = dataset[dataset['center'] == hospital_id]
        tumor = hospital_data[hospital_data['tumor'] == 1].head(50)
        non_tumor = hospital_data[hospital_data['tumor'] == 0].head(50)
        balanced_data = pd.concat([tumor, non_tumor])
        test_datasets.append(
            Camelyon17Dataset(root_dir, metadata_path, hospital_id, balanced_data.index)
        )
    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=16, shuffle=True)
    test_loader = DataLoader(ConcatDataset(test_datasets), batch_size=16, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CamelyonModel().to(device)
    trainer = CamelyonTrainer(model)
    for epoch in range(100):
        train_metrics, train_acc = trainer.train_epoch(train_loader, device)
        test_loss, test_acc = trainer.evaluate(test_loader, device)
        print(f'Epoch {epoch + 1}:')
        print(f'Train - Loss: {train_metrics["total_loss"]:.4f}, '
              f'Pred Loss: {train_metrics["pred_loss"]:.4f}, '
              f'R1: {train_metrics["R1"]:.4f}, '
              f'R2: {train_metrics["R2"]:.4f}, '
              f'Acc: {train_acc:.2f}%')
        print(f'Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')
        if epoch % 10 == 0:
            cvisualize_representations(model, test_loader, epoch)

def ball_agent():
    ball_data = BallAgentDataset(n_balls=4, n_samples=20000)
    train_data = BallAgentEnvironment(ball_data, is_train=True)
    test_data = BallAgentEnvironment(ball_data, is_train=False)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BallCausalModel().to(device)
    os.makedirs('results', exist_ok=True)
    metrics = ball_train_model(train_loader, test_loader, model, n_epochs=20)
    print("\nFinal Results for Ball Agent Dataset:")
    print(f"Accuracy (%): {metrics['accuracy']:.2f}")
    print(f"Environment Independence: {metrics['env_independence']:.4f}")
    print(f"Low-level: {metrics['low_level']:.4f}")
    print(f"Intervention: {metrics['intervention']:.4f}")

    return metrics


if __name__ == "__main__":
    # cmnist()
    # rmnist()
    camelyon17(ROOT_DIR, METADATA_PATH)
    # metrics = ball_agent()