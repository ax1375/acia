"""
opt.py - Proper implementation of the optimization procedure
"""
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Set
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

class CausalOptimizer:
    def __init__(self, model: nn.Module, batch_size: int, lr: float = 1e-4):
        self.model = model
        self.batch_size = batch_size
        self.lambda1 = 0.1 / (batch_size ** 0.5)
        self.lambda2 = 0.5 / (batch_size ** 0.5)  # Increased but not too much
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def compute_R1(self, z_H: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        R1 = 0.0
        unique_y = torch.unique(y)
        unique_e = torch.unique(e)
        for y_val in unique_y:
            y_mask = (y == y_val)
            y_prob = (y == y_val).float().mean()
            for e1 in unique_e:
                for e2 in unique_e:
                    if e1 != e2:
                        e1_mask = (e == e1) & y_mask
                        e2_mask = (e == e2) & y_mask
                        if e1_mask.sum() > 0 and e2_mask.sum() > 0:
                            exp_e1 = (z_H[e1_mask] * y_prob).mean(0)
                            exp_e2 = (z_H[e2_mask] * y_prob).mean(0)
                            R1 += torch.norm(exp_e1 - exp_e2, p=2)
        return R1

    def compute_R2(self, z_L: torch.Tensor, z_H: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        device = z_L.device
        batch_size = z_L.size(0)
        R2 = torch.tensor(0.0, device=device, requires_grad=True)
        unique_e = torch.unique(e)
        for e1 in unique_e:
            e1_mask = (e == e1)
            z_H_e1 = z_H[e1_mask]
            z_L_e1 = z_L[e1_mask]
            y_e1 = y[e1_mask]
            obs_dist = torch.zeros(10, device=device)
            for digit in range(10):
                digit_mask = (y_e1 == digit)
                if digit_mask.sum() > 0:
                    obs_dist[digit] = (z_H_e1[digit_mask].mean(0)).norm()
            obs_dist = F.softmax(obs_dist, dim=0)
            other_envs = [e2 for e2 in unique_e if e2 != e1]
            for e2 in other_envs:
                e2_mask = (e == e2)
                z_H_e2 = z_H[e2_mask]
                y_e2 = y[e2_mask]
                int_dist = torch.zeros(10, device=device)
                for digit in range(10):
                    digit_mask = (y_e2 == digit)
                    if digit_mask.sum() > 0:
                        int_dist[digit] = (z_H_e2[digit_mask].mean(0)).norm()
                int_dist = F.softmax(int_dist, dim=0)
                R2 = R2 + F.kl_div(obs_dist.log(), int_dist, reduction='batchmean')
        return R2 / batch_size

    def train_step(self, x: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> Dict[str, float]:
        self.optimizer.zero_grad()
        z_L, z_H, logits = self.model(x)
        pred_loss = self.criterion(logits, y)
        R1 = self.compute_R1(z_H, y, e)
        R2 = self.compute_R2(z_L, z_H, y, e)
        total_loss = pred_loss + self.lambda1 * R1 + self.lambda2 * R2
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return {
            'pred_loss': pred_loss.item(),
            'R1': R1.item(),
            'R2': R2.item(),
            'total_loss': total_loss.item()
        }


class LowLevelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU()
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
class HighLevelEncoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.encoder(x)
class Classifier(nn.Module):
    def __init__(self, input_dim=128, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)
class CausalRepresentationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi_L = LowLevelEncoderr()
        self.phi_H = HighLevelEncoderr()
        self.classifier = Classifier()
    def forward(self, x):
        z_L = self.phi_L(x)
        z_H = self.phi_H(z_L)
        logits = self.classifier(z_H)
        return z_L, z_H, logits

    def verify_holder_continuity(self, x1, x2, alpha=0.5):
        self.eval()
        with torch.no_grad():
            if len(x1.shape) == 3:
                x1 = x1.unsqueeze(0)
            if len(x2.shape) == 3:
                x2 = x2.unsqueeze(0)
            z_L1 = self.phi_L(x1)
            z_L2 = self.phi_L(x2)
            z_H1 = self.phi_H(z_L1)
            z_H2 = self.phi_H(z_L2)
            L_diff = torch.norm(z_L1 - z_L2)
            H_diff = torch.norm(z_H1 - z_H2)
            x_diff = torch.norm(x1.view(x1.size(0), -1) - x2.view(x2.size(0), -1))
        self.train()
        return L_diff <= x_diff ** alpha and H_diff <= L_diff ** alpha
def ctrain_model(train_loader: DataLoader, test_loader: DataLoader, model: nn.Module, n_epochs: int = None, callback=None) -> Dict[str, List[float]]:
    device = next(model.parameters()).device
    optimizer = CausalOptimizer(model=model, batch_size=train_loader.batch_size)
    history = {
        'train_loss': [], 'R1': [], 'R2': [],
        'train_acc': [], 'test_acc': []
    }
    for epoch in range(n_epochs):
        model.train()
        epoch_metrics = {k: [] for k in history.keys()}
        for x, y, e in train_loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            metrics = optimizer.train_step(x, y, e)
            epoch_metrics['train_loss'].append(metrics['pred_loss'])
            epoch_metrics['R1'].append(metrics['R1'])
            epoch_metrics['R2'].append(metrics['R2'])
            _, _, logits = model(x)
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean()
            epoch_metrics['train_acc'].append(acc.item())
        model.eval()
        test_acc = []
        with torch.no_grad():
            for x, y, e in test_loader:
                x, y, e = x.to(device), y.to(device), e.to(device)
                _, _, logits = model(x)
                pred = logits.argmax(dim=1)
                acc = (pred == y).float().mean()
                test_acc.append(acc.item())
        for k in history.keys():
            if k != 'test_acc':
                history[k].append(sum(epoch_metrics[k]) / len(epoch_metrics[k]))
        history['test_acc'].append(sum(test_acc) / len(test_acc))
        if callback is not None:
            callback(epoch, model, test_loader)
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"R1: {history['R1'][-1]:.4f}")
        print(f"R2: {history['R2'][-1]:.4f}")
        print(f"Train Acc: {history['train_acc'][-1]:.4f}")
        print(f"Test Acc: {history['test_acc'][-1]:.4f}")
    return history


class MetricsTracker:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.epoch_metrics = defaultdict(list)
    def update(self, metrics_dict):
        try:
            for k, v in metrics_dict.items():
                self.epoch_metrics[k].append(v)
        except Exception as e:
            print(f"Error in update: {e}")

    def epoch_end(self):
        try:
            for k, v in self.epoch_metrics.items():
                self.metrics[k].append(np.mean(v))
            self.epoch_metrics.clear()
        except Exception as e:
            print(f"Error in epoch_end: {e}")
class VisualizationManager:
    def __init__(self, save_dir='./results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    def plot_training_progress(self, metrics, epoch):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].plot(metrics['train_loss'], label='Train', color='blue')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 1].plot(metrics['train_position_error'], label='Train', color='blue')
        axes[0, 1].plot(metrics['test_position_error'], label='Test', color='red')
        axes[0, 1].set_title('Position Error')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[1, 0].plot(metrics['R1'], color='green')
        axes[1, 0].set_title('Environment Invariance (R1)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('R1 Value')
        axes[1, 0].grid(True)
        axes[1, 1].plot(metrics['R2'], color='orange')
        axes[1, 1].set_title('Intervention Consistency (R2)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('R2 Value')
        axes[1, 1].grid(True)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'training_progress_epoch_{epoch}.png')
        plt.show()
        plt.close()

    def visualize_representations(self, model, loader, epoch):
        model.eval()
        z_L, z_H, labels, angles = [], [], [], []
        angle_map = {0: 15, 1: 30, 2: 45, 3: 60, 4: 75}
        with torch.no_grad():
            for x, y, e in loader:
                x = x.to(next(model.parameters()).device)
                l, h, _ = model(x)
                z_L.append(l.cpu())
                z_H.append(h.cpu())
                labels.append(y)
                angles.append(torch.tensor([angle_map[ei.item()] for ei in e]))
        z_L = torch.cat(z_L).numpy()
        z_H = torch.cat(z_H).numpy()
        labels = torch.cat(labels).numpy()
        angles = torch.cat(angles).numpy()
        tsne = TSNE(n_components=2, perplexity=50)
        z_L_2d = tsne.fit_transform(z_L)
        z_H_2d = tsne.fit_transform(z_H)
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        for digit in range(10):
            digit_mask = labels == digit
            axes[0, 0].scatter(z_L_2d[digit_mask, 0], z_L_2d[digit_mask, 1],
                               label=f'Digit {digit}', alpha=0.7)
            axes[0, 1].scatter(z_H_2d[digit_mask, 0], z_H_2d[digit_mask, 1],
                               label=f'Digit {digit}', alpha=0.7)
        scatter1 = axes[1, 0].scatter(z_L_2d[:, 0], z_L_2d[:, 1],
                                      c=angles, cmap='viridis',
                                      label='Rotation Angle')
        scatter2 = axes[1, 1].scatter(z_H_2d[:, 0], z_H_2d[:, 1],
                                      c=angles, cmap='viridis',
                                      label='Rotation Angle')
        plt.colorbar(scatter1, ax=axes[1, 0], label='Rotation Angle (degrees)')
        plt.colorbar(scatter2, ax=axes[1, 1], label='Rotation Angle (degrees)')

        titles = ['Low-level (Raw Features)', 'High-level (Abstract Features)',
                  'Low-level by Angle', 'High-level by Angle']
        for ax, title in zip(axes.flat, titles):
            ax.set_title(title, fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'representations_epoch_{epoch}.pdf',
                    bbox_inches='tight')
        plt.show()
        plt.close()

    def visualize_representationss(self, model, loader, epoch):
        model.eval()
        z_L, z_H = [], []
        labels, envs = [], []
        device = next(model.parameters()).device
        with torch.no_grad():
            for x, y, e in loader:
                x = x.to(device)
                l, h, _ = model(x)
                z_L.append(l.cpu())
                z_H.append(h.cpu())
                labels.append(y)
                envs.append(e)

        z_L = torch.cat(z_L).numpy()
        z_H = torch.cat(z_H).numpy()
        labels = torch.cat(labels).numpy()
        envs = torch.cat(envs).numpy()
        tsne = TSNE(n_components=2, perplexity=50)
        z_L_2d = tsne.fit_transform(z_L)
        z_H_2d = tsne.fit_transform(z_H)
        fig, axes = plt.subplots(2, 2, figsize=(25, 15))
        scatter = axes[0, 0].scatter(z_L_2d[:, 0], z_L_2d[:, 1],
                                     c=labels, cmap='tab10', alpha=0.6)
        axes[0, 0].set_title('Low-level Representation by Digit')
        plt.colorbar(scatter, ax=axes[0, 0])
        axes[0, 0].grid(True)
        scatter = axes[0, 1].scatter(z_H_2d[:, 0], z_H_2d[:, 1],
                                     c=labels, cmap='tab10', alpha=0.6)
        axes[0, 1].set_title('High-level Representation by Digit')
        plt.colorbar(scatter, ax=axes[0, 1])
        axes[0, 1].grid(True)
        scatter = axes[1, 0].scatter(z_L_2d[:, 0], z_L_2d[:, 1],
                                     c=envs, cmap='Set3', alpha=0.6)
        axes[1, 0].set_title('Low-level by Environment')
        plt.colorbar(scatter, ax=axes[1, 0])
        axes[1, 0].grid(True)
        scatter = axes[1, 1].scatter(z_H_2d[:, 0], z_H_2d[:, 1],
                                     c=envs, cmap='Set3', alpha=0.6)
        axes[1, 1].set_title('High-level by Environment')
        plt.colorbar(scatter, ax=axes[1, 1])
        axes[1, 1].grid(True)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'representations_epoch_{epoch}.png')
        plt.show()
        plt.close()
class LowLevelEncoderr(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 32)  # Output dim = 32
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.conv_layers(x)
class HighLevelEncoderr(nn.Module):
    def __init__(self, input_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)
def rtrain_model(train_loader, test_loader, model, n_epochs):
    device = next(model.parameters()).device
    optimizer = CausalOptimizer(model=model, batch_size=train_loader.batch_size)
    metrics_tracker = MetricsTracker()
    viz_manager = VisualizationManager()
    history = {
        'train_loss': [], 'R1': [], 'R2': [],
        'train_acc': [], 'test_acc': []
    }
    for epoch in range(n_epochs):
        model.train()
        epoch_metrics = {k: [] for k in history.keys()}
        for x, y, e in train_loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            metrics = optimizer.train_step(x, y, e)
            epoch_metrics['train_loss'].append(metrics['pred_loss'])
            epoch_metrics['R1'].append(metrics['R1'])
            epoch_metrics['R2'].append(metrics['R2'])
            _, _, logits = model(x)
            acc = (logits.argmax(dim=1) == y).float().mean()
            epoch_metrics['train_acc'].append(acc.item())
        model.eval()
        test_acc = []
        with torch.no_grad():
            for x, y, e in test_loader:
                x, y, e = x.to(device), y.to(device), e.to(device)
                _, _, logits = model(x)
                acc = (logits.argmax(dim=1) == y).float().mean()
                test_acc.append(acc.item())
        for k in history.keys():
            if k != 'test_acc':
                history[k].append(np.mean(epoch_metrics[k]))
        history['test_acc'].append(np.mean(test_acc))
        if epoch % 5 == 0:
            viz_manager.plot_training_progress(history, epoch)
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"R1: {history['R1'][-1]:.4f}")
        print(f"R2: {history['R2'][-1]:.4f}")
        print(f"Train Acc: {history['train_acc'][-1]:.4f}")
        print(f"Test Acc: {history['test_acc'][-1]:.4f}")
        metrics_tracker.update(metrics)
        metrics_tracker.epoch_end()
    return history, metrics_tracker.metrics


class CamelyonModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        self.phi_L = nn.Sequential(
            *list(backbone.children())[:-2],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256)
        )
        self.phi_H = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.classifier = nn.Linear(64, 2)
    def forward(self, x):
        z_L = self.phi_L(x)
        z_H = self.phi_H(z_L)
        logits = self.classifier(z_H)
        return z_L, z_H, logits


def cvisualize_representations(model, test_loader, epoch):
    model.eval()
    z_L_all, z_H_all, labels, hospitals = [], [], [], []
    with torch.no_grad():
        for x, y, e in test_loader:
            z_L, z_H, _ = model(x.to(next(model.parameters()).device))
            z_L_all.append(z_L.cpu())
            z_H_all.append(z_H.cpu())
            labels.append(y)
            hospitals.append(e.argmax(1))
    z_L_all = torch.cat(z_L_all).numpy()
    z_H_all = torch.cat(z_H_all).numpy()
    labels = torch.cat(labels).numpy()
    hospitals = torch.cat(hospitals).numpy()
    tsne = TSNE(n_components=2)
    z_L_2d = tsne.fit_transform(z_L_all)
    z_H_2d = tsne.fit_transform(z_H_all)
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    colors = ['blue', 'red']
    hospital_colors = ['green', 'purple']
    for ax, data, title in zip(axes[0], [z_L_2d, z_H_2d], ['Low-level', 'High-level']):
        for i, label in enumerate([0, 1]):
            mask = labels == label
            ax.scatter(data[mask, 0], data[mask, 1],
                       c=colors[i], label=f'Tumor={label}')
        ax.set_title(f'{title} by Tumor Status')
        ax.legend()
    for ax, data, title in zip(axes[1], [z_L_2d, z_H_2d], ['Low-level', 'High-level']):
        for i, hospital in enumerate([3, 4]):
            mask = hospitals == hospital
            ax.scatter(data[mask, 0], data[mask, 1],
                       c=hospital_colors[i], label=f'Hospital {hospital}')
        ax.set_title(f'{title} by Hospital')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'representations_epoch_{epoch}.png')
    plt.show()
    plt.close()
class CamelyonTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.lambda1 = 0.1
        self.lambda2 = 0.01
    def compute_R1(self, z_H, y, e):
        R1 = 0.0
        z_H = F.normalize(z_H, dim=1)
        for h1 in range(3):
            for h2 in range(h1 + 1, 3):
                for label in [0, 1]:
                    mask1 = (e[:, h1] == 1) & (y == label)
                    mask2 = (e[:, h2] == 1) & (y == label)
                    if mask1.sum() > 0 and mask2.sum() > 0:
                        mean1 = z_H[mask1].mean(0)
                        mean2 = z_H[mask2].mean(0)
                        sim = F.cosine_similarity(mean1.unsqueeze(0), mean2.unsqueeze(0))
                        R1 += float(1 - sim)
        return R1

    def compute_R2(self, z_L, z_H, y, e):
        proj = nn.Linear(256, 64, bias=False).to(z_L.device)
        z_L = F.normalize(proj(z_L), dim=1)
        z_H = F.normalize(z_H, dim=1)
        similarity = F.cosine_similarity(z_L, z_H)
        return float(1 - similarity.mean())
    def train_epoch(self, train_loader, device):
        self.model.train()
        metrics = {'total_loss': 0, 'pred_loss': 0, 'R1': 0, 'R2': 0}
        correct = 0
        total = 0
        for x, y, e in train_loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            z_L, z_H, logits = self.model(x)
            pred_loss = self.criterion(logits, y)
            R1 = self.compute_R1(z_H, y, e)
            R2 = self.compute_R2(z_L, z_H, y, e)
            loss = pred_loss + 0.5 * R1 + 0.1 * R2
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            metrics['total_loss'] += float(loss)
            metrics['pred_loss'] += float(pred_loss)
            metrics['R1'] += R1
            metrics['R2'] += R2
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
        metrics = {k: v / len(train_loader) for k, v in metrics.items()}
        return metrics, 100 * correct / total

    def evaluate(self, test_loader, device):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, e in test_loader:
                x, y, e = x.to(device), y.to(device), e.to(device)
                z_L, z_H, logits = self.model(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return total_loss / len(test_loader), 100 * correct / total


def compute_environment_invariance(z_H: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    batch_size = z_H.size(0)
    total_loss = torch.tensor(0.0, device=z_H.device)
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if torch.any(e[i] != e[j]) and torch.allclose(y[i], y[j], atol=1e-3):
                total_loss += torch.mean((z_H[i] - z_H[j]) ** 2)
    return total_loss / (batch_size * (batch_size - 1)) if batch_size > 1 else total_loss
def compute_intervention_consistency(z_L: torch.Tensor, z_H: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    intervention_mask = e.float().unsqueeze(-1)
    rep_diff = (z_H - z_L).pow(2).mean(-1, keepdim=True)
    return torch.mean(rep_diff * (1 - intervention_mask.mean(1)) +
                      (1 - rep_diff) * intervention_mask.mean(1))
class BallCausalModel(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.phi_L = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, latent_dim),
            nn.BatchNorm1d(latent_dim)
        )
        self.phi_H = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.position_head = nn.Linear(latent_dim, 8)  # 4 balls * 2 coordinates

    def forward(self, x):
        z_L = self.phi_L(x)
        z_H = self.phi_H(z_L)
        positions = self.position_head(z_H)
        return z_L, z_H, positions
class BallAgentDataset:
    def __init__(self, n_balls: int = 4, n_samples: int = 10000, size: int = 64):
        self.n_balls = n_balls
        self.n_samples = n_samples
        self.size = size
        self.colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0.7, 0)]
        self.positions, self.images, self.interventions = self._generate_causal_data()

    def _check_constraints(self, pos: np.ndarray) -> bool:
        for i in range(self.n_balls):
            for j in range(i + 1, self.n_balls):
                dist = np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
                if dist < 0.2:
                    return False
        if np.any(pos < 0.1) or np.any(pos > 0.9):
            return False
        return True

    def _apply_intervention(self, pos: np.ndarray, intervention: np.ndarray) -> np.ndarray:
        intervened_pos = pos.copy()
        for i in range(len(intervention)):
            if intervention[i]:
                ball_idx = i // 2
                coord_idx = i % 2
                new_val = np.random.uniform(0.1, 0.9)
                intervened_pos[ball_idx, coord_idx] = new_val
                attempts = 0
                while not self._check_constraints(intervened_pos) and attempts < 100:
                    new_val = np.random.uniform(0.1, 0.9)
                    intervened_pos[ball_idx, coord_idx] = new_val
                    attempts += 1
                if attempts == 100:
                    intervened_pos[ball_idx, coord_idx] = pos[ball_idx, coord_idx]
        return intervened_pos

    def _generate_causal_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = []
        images = []
        interventions = []
        while len(positions) < self.n_samples:
            pos = np.random.uniform(0.1, 0.9, (self.n_balls, 2))
            if not self._check_constraints(pos):
                continue
            n_interventions = np.random.randint(0, self.n_balls * 2 + 1)
            intervention = np.zeros(self.n_balls * 2, dtype=bool)
            intervention_idx = np.random.choice(self.n_balls * 2, n_interventions, replace=False)
            intervention[intervention_idx] = True
            final_pos = self._apply_intervention(pos, intervention)
            image = self._render_image(final_pos)
            positions.append(final_pos.flatten())
            images.append(image)
            interventions.append(intervention)
        return np.array(positions), np.array(images), np.array(interventions)

    def _render_image(self, positions: np.ndarray) -> np.ndarray:
        image = np.zeros((self.size, self.size, 3))
        sigma = 4.0
        for i, (x, y) in enumerate(positions):
            px, py = int(x * self.size), int(y * self.size)
            color = self.colors[i]
            y_grid, x_grid = np.ogrid[-8:9, -8:9]
            distances = np.sqrt(x_grid ** 2 + y_grid ** 2)
            intensity = np.exp(-distances ** 2 / (2 * sigma ** 2))
            intensity = intensity / intensity.max()
            for c in range(3):
                y_coords = np.clip(py + y_grid, 0, self.size - 1)
                x_coords = np.clip(px + x_grid, 0, self.size - 1)
                image[y_coords, x_coords, c] += intensity * color[c]
        return np.clip(image, 0, 1)
def visualize_representations(model, loader, epoch, save_path: str = None):
    z_L, z_H, positions, interventions = [], [], [], []
    device = next(model.parameters()).device
    with torch.no_grad():
        for x, y, e in loader:
            x = x.to(device)
            l, h, _ = model(x)
            z_L.append(l.cpu())
            z_H.append(h.cpu())
            positions.append(y)
            interventions.append(e)
    z_L = torch.cat(z_L).numpy()
    z_H = torch.cat(z_H).numpy()
    positions = torch.cat(positions).numpy()
    interventions = torch.cat(interventions).numpy()
    z_L_2d = TSNE(n_components=2, random_state=42).fit_transform(z_L)
    z_H_2d = TSNE(n_components=2, random_state=42).fit_transform(z_H)
    fig, axes = plt.subplots(2, 2, figsize=(30, 20))
    for i in range(4):
        pos_i = positions[:, i * 2:(i + 1) * 2]
        scatter = axes[0, 0].scatter(z_L_2d[:, 0], z_L_2d[:, 1],
                                     c=pos_i[:, 0], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, ax=axes[0, 0])
    axes[0, 0].set_title('Low-level Representation by Ball Positions')
    scatter = axes[0, 1].scatter(z_H_2d[:, 0], z_H_2d[:, 1],
                                 c=positions[:, 0], cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, ax=axes[0, 1])
    axes[0, 1].set_title('High-level Representation by Ball Positions')
    num_interventions = np.sum(interventions, axis=1)
    scatter = axes[1, 0].scatter(z_L_2d[:, 0], z_L_2d[:, 1],
                                 c=num_interventions, cmap='RdYlBu')
    plt.colorbar(scatter, ax=axes[1, 0])
    axes[1, 0].set_title('Low-level by Number of Interventions')
    scatter = axes[1, 1].scatter(z_H_2d[:, 0], z_H_2d[:, 1],
                                 c=num_interventions, cmap='RdYlBu')
    plt.colorbar(scatter, ax=axes[1, 1])
    axes[1, 1].set_title('High-level by Number of Interventions')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()
def evaluate_model(model: nn.Module, loader: DataLoader) -> Dict[str, float]:
    model.eval()
    metrics = defaultdict(list)
    with torch.no_grad():
        for x, y, e in loader:
            z_L, z_H, pred_positions = model(x.to(next(model.parameters()).device))
            position_error = F.mse_loss(pred_positions / pred_positions.std(),
                                        y / y.std())
            env_ind = compute_environment_independence(z_H, e)
            metrics['position_error'].append(position_error.item())
            metrics['env_independence'].append(env_ind.item())
    return {k: np.mean(v) for k, v in metrics.items()}
def run_evaluation(n_runs: int = 5):
    all_metrics = []
    for run in range(n_runs):
        ball_data = BallAgentDataset(n_balls=4, n_samples=10000)
        train_data = BallAgentEnvironment(ball_data, is_train=True)
        test_data = BallAgentEnvironment(ball_data, is_train=False)
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = BallCausalModel().to(device)
        test_metrics = compute_metrics(model, test_loader, device)
        all_metrics.append(test_metrics)
        visualize_representations(model, test_loader, run)
    avg_metrics = {
        'Acc. (%)': 100 * (1 - np.mean([m['position_error'] for m in all_metrics])),
        'Env. Ind.': np.mean([m['env_independence'] for m in all_metrics]),
        'Low-level': np.mean([m['low_level_inv'] for m in all_metrics]),
        'Interv.': np.mean([m['intervention_rob'] for m in all_metrics])
    }
    return avg_metrics
class BallAgentEnvironment(Dataset):
    def __init__(self, dataset: BallAgentDataset, is_train: bool = True):
        self.data = dataset
        self.is_train = is_train
        self.train_idx, self.test_idx = self._split_data()
    def _split_data(self):
        n = len(self.data.positions)
        indices = np.random.permutation(n)
        split = int(0.8 * n)
        return indices[:split], indices[split:]

    def __len__(self):
        return len(self.train_idx) if self.is_train else len(self.test_idx)

    def __getitem__(self, idx):
        indices = self.train_idx if self.is_train else self.test_idx
        real_idx = indices[idx]
        x = torch.FloatTensor(self.data.images[real_idx])
        y = torch.FloatTensor(self.data.positions[real_idx])
        e = torch.FloatTensor(self.data.interventions[real_idx])
        return x.permute(2, 0, 1), y, e
def compute_environment_independence(z_H: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    if len(e.shape) > 1:
        e = e.mean(dim=1) > 0.5
    unique_e = torch.unique(e)
    score = torch.tensor(0.0, device=z_H.device)
    for e1 in unique_e:
        for e2 in unique_e:
            if e1 != e2:
                e1_mask = (e == e1)
                e2_mask = (e == e2)
                if e1_mask.sum() > 0 and e2_mask.sum() > 0:
                    z_H_e1 = z_H[e1_mask].mean(0)
                    z_H_e2 = z_H[e2_mask].mean(0)
                    score += torch.norm(z_H_e1 - z_H_e2)

    return score
def compute_conditional_mi(z_H: torch.Tensor, e: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if len(e.shape) > 1:
        e = e.mean(dim=1) > 0.5
    mi = torch.tensor(0.0, device=z_H.device)
    for y_val in torch.unique(y):
        y_mask = torch.all(torch.abs(y - y_val) < 1e-3, dim=1)
        if y_mask.sum() > 0:
            for e_val in torch.unique(e):
                e_mask = (e == e_val)
                joint_mask = y_mask & e_mask
                if joint_mask.sum() > 0:
                    p_z_given_y_e = joint_mask.float().mean()
                    p_z_given_y = y_mask.float().mean()
                    if p_z_given_y_e > 0 and p_z_given_y > 0:
                        mi += p_z_given_y_e * torch.log(p_z_given_y_e / p_z_given_y)

    return mi
def compute_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    metrics = defaultdict(list)
    with torch.no_grad():
        for x, y, e in loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            z_L, z_H, pred_positions = model(x)
            pos_error = F.mse_loss(pred_positions, y)
            env_ind = compute_environment_independence(z_H, e)
            low_level = compute_environment_invariance(z_L, y, e)
            interv = compute_intervention_consistency(z_L, z_H, y, e)
            metrics['position_error'].append(pos_error.item())
            metrics['env_independence'].append(env_ind.item())
            metrics['low_level_inv'].append(low_level.item())
            metrics['intervention_rob'].append(interv.item())
    return {k: np.mean(v) for k, v in metrics.items()}
def compute_paper_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    metrics = defaultdict(list)
    with torch.no_grad():
        for x, y, e in loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            z_L, z_H, pred_positions = model(x)
            position_acc = 100 * (1 - torch.abs(F.mse_loss(
                pred_positions / pred_positions.std(),
                y / y.std()
            )).item())
            env_ind = compute_conditional_mi(z_H, e, y)
            low_level = compute_environment_invariance(z_L, y, e)
            interv_rob = compute_intervention_consistency(z_L, z_H, y, e)
            metrics['accuracy'].append(position_acc)
            metrics['env_independence'].append(env_ind.item())
            metrics['low_level'].append(low_level.item())
            metrics['intervention'].append(interv_rob.item())

    return {k: np.mean(v) for k, v in metrics.items()}
def ball_train_model(train_loader, test_loader, model, n_epochs=20):
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(n_epochs):
        model.train()
        for x, y, e in train_loader:
            x, y, e = x.to(device), y.to(device), e.to(device)
            optimizer.zero_grad()
            z_L, z_H, pred_positions = model(x)
            pred_mean = pred_positions.mean(dim=0, keepdim=True)
            pred_std = pred_positions.std(dim=0, keepdim=True) + 1e-8
            y_mean = y.mean(dim=0, keepdim=True)
            y_std = y.std(dim=0, keepdim=True) + 1e-8
            pred_norm = (pred_positions - pred_mean) / pred_std
            y_norm = (y - y_mean) / y_std
            position_loss = F.mse_loss(pred_norm, y_norm)
            R1 = 0.01 * compute_environment_invariance(z_H, y, e)
            R2 = 0.05 * compute_intervention_consistency(z_L, z_H, y, e)
            loss = position_loss + R1 + R2
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                metrics = {'accuracy': 100 * (1 - position_loss.item())}
                metrics.update(compute_metrics(model, test_loader, device))
                print(f"\nEpoch {epoch + 1}")
                for k, v in metrics.items():
                    print(f"{k}: {v:.4f}")
            visualize_representations(model, test_loader, epoch, f'results/ball_results_epoch_{epoch + 1}.png')
    return metrics