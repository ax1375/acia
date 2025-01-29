"""
opt.py - Proper implementation of the optimization procedure
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple
from anticausal import AntiCausalKernel
import torch
import torch.nn as nn
from typing import Dict, List
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F


class CausalOptimizer:
    def __init__(self, model: nn.Module, batch_size: int, lr: float = 1e-4):
        self.model = model
        self.batch_size = batch_size
        # Better balance between R1 and R2
        self.lambda1 = 0.1 / (batch_size ** 0.5)
        self.lambda2 = 0.5 / (batch_size ** 0.5)  # Increased but not too much
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def compute_R1(self, z_H: torch.Tensor, y: torch.Tensor,
                  e: torch.Tensor) -> torch.Tensor:
        """
        Compute R1 regularization term:
        R1 = ∑_{e_i,e_j} ||∫_Y ∫_Ω φ_H(ω) dP_ei(ω|y) dμ_Y(y) -
                          ∫_Y ∫_Ω φ_H(ω) dP_ej(ω|y) dμ_Y(y)||_2
        """
        R1 = 0.0
        unique_y = torch.unique(y)
        unique_e = torch.unique(e)

        for y_val in unique_y:
            y_mask = (y == y_val)
            # Compute μ_Y(y)
            y_prob = (y == y_val).float().mean()

            for e1 in unique_e:
                for e2 in unique_e:
                    if e1 != e2:
                        e1_mask = (e == e1) & y_mask
                        e2_mask = (e == e2) & y_mask

                        if e1_mask.sum() > 0 and e2_mask.sum() > 0:
                            # Compute conditional expectations
                            exp_e1 = (z_H[e1_mask] * y_prob).mean(0)
                            exp_e2 = (z_H[e2_mask] * y_prob).mean(0)
                            R1 += torch.norm(exp_e1 - exp_e2, p=2)

        return R1

    def compute_R2(self, z_L: torch.Tensor, z_H: torch.Tensor,
                   y: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """
        Improved R2 computation that better captures intervention dynamics
        """
        device = z_L.device
        batch_size = z_L.size(0)
        R2 = torch.tensor(0.0, device=device, requires_grad=True)

        # Split data by environment
        unique_e = torch.unique(e)
        for e1 in unique_e:
            e1_mask = (e == e1)

            # Get environment-specific representations and labels
            z_H_e1 = z_H[e1_mask]
            z_L_e1 = z_L[e1_mask]
            y_e1 = y[e1_mask]

            # Compute observational distribution
            obs_dist = torch.zeros(10, device=device)  # For 10 digits
            for digit in range(10):
                digit_mask = (y_e1 == digit)
                if digit_mask.sum() > 0:
                    obs_dist[digit] = (z_H_e1[digit_mask].mean(0)).norm()

            # Normalize observational distribution
            obs_dist = F.softmax(obs_dist, dim=0)

            # Compute interventional distribution
            # Use a different environment's data to simulate intervention
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

                # Add KL divergence between distributions
                R2 = R2 + F.kl_div(obs_dist.log(), int_dist, reduction='batchmean')

        return R2 / batch_size

    def train_step(self, x: torch.Tensor, y: torch.Tensor, e: torch.Tensor) -> Dict[str, float]:
        """Execute training step following paper's optimization procedure"""
        self.optimizer.zero_grad()

        # Forward pass
        z_L, z_H, logits = self.model(x)

        # Prediction loss
        pred_loss = self.criterion(logits, y)

        # Regularization terms
        R1 = self.compute_R1(z_H, y, e)
        R2 = self.compute_R2(z_L, z_H, y, e)

        # Total loss following paper's formulation
        total_loss = pred_loss + self.lambda1 * R1 + self.lambda2 * R2

        # Do the backward pass without Hölder continuity check during training
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            'pred_loss': pred_loss.item(),
            'R1': R1.item(),
            'R2': R2.item(),
            'total_loss': total_loss.item()
        }


class LowLevelEncoder(nn.Module):
    """φ_L: Low-level representation (Causal Latent Dynamics)"""

    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14
            nn.Conv2d(32, 64, 3, padding=1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
            nn.Conv2d(64, 64, 3, padding=1),  # 7x7 -> 7x7
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Convert to fixed size
            nn.Flatten()  # 64 * 4 * 4 = 1024
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU()
        )

    def forward(self, x):
        # Ensure input is in the right format
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class HighLevelEncoder(nn.Module):
    """φ_H: High-level representation (Causal Latent Abstraction)"""

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
    """w: Final classifier"""

    def __init__(self, input_dim=128, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


class CausalRepresentationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi_L = LowLevelEncoder()
        self.phi_H = HighLevelEncoder()
        self.classifier = Classifier()

    def forward(self, x):
        z_L = self.phi_L(x)
        z_H = self.phi_H(z_L)
        logits = self.classifier(z_H)
        return z_L, z_H, logits

    def verify_holder_continuity(self, x1, x2, alpha=0.5):
        """Verify Hölder continuity condition"""
        # Turn off BatchNorm during verification
        self.eval()
        with torch.no_grad():
            # Handle single image case
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


def train_model(train_loader: DataLoader, test_loader: DataLoader, model: nn.Module, n_epochs: int = None, callback=None) -> Dict[str, List[float]]:
    """Train model with visualization callback"""
    # Get device from model parameters
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

            # Training step
            metrics = optimizer.train_step(x, y, e)

            # Record metrics
            epoch_metrics['train_loss'].append(metrics['pred_loss'])
            epoch_metrics['R1'].append(metrics['R1'])
            epoch_metrics['R2'].append(metrics['R2'])

            # Compute accuracy
            _, _, logits = model(x)
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean()
            epoch_metrics['train_acc'].append(acc.item())

        # Evaluation
        model.eval()
        test_acc = []
        with torch.no_grad():
            for x, y, e in test_loader:
                x, y, e = x.to(device), y.to(device), e.to(device)
                _, _, logits = model(x)
                pred = logits.argmax(dim=1)
                acc = (pred == y).float().mean()
                test_acc.append(acc.item())

        # Update history
        for k in history.keys():
            if k != 'test_acc':
                history[k].append(sum(epoch_metrics[k]) / len(epoch_metrics[k]))
        history['test_acc'].append(sum(test_acc) / len(test_acc))

        # Call callback if provided
        if callback is not None:
            callback(epoch, model, test_loader)

        # Print progress
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"R1: {history['R1'][-1]:.4f}")
        print(f"R2: {history['R2'][-1]:.4f}")
        print(f"Train Acc: {history['train_acc'][-1]:.4f}")
        print(f"Test Acc: {history['test_acc'][-1]:.4f}")

    return history