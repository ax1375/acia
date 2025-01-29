"""
anticausal.py - Implements anti-causal kernel characterization and ColoredMNIST
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from torchvision import transforms
import torch
import torch.nn as nn
from typing import Dict, Set

class ColoredMNIST(Dataset):
    """
    ColoredMNIST dataset implementing the anti-causal structure Y → X ← E
    Now properly returns environment labels
    """

    def __init__(self, env: str, root='./data', train=True):
        super().__init__()
        self.env = env
        # Load original MNIST
        mnist = datasets.MNIST(root=root, train=train, download=True)
        self.images = mnist.data.float() / 255.0
        self.labels = mnist.targets

        # Create colored version following anti-causal mechanism
        self.colored_images = self._color_images()
        # Create environment labels (0 for e1, 1 for e2)
        self.env_labels = torch.full_like(self.labels, float(env == 'e2'))

    def _color_images(self) -> torch.Tensor:
        """Implement anti-causal coloring mechanism"""
        n_images = len(self.images)
        colored = torch.zeros((n_images, 3, 28, 28))

        for i, (img, label) in enumerate(zip(self.images, self.labels)):
            # Following the paper's specification
            p_red = 0.75 if ((label % 2 == 0) == (self.env == 'e1')) else 0.25
            is_red = torch.rand(1) < p_red

            # Convert to CHW format and add color
            if is_red:
                colored[i, 0] = img  # Red channel
            else:
                colored[i, 1] = img  # Green channel

        return colored

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Now returns environment label as well"""
        return self.colored_images[idx], self.labels[idx], self.env_labels[idx]


class AntiCausalKernel:
    """
    Implementation of Theorem 3: Anti-Causal Kernel Characterization
    """

    def __init__(self, X: torch.Tensor, Y: torch.Tensor, E: torch.Tensor, epsilon: float = 1e-6):
        self.X = X.view(X.size(0), -1)  # Flatten features to 2D: [batch_size, feature_dim]
        self.Y = Y
        self.E = E
        self.epsilon = epsilon
        self.mu_Y = self._compute_Y_measure()

    def _compute_Y_measure(self) -> callable:
        """Compute marginal measure μ_Y on label space"""
        counts = torch.bincount(self.Y)
        probs = counts.float() / len(self.Y)

        def mu_Y(y: torch.Tensor) -> float:
            return probs[y].item()

        return mu_Y

    def compute_kernel(self, omega: torch.Tensor, A: torch.Tensor, s: Set[str]) -> float:
        """
        Compute K_s(ω,A) = ∫_Y P(X ∈ A|Y=y, E ∈ s)dμ_Y(y)
        """
        # Flatten input tensors
        omega = omega.view(-1)
        A = A.view(-1)

        result = 0.0
        for y_val in torch.unique(self.Y):
            # P(X ∈ A|Y=y, E ∈ s)
            cond_prob = self._compute_conditional_prob(A, y_val, s)
            # Weight by Y measure
            result += cond_prob * self.mu_Y(y_val)

        return result

    def _compute_conditional_prob(self, A: torch.Tensor, y: torch.Tensor, s: Set[str]) -> float:
        """Compute P(X ∈ A|Y=y, E ∈ s)"""
        # Create masks for conditioning
        y_mask = (self.Y == y)
        e_mask = torch.zeros_like(self.Y, dtype=torch.bool)
        for e in s:
            e_mask |= (self.E == (e == 'e2'))

        # Combined condition
        condition = y_mask & e_mask
        if condition.sum() == 0:
            return 0.0

        # Compare flattened features
        X_cond = self.X[condition]
        A_expanded = A.view(1, -1).expand(X_cond.size(0), -1)
        matching = torch.all(torch.isclose(X_cond, A_expanded), dim=1)

        return float(matching.sum()) / float(condition.sum())

    def verify_d_separation(self, y: torch.Tensor) -> bool:
        """Verify d-separation criterion"""
        y_mask = (self.Y == y)
        x_given_y = self.X[y_mask]
        e_given_y = self.E[y_mask]

        for e in [0, 1]:
            e_mask = (e_given_y == e)
            if e_mask.sum() == 0:
                continue

            dist_e = x_given_y[e_mask].mean(0)
            dist_not_e = x_given_y[~e_mask].mean(0)

            if torch.norm(dist_e - dist_not_e) > self.epsilon:
                return False

        return True

    def _compute_do_Y_kernel(self, omega: torch.Tensor, A: torch.Tensor) -> float:
        """Compute kernel under do(Y) intervention"""
        omega = omega.view(-1)
        A = A.view(-1)

        # In do(Y) intervention, we break causal link Y → X
        matching = torch.all(torch.isclose(self.X.view(len(self.X), -1),
                                           A.view(1, -1).expand(len(self.X), -1)), dim=1)
        return float(matching.sum()) / len(self.X)

    def compute_intervention_probs(self, z_H: torch.Tensor, y: torch.Tensor,
                                   e: torch.Tensor) -> torch.Tensor:
        """
        Compute intervention probabilities P(y|do(φ_H(ω)))
        """
        device = z_H.device
        probs = torch.zeros((10,), device=device)  # 10 classes

        # For intervention, break dependency on environment
        for label in range(10):
            mask = (y == label)
            if mask.sum() > 0:
                probs[label] = mask.float().mean()

        return probs / probs.sum()



def create_cmnist_environments(root='./data', train=True) -> Tuple[ColoredMNIST, ColoredMNIST]:
    """Create both CMNIST environments"""
    env1 = ColoredMNIST('e1', root=root, train=train)
    env2 = ColoredMNIST('e2', root=root, train=train)
    return env1, env2



def verify_cmnist_properties(env1: ColoredMNIST, env2: ColoredMNIST) -> Dict[str, bool]:
    """Verify CMNIST satisfies theoretical properties from paper"""
    # Combine data from both environments
    X = torch.cat([env1.colored_images, env2.colored_images])
    Y = torch.cat([env1.labels, env2.labels])
    E = torch.cat([torch.zeros_like(env1.labels), torch.ones_like(env2.labels)])

    # Create kernel
    kernel = AntiCausalKernel(X, Y, E)

    results = {}

    # Test d-separation
    d_sep = all(kernel.verify_d_separation(y) for y in torch.unique(Y))
    results['d_separation'] = d_sep

    # Test kernel independence
    k_indep = all(
        kernel.verify_kernel_independence(X[i], X[j], X[k])
        for i in range(min(100, len(X)))
        for j in range(i+1, min(100, len(X)))
        for k in range(min(10, len(X)))
        if Y[i] == Y[j]
    )
    results['kernel_independence'] = k_indep

    # Test intervention invariance
    inv_results = kernel.verify_intervention_invariance(X[0], X[1])
    results.update(inv_results)

    return results