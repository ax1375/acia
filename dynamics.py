"""
dynamic.py
"""
from torch.utils.data import Dataset
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
from torchvision import datasets
from causalspace import MeasurableSpace, CausalSpace, ProductCausalSpace, MeasurableSet
from causalkernel import CausalKernel
from typing import Dict, List, Tuple, Callable, Set
from dataclasses import dataclass
import torch
import torch.nn as nn
from causalspace import MeasurableSpace, CausalSpace, ProductCausalSpace
from causalkernel import CausalKernel


class CausalDynamics:
    """Implementation of Algorithm 1: Causal Dynamics"""

    def __init__(self):
        self.causal_spaces = {}
        self.product_space = None
        self.empirical_measure = None
        self.interventional_kernel = None

    def create_causal_space(self, data: torch.Tensor, labels: torch.Tensor,
                            env: str) -> Tuple[torch.Tensor, List[MeasurableSet],
    Callable, CausalKernel]:
        """Create causal space (Ω_ei, H_ei, P_ei, K_ei) for environment ei"""
        # Generate sample space
        sample_space = data

        # Generate σ-algebra
        kernel = CausalKernel(sample_space, labels,
                              torch.full_like(labels, float(env == 'e2')))
        sigma_algebra = kernel._generate_sigma_algebra()

        # Define probability measure
        probability_measure = kernel._compute_probability_measure()

        return sample_space, sigma_algebra, probability_measure, kernel

    def compute_empirical_measure(self, V_L: torch.Tensor) -> Callable:
        """Compute empirical measure Q(A) = |{v ∈ V_L : v ∈ A}| / |V_L|"""

        def Q(A: MeasurableSet) -> float:
            indicator = A.data
            return float(torch.sum(indicator)) / len(V_L)

        return Q

    def compute_interventional_kernel(self, omega: torch.Tensor,
                                      A: MeasurableSet, s: Set[str]) -> float:
        """
        Compute interventional kernel k_s^do(V_L,Q,L) using exact integration
        """
        kernels = [self.causal_spaces[env][3] for env in s]
        measures = [self.causal_spaces[env][2] for env in s]

        # Product kernel computation
        kernel_values = []
        for k, p in zip(kernels, measures):
            k_value = k.compute_kernel(omega, A)
            p_value = p(A)
            kernel_values.append(k_value * p_value)

        return sum(kernel_values) / len(kernel_values)

    def construct_low_level_representation(self, V_L: torch.Tensor,
                                           envs: Dict[str, Tuple[torch.Tensor, torch.Tensor]]):
        """
        Construct Z_L = ⟨V_L, Q, k_s^do(V_L,Q,L)⟩
        """
        # Create causal spaces for each environment
        for env, (data, labels) in envs.items():
            self.causal_spaces[env] = self.create_causal_space(data, labels, env)

        # Compute empirical measure
        self.empirical_measure = self.compute_empirical_measure(V_L)

        # Return low-level representation tuple
        return (V_L,
                self.empirical_measure,
                lambda omega, A, s: self.compute_interventional_kernel(omega, A, s))

class LowLevelEncoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim))

    def forward(self, x):
        return self.encoder(x)