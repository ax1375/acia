"""
abstraction.py - Implementation of Causal Abstraction (Algorithm 2)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from causalspace import MeasurableSpace, CausalSpace, ProductCausalSpace
from causalkernel import CausalKernel
from anticausal import *
from typing import Dict, List, Tuple, Callable

@dataclass
class HighLevelRepresentation:
    V_H: torch.Tensor
    k_H: 'HighLevelKernel'

class DynamicsFunction(nn.Module):
    """
    Implements τ: D_V_L → D_Z_L mapping from input space to low-level representation
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class HighLevelKernel:
    def __init__(self, dynamics_func: DynamicsFunction):
        self.tau = dynamics_func
        self.weights = {}  # Store optimal weights αk

    def compute_kernel(self, Z_L: torch.Tensor, A: torch.Tensor, s_k: int) -> torch.Tensor:
        """
        Compute k_H^Z_L_k for dataset s_k
        """
        # Map to low-level space using inverse dynamics
        V_L = self.tau(Z_L)

        # Compute kernel integral
        return self._compute_kernel_integral(V_L, A, s_k)

    def _compute_kernel_integral(self, V_L: torch.Tensor, A: torch.Tensor, s_k: int) -> torch.Tensor:
        # Implement integral computation based on paper formula
        # k_H^Z_L_k = ∫_A k_s_k^do(V_L,Q,L)(ω,τ^-1(A_k)) dV_L
        # This is approximated using Monte Carlo integration
        n_samples = 1000
        samples = torch.randn_like(V_L.unsqueeze(0).repeat(n_samples, 1))
        kernel_values = torch.zeros(n_samples)

        for i in range(n_samples):
            kernel_values[i] = self._evaluate_interventional_kernel(
                samples[i], A, s_k
            )

        return kernel_values.mean()


class CausalAbstraction:
    """Implementation of Algorithm 2: Causal Abstraction using measure theory"""

    def __init__(self, phi_L: List[Tuple[torch.Tensor, Callable, Callable]]):
        """
        Initialize with set of low-level representations
        phi_L: List of (V_L, Q, k_s) tuples from CausalDynamics
        """
        self.phi_L = phi_L
        self.dynamics = self._construct_dynamics_function()
        self.high_level_kernel = None

    def _construct_dynamics_function(self) -> Callable:
        """
        Construct dynamics function τ: D_V_L → D_Z_L through measure-theoretic mapping
        """

        def tau(v_l: torch.Tensor) -> torch.Tensor:
            # Map from input space to low-level representation space
            # using proper measure-theoretic transformation
            measurable_sets = [MeasurableSet(v_l == v, f"set_{i}")
                               for i, v in enumerate(torch.unique(v_l))]

            # Compute pushforward measure
            pushforward = torch.zeros_like(v_l)
            for set_idx, mset in enumerate(measurable_sets):
                for phi in self.phi_L:
                    V_L, Q, k_s = phi
                    pushforward[mset.data] = Q(mset)

            return pushforward

        return tau

    def compute_high_level_kernel(self, Z_L: torch.Tensor,
                                  A: MeasurableSet,
                                  s_k: int) -> float:
        """
        Compute k_H^Z_L_k using exact integration over A_k
        """
        # Get low-level representation components
        V_L, Q, k_s = self.phi_L[s_k]

        # Compute preimage under τ
        tau_inverse = self.dynamics(Z_L)

        # Compute integral using proper measure-theoretic definition
        integral = 0.0
        measurable_sets = self._generate_partition(A)

        for mset in measurable_sets:
            # Compute k_s^do(V_L,Q,L) for each partition
            kernel_value = k_s(tau_inverse, mset, {s_k})
            # Weight by measure of partition
            integral += kernel_value * Q(mset)

        return integral

    def _generate_partition(self, A: MeasurableSet) -> List[MeasurableSet]:
        """Generate finite partition of measurable set A"""
        # Implementation depends on the structure of your σ-algebra
        # This is a simplified version
        indicator = A.data
        unique_values = torch.unique(indicator)
        return [MeasurableSet(indicator == v, f"partition_{v}")
                for v in unique_values]

    def optimize_weights(self) -> Dict[int, float]:
        """
        Find optimal weights α_k* through exact optimization
        Uses convex optimization instead of gradient descent
        """
        n_datasets = len(self.phi_L)

        # Set up quadratic program
        from cvxopt import matrix, solvers

        # Compute kernel differences matrix
        P = torch.zeros((n_datasets, n_datasets))
        for i in range(n_datasets):
            for j in range(n_datasets):
                for A in self._generate_base_sets():
                    k_i = self.compute_high_level_kernel(self.phi_L[i][0], A, i)
                    k_j = self.compute_high_level_kernel(self.phi_L[j][0], A, j)
                    P[i, j] += (k_i - k_j) ** 2

        # Solve quadratic program with constraints:
        # Σ_k α_k = 1, α_k ≥ 0
        P = matrix(P.numpy())
        q = matrix(torch.zeros(n_datasets).numpy())
        G = matrix(-torch.eye(n_datasets).numpy())
        h = matrix(torch.zeros(n_datasets).numpy())
        A = matrix(torch.ones(1, n_datasets).numpy())
        b = matrix([1.0])

        solution = solvers.qp(P, q, G, h, A, b)
        optimal_weights = torch.tensor(solution['x']).squeeze()

        return {k: w.item() for k, w in enumerate(optimal_weights)}

    def _generate_base_sets(self) -> List[MeasurableSet]:
        """Generate base measurable sets for kernel computation"""
        base_sets = []
        for V_L, _, _ in self.phi_L:
            values = torch.unique(V_L)
            for v in values:
                base_sets.append(MeasurableSet(V_L == v, f"base_{v}"))
        return base_sets

    def construct_high_level_representation(self) -> Tuple[torch.Tensor, Callable]:
        """
        Construct Z_H = ⟨V_H, k_H^Z_H⟩ using optimal weights
        """
        # Optimize weights
        optimal_weights = self.optimize_weights()

        # Construct V_H as weighted combination
        V_H = sum(w * phi[0] for w, phi in zip(optimal_weights.values(), self.phi_L))

        # Construct high-level kernel
        def k_H(omega: torch.Tensor, A: MeasurableSet) -> float:
            return sum(w * self.compute_high_level_kernel(omega, A, k)
                       for k, w in optimal_weights.items())

        return V_H, k_H


class InterventionalKernel:
    def __init__(self, dataset_e1: ColoredMNIST, dataset_e2: ColoredMNIST):
        self.e1_data = dataset_e1
        self.e2_data = dataset_e2
        self.base_kernel = ColoredMNISTKernel(dataset_e1, dataset_e2)

    def compute_do_X_kernel(self, omega: torch.Tensor, A: torch.Tensor, env: str) -> float:
        kernel = self.base_kernel.kernel_e1 if env == 'e1' else self.base_kernel.kernel_e2
        omega_flat = omega.reshape(-1)

        # Get color and label information
        red_sum = omega_flat[:784].sum()
        label_idx = (kernel.sample_space @ omega_flat.float()) / (kernel.sample_space @ kernel.sample_space[0])
        label = kernel.Y[label_idx.argmax()]

        # Since X is intervened, Y is independent of X
        return float(torch.sum(kernel.Y == label)) / len(kernel.Y)

        # Since X is intervened, Y is independent of X
        # Therefore, just return empirical distribution of Y
        return float(torch.sum(kernel.Y == label)) / len(kernel.Y)

    def compute_do_Y_kernel(self, omega: torch.Tensor, A: torch.Tensor, env: str) -> float:
        """
        Compute K_s^do(Y)(omega, {X ∈ A}) ≠ K_s(omega, {X ∈ A})
        Intervening on Y breaks the causal link Y → X
        """
        kernel = self.base_kernel.kernel_e1 if env == 'e1' else self.base_kernel.kernel_e2

        # Since Y is intervened, X only depends on E
        # The color distribution should now only depend on environment
        if env == 'e1':
            p_red = 0.5  # Without Y's influence, color probability becomes uniform
        else:
            p_red = 0.5

        # Check if omega is red
        omega_flat = omega.reshape(-1)
        is_red = omega_flat[:784].sum() > omega_flat[784:1568].sum()

        # Check if A contains mostly red images
        A_flat = A.reshape(A.size(0), -1)
        A_red_sum = A_flat[:, :784].sum(1)
        A_green_sum = A_flat[:, 784:1568].sum(1)
        A_is_red = A_red_sum > A_green_sum

        return p_red if is_red == A_is_red.any() else (1 - p_red)

    def verify_intervention_effects(self, omega: torch.Tensor, A: torch.Tensor):
        """
        Verify the properties from the Intervention Invariance Criteria theorem:
        1. K_s^do(X)(omega, {Y ∈ B}) = K_s(omega, {Y ∈ B})
        2. K_s^do(Y)(omega, {X ∈ A}) ≠ K_s(omega, {X ∈ A})
        """
        results = {}
        for env in ['e1', 'e2']:
            # Observational kernels
            obs_kernel = self.base_kernel.compute_environment_kernel(omega, A, env)

            # Interventional kernels
            do_x_kernel = self.compute_do_X_kernel(omega, A, env)
            do_y_kernel = self.compute_do_Y_kernel(omega, A, env)

            results[f'{env}_obs'] = obs_kernel
            results[f'{env}_do_x'] = do_x_kernel
            results[f'{env}_do_y'] = do_y_kernel

            # Check invariance properties
            results[f'{env}_x_invariant'] = abs(do_x_kernel - obs_kernel) < 1e-6
            results[f'{env}_y_different'] = abs(do_y_kernel - obs_kernel) >= 1e-6

        return results