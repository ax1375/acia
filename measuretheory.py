"""
measuretheory.py - Implements core measure-theoretic components
"""
from typing import List, Tuple, Set, Dict
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset

@dataclass
class MeasurableSet:
    """
    Measurable set in σ-algebra as per Definition 1
    """
    data: torch.Tensor  # Indicator function of the set
    name: str  # Description

    def __post_init__(self):
        if not isinstance(self.data, torch.bool):
            self.data = self.data.bool()

    def intersection(self, other: 'MeasurableSet') -> 'MeasurableSet':
        """Implements A ∩ B"""
        return MeasurableSet(
            data=self.data & other.data,
            name=f"({self.name} ∩ {other.name})"
        )

    def union(self, other: 'MeasurableSet') -> 'MeasurableSet':
        """Implements A ∪ B"""
        return MeasurableSet(
            data=self.data | other.data,
            name=f"({self.name} ∪ {other.name})"
        )

    def complement(self) -> 'MeasurableSet':
        """Implements A^c"""
        return MeasurableSet(
            data=~self.data,
            name=f"({self.name})^c"
        )

class MeasurableSpace:
    """
    Implementation of Definition 1: Environmental Measurable Space
    (X_e, F_X_e), (Y_e, F_Y_e) and probability P_e
    """
    def __init__(self, X_e: torch.Tensor, Y_e: torch.Tensor):
        self.X_e = X_e  # Input space
        self.Y_e = Y_e  # Output space
        self.F_X = self._generate_sigma_algebra(X_e)  # σ-algebra on X_e
        self.F_Y = self._generate_sigma_algebra(Y_e)  # σ-algebra on Y_e
        self.F_prod = self._generate_product_sigma()  # Product σ-algebra
        self.P_e = self._compute_probability_measure()  # Probability measure

    def _generate_sigma_algebra(self, space: torch.Tensor) -> List[MeasurableSet]:
        """Generate σ-algebra following measure theory"""
        n = len(space)
        base_sets = []
        
        # Empty set and full space
        base_sets.append(MeasurableSet(torch.zeros(n, dtype=torch.bool), "∅"))
        base_sets.append(MeasurableSet(torch.ones(n, dtype=torch.bool), "Ω"))
        
        # Generate atomic sets
        for i in range(n):
            indicator = torch.zeros(n, dtype=torch.bool)
            indicator[i] = True
            base_sets.append(MeasurableSet(indicator, f"A_{i}"))
            
        return base_sets

    def _generate_product_sigma(self) -> List[MeasurableSet]:
        """Generate F_X_e ⊗ F_Y_e as product σ-algebra"""
        product_sets = []
        for set_x in self.F_X:
            for set_y in self.F_Y:
                # Create measurable rectangle A × B
                rect = torch.outer(set_x.data, set_y.data)
                product_sets.append(
                    MeasurableSet(rect, f"{set_x.name}×{set_y.name}")
                )
        return product_sets

    def _compute_probability_measure(self) -> callable:
        """Define probability measure P_e on product space"""
        def P_e(A: MeasurableSet) -> float:
            if A.data.dim() == 1:  # Marginal measure
                return float(A.data.sum()) / len(A.data)
            else:  # Joint measure
                return float(A.data.sum()) / A.data.numel()
        return P_e

    def verify_measure_properties(self) -> bool:
        """Verify measure-theoretic properties"""
        # Check σ-algebra properties
        for F in [self.F_X, self.F_Y, self.F_prod]:
            # Closed under complementation
            for A in F:
                assert A.complement() in F
            
            # Closed under countable unions
            for A in F:
                for B in F:
                    assert A.union(B) in F

        # Check probability measure properties
        empty_set = MeasurableSet(torch.zeros_like(self.X_e, dtype=torch.bool), "∅")
        full_set = MeasurableSet(torch.ones_like(self.X_e, dtype=torch.bool), "Ω")
        
        assert self.P_e(empty_set) == 0
        assert abs(self.P_e(full_set) - 1.0) < 1e-6
        
        return True

class ColoredMNIST(Dataset):
    """
    ColoredMNIST dataset implementing measure-theoretic properties
    """
    def __init__(self, env: str, root='./data', train=True):
        super().__init__()
        self.env = env
        self.mnist = datasets.MNIST(root, train=train, download=True)
        self.images = self.mnist.data.float() / 255.0
        self.labels = self.mnist.targets
        
        # Create measure space
        self.measurable_space = self._create_measure_space()

    def _create_measure_space(self) -> MeasurableSpace:
        """Create measure space for this environment"""
        # Generate colored images following anti-causal mechanism
        n_images = len(self.images)
        colored = torch.zeros((n_images, 3, 28, 28))
        
        for i, (img, label) in enumerate(zip(self.images, self.labels)):
            # Anti-causal coloring: Y → X ← E
            p_red = 0.75 if ((label % 2 == 0) == (self.env == 'e1')) else 0.25
            is_red = torch.rand(1) < p_red
            
            if is_red:
                colored[i, 0] = img  # Red channel
            else:
                colored[i, 1] = img  # Green channel
            
        # Create measurable space
        return MeasurableSpace(colored, self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.measurable_space.X_e[idx], self.measurable_space.Y_e[idx]
