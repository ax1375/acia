"""
causalspace.py - Implements causal space structure
"""
from measuretheory import MeasurableSet, MeasurableSpace
import torch
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass


class CausalSpace:
    """
    Implementation of Definition 2: Causal Space (Ω_ei, H_ei, P_ei, K_ei)
    """

    def __init__(self, sample_space: torch.Tensor, index_set: torch.Tensor):
        self.sample_space = sample_space  # Ω_ei = ×_{t ∈ T_ei} E_t
        self.index_set = index_set  # T_ei
        self.H_ei = self._generate_sigma_algebra()  # σ-algebra H_ei
        self.P_ei = self._compute_probability_measure()  # Probability measure
        self.K_ei = self._init_causal_mechanism()  # Causal kernel K_ei

    def _generate_sigma_algebra(self) -> List[MeasurableSet]:
        """Generate H_ei = ⊗_{t ∈ T_ei} A_t"""
        sigma_sets = []

        # Generate base sets for each dimension in index set
        for t in self.index_set:
            E_t = self.sample_space[:, t]
            for value in torch.unique(E_t):
                indicator = (E_t == value)
                sigma_sets.append(MeasurableSet(indicator, f"E_{t}_{value}"))

        # Generate closures under operations
        n_base = len(sigma_sets)
        for i in range(n_base):
            for j in range(i + 1, n_base):
                # Add unions
                sigma_sets.append(sigma_sets[i].union(sigma_sets[j]))
                # Add intersections
                sigma_sets.append(sigma_sets[i].intersection(sigma_sets[j]))

        return sigma_sets

    def _compute_probability_measure(self) -> callable:
        """Define P_ei on (Ω_ei, H_ei)"""

        def P_ei(A: MeasurableSet) -> float:
            if not self.is_measurable(A):
                raise ValueError("Set is not measurable")
            return float(A.data.sum()) / len(self.sample_space)

        return P_ei

    def _init_causal_mechanism(self) -> callable:
        """
        Initialize K_ei following anti-causal structure
        K_ei(ω,A) = P(X ∈ A | Y=y, E=e)
        """

        def K_ei(omega: torch.Tensor, A: MeasurableSet) -> float:
            # Get Y-component from omega
            y = self.get_Y_component(omega)

            # Get E-component
            e = self.get_E_component(omega)

            # Compute conditional probability
            y_mask = (self.get_Y_component(self.sample_space) == y)
            e_mask = (self.get_E_component(self.sample_space) == e)

            # P(X ∈ A | Y=y, E=e)
            conditional_mask = y_mask & e_mask
            if conditional_mask.sum() == 0:
                return 0.0

            return float((conditional_mask & A.data).sum()) / float(conditional_mask.sum())

        return K_ei

    def is_measurable(self, A: MeasurableSet) -> bool:
        """Check if A is in H_ei"""
        return any((A.data == H.data).all() for H in self.H_ei)

    def get_Y_component(self, omega: torch.Tensor) -> torch.Tensor:
        """Extract Y component from sample point"""
        return omega[..., -1]  # Assuming Y is last component

    def get_E_component(self, omega: torch.Tensor) -> torch.Tensor:
        """Extract E component from sample point"""
        return omega[..., -2]  # Assuming E is second-to-last component


class ProductCausalSpace:
    """
    Implementation of Proposition 1: Product Causal Space
    """

    def __init__(self, spaces: List[CausalSpace]):
        self.spaces = spaces
        self.sample_space = self._product_sample_space()
        self.sigma_algebra = self._product_sigma_algebra()
        self.probability = self._product_probability()
        self.kernel = self._product_kernel()

    def _product_sample_space(self) -> torch.Tensor:
        """Compute Ω = Ω_e1 × Ω_e2"""
        return torch.cartesian_prod(*[space.sample_space for space in self.spaces])

    def _product_sigma_algebra(self) -> List[MeasurableSet]:
        """Generate H = H_e1 ⊗ H_e2"""
        product_sets = []

        # Generate measurable rectangles
        for sets in zip(*[space.H_ei for space in self.spaces]):
            indicator = torch.ones(len(self.sample_space), dtype=torch.bool)
            for set_i in sets:
                indicator &= set_i.data
            product_sets.append(
                MeasurableSet(indicator, "× ".join(s.name for s in sets))
            )

        return product_sets

    def _product_probability(self) -> callable:
        """Define P = P_e1 ⊗ P_e2"""

        def P(A: MeasurableSet) -> float:
            if not self.is_measurable(A):
                raise ValueError("Set is not measurable in product space")
            return float(A.data.sum()) / len(self.sample_space)

        return P

    def _product_kernel(self) -> callable:
        """Define K = {K_s : s ∈ P(T)}"""

        def K(omega: torch.Tensor, A: MeasurableSet, s: Set[int]) -> float:
            # Implement kernel following Theorem 1
            result = 0.0
            for i, space in enumerate(self.spaces):
                if i in s:
                    k_value = space.K_ei(omega, A)
                    result += k_value
            return result / len(s)

        return K

    def is_measurable(self, A: MeasurableSet) -> bool:
        """Check if A is measurable in product space"""
        return any((A.data == H.data).all() for H in self.sigma_algebra)

    def verify_properties(self) -> bool:
        """Verify product space properties"""
        # Check σ-algebra properties
        for A in self.sigma_algebra:
            assert A.complement() in self.sigma_algebra

        # Check probability measure properties
        empty_set = MeasurableSet(
            torch.zeros(len(self.sample_space), dtype=torch.bool),
            "∅"
        )
        full_set = MeasurableSet(
            torch.ones(len(self.sample_space), dtype=torch.bool),
            "Ω"
        )

        assert self.probability(empty_set) == 0
        assert abs(self.probability(full_set) - 1.0) < 1e-6

        return True