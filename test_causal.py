"""
test_causal.py - Comprehensive test suite for causal representation learning framework
"""

import unittest
import torch
import numpy as np
from torch.utils.data import DataLoader
from anticausal import AntiCausalKernel, ColoredMNIST
from optimization import RepresentationNetwork, CausalOptimizer
from utils import AnalysisUtils, VisualizationUtils

class TestCausalProperties(unittest.TestCase):
    """Test theoretical properties from the paper"""
    
    def setUp(self):
        """Setup test environment"""
        # Create test datasets
        self.env1 = ColoredMNIST('e1', train=True)
        self.env2 = ColoredMNIST('e2', train=True)
        self.n_test = 100
        
        # Create test tensors
        self.X = torch.cat([
            self.env1.colored_images[:self.n_test],
            self.env2.colored_images[:self.n_test]
        ])
        self.Y = torch.cat([
            self.env1.labels[:self.n_test],
            self.env2.labels[:self.n_test]
        ])
        self.E = torch.cat([
            torch.zeros(self.n_test),
            torch.ones(self.n_test)
        ])
        
        # Initialize models
        self.model = RepresentationNetwork()
        self.kernel = AntiCausalKernel(self.X, self.Y, self.E)
        self.optimizer = CausalOptimizer(self.model)
        
        # Test parameters
        self.epsilon = 1e-6  # Numerical precision threshold

    def test_kernel_properties(self):
        """Test Theorem 3: Anti-Causal Kernel Characterization"""
        print("\nTesting Kernel Properties...")
        
        # Test kernel computation
        omega = self.X[0]
        A = self.X[1]
        k_value = self.kernel.compute_kernel(omega, A, {'e1', 'e2'})
        
        # Verify kernel properties
        self.assertTrue(0 <= k_value <= 1, "Kernel value not in [0,1]")
        self.assertIsInstance(k_value, float, "Kernel value not float")
        
        # Test d-separation
        for y in torch.unique(self.Y):
            is_separated = self.kernel.verify_d_separation(y)
            self.assertTrue(is_separated, f"D-separation failed for digit {y}")
        
        print("✓ Kernel properties verified")

    def test_independence_property(self):
        """Test Theorem 4: Kernel Independence Property"""
        print("\nTesting Independence Property...")
        
        # Find two points with same label
        y0 = self.Y[0]
        same_label_idx = (self.Y == y0).nonzero()[1]
        
        omega1 = self.X[0]
        omega2 = self.X[same_label_idx]
        A = self.X[2]
        
        # Test kernel independence
        is_independent = self.kernel.verify_kernel_independence(omega1, omega2, A)
        self.assertTrue(is_independent, "Kernel independence property failed")
        
        print("✓ Independence property verified")

    def test_intervention_criteria(self):
        """Test Intervention Invariance Criteria"""
        print("\nTesting Intervention Criteria...")
        
        omega = self.X[0]
        A = self.X[1]
        results = self.kernel.verify_intervention_invariance(omega, A)
        
        self.assertTrue(results['do_X_invariant'], 
                       "do(X) intervention should not affect Y")
        self.assertTrue(results['do_Y_different'],
                       "do(Y) intervention should affect X")
        
        print("✓ Intervention criteria verified")

    def test_optimization_properties(self):
        """Test Theorem 5: Optimization Properties"""
        print("\nTesting Optimization Properties...")
        
        # Create test dataloader
        test_loader = DataLoader(
            torch.utils.data.ConcatDataset([self.env1, self.env2]),
            batch_size=32,
            shuffle=True
        )
        
        # Initial evaluation
        init_losses = self._evaluate_model(test_loader)
        
        # Train for a few steps
        n_steps = 5
        for _ in range(n_steps):
            x, y, e = next(iter(test_loader))
            self.optimizer.train_step(x, y, e)
        
        # Final evaluation
        final_losses = self._evaluate_model(test_loader)
        
        # Verify improvement
        self.assertLess(final_losses['total_loss'], init_losses['total_loss'],
                       "Optimization did not improve total loss")
        self.assertLess(final_losses['R1_loss'], init_losses['R1_loss'],
                       "R1 regularization not effective")
        self.assertLess(final_losses['R2_loss'], init_losses['R2_loss'],
                       "R2 regularization not effective")
        
        print("✓ Optimization properties verified")

    def test_representation_properties(self):
        """Test Learned Representation Properties"""
        print("\nTesting Representation Properties...")
        
        # Get representations
        self.model.eval()
        with torch.no_grad():
            z_L, z_H, _ = self.model(self.X)
        
        # Test environment independence
        mi_score = AnalysisUtils.analyze_mutual_information(z_H, self.Y, self.E)
        self.assertLess(mi_score, 0.1, 
                       "High-level representation not environment independent")
        
        # Test low-level invariance
        for y in torch.unique(self.Y):
            y_mask = (self.Y == y)
            for e in [0, 1]:
                e_mask = (self.E == e)
                mask = y_mask & e_mask
                if mask.sum() > 0:
                    z_L_mean = z_L[mask].mean(0)
                    z_L_std = z_L[mask].std(0)
                    self.assertLess(z_L_std.mean(), 0.5,
                                  "Low-level representation not stable")
        
        print("✓ Representation properties verified")

    def _evaluate_model(self, loader):
        """Helper function to evaluate model"""
        self.model.eval()
        losses = {'total_loss': 0, 'R1_loss': 0, 'R2_loss': 0}
        n_batches = 0
        
        with torch.no_grad():
            for x, y, e in loader:
                z_L, z_H, logits = self.model(x)
                
                # Compute losses
                losses['total_loss'] += self.optimizer.compute_prediction_loss(logits, y)
                losses['R1_loss'] += self.optimizer.compute_R1(z_H, y, e)
                losses['R2_loss'] += self.optimizer.compute_R2(z_L, z_H, y)
                n_batches += 1
        
        return {k: v/n_batches for k, v in losses.items()}

if __name__ == '__main__':
    unittest.main(verbosity=2)
