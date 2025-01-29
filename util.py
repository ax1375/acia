"""
utils.py - Visualization and analysis utilities for causal representation learning
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader

class VisualizationUtils:
    """Utilities for visualizing training progress and representations"""

    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], save_path: str = None):
        """Plot training metrics over time"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Loss plot
        axes[0,0].plot(history['train_loss'], label='Training Loss')
        axes[0,0].set_title('Training Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()

        # Accuracy plot
        axes[0,1].plot(history['train_acc'], label='Train Accuracy')
        axes[0,1].plot(history['test_acc'], label='Test Accuracy')
        axes[0,1].set_title('Model Accuracy')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].legend()

        # R1 and R2 metrics
        axes[1,0].plot(history['R1_metric'], label='R1 Metric')
        axes[1,0].set_title('Environment Invariance (R1)')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('R1 Value')
        axes[1,0].legend()

        axes[1,1].plot(history['R2_metric'], label='R2 Metric')
        axes[1,1].set_title('Intervention Consistency (R2)')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('R2 Value')
        axes[1,1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def visualize_representations(Z_L: torch.Tensor, Z_H: torch.Tensor,
                                Y: torch.Tensor, E: torch.Tensor,
                                save_path: str = None):
        """Visualize learned representations using t-SNE"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Low-level representations by digit
        tsne_L = TSNE(n_components=2, random_state=42)
        Z_L_2d = tsne_L.fit_transform(Z_L.detach().cpu().numpy())

        scatter = axes[0,0].scatter(Z_L_2d[:, 0], Z_L_2d[:, 1],
                                  c=Y.cpu().numpy(), cmap='tab10')
        axes[0,0].set_title('Low-level (Z_L) by Digit')
        plt.colorbar(scatter, ax=axes[0,0])

        # High-level representations by digit
        tsne_H = TSNE(n_components=2, random_state=42)
        Z_H_2d = tsne_H.fit_transform(Z_H.detach().cpu().numpy())

        scatter = axes[0,1].scatter(Z_H_2d[:, 0], Z_H_2d[:, 1],
                                  c=Y.cpu().numpy(), cmap='tab10')
        axes[0,1].set_title('High-level (Z_H) by Digit')
        plt.colorbar(scatter, ax=axes[0,1])

        # Low-level by environment
        scatter = axes[1,0].scatter(Z_L_2d[:, 0], Z_L_2d[:, 1],
                                  c=E.cpu().numpy(), cmap='RdYlBu')
        axes[1,0].set_title('Z_L by Environment')
        plt.colorbar(scatter, ax=axes[1,0])

        # High-level by environment
        scatter = axes[1,1].scatter(Z_H_2d[:, 0], Z_H_2d[:, 1],
                                  c=E.cpu().numpy(), cmap='RdYlBu')
        axes[1,1].set_title('Z_H by Environment')
        plt.colorbar(scatter, ax=axes[1,1])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

class AnalysisUtils:
    """Utilities for analyzing causal properties"""

    @staticmethod
    def analyze_mutual_information(Z: torch.Tensor, Y: torch.Tensor,
                                 E: torch.Tensor, n_bins: int = 20) -> float:
        """Compute mutual information between representation and environment"""
        # Project to 1D for binning
        Z_proj = Z @ torch.randn_like(Z[0])
        Z_bins = torch.linspace(Z_proj.min(), Z_proj.max(), n_bins+1)

        # Compute MI for each digit
        mi_scores = []
        for y in torch.unique(Y):
            y_mask = (Y == y)
            if y_mask.sum() > 0:
                Z_y = Z_proj[y_mask]
                E_y = E[y_mask]

                # Compute histograms
                joint_hist = torch.zeros(n_bins, 2)
                for i in range(n_bins):
                    for j in range(2):
                        mask = (Z_y >= Z_bins[i]) & (Z_y < Z_bins[i+1]) & (E_y == j)
                        joint_hist[i,j] = mask.sum()

                # Normalize
                joint_hist = joint_hist / joint_hist.sum()
                z_hist = joint_hist.sum(1)
                e_hist = joint_hist.sum(0)

                # Compute MI
                mi = 0.0
                for i in range(n_bins):
                    for j in range(2):
                        if joint_hist[i,j] > 0:
                            mi += joint_hist[i,j] * torch.log(
                                joint_hist[i,j] / (z_hist[i] * e_hist[j])
                            )
                mi_scores.append(mi.item())

        return np.mean(mi_scores)

    @staticmethod
    def verify_causal_properties(model, dataloader: DataLoader,
                               kernel, epsilon: float = 1e-6) -> Dict[str, bool]:
        """Verify theoretical properties from the paper"""
        model.eval()
        results = {}

        with torch.no_grad():
            # Get representations for a batch
            x, y, e = next(iter(dataloader))
            z_L, z_H, _ = model(x)

            # Test environment independence
            mi_H = AnalysisUtils.analyze_mutual_information(z_H, y, e)
            results['env_independence'] = mi_H < epsilon

            # Test low-level invariance
            low_level_inv = 0.0
            for y_val in torch.unique(y):
                y_mask = (y == y_val)
                for e1 in torch.unique(e):
                    for e2 in torch.unique(e):
                        if e1 != e2:
                            e1_mask = (e == e1) & y_mask
                            e2_mask = (e == e2) & y_mask
                            if e1_mask.sum() > 0 and e2_mask.sum() > 0:
                                z_L_e1 = z_L[e1_mask].mean(0)
                                z_L_e2 = z_L[e2_mask].mean(0)
                                low_level_inv += torch.norm(z_L_e1 - z_L_e2)

            results['low_level_invariance'] = low_level_inv < epsilon

            # Test kernel properties
            kernel_props = kernel.verify_kernel_independence(z_H[0], z_H[1], x[0])
            results.update(kernel_props)

        return results

def analyze_color_distribution(dataset, env: str):
    """Analyze color distribution in CMNIST dataset"""
    counts = torch.zeros(10, 2)  # 10 digits, 2 colors (red/green)

    for x, y in dataset:
        is_red = x[0].sum() > x[1].sum()
        counts[y, int(is_red)] += 1

    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(counts.numpy() / counts.sum(1, keepdim=True).numpy(),
                annot=True, fmt='.2f', cmap='RdYlBu',
                xticklabels=['Green', 'Red'],
                yticklabels=range(10))
    plt.title(f'Color Distribution by Digit (Environment {env})')
    plt.xlabel('Color')
    plt.ylabel('Digit')
    plt.show()


def visualize_results(model, dataloader, epoch, save_path=None):
    device = next(model.parameters()).device
    model.eval()
    all_Z_L, all_Z_H, all_Y, all_E = [], [], [], []

    with torch.no_grad():
        for x, y, e in dataloader:
            x = x.to(device)
            z_L, z_H, _ = model(x)
            all_Z_L.append(z_L.cpu())
            all_Z_H.append(z_H.cpu())
            all_Y.append(y)
            all_E.append(e)

    # Concatenate all batches
    Z_L = torch.cat(all_Z_L, 0).numpy()
    Z_H = torch.cat(all_Z_H, 0).numpy()
    Y = torch.cat(all_Y, 0).numpy()
    E = torch.cat(all_E, 0).numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    Z_L_2d = tsne.fit_transform(Z_L)
    Z_H_2d = tsne.fit_transform(Z_H)

    # Create parity labels
    parity = ['Even' if y % 2 == 0 else 'Odd' for y in Y]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Plot low-level representations by digit
    sns.scatterplot(data=pd.DataFrame({
        'x': Z_L_2d[:, 0],
        'y': Z_L_2d[:, 1],
        'Digit': Y}),
        x='x', y='y', hue='Digit', ax=axes[0])
    axes[0].set_title('Low-level (Z_L) by Digit')

    # Plot high-level representations by digit
    sns.scatterplot(data=pd.DataFrame({
        'x': Z_H_2d[:, 0],
        'y': Z_H_2d[:, 1],
        'Digit': Y}),
        x='x', y='y', hue='Digit', ax=axes[1])
    axes[1].set_title('High-level (Z_H) by Digit')

    # Plot by environment
    sns.scatterplot(data=pd.DataFrame({
        'x': Z_H_2d[:, 0],
        'y': Z_H_2d[:, 1],
        'Environment': ['Env 1' if e == 0 else 'Env 2' for e in E]}),
        x='x', y='y', hue='Environment', ax=axes[2])
    axes[2].set_title('Z_H by Environment')

    # Plot by parity
    sns.scatterplot(data=pd.DataFrame({
        'x': Z_H_2d[:, 0],
        'y': Z_H_2d[:, 1],
        'Parity': parity}),
        x='x', y='y', hue='Parity', ax=axes[3])
    axes[3].set_title('Z_H by Parity')

    plt.suptitle(f'Representation Analysis at Epoch {epoch}')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def training_callback(epoch, model, test_loader):  # Changed to match required args
    if epoch % 5 == 0:  # Visualize every 5 epochs
        visualize_results(model, test_loader, epoch)