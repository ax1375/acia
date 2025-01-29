"""
lca.py
"""
from scipy.stats import entropy
from typing import Dict
from LCA import hier
import torchvision
from dataset import RotatedMNIST
from torch.utils.data import TensorDataset
from torchvision import datasets
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mutual_info_score

class ColoredMNIST(torch.utils.data.Dataset):
    def __init__(self, env):
        self.mnist = torchvision.datasets.MNIST(
            root='./data',
            train=True if env == 'e1' else False,
            download=True,
            transform=transforms.ToTensor()
        )
        self.env = env

        # Simple CNN for digit recognition
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 10)
        ).to('cuda' if torch.cuda.is_available() else 'cpu')

        # Train model if in environment 1
        if env == 'e1':
            self._train_model()

        self.images = []
        self.labels = []
        self.predictions = []

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.no_grad():
            for img, label in self.mnist:
                # Get prediction
                pred = self.model(img.unsqueeze(0).to(device)).argmax().item()

                # Create colored version
                img_np = img.squeeze().numpy()
                colored = np.zeros((28, 28, 3))
                if label % 2 == 0:
                    colored[:, :, 1] = img_np  # Green
                else:
                    colored[:, :, 0] = img_np  # Red

                self.images.append(colored)
                self.labels.append(label)
                self.predictions.append(pred)

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.predictions = np.array(self.predictions)

    def _train_model(self):
        optimizer = torch.optim.Adam(self.model.parameters())
        criterion = nn.CrossEntropyLoss()

        dataloader = torch.utils.data.DataLoader(self.mnist, batch_size=128, shuffle=True)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for epoch in range(2):  # Quick training
            for img, label in dataloader:
                img, label = img.to(device), label.to(device)
                pred = self.model(img)
                loss = criterion(pred, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def get_color(self, label):
        # Even digits are green (0), odd are red (1)
        return [1, 0] if label % 2 == 1 else [0, 1]
class CMNISTHierarchy:
    def __init__(self):
        self.parents = np.array([
            -1,  # Root (0)
            0,  # Odd group (1)
            0,  # Even group (2)
            1,  # 1,3 (3)
            1,  # 5,7 (4)
            1,  # 9 (5)
            2,  # 0,2 (6)
            2,  # 4,6 (7)
            2,  # 8 (8)
            3,  # 1 (9)
            3,  # 3 (10)
            4,  # 5 (11)
            4,  # 7 (12)
            5,  # 9 (13)
            6,  # 0 (14)
            6,  # 2 (15)
            7,  # 4 (16)
            7,  # 6 (17)
            8  # 8 (18)
        ])
        self.tree = hier.Hierarchy(self.parents)

        self.digit_to_leaf = {
            1: 9,
            3: 10,
            5: 11,
            7: 12,
            9: 13,
            0: 14,
            2: 15,
            4: 16,
            6: 17,
            8: 18
        }
        self.leaf_to_digit = {v: k for k, v in self.digit_to_leaf.items()}
class CMNISTAnalyzer:
    """Analyzer for Colored MNIST dataset with LCA metrics"""

    def __init__(self, env: str):
        super().__init__()
        self.dataset = ColoredMNIST(env)
        self.hierarchy = CMNISTHierarchy()
        self.results = {}
        self.analyze()

    def analyze(self):
        # Compute basic stats
        self.results['label_stats'] = self.compute_label_statistics(self.dataset.labels)

        # Analyze color distribution
        colors = np.array([1 if img[:, :, 0].max() > 0 else 0 for img in self.dataset.images])
        self.results['color_stats'] = self.analyze_color_distribution(colors)

        # Compute LCA metrics
        self.results['lca_metrics'] = self.compute_lca_metrics(
            self.dataset.labels,
            self.dataset.predictions
        )

    def compute_label_statistics(self, labels: np.ndarray) -> Dict:
        unique_labels, counts = np.unique(labels, return_counts=True)
        return {
            'label_dist': dict(zip(unique_labels, counts)),
            'total_samples': len(labels),
            'entropy': entropy(counts / len(labels))
        }

    def analyze_color_distribution(self, colors: np.ndarray) -> Dict:
        total = len(colors)
        red_count = np.sum(colors == 1)
        return {
            'red_ratio': red_count / total,
            'green_ratio': (total - red_count) / total,
            'color_entropy': entropy([red_count / total, (total - red_count) / total])
        }

    def compute_lca_metrics(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict:
        """Compute LCA-based metrics between true and predicted labels"""
        # Convert digit labels to tree indices
        tree_true = np.array([self.hierarchy.digit_to_leaf[l] for l in true_labels])
        tree_pred = np.array([self.hierarchy.digit_to_leaf[p] for p in pred_labels])

        # Find LCA nodes
        find_lca = hier.FindLCA(self.hierarchy.tree)
        lca_nodes = find_lca(tree_true, tree_pred)

        # Compute metrics
        wrong_mask = (true_labels != pred_labels)
        if wrong_mask.sum() > 0:
            # Get depths of LCA nodes for incorrect predictions
            lca_depths = self.hierarchy.tree.depths()[lca_nodes[wrong_mask]]
            mean_lca_depth = float(lca_depths.mean())
        else:
            mean_lca_depth = 0.0

        return {
            'lca_distance': mean_lca_depth,
            'accuracy': (true_labels == pred_labels).mean(),
            'lca_by_digit': self.compute_per_digit_lca(true_labels, pred_labels)
        }

    def compute_per_digit_lca(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict:
        """Compute LCA metrics for each digit separately"""
        per_digit_lca = {}
        for digit in range(10):
            digit_mask = (true_labels == digit)
            if digit_mask.sum() > 0:
                digit_true = true_labels[digit_mask]
                digit_pred = pred_labels[digit_mask]

                # Convert to tree indices and compute LCA
                tree_true = np.array([self.hierarchy.digit_to_leaf[l] for l in digit_true])
                tree_pred = np.array([self.hierarchy.digit_to_leaf[p] for p in digit_pred])

                find_lca = hier.FindLCA(self.hierarchy.tree)
                lca_nodes = find_lca(tree_true, tree_pred)

                wrong_mask = (digit_true != digit_pred)
                if wrong_mask.sum() > 0:
                    lca_depths = self.hierarchy.tree.depths()[lca_nodes[wrong_mask]]
                    mean_lca_depth = float(lca_depths.mean())
                else:
                    mean_lca_depth = 0.0

                per_digit_lca[digit] = {
                    'lca_distance': mean_lca_depth,
                    'accuracy': (digit_true == digit_pred).mean()
                }

        return per_digit_lca

    def plot_lca_results(self):
        metrics = self.results['lca_metrics']['lca_by_digit']
        digits = sorted(metrics.keys())
        lca_distances = [metrics[d]['lca_distance'] for d in digits]
        accuracies = [metrics[d]['accuracy'] for d in digits]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # LCA Distance Plot
        colors = ['#e74c3c' if d % 2 == 1 else '#2ecc71' for d in digits]
        bars = ax1.bar(digits, lca_distances, color=colors)
        ax1.set_title('LCA Distance by Digit Type')
        ax1.set_xlabel('Digit')
        ax1.set_ylabel('LCA Distance')

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.2f}', ha='center', va='bottom')

        # Legend
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Even Digits'),
            Patch(facecolor='#e74c3c', label='Odd Digits')
        ]
        ax1.legend(handles=legend_elements)

        # Accuracy vs LCA Plot
        ax2.scatter(lca_distances, accuracies, c=colors)
        for i, digit in enumerate(digits):
            ax2.annotate(str(digit), (lca_distances[i], accuracies[i]))

        z = np.polyfit(lca_distances, accuracies, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(lca_distances), max(lca_distances), 100)
        ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8)

        ax2.set_title('Accuracy vs LCA Distance')
        ax2.set_xlabel('LCA Distance')
        ax2.set_ylabel('Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.legend(handles=legend_elements)

        plt.tight_layout()
        plt.savefig('lca_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
def cmnist():
    analyzer = CMNISTAnalyzer('e1')
    print("Overall LCA metrics:", analyzer.results['lca_metrics'])
    print("Per-digit LCA metrics:", analyzer.results['lca_metrics']['lca_by_digit'])
    analyzer.plot_lca_results()


class RotatedMNIST:
    ENVIRONMENTS = [0, 15, 30, 45, 60, 75]
    def __init__(self, root='./data', split='train', indices=None):
        mnist = datasets.MNIST(root, train=(split == 'train'), download=True)
        self.images, self.labels = mnist.data.numpy(), mnist.targets.numpy()
        self.images = self.images.astype(np.float32) / 255.0
        self.envs = {}

        if indices is not None:
            self.images = self.images[indices]
            self.labels = self.labels[indices]
        elif split == 'train':
            self.images = self.images[:50000]
            self.labels = self.labels[:50000]
        else:
            self.images = self.images[50000:]
            self.labels = self.labels[50000:]

        for angle in self.ENVIRONMENTS:
            rotated_images = np.zeros_like(self.images)
            for i, img in enumerate(self.images):
                rotated_images[i] = rotate(img, angle, reshape=False)
            self.envs[angle] = TensorDataset(torch.tensor(rotated_images), torch.tensor(self.labels))

    def __getitem__(self, env):
        return self.envs[env]
class RMNISTHierarchy:
    def __init__(self):
        self.parents = np.array([
            -1,  # Root (0)
            0,  # 0-30 degrees group (1)
            0,  # 31-75 degrees group (2)
            1,  # 0-15 degrees (3)
            1,  # 16-30 degrees (4)
            2,  # 31-45 degrees (5)
            2,  # 46-75 degrees (6)
            3,  # 0 degrees (7)
            3,  # 15 degrees (8)
            4,  # 30 degrees (9)
            5,  # 45 degrees (10)
            6,  # 60 degrees (11)
            6  # 75 degrees (12)
        ])
        self.tree = hier.Hierarchy(self.parents)

        self.angle_to_leaf = {
            0: 7,
            15: 8,
            30: 9,
            45: 10,
            60: 11,
            75: 12
        }
        self.leaf_to_angle = {v: k for k, v in self.angle_to_leaf.items()}
class RMNISTAnalyzer:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Added device definition
        mnist = datasets.MNIST('./data', train=True, download=True)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        all_indices = np.arange(len(mnist.data))
        train_idx, val_idx = next(kf.split(all_indices))
        self.train_dataset = RotatedMNIST(split='train', indices=train_idx)
        self.val_dataset = RotatedMNIST(split='train', indices=val_idx)
        self.test_dataset = RotatedMNIST(split='test')
        self.hierarchy = RMNISTHierarchy()
        self.results = {}

        self.model = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Dropout(0.5),
                nn.Linear(128 * 7 * 7, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, len(RotatedMNIST.ENVIRONMENTS))
            ).to(self.device)  # Use self.device

        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        self._train_model()
        self.analyze()

    def plot_lca_results(self):
        metrics = self.results['lca_metrics']['lca_by_angle']
        angles = sorted(metrics.keys())
        lca_distances = [metrics[a]['lca_distance'] for a in angles]
        accuracies = [metrics[a]['accuracy'] for a in angles]

        if all(d == 0 for d in lca_distances) and all(a == 0 for a in accuracies):
            print("No valid metrics to plot - model may need more training")
            return

        fig, ax1 = plt.subplots(figsize=(10, 6))

        colors = ['#2ecc71' if a <= 30 else '#e74c3c' for a in angles]
        bars = ax1.bar(angles, lca_distances, color=colors)
        ax1.set_title('LCA Distance by Rotation Angle')
        ax1.set_xlabel('Angle')
        ax1.set_ylabel('LCA Distance')

        legend_elements = [
            Patch(facecolor='#2ecc71', label='0-30°'),
            Patch(facecolor='#e74c3c', label='31-75°')
        ]
        ax1.legend(handles=legend_elements)
        plt.show()

    def _validate(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = nn.CrossEntropyLoss(reduction='mean')
        self.model.eval()
        total_loss = 0
        total_samples = 0

        with torch.no_grad():
            for angle_idx, angle in enumerate(self.val_dataset.ENVIRONMENTS):
                loader = torch.utils.data.DataLoader(self.val_dataset[angle], batch_size=128, shuffle=False)
                if len(loader) == 0:
                    continue

                for img, _ in loader:
                    img = img.unsqueeze(1).float().to(device)
                    label = torch.full((img.shape[0],), angle_idx).long().to(device)
                    pred = self.model(img)  # Changed here
                    loss = criterion(pred, label)
                    total_loss += loss.item() * img.shape[0]
                    total_samples += img.shape[0]

        return total_loss / total_samples if total_samples > 0 else float('inf')

    def _train_model(self):
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.7)
        criterion = nn.CrossEntropyLoss()
        max_norm = 0.25

        for epoch in range(100):
            self.model.train()
            train_loss = 0
            total_samples = 0
            batch_count = 0
            total_batches = sum(len(torch.utils.data.DataLoader(self.train_dataset[angle], batch_size=128))
                                for angle in self.train_dataset.ENVIRONMENTS)

            for angle_idx, angle in enumerate(self.train_dataset.ENVIRONMENTS):
                dataloader = torch.utils.data.DataLoader(self.train_dataset[angle], batch_size=128, shuffle=True)
                for img, _ in dataloader:
                    batch_count += 1
                    print(f"\rEpoch {epoch}: Processing batch {batch_count}/{total_batches} "
                          f"({(batch_count / total_batches) * 100:.1f}%)", end="")

                    img = img.unsqueeze(1).float().to(self.device)
                    label = torch.full((img.shape[0],), angle_idx).long().to(self.device)

                    optimizer.zero_grad()
                    pred = self.model(img)
                    loss = criterion(pred, label)

                    if not torch.isfinite(loss):
                        print(f"\nWarning: Non-finite loss detected: {loss.item()}")
                        continue

                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                    if torch.isfinite(grad_norm):
                        optimizer.step()
                    else:
                        print(f"\nWarning: Non-finite gradient detected")

                    train_loss += loss.item() * img.shape[0]
                    total_samples += img.shape[0]

            train_loss = train_loss / total_samples
            val_loss = self._validate()
            metrics = self.compute_metrics()
            scheduler.step()

            print(f"\nEpoch {epoch}:")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Accuracy: {metrics['Acc']:.2f}%")
            print(f"Per-class accuracies: {metrics['class_acc']}")
            print("-" * 80)
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            print(f"Accuracy: {metrics['Acc']:.2f}%")
            print(f"Env Independence: {metrics['Env_Ind']:.4f}")
            print(f"Low-level Inv: {metrics['Low_level']:.4f}")
            print(f"Intervention Rob: {metrics['Interv']:.4f}\n")
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')


    def compute_per_angle_lca(self, true_angles, pred_angles):
        per_angle_lca = {}
        batch_size = 1000  # Process in batches to avoid memory issues

        for angle in self.test_dataset.ENVIRONMENTS:
            angle_mask = (true_angles == angle)
            if angle_mask.sum() > 0:
                angle_true = true_angles[angle_mask]
                angle_pred = pred_angles[angle_mask]

                # Process in batches
                lca_depths_list = []
                for i in range(0, len(angle_true), batch_size):
                    batch_true = angle_true[i:i + batch_size]
                    batch_pred = angle_pred[i:i + batch_size]

                    tree_true = np.array([self.hierarchy.angle_to_leaf[a] for a in batch_true])
                    tree_pred = np.array([self.hierarchy.angle_to_leaf[p] for p in batch_pred])

                    find_lca = hier.FindLCA(self.hierarchy.tree)
                    lca_nodes = find_lca(tree_true, tree_pred)

                    wrong_mask = (batch_true != batch_pred)
                    if wrong_mask.sum() > 0:
                        lca_depths = self.hierarchy.tree.depths()[lca_nodes[wrong_mask]]
                        lca_depths_list.append(lca_depths)

                if lca_depths_list:
                    mean_lca_depth = float(np.concatenate(lca_depths_list).mean())
                else:
                    mean_lca_depth = 0.0

                per_angle_lca[angle] = {
                    'lca_distance': mean_lca_depth,
                    'accuracy': (angle_true == angle_pred).mean()
                }
        return per_angle_lca

    def analyze(self):
        self.model.to(self.device)
        self.model.eval()
        angle_metrics = {}

        for angle in self.test_dataset.ENVIRONMENTS:
            correct = 0
            total = 0
            lca_depths = []

            with torch.no_grad():
                loader = torch.utils.data.DataLoader(self.test_dataset[angle], batch_size=128)
                for data, _ in loader:
                    data = data.unsqueeze(1).to(self.device)
                    pred = self.model(data).argmax(dim=1)
                    pred_angles = np.array([self.test_dataset.ENVIRONMENTS[p.item()] for p in pred])

                    true_idx = np.array([self.hierarchy.angle_to_leaf[angle]] * len(pred_angles))
                    pred_idx = np.array([self.hierarchy.angle_to_leaf[a] for a in pred_angles])

                    find_lca = hier.FindLCA(self.hierarchy.tree)
                    lca_nodes = find_lca(true_idx, pred_idx)
                    lca_depths.extend(self.hierarchy.tree.depths()[lca_nodes].tolist())

                    correct += (pred == self.test_dataset.ENVIRONMENTS.index(angle)).sum().item()
                    total += len(data)

            angle_metrics[angle] = {
                'lca_distance': np.mean(lca_depths) if lca_depths else 0,
                'accuracy': (correct / total) if total > 0 else 0
            }

        self.results['lca_metrics'] = {'lca_by_angle': angle_metrics}

    def get_features(self, angle):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features = []
        self.model.eval()

        with torch.no_grad():
            loader = torch.utils.data.DataLoader(self.test_dataset[angle], batch_size=128)
            for data, _ in loader:
                data = data.unsqueeze(1).float().to(device)
                # Get features from second to last layer
                feat = data
                for layer in list(self.model.children())[:-1]:
                    feat = layer(feat)
                features.append(feat.cpu().numpy())

        return np.concatenate(features)

    def get_predictions(self, angle):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        predictions = []
        self.model.eval()

        with torch.no_grad():
            loader = torch.utils.data.DataLoader(self.test_dataset[angle], batch_size=128)
            for data, _ in loader:
                data = data.unsqueeze(1).float().to(device)
                pred = self.model(data).cpu().numpy()
                predictions.append(pred)

        return np.concatenate(predictions)

    def compute_metrics(self):
        results = {'Acc': 0, 'Env_Ind': 0, 'Low_level': 0, 'Interv': 0}
        class_correct = torch.zeros(len(self.test_dataset.ENVIRONMENTS), device=self.device)
        class_total = torch.zeros(len(self.test_dataset.ENVIRONMENTS), device=self.device)

        self.model.eval()
        with torch.no_grad():
            for angle_idx, angle in enumerate(self.test_dataset.ENVIRONMENTS):
                loader = torch.utils.data.DataLoader(self.test_dataset[angle], batch_size=128)
                for data, _ in loader:
                    data = data.unsqueeze(1).to(self.device)
                    outputs = self.model(data)
                    pred = outputs.argmax(dim=1)
                    class_correct[angle_idx] += (pred == angle_idx).sum()
                    class_total[angle_idx] += data.size(0)

        class_acc = (class_correct / class_total * 100).cpu().numpy()
        results['Acc'] = class_acc.mean()
        results['class_acc'] = class_acc
        return results

    def compute_feature_distances(self):
        # Simplified R1 metric
        features = []
        with torch.no_grad():
            for angle in self.test_dataset.ENVIRONMENTS:
                feat = self.get_features(angle)
                features.append(feat)

        total_dist = 0
        count = 0
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                dist = np.mean(np.abs(features[i] - features[j]))
                total_dist += dist
                count += 1
        return total_dist / count if count > 0 else 0

    def compute_intervention_robustness(self):
        # Simplified R2 metric
        angle_preds = {}
        for angle in self.test_dataset.ENVIRONMENTS:
            preds = self.get_predictions(angle)
            angle_preds[angle] = preds

        total_diff = 0
        count = 0
        for a1 in angle_preds:
            for a2 in angle_preds:
                if a1 != a2:
                    diff = np.mean(np.abs(angle_preds[a1] - angle_preds[a2]))
                    total_diff += diff
                    count += 1
        return total_diff / count if count > 0 else 0
def rmnist():
    analyzer = RMNISTAnalyzer()
    print("RMNIST LCA metrics:", analyzer.results['lca_metrics'])
    analyzer.plot_lca_results()


class Camelyon17Dataset(Dataset):
    def __init__(self, root_dir: str, metadata_path: str, hospital_id: int = None):
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3]),  # Keep only RGB channels
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.metadata = pd.read_csv(metadata_path)
        if hospital_id is not None:
            self.metadata = self.metadata[self.metadata['center'] == hospital_id]

        self.data = []
        self._prepare_data()

    def _prepare_data(self):
        for _, row in self.metadata.iterrows():
            patient_id = f"patient_{row['patient']:03d}"
            node_id = f"node_{row['node']}"
            patch_filename = f"patch_{patient_id}_{node_id}_x_{row['x_coord']}_y_{row['y_coord']}.png"
            patch_path = self.root_dir / f"{patient_id}_{node_id}" / patch_filename

            if patch_path.exists():
                self.data.append({
                    'path': str(patch_path),
                    'label': row['tumor'],
                    'hospital': row['center']
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = Image.open(sample['path']).convert('RGB')
        image = self.transform(image)

        return {
            'image': image,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'hospital': torch.tensor(sample['hospital'], dtype=torch.long)
        }
class CausalKernel:
    def __init__(self, dataset: Camelyon17Dataset):
        self.dataset = dataset
        self.hospital_data = {h: {} for h in range(5)}
        self._compute_conditionals()

    def _compute_conditionals(self):
        dataloader = DataLoader(self.dataset, batch_size=32, shuffle=False)

        for batch in dataloader:
            images = batch['image']
            labels = batch['label']
            hospitals = batch['hospital']

            for h in range(5):
                for y in [0, 1]:
                    mask = (hospitals == h) & (labels == y)
                    if mask.any():
                        if y not in self.hospital_data[h]:
                            self.hospital_data[h][y] = []
                        self.hospital_data[h][y].append(images[mask])

        for h in range(5):
            for y in [0, 1]:
                if y in self.hospital_data[h] and self.hospital_data[h][y]:
                    features = torch.cat(self.hospital_data[h][y])
                    self.hospital_data[h][y] = {
                        'mean': features.mean(0),
                        'std': features.std(0)
                    }

    def sample(self, hospital: int, label: int, n_samples: int = 1):
        params = self.hospital_data[hospital][label]
        mean = params['mean']
        std = params['std']
        noise = torch.randn(n_samples, *mean.shape)
        return mean + noise * std
def create_causal_space(patches_dir: str, metadata_path: str):
    dataset = Camelyon17Dataset(patches_dir, metadata_path)
    kernel = CausalKernel(dataset)
    return {
        'dataset': dataset,
        'kernel': kernel,
        'sample_space': {
            'Y': torch.tensor([0, 1]),
            'E': torch.arange(5),
            'X_shape': (3, 96, 96)
        }
    }
class Metrics:
    @staticmethod
    def accuracy(outputs, labels):
        """Classification Accuracy"""
        pred = torch.argmax(outputs, dim=1)
        return (pred == labels).float().mean().item()

    @staticmethod
    def environment_independence(representations, environments, labels):
        """Environment Independence via Mutual Information"""
        rep_np = representations.detach().cpu().numpy()
        env_np = environments.cpu().numpy()
        lab_np = labels.cpu().numpy()

        mi_sum = 0
        for y in np.unique(lab_np):
            mask = lab_np == y
            if np.sum(mask) > 0:
                mi = mutual_info_score(
                    rep_np[mask].argmax(axis=1),
                    env_np[mask]
                )
                mi_sum += mi * np.mean(mask)
        return mi_sum

    @staticmethod
    def low_level_invariance(representations, environments):
        """R1: Low-level Invariance"""
        rep_mean_per_env = []
        for e in torch.unique(environments):
            mask = environments == e
            if mask.any():
                rep_mean_per_env.append(representations[mask].mean(0))

        rep_mean_per_env = torch.stack(rep_mean_per_env)
        return torch.cdist(rep_mean_per_env, rep_mean_per_env).mean().item()

    @staticmethod
    def intervention_robustness(model, obs_data, int_data):
        """R2: Intervention Robustness"""
        with torch.no_grad():
            obs_outputs = model(obs_data)
            int_outputs = model(int_data)
        return F.kl_div(
            F.log_softmax(obs_outputs, dim=1),
            F.softmax(int_outputs, dim=1),
            reduction='batchmean'
        ).item()
class SimpleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 11 * 11, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
def evaluate_model(model, dataloader, metrics):
    model.eval()
    total_acc = 0
    total_env_ind = 0
    total_r1 = 0
    total_r2 = 0
    batches = 0

    for batch in dataloader:
        images = batch['image']
        labels = batch['label']
        hospitals = batch['hospital']

        with torch.no_grad():
            outputs = model(images)
            features = model.fc1(images.view(images.size(0), -1))

        total_acc += metrics.accuracy(outputs, labels)
        total_env_ind += metrics.environment_independence(features, hospitals, labels)
        total_r1 += metrics.low_level_invariance(features, hospitals)

        # Simulate intervention by adding noise
        int_images = images + 0.1 * torch.randn_like(images)
        total_r2 += metrics.intervention_robustness(model, images, int_images)

        batches += 1

    return {
        'accuracy': total_acc / batches,
        'env_independence': total_env_ind / batches,
        'r1': total_r1 / batches,
        'r2': total_r2 / batches
    }
def camelyon17():
    BASE_DIR = Path(r"C:\University\Spring2025\Research\Session9\baseline")
    PATCHES_DIR = BASE_DIR / "data" / "camelyon17_v1.0" / "patches"
    METADATA_PATH = BASE_DIR / "data" / "camelyon17_v1.0" / "metadata.csv"

    print("Creating dataset...")
    dataset = Camelyon17Dataset(PATCHES_DIR, METADATA_PATH)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    print("Initializing model and metrics...")
    model = SimpleConvNet()
    metrics = Metrics()

    print("Training and evaluating...")
    results = evaluate_model(model, dataloader, metrics)

    print("\nResults:")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Environment Independence: {results['env_independence']:.3f}")
    print(f"Low-level Invariance (R1): {results['r1']:.3f}")
    print(f"Intervention Robustness (R2): {results['r2']:.3f}")

if __name__ == "__main__":
    # cmnist()
    # rmnist()
    camelyon17()
    # ballagent()