import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class IRMv1Loss(nn.Module):
    def __init__(self, lambda_irm: float = 1.0):
        super().__init__()
        self.lambda_irm = lambda_irm

    def forward(self, logits: torch.Tensor, y: torch.Tensor, env_idx: torch.Tensor) -> torch.Tensor:
        loss = 0
        penalty = 0
        n_envs = env_idx.unique().numel()

        for i in range(n_envs):
            env_mask = env_idx == i
            env_logits = logits[env_mask]
            env_y = y[env_mask]

            # ERM loss for this environment
            env_loss = F.cross_entropy(env_logits, env_y)
            loss += env_loss

            # Compute gradient penalty
            scale = torch.ones_like(env_loss).requires_grad_()
            grad = torch.autograd.grad(env_loss * scale, [logits], create_graph=True)[0]
            penalty += (grad ** 2).sum()

        loss = loss / n_envs + self.lambda_irm * penalty
        return loss


class gIRMv1Loss(IRMv1Loss):
    def forward(self, logits: torch.Tensor, y: torch.Tensor, env_idx: torch.Tensor) -> torch.Tensor:
        loss = 0
        penalty = 0
        n_envs = env_idx.unique().numel()

        for i in range(n_envs):
            env_mask = env_idx == i
            env_logits = logits[env_mask]
            env_y = y[env_mask]

            # Compute class weights for this environment
            unique_y, counts = torch.unique(env_y, return_counts=True)
            weights = torch.zeros_like(env_y, dtype=torch.float)
            for y_val, count in zip(unique_y, counts):
                weights[env_y == y_val] = 1.0 / count
            weights = weights / weights.sum()

            # Weighted ERM loss
            env_loss = F.cross_entropy(env_logits, env_y, reduction='none')
            env_loss = (env_loss * weights).sum()
            loss += env_loss

            # Compute gradient penalty with weights
            scale = torch.ones_like(env_loss).requires_grad_()
            grad = torch.autograd.grad(env_loss * scale, [logits], create_graph=True)[0]
            penalty += (grad ** 2).sum()

        loss = loss / n_envs + self.lambda_irm * penalty
        return loss


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, input_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, num_epochs: int = 10, device: str = 'cuda') -> Tuple[List[float], List[float]]:
    model = model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, labels, env_idx) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            env_idx = env_idx.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels, env_idx)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, labels, env_idx in val_loader:
                data, labels = data.to(device), labels.to(device)
                env_idx = env_idx.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels, env_idx)
                val_loss += loss.item()

        val_losses.append(val_loss / len(val_loader))

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Training Loss: {train_losses[-1]:.4f}')
        print(f'Validation Loss: {val_losses[-1]:.4f}')

    return train_losses, val_losses


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cuda') -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    test_loss = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, labels, _ in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)

    return accuracy, avg_loss


def plot_training_curves(train_losses: List[float], val_losses: List[float], title: str = 'Training Curves'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


class ColoredMNISTDataset(Dataset):
    def __init__(self, is_train: bool = True, env: str = 'e1'):
        super().__init__()
        self.is_train = is_train
        self.env = env

        # Load base MNIST data
        from torchvision.datasets import MNIST
        mnist = MNIST('./data', train=is_train, download=True)
        self.images = mnist.data.numpy()
        self.labels = mnist.targets.numpy()

        # Apply coloring based on environment rules
        self.colored_images = self._color_images()

    def _color_images(self):
        colored = np.zeros((len(self.images), 28, 28, 3), dtype=np.uint8)
        for i, (img, label) in enumerate(zip(self.images, self.labels)):
            if self.env == 'e1':
                p_red = 0.75 if label % 2 == 0 else 0.25
            else:  # e2
                p_red = 0.25 if label % 2 == 0 else 0.75

            # Apply color
            if np.random.random() < p_red:
                colored[i, :, :, 0] = img  # Red
            else:
                colored[i, :, :, 1] = img  # Green

        return colored

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.colored_images[idx]
        img = torch.FloatTensor(img.transpose(2, 0, 1)) / 255.0
        label = torch.LongTensor([self.labels[idx]])[0]
        env_idx = torch.LongTensor([0 if self.env == 'e1' else 1])[0]
        return img, label, env_idx


class RotatedMNISTDataset(Dataset):
    def __init__(self, is_train: bool = True, angle: float = 15.0):
        super().__init__()
        self.is_train = is_train
        self.angle = angle

        # Load base MNIST data
        from torchvision.datasets import MNIST
        mnist = MNIST('./data', train=is_train, download=True)
        self.images = mnist.data.numpy()
        self.labels = mnist.targets.numpy()

        # Apply rotation
        self.rotated_images = self._rotate_images()

    def _rotate_images(self):
        rotated = []
        for img in self.images:
            img_pil = Image.fromarray(img.astype(np.uint8))
            rotated_pil = img_pil.rotate(self.angle)
            rotated.append(np.array(rotated_pil))
        return np.array(rotated)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.rotated_images[idx]
        img = torch.FloatTensor(img).unsqueeze(0) / 255.0
        label = torch.LongTensor([self.labels[idx]])[0]
        env_idx = torch.LongTensor([int(self.angle / 15)])[0]
        return img, label, env_idx


class Camelyon17Dataset(Dataset):
    def __init__(self, hospital_idx: int = 1):
        super().__init__()
        self.hospital_idx = hospital_idx

        # Generate synthetic data following the paper's specifications
        self.images = []
        self.labels = []
        self.generate_data()

    def generate_data(self):
        # Following the distribution from the paper
        distributions = {
            1: {'total': 17934, 'tumor': 7786},
            2: {'total': 15987, 'tumor': 6446},
            3: {'total': 16828, 'tumor': 7212},
            4: {'total': 17155, 'tumor': 7502},
            5: {'total': 16960, 'tumor': 7089}
        }

        dist = distributions[self.hospital_idx]
        num_tumor = dist['tumor']
        num_normal = dist['total'] - dist['tumor']

        # Generate data with hospital-specific characteristics
        bg_color = np.array([
            [0.98, 0.95, 0.95],  # h1
            [0.97, 0.95, 0.98],  # h2
            [0.98, 0.98, 0.92],  # h3
            [0.97, 0.95, 0.93],  # h4
            [0.95, 0.97, 0.97]  # h5
        ])[self.hospital_idx - 1]

        # Generate synthetic images
        for i in range(num_tumor):
            self.images.append(self._generate_image(True, bg_color))
            self.labels.append(1)

        for i in range(num_normal):
            self.images.append(self._generate_image(False, bg_color))
            self.labels.append(0)

    def _generate_image(self, is_tumor: bool, bg_color: np.ndarray) -> np.ndarray:
        size = 96
        image = np.ones((size, size, 3)) * bg_color

        if is_tumor:
            # Add tumor-like features
            center = size // 2
            radius = size // 5
            y, x = np.ogrid[-center:size - center, -center:size - center]
            mask = x * x + y * y <= radius * radius

            # Hospital-specific tumor coloring
            tumor_colors = [
                [0.6, 0.2, 0.2],  # h1
                [0.5, 0.15, 0.4],  # h2
                [0.5, 0.4, 0.15],  # h3
                [0.5, 0.3, 0.2],  # h4
                [0.3, 0.4, 0.4]  # h5
            ]

            tumor_color = np.array(tumor_colors[self.hospital_idx - 1])
            image[mask] = tumor_color

        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.FloatTensor(self.images[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        env_idx = torch.LongTensor([self.hospital_idx - 1])[0]
        return img, label, env_idx
