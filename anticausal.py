"""
anticausal.py - Implements anti-causal kernel characterization and ColoredMNIST
"""
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torchvision import transforms
import torch
from torch.utils.data import TensorDataset
from scipy.ndimage import rotate
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd


class ColoredMNIST(Dataset):
    def __init__(self, env: str, root='./data', train=True):
        super().__init__()
        self.env = env
        mnist = datasets.MNIST(root=root, train=train, download=True)
        self.images = mnist.data.float() / 255.0
        self.labels = mnist.targets
        self.colored_images = self._color_images()
        self.env_labels = torch.full_like(self.labels, float(env == 'e2'))

    def _color_images(self) -> torch.Tensor:
        n_images = len(self.images)
        colored = torch.zeros((n_images, 3, 28, 28))
        for i, (img, label) in enumerate(zip(self.images, self.labels)):
            p_red = 0.75 if ((label % 2 == 0) == (self.env == 'e1')) else 0.25
            is_red = torch.rand(1) < p_red
            if is_red:
                colored[i, 0] = img
            else:
                colored[i, 1] = img
        return colored

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.colored_images[idx], self.labels[idx], self.env_labels[idx]


class RotatedMNIST:
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']
    def __init__(self, root='./data'):
        mnist = datasets.MNIST(root, train=True, download=True)
        self.images, self.labels = mnist.data.numpy(), mnist.targets.numpy()
        self.envs = {}
        for i, angle in enumerate([0, 15, 30, 45, 60, 75]):
            rotated_images = np.array([rotate(img, angle, reshape=False) for img in self.images])
            env_labels = torch.full_like(torch.tensor(self.labels), i)
            self.envs[str(angle)] = TensorDataset(
                torch.tensor(rotated_images, dtype=torch.float32) / 255.0,
                torch.tensor(self.labels),
                env_labels
            )

    def __getitem__(self, env):
        return self.envs[env]

    def rotate_dataset(self, images, labels, angle):
        rotated_images = np.zeros_like(images)
        for i, img in enumerate(images):
            rotated_images[i] = rotate(img, angle, reshape=False)
        return TensorDataset(torch.tensor(rotated_images), torch.tensor(labels))


class Camelyon17Dataset(Dataset):
    def __init__(self, root_dir, metadata_path, hospital_id, indices):
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata['center'] == hospital_id]
        self.metadata = self.metadata.loc[indices]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_name = f"patch_patient_{row['patient']:03d}_node_{row['node']}_x_{row['x_coord']}_y_{row['y_coord']}.png"
        img_path = self.root_dir / f"patient_{row['patient']:03d}_node_{row['node']}" / img_name
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        tumor = torch.tensor(row['tumor'], dtype=torch.long)
        hospital = torch.zeros(5)
        hospital[row['center']] = 1
        return image, tumor, hospital

class BallAgentDataset:
    def __init__(self, n_balls=3, n_samples=10000):
        self.n_balls = n_balls
        self.n_samples = n_samples
        self.colors = [(1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0.7, 0)]
        self.size = 64
        self.positions, self.images, self.interventions = self._generate_data()

    def _generate_data(self):
        positions = []
        images = []
        interventions = []
        while len(positions) < self.n_samples:
            pos = np.random.uniform(0.1, 0.9, (self.n_balls, 2))
            valid = True
            for i in range(self.n_balls):
                for j in range(i + 1, self.n_balls):
                    dist = np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
                    if dist < 0.2:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            n_interventions = np.random.randint(0, self.n_balls + 1)
            intervention_idx = np.random.choice(self.n_balls, n_interventions, replace=False)
            intervention = np.zeros(self.n_balls, dtype=bool)
            intervention[intervention_idx] = True
            positions.append(pos.flatten())
            images.append(self._generate_ball_image(pos))
            interventions.append(intervention)
        return np.array(positions), np.array(images), np.array(interventions)

    def _generate_ball_image(self, positions):
        img = np.zeros((self.size, self.size, 3))
        for i, (x, y) in enumerate(positions):
            px, py = int(x * self.size), int(y * self.size)
            color = self.colors[i]
            y_grid, x_grid = np.ogrid[-8:9, -8:9]
            distances = np.sqrt(x_grid ** 2 + y_grid ** 2)
            intensity = np.exp(-distances ** 2 / 16)
            for c in range(3):
                y_coords = np.clip(py + y_grid, 0, self.size - 1)
                x_coords = np.clip(px + x_grid, 0, self.size - 1)
                img[y_coords, x_coords, c] += intensity * color[c]
        return np.clip(img, 0, 1)
class BallAgentEnvironment(Dataset):
    def __init__(self, dataset, is_train=True):
        self.data = dataset
        self.is_train = is_train
        self.train_idx, self.test_idx = self._split_data()
    def _split_data(self):
        n = len(self.data.positions)
        indices = np.random.permutation(n)
        split = int(0.8 * n)
        return indices[:split], indices[split:]

    def __len__(self):
        return len(self.train_idx) if self.is_train else len(self.test_idx)

    def __getitem__(self, idx):
        indices = self.train_idx if self.is_train else self.test_idx
        real_idx = indices[idx]
        x = torch.FloatTensor(self.data.images[real_idx])
        y = torch.FloatTensor(self.data.positions[real_idx])
        e = torch.FloatTensor(self.data.interventions[real_idx])
        return x.permute(2, 0, 1), y, e


