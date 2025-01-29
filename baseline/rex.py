"""
rex.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import argparse
from torch.optim import lr_scheduler
from scipy.ndimage import rotate
from sklearn.metrics import mutual_info_score
import os
import gdown
import zipfile
import wilds
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class MLP(nn.Module):
    def __init__(self, hidden_dim=256):  # Made hidden dim configurable
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),  # Added BatchNorm
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),  # Added BatchNorm
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),  # Added BatchNorm
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),  # Added dropout
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        return self.net(x).squeeze()
def mean_nll(logits, y):
    return F.cross_entropy(logits, y, reduction='none')
def rex_penalty(losses, use_mse=True):
    mean = losses.mean()
    if use_mse:
        return ((losses - mean) ** 2).mean()
    else:
        return (losses - mean).abs().mean()
def pretty_print(*values):
    col_width = 13

    def format_val(v):
        if isinstance(v, float):
            return "{:.5f}".format(v).ljust(col_width)
        elif isinstance(v, np.ndarray):
            return np.array2string(v, precision=5, floatmode='fixed').ljust(col_width)
        else:
            return str(v).ljust(col_width)

    print("   ".join(format_val(v) for v in values))
def calculate_mutual_information(model, loader1, loader2, device):
    reps = []
    envs = []
    with torch.no_grad():
        for (x1, _), (x2, _) in zip(loader1, loader2):
            x1, x2 = x1.to(device), x2.to(device)
            rep1 = model.net[:11](x1).view(x1.size(0), -1)
            rep2 = model.net[:11](x2).view(x2.size(0), -1)
            reps.extend([rep1.mean(1).cpu(), rep2.mean(1).cpu()])
            envs.extend([torch.zeros(len(x1)), torch.ones(len(x2))])
    reps = torch.cat(reps).numpy()
    envs = torch.cat(envs).long().numpy()
    return mutual_info_score(envs, np.digitize(reps, np.linspace(reps.min(), reps.max(), 20)))
def calculate_invariance(model, loader1, loader2, device):
    """Calculate representation stability across environments"""
    model.eval()
    dists = []
    with torch.no_grad():
        for (x1, y1), (x2, y2) in zip(loader1, loader2):
            x1, x2 = x1.to(device), x2.to(device)
            rep1 = model.net[:11](x1)
            rep2 = model.net[:11](x2)

            # Calculate Wasserstein distance between distributions
            dist = torch.norm(rep1.mean(0) - rep2.mean(0))
            dists.append(dist.item())

    return np.mean(dists)


class ColoredMNIST(Dataset):
    def __init__(self, env, root='./data', train_data=None, indices=None):
        super().__init__()
        self.env = env

        if train_data is not None:
            # Use provided training data
            self.images = train_data['images'][indices].float() / 255.0
            self.labels = train_data['labels'][indices]
        else:
            # Load MNIST for test set
            mnist = datasets.MNIST(root, train=(env != 'test'), download=True)
            self.images = mnist.data.float() / 255.0
            self.labels = mnist.targets

        # Convert to binary labels and rest of the processing remains same
        # self.labels = (self.labels < 5).float()
        self.labels = torch.tensor(self.labels).long()

        # Flip labels with 25% probability
        flip_mask = (torch.rand(len(self.labels)) < 0.25).float()
        self.labels = (self.labels + flip_mask) % 2

        # Add color with environment-dependent correlation
        if env == 'e1':
            # First environment: strong correlation (0.9)
            color_mask = (torch.rand(len(self.labels)) < 0.1).float()
        else:
            # Second environment and test: weak correlation (0.2)
            color_mask = (torch.rand(len(self.labels)) < 0.8).float()

        self.colors = (self.labels + color_mask) % 2

        # Expand colors to match image dimensions
        self.colors = self.colors.view(-1, 1, 1).expand(-1, 28, 28)

        # Create colored images: (batch_size, channels, height, width)
        self.images = torch.stack([
            self.images * self.colors,  # Red channel
            self.images * (1 - self.colors),  # Green channel
            torch.zeros_like(self.images)  # Blue channel
        ], dim=1)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)
def calculate_intervention_robustness(model, loader_test, device):
    """Compare predictions under different interventions"""
    model.eval()
    orig_preds = []
    color_flipped_preds = []

    with torch.no_grad():
        for x, y in loader_test:
            x = x.to(device)
            # Original prediction
            orig_pred = model(x)

            # Flip colors and predict
            x_flip = torch.stack([
                x[:, 1],  # Green channel becomes red
                x[:, 0],  # Red channel becomes green
                x[:, 2]  # Blue stays same
            ], dim=1)
            flip_pred = model(x_flip)

            orig_preds.extend(orig_pred.cpu())
            color_flipped_preds.extend(flip_pred.cpu())

    # Calculate prediction consistency
    consistency = np.mean(np.abs(np.array(orig_preds) - np.array(color_flipped_preds)))
    return consistency
def train_rex(model, train_loaders, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_regularizer_weight)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    train_iter1 = iter(train_loaders[0])
    train_iter2 = iter(train_loaders[1])

    pretty_print('step', 'train nll', 'train acc', 'rex penalty', 'test acc', 'env_ind', 'low_inv', 'int_rob')
    best_test_acc = 0
    for step in range(args.steps):
        model.train()

        # Get batch from each environment
        try:
            x1, y1 = next(train_iter1)
        except StopIteration:
            train_iter1 = iter(train_loaders[0])
            x1, y1 = next(train_iter1)

        try:
            x2, y2 = next(train_iter2)
        except StopIteration:
            train_iter2 = iter(train_loaders[1])
            x2, y2 = next(train_iter2)

        x1, y1 = x1.to(args.device), y1.to(args.device)
        x2, y2 = x2.to(args.device), y2.to(args.device)

        # Compute losses for each environment
        logits1 = model(x1)
        logits2 = model(x2)

        loss1 = mean_nll(logits1, y1).mean()
        loss2 = mean_nll(logits2, y2).mean()

        train_nll = (loss1 + loss2) / 2
        train_acc = (mean_accuracy(logits1, y1) + mean_accuracy(logits2, y2)) / 2

        # Compute REx penalty
        penalty_weight = (args.penalty_weight if step >= args.penalty_anneal_iters else 1.0)
        rex_pen = ((loss1.mean() - loss2.mean()) ** 2)  # Simplified penalty computation

        # Total loss with scaling
        loss = train_nll
        if penalty_weight > 1.0:
            loss += (penalty_weight * rex_pen) / penalty_weight
        else:
            loss += penalty_weight * rex_pen

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % args.eval_interval == 0:
            test_acc = evaluate(model, args.test_loader, args.device)
            env_ind = calculate_mutual_information(model, train_loaders[0], train_loaders[1], args.device)
            low_inv = calculate_invariance(model, train_loaders[0], train_loaders[1], args.device)
            int_rob = calculate_intervention_robustness(model, args.test_loader, args.device)

            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                rex_pen.detach().cpu().numpy(),
                test_acc,
                env_ind,
                low_inv,
                int_rob
            )

    return best_test_acc
def mean_accuracy(logits, y):
    preds = (logits > 0.).float()
    return ((preds - y).abs() < 1e-2).float().mean()
def evaluate(model, loader, device):
    model.eval()
    acc = 0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = (model(x) > 0.).float()
            acc += (p == y).sum().item()
            n += len(x)
    model.train()
    return acc / n
def cmnist():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.002)  # Slightly higher learning rate
    parser.add_argument('--l2_regularizer_weight', type=float, default=1e-4)  # Reduced L2
    parser.add_argument('--steps', type=int, default=5001)
    parser.add_argument('--penalty_anneal_iters', type=int, default=1500)  # Increased warmup
    parser.add_argument('--penalty_weight', type=float, default=12000.0)  # Increased penalty
    parser.add_argument('--batch_size', type=int, default=512)  # Larger batch size
    parser.add_argument('--eval_interval', type=int, default=100)
    args = parser.parse_args()

    # Setup device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets with proper splits
    train_size = 50000
    mnist = datasets.MNIST('./data', train=True, download=True)
    train_data = {
        'images': mnist.data[:train_size],
        'labels': mnist.targets[:train_size]
    }

    # Split training data for two environments
    train_e1 = ColoredMNIST('e1', train_data=train_data, indices=range(0, train_size, 2))
    train_e2 = ColoredMNIST('e2', train_data=train_data, indices=range(1, train_size, 2))
    test_set = ColoredMNIST('test')

    # Create dataloaders
    train_loader1 = DataLoader(train_e1, batch_size=args.batch_size, shuffle=True)
    train_loader2 = DataLoader(train_e2, batch_size=args.batch_size, shuffle=True)
    args.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Create model
    model = MLP().to(args.device)

    # Train model
    final_acc = train_rex(model, [train_loader1, train_loader2], args)

    print(f"\nTraining completed!")
    print(f"Final test accuracy: {final_acc:.4f}")


class RotatedMNIST(Dataset):
    def __init__(self, env, root='./data', train_data=None, indices=None, transform=None):
        super().__init__()
        self.env = env
        self.transform = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])

        if train_data is not None:
            self.images = train_data['images'][indices]
            self.labels = train_data['labels'][indices]
        else:
            mnist = datasets.MNIST(root, train=(env != 'test'), download=True)
            self.images = mnist.data
            self.labels = mnist.targets

        self.labels = self.labels.clone().detach()

        self.images = self.images.numpy()
        angle_map = {
            'e1': 0, 'e2': 15, 'e3': 30,
            'e4': 45, 'e5': 60, 'test': 75
        }
        angle = angle_map.get(env, 75)

        rotated_images = np.zeros_like(self.images)
        for i, img in enumerate(self.images):
            rotated_images[i] = rotate(img, angle, reshape=False)

        self.images = torch.tensor(rotated_images).float() / 255.0
        self.images = self.images.unsqueeze(1).repeat(1, 3, 1, 1)
        if self.transform:
            self.images = self.transform(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)
def revaluate(model, loader, device):
    model.eval()
    acc = 0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            acc += (preds == y).sum().item()
            n += len(x)
    model.train()
    return acc / n
def rcalculate_intervention_robustness(model, loader_test, device):
    model.eval()
    orig_preds = []
    rotated_preds = []

    with torch.no_grad():
        for x, y in loader_test:
            x = x.to(device)
            orig_pred = model(x).argmax(dim=1)

            x_rot = x.cpu().numpy()
            for i in range(len(x_rot)):
                x_rot[i] = rotate(x_rot[i].transpose(1,2,0), 15, reshape=False).transpose(2,0,1)
            x_rot = torch.tensor(x_rot).float().to(device)

            rot_pred = model(x_rot).argmax(dim=1)
            orig_preds.extend(orig_pred.cpu())
            rotated_preds.extend(rot_pred.cpu())

    consistency = np.mean(np.abs(np.array(orig_preds) - np.array(rotated_preds)))
    return consistency
def rmean_accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean()
def train_rexr(model, train_loaders, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.steps)
    pretty_print('step', 'rex penalty', 'test acc', 'env_ind', 'low_inv', 'int_rob')
    best_acc = 0

    for step in range(args.steps):
        model.train()
        train_iters = [iter(loader) for loader in train_loaders]
        losses = []

        for i, train_iter in enumerate(train_iters):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loaders[i])
                x, y = next(train_iter)

            x, y = x.to(args.device), y.to(args.device)
            logits = model(x)
            env_loss = F.cross_entropy(logits, y)
            losses.append(env_loss)

        losses = torch.stack(losses)
        mean_loss = losses.mean()
        rex_penalty = ((losses - mean_loss) ** 2).mean()
        loss = mean_loss + 1000.0 * rex_penalty  # Reduced penalty weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % args.eval_interval == 0:
            acc = revaluate(model, args.test_loader, args.device)
            env_ind = calculate_mutual_information(model, train_loaders[0], train_loaders[1], args.device)
            low_inv = calculate_invariance(model, train_loaders[0], train_loaders[1], args.device)
            int_rob = rcalculate_intervention_robustness(model, args.test_loader, args.device)
            best_acc = max(best_acc, acc)
            pretty_print(step, rex_penalty.detach().cpu().numpy(), acc, env_ind, low_inv, int_rob)

    return best_acc
def rmnist():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--l2_regularizer_weight', type=float, default=1e-4)
    parser.add_argument('--steps', type=int, default=5001)
    parser.add_argument('--penalty_anneal_iters', type=int, default=1500)
    parser.add_argument('--penalty_weight', type=float, default=12000.0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--eval_interval', type=int, default=100)
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets
    train_size = 50000
    mnist = datasets.MNIST('./data', train=True, download=True)
    train_data = {
        'images': mnist.data[:train_size],
        'labels': mnist.targets[:train_size]
    }

    # Create all training environments
    train_envs = []
    train_loaders = []
    for i in range(1, 6):
        env = RotatedMNIST(f'e{i}', train_data=train_data,
                          indices=range(i - 1, train_size, 5))
        train_envs.append(env)
        train_loaders.append(DataLoader(env, batch_size=args.batch_size, shuffle=True))

    test_set = RotatedMNIST('test')
    args.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Create and train model
    model = MLP().to(args.device)
    final_acc = train_rexr(model, train_loaders, args)

    print(f"\nTraining completed!")
    print(f"Final test accuracy: {final_acc:.4f}")


class Camelyon17Dataset(Dataset):
    def __init__(self, center_id, transform=None):
        import pandas as pd
        base_path = os.path.dirname(os.path.abspath(__file__))
        metadata_path = os.path.join(base_path, "data", "camelyon17_v1.0", "metadata.csv")
        self.patches_path = os.path.join(base_path, "data", "camelyon17_v1.0", "patches")

        print(f"Looking for metadata at: {metadata_path}")
        print(f"Looking for patches at: {self.patches_path}")

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")

        metadata = pd.read_csv(metadata_path)
        center_data = metadata[metadata['center'] == center_id]

        if len(center_data) == 0:
            raise ValueError(f"No data found for center_id {center_id}")

        self.image_paths = []
        self.labels = []

        for _, row in center_data.iterrows():
            img_path = os.path.join(
                self.patches_path,
                f"patient_{str(row['patient']).zfill(3)}_node_{row['node']}",
                f"patch_patient_{str(row['patient']).zfill(3)}_node_{row['node']}_x_{row['x_coord']}_y_{row['y_coord']}.png"
            )
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                self.labels.append(row['tumor'])

        if not self.image_paths:
            raise ValueError(f"No valid images found for center_id {center_id}")

        self.labels = torch.tensor(self.labels)
        self.transform = transform if transform else transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {str(e)}")

    def __len__(self):
        return len(self.image_paths)
class CamelyonMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)
def train_camelyon(model, train_loaders, args, test_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.steps)

    for step in range(args.steps):
        losses = []
        model.train()

        for loader in train_loaders:
            try:
                x, y = next(iter(loader))
            except StopIteration:
                continue

            x, y = x.to(args.device), y.to(args.device).float()
            logits = model(x).squeeze()
            loss = F.binary_cross_entropy_with_logits(logits, y)
            losses.append(loss)

        losses = torch.stack(losses)
        mean_loss = losses.mean()
        penalty = ((losses - mean_loss) ** 2).mean()

        loss = mean_loss + args.penalty_weight * penalty

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % args.eval_interval == 0 and test_loader is not None:
            test_acc = evaluate_camelyon(model, test_loader, args.device)
            pretty_print(step, mean_loss.item(), test_acc, penalty.item())
def evaluate_camelyon(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x).squeeze()
            preds = (logits > 0).float()
            correct += (preds == y).sum().item()
            total += len(y)
    return correct / total
def download_camelyon17():
    # Create data directory
    os.makedirs('./data/camelyon17', exist_ok=True)

    # Download from Google Drive
    url = 'https://drive.google.com/uc?id=<FILE_ID>'  # Need valid file ID
    output = './data/camelyon17/data.zip'
    gdown.download(url, output, quiet=False)

    # Unzip
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('./data/camelyon17')
def setup_camelyon17():
    # Download dataset through Wilds
    dataset = wilds.get_dataset('camelyon17', download=True)

    # Create normalized datasets for each hospital
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
def camelyon17():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--eval_interval', type=int, default=100)
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets for each hospital
    train_loaders = []
    for hospital in range(1, 5):  # Changed to 1-4 for training
        try:
            dataset = Camelyon17Dataset(hospital)
            if len(dataset) == 0:
                print(f"Warning: Empty dataset for hospital {hospital}")
                continue
            train_loaders.append(DataLoader(dataset,
                                          batch_size=min(args.batch_size, len(dataset)),
                                          shuffle=True))
        except Exception as e:
            print(f"Error creating dataset for hospital {hospital}: {str(e)}")
            continue

    # Create test dataset
    try:
        test_dataset = Camelyon17Dataset(4)  # Using hospital 4 as test set
        test_loader = DataLoader(test_dataset,
                               batch_size=min(args.batch_size, len(test_dataset)),
                               shuffle=False)
    except Exception as e:
        print(f"Error creating test dataset: {str(e)}")
        test_loader = None

    if not train_loaders:
        raise RuntimeError("No valid training datasets found")

    model = CamelyonMLP().to(args.device)
    train_camelyon(model, train_loaders, args, test_loader)


class BallEnvDataset(Dataset):
    def __init__(self, num_balls=4, num_samples=10000, image_size=64, env_id=0):
        self.num_balls = num_balls
        self.image_size = image_size

        # Generate positions and images
        self.positions = torch.rand(num_samples, 2 * num_balls) * 0.8 + 0.1  # U(0.1, 0.9)
        self.images = torch.zeros(num_samples, 3, image_size, image_size)

        # Apply environment-specific interventions
        self.intervened_positions = self.positions.clone()
        if env_id > 0:
            # Randomly select coordinates to intervene
            intervention_mask = torch.rand(num_samples, 2 * num_balls) < 0.5
            interventions = torch.rand_like(self.positions) * 0.8 + 0.1
            self.intervened_positions[intervention_mask] = interventions[intervention_mask]

        # Render images
        self.render_balls()

    def render_balls(self):
        for i in range(len(self.positions)):
            for b in range(self.num_balls):
                x = int(self.intervened_positions[i, 2 * b] * self.image_size)
                y = int(self.intervened_positions[i, 2 * b + 1] * self.image_size)
                color = torch.tensor([1.0 if j == b % 3 else 0.0 for j in range(3)])

                # Draw ball as a 5x5 circle
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx * dx + dy * dy <= 4:
                            px = max(0, min(x + dx, self.image_size - 1))
                            py = max(0, min(y + dy, self.image_size - 1))
                            self.images[i, :, px, py] = color

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.images[idx], self.positions[idx]
class BallPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        self.predictor = nn.Linear(128, 8)  # 4 balls * 2 coordinates

    def forward(self, x):
        features = self.encoder(x)
        positions = self.predictor(features)
        return positions
def ballagent():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create datasets
    train_loaders = []
    for env_id in range(3):
        dataset = BallEnvDataset(env_id=env_id)
        train_loaders.append(DataLoader(dataset, batch_size=args.batch_size, shuffle=True))

    test_dataset = BallEnvDataset(env_id=3)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = BallPredictor().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_losses, test_losses, penalties = [], [], []
    steps = []
    for step in range(args.steps):
        losses = []
        model.train()

        for loader in train_loaders:
            x, y = next(iter(loader))
            x, y = x.to(args.device), y.to(args.device)

            pred = model(x)
            loss = F.mse_loss(pred, y)
            losses.append(loss)

        losses = torch.stack(losses)
        mean_loss = losses.mean()
        penalty = ((losses - mean_loss) ** 2).mean()

        total_loss = mean_loss + args.penalty_weight * penalty

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 100 == 0:
            test_loss = evaluate_ball_agent(model, test_loader, args.device)
            print(f"{step}\t{mean_loss:.5f}\t{test_loss:.5f}\t{penalty:.5f}")
            steps.append(step)
            train_losses.append(mean_loss.item())
            test_losses.append(test_loss)
            penalties.append(penalty.item())

            # Visualize predictions periodically
            if step % 1000 == 0:
                fig = visualize_ball_predictions(model, test_dataset)
                plt.savefig(f'predictions_step_{step}.png')
                plt.close()
    # plot_training_metrics(steps, train_losses, test_losses, penalties)
def evaluate_ball_agent(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = F.mse_loss(pred, y)
            total_loss += loss.item()
    return total_loss / len(loader)
def visualize_ball_predictions(model, dataset, num_samples=5, device='cuda'):
    """Visualize original images, predicted ball positions, and ground truth"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3 * num_samples))

    for i in range(num_samples):
        image, true_pos = dataset[i]

        # Original image
        axes[i, 0].imshow(image.permute(1, 2, 0))
        axes[i, 0].set_title('Original Image')

        # Predicted positions
        with torch.no_grad():
            pred_pos = model(image.unsqueeze(0).to(device)).cpu()

        # Plot ground truth vs predicted positions
        axes[i, 1].imshow(image.permute(1, 2, 0))
        for b in range(dataset.num_balls):
            # Ground truth positions
            axes[i, 1].plot(true_pos[2 * b] * dataset.image_size,
                            true_pos[2 * b + 1] * dataset.image_size,
                            'go', label='Ground Truth' if b == 0 else None)

            # Predicted positions
            axes[i, 1].plot(pred_pos[0, 2 * b] * dataset.image_size,
                            pred_pos[0, 2 * b + 1] * dataset.image_size,
                            'rx', label='Predicted' if b == 0 else None)
        axes[i, 1].legend()
        axes[i, 1].set_title('Positions Comparison')

        # Position error heatmap
        error_map = np.zeros((dataset.image_size, dataset.image_size))
        for b in range(dataset.num_balls):
            tx, ty = true_pos[2 * b:2 * b + 2] * dataset.image_size
            px, py = pred_pos[0, 2 * b:2 * b + 2] * dataset.image_size
            error_map[int(ty), int(tx)] = np.sqrt((tx - px) ** 2 + (ty - py) ** 2)

        im = axes[i, 2].imshow(error_map, cmap='hot')
        plt.colorbar(im, ax=axes[i, 2])
        axes[i, 2].set_title('Position Error Heatmap')

    plt.tight_layout()
    return fig
def plot_training_metrics(steps, train_losses, test_losses, penalties):
    """Plot training metrics over time"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Training and test losses
    ax1.plot(steps, train_losses, label='Train Loss')
    ax1.plot(steps, test_losses, label='Test Loss')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.set_title('Training and Test Losses')

    # REx penalty
    ax2.plot(steps, penalties, color='red')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('REx Penalty')
    ax2.set_title('REx Penalty Over Time')

    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # cmnist()
    # rmnist()
    # setup_camelyon17()
    # camelyon17()
    ballagent()