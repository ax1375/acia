"""
actir.py
"""
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from wilds import get_dataset
from torchvision import datasets,transforms
from PIL import Image
import pandas as pd


class cFeatureExtractor1(nn.Module):
    def __init__(self):
        super().__init__()
        # Adjust for 96x96x3 input
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate correct dimensions
        self.fc1 = nn.Linear(128 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, 8)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Add dimension checks
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 48x48
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 24x24
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 12x12

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
def ctrain_model(model, train_loaders, test_loader, device, args):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for env_id, loader in enumerate(train_loaders):
            for batch_idx, (x, y) in enumerate(loader):
                try:
                    x, y = x.to(device), y.to(device)

                    optimizer.zero_grad()
                    out, f_beta, f_eta, phi = model(x, env_id)

                    # IRM penalty
                    task_loss = criterion(out, y)
                    scale = torch.tensor(1.).to(device).requires_grad_()
                    grad_beta = torch.autograd.grad(
                        criterion(f_beta * scale, y),
                        [scale], create_graph=True)[0]
                    penalty = grad_beta.pow(2).mean()

                    loss = task_loss + args.irm_lambda * penalty
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                    if batch_idx % 100 == 0:
                        print(f'Epoch {epoch}, Env {env_id}, Batch {batch_idx}, Loss: {loss.item():.4f}')

                except Exception as e:
                    print(f"Error in batch {batch_idx} of env {env_id}: {str(e)}")
                    continue
        # avg_loss = total_loss / num_batches
        test_acc = cevaluate(model, test_loader, device)
        env_ind = compute_env_independence(model, train_loaders, device)
        low_level_inv = compute_low_level_invariance(model, train_loaders, device)
        interv_rob = compute_intervention_robustness(model, train_loaders[0], test_loader, device)

        print(f'Epoch {epoch}: Loss = {total_loss / num_batches:.4f}, '
              f'Test Acc = {test_acc:.2f}%, '
              f'Env Ind = {env_ind:.4f}, '
              f'Low-level Inv = {low_level_inv:.4f}, '
              f'Interv Rob = {interv_rob:.4f}')
class Classifier(nn.Module):
    def __init__(self, n_environments):
        super(Classifier, self).__init__()
        self.phi = FeatureExtractor()

        # # Base predictor β
        # self.beta = nn.Parameter(torch.zeros(8, 2))
        # with torch.no_grad():
        #     self.beta[0, 0] = 1.0
        #     self.beta[1, 1] = 1.0
        #
        # # Environment-specific predictors η
        # self.etas = nn.ParameterList([
        #     nn.Parameter(torch.zeros(8, 2)) for _ in range(n_environments)
        # ])

        self.beta = nn.Parameter(torch.zeros(8, 10))  # 10 classes
        self.etas = nn.ParameterList([
        nn.Parameter(torch.zeros(8, 10)) for _ in range(n_environments)
        ])
    def forward(self, x, env_id):
        phi = self.phi(x)
        f_beta = phi @ self.beta
        f_eta = phi @ self.etas[env_id]
        return f_beta + f_eta, f_beta, f_eta, phi
def ccompute_env_independence(model, loaders):
    env_representations = []
    labels = []

    for env_id, loader in enumerate(loaders):
        env_reps = []
        env_labels = []
        for x, y in loader:
            _, _, _, phi = model(x, env_id)
            env_reps.append(phi.detach())
            env_labels.append(y)
        env_representations.append(torch.cat(env_reps))
        labels.append(torch.cat(env_labels))

    # Calculate MI between representations and environments
    return mutual_information(env_representations, labels)
def cevaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out, _, _, _ = model(x, 0)
            _, predicted = torch.max(out.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return 100 * correct / total
def mutual_information(env_representations, labels):
    """Calculate mutual information between representations and environments"""
    import numpy as np
    from sklearn.metrics import mutual_info_score

    # Convert representations to numpy arrays
    reps = torch.cat(env_representations).cpu().numpy()
    labs = torch.cat(labels).cpu().numpy()

    # Discretize representations for MI calculation
    from sklearn.preprocessing import KBinsDiscretizer
    kbd = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='uniform')
    reps_discrete = kbd.fit_transform(reps)

    # Calculate MI for each dimension
    mi_scores = []
    for i in range(reps_discrete.shape[1]):
        mi = mutual_info_score(reps_discrete[:, i], labs)
        mi_scores.append(mi)

    return np.mean(mi_scores)
def ccompute_low_level_invariance(model, loaders):
    reps = []
    for loader in loaders:
        batch_reps = []
        for x, _ in loader:
            x = x.to(next(model.parameters()).device)
            _, _, _, phi = model(x, 0)
            batch_reps.append(phi.detach())
        reps.append(torch.cat(batch_reps))
    return torch.mean(torch.var(torch.stack(reps), dim=0)).item()
def ccompute_intervention_robustness(model, loader_obs, loader_int):
    model.eval()
    obs_preds = []
    int_preds = []

    with torch.no_grad():
        # Observational predictions
        for x, _ in loader_obs:
            x = x.to(model.device)
            out, _, _, _ = model(x, 0)
            obs_preds.append(F.softmax(out, dim=1))

        # Interventional predictions
        for x, _ in loader_int:
            x = x.to(model.device)
            out, _, _, _ = model(x, 0)
            int_preds.append(F.softmax(out, dim=1))

    # Calculate KL divergence between distributions
    obs = torch.cat(obs_preds)
    int = torch.cat(int_preds)
    return F.kl_div(obs.log(), int, reduction='batchmean').item()
class ColoredMNIST(Dataset):
    def __init__(self, env_id, train=True):
        super(ColoredMNIST, self).__init__()
        mnist = datasets.MNIST('~/data', train=train, download=True)

        if train:
            start = 0
            end = 50000
        else:
            start = 0
            end = 10000

        self.images = mnist.data[start:end].float()
        self.labels = mnist.targets[start:end]
        label_noise = torch.bernoulli(0.25 * torch.ones_like(self.labels))
        self.labels = torch.where(label_noise == 1, (9 - self.labels), self.labels)

        color_corr = torch.zeros_like(self.labels.float())
        for digit in range(10):
            mask = (self.labels == digit)
            if env_id == 0:
                color_corr[mask] = 0.75 if digit < 5 else 0.25  # Changed from 0.8/0.2
            elif env_id == 1:
                color_corr[mask] = 0.75 if digit < 5 else 0.25  # Changed from 0.9/0.1
            else:  # Test
                color_corr[mask] = 0.25 if digit < 5 else 0.75  # Changed from 0.1/0.9

        self.colors = torch.bernoulli(color_corr)

        self.images = torch.stack([self.images, self.images], dim=1)
        self.images = self.images.float() / 255.0

        for i in range(len(self.images)):
            if self.colors[i] == 1:
                self.images[i, 1] *= 0
            else:
                self.images[i, 0] *= 0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx].long()
def cmnist():
    class Args:
        def __init__(self):
            self.batch_size = 256
            self.epochs = 100
            self.lr = 0.0001
            self.irm_lambda = 1000.0
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = Args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_env1 = DataLoader(ColoredMNIST(0), batch_size=args.batch_size, shuffle=True)
    train_env2 = DataLoader(ColoredMNIST(1), batch_size=args.batch_size, shuffle=True)
    test_env = DataLoader(ColoredMNIST(2, train=False), batch_size=args.batch_size)

    model = Classifier(n_environments=2).to(device)
    ctrain_model(model, [train_env1, train_env2], test_env, device, args)


class Args:
    def __init__(self):
        self.batch_size = 256
        self.epochs = 100
        self.lr = 0.0001
        self.irm_lambda = 1000.0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class RotatedMNIST(Dataset):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, env_id, train=True, device=None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mnist = datasets.MNIST('~/data', train=True, download=True)
        self.angle = int(self.ENVIRONMENTS[env_id])

        # Use same size (10000) for both train and test
        if train:
            self.images = mnist.data[:10000].float().to(self.device)
            self.labels = mnist.targets[:10000].to(self.device)
        else:
            self.images = mnist.data[50000:60000].float().to(self.device)
            self.labels = mnist.targets[50000:60000].to(self.device)

        self.images = self.images.unsqueeze(1)
        self.images = TF.rotate(self.images, self.angle) / 255.0

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
def rcompute_intervention_robustness(model, loader_obs, loader_int):
    device = next(model.parameters()).device
    with torch.no_grad():
        obs_preds = []
        int_preds = []
        for x, _ in loader_obs:
            x = x.to(device)
            out, _, _, _ = model(x, 0)
            obs_preds.append(F.softmax(out, dim=1))
        for x, _ in loader_int:
            x = x.to(device)
            out, _, _, _ = model(x, 0)
            int_preds.append(F.softmax(out, dim=1))
    obs = torch.cat(obs_preds)
    int = torch.cat(int_preds)
    return F.kl_div(obs.log(), int, reduction='batchmean').item()
def rmnist():
    args = Args()
    model = Classifier(n_environments=6).to(args.device)

    train_loaders = [
        DataLoader(RotatedMNIST(i), batch_size=args.batch_size, shuffle=True)
        for i in range(5)
    ]
    test_loader = DataLoader(RotatedMNIST(5, train=False), batch_size=args.batch_size)

    ctrain_model(model, train_loaders, test_loader, args.device, args)


class Camelyon17Dataset(Dataset):
    def __init__(self, env_id, train=True):
        super().__init__()
        self.root = "C:/University/Spring2025/Research/Session9/baseline/data/camelyon17_v1.0"
        self.metadata = pd.read_csv(f"{self.root}/metadata.csv")

        # Balance the dataset
        env_data = self.metadata[self.metadata['center'] == env_id]
        env_data = env_data[env_data['split'] == (0 if train else 1)]

        # Subsample to balance classes
        tumor = env_data[env_data['tumor'] == 1]
        normal = env_data[env_data['tumor'] == 0]
        min_samples = min(len(tumor), len(normal))

        tumor = tumor.sample(min_samples, random_state=42)
        normal = normal.sample(min_samples, random_state=42)
        self.metadata = pd.concat([tumor, normal])

        print(f"Environment {env_id}: {len(self.metadata)} samples ({min_samples} per class)")

        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[:3]),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = f"{self.root}/patches/patient_{row['patient']:03d}_node_{row['node']}/patch_patient_{row['patient']:03d}_node_{row['node']}_x_{row['x_coord']}_y_{row['y_coord']}.png"
        image = Image.open(img_path)
        image = self.transform(image)
        label = row['tumor']
        return image, torch.tensor(label, dtype=torch.long)
class Classifier(nn.Module):
    def __init__(self, n_environments):
        super().__init__()
        self.phi = FeatureExtractor()
        self.beta = nn.Parameter(torch.zeros(8, 2))
        self.etas = nn.ParameterList([nn.Parameter(torch.zeros(8, 2)) for _ in range(n_environments)])

    def forward(self, x, env_id):
        phi = self.phi(x)
        f_beta = phi @ self.beta
        f_eta = phi @ self.etas[env_id]
        return f_beta + f_eta, f_beta, f_eta, phi
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Added padding
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 12 * 12, 512)  # Adjusted dimensions
        self.fc2 = nn.Linear(512, 8)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # More robust flattening
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
def compute_env_independence(model, train_loaders, device):
    """Mutual information between representations and environments"""
    env_reps = []
    env_labels = []

    for env_id, loader in enumerate(train_loaders):
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                _, _, _, phi = model(x, env_id)
                env_reps.append(phi.cpu().numpy())
                env_labels.extend([env_id] * len(y))

    # Combine representations and discretize
    reps = np.concatenate(env_reps)
    env_labels = np.array(env_labels)

    # Calculate MI between each feature dimension and environment
    mi_scores = []
    for dim in range(reps.shape[1]):
        # Discretize continuous representations
        bins = np.histogram_bin_edges(reps[:, dim], bins=20)
        discretized = np.digitize(reps[:, dim], bins)
        mi = mutual_info_score(discretized, env_labels)
        mi_scores.append(mi)

    return np.mean(mi_scores)
def compute_low_level_invariance(model, train_loaders, device):
    """Measure stability of representations across environments"""
    env_means = []

    for loader in train_loaders:
        env_reps = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                _, _, _, phi = model(x, 0)
                env_reps.append(phi)
        env_mean = torch.cat(env_reps).mean(dim=0)
        env_means.append(env_mean)

    # Calculate variance across environment means
    env_means = torch.stack(env_means)
    return torch.var(env_means, dim=0).mean().item()
def compute_intervention_robustness(model, obs_loader, int_loader, device):
    """KL divergence between observational and interventional predictions"""
    obs_preds = []
    int_preds = []

    with torch.no_grad():
        for x, _ in obs_loader:
            x = x.to(device)
            pred, _, _, _ = model(x, 0)
            obs_preds.append(F.softmax(pred, dim=1))

        for x, _ in int_loader:
            x = x.to(device)
            pred, _, _, _ = model(x, 0)
            int_preds.append(F.softmax(pred, dim=1))

    obs_dist = torch.cat(obs_preds).mean(dim=0)
    int_dist = torch.cat(int_preds).mean(dim=0)

    return F.kl_div(obs_dist.log(), int_dist).item()
def camelyon17():
    args = Args()
    model = Classifier(n_environments=4).to(args.device)
    print(1)
    # Create train loaders for first 4 hospitals
    train_loaders = [
        DataLoader(Camelyon17Dataset(i, train=True),
                   batch_size=args.batch_size,
                   shuffle=True)
        for i in range(4)
    ]
    print(2)
    # Use 5th hospital (index 4) for testing
    test_loader = DataLoader(
        Camelyon17Dataset(4, train=False),
        batch_size=args.batch_size
    )
    print(3)
    ctrain_model(model, train_loaders, test_loader, args.device, args)
class Args:
    def __init__(self):
        self.batch_size = 32  # Reduced batch size
        self.epochs = 2
        self.lr = 1e-4
        self.irm_lambda = 1.0
        self.reg_lambda = 0.1
        self.gamma = 0.9
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BallAgentDataset(Dataset):
    def __init__(self, n_balls=4, n_samples=10000, img_size=64, env_id=None):
        self.n_balls = n_balls
        self.img_size = img_size

        # Generate data
        self.positions = torch.rand(n_samples, 2 * n_balls) * 0.8 + 0.1  # [0.1, 0.9]
        self.images = torch.zeros(n_samples, 3, img_size, img_size)

        # Generate interventions
        if env_id is not None:
            # Create environment-specific interventions
            self.interventions = torch.zeros_like(self.positions)
            intervention_mask = torch.randint(0, 2, (n_samples, n_balls))
            for i in range(n_balls):
                if env_id % (n_balls + 1) == i:  # Different intervention pattern per environment
                    idx = intervention_mask[:, i] == 1
                    self.positions[idx, 2 * i:2 * i + 2] += torch.randn(sum(idx), 2) * 0.1

        # Render images
        for i in range(n_samples):
            self.images[i] = self._render_balls(self.positions[i])

    def _render_balls(self, positions):
        image = torch.zeros(3, self.img_size, self.img_size)
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]  # RGB colors for balls

        for i in range(self.n_balls):
            x, y = positions[2 * i:2 * i + 2] * self.img_size
            x, y = int(x), int(y)

            # Draw circle
            for dx in range(-3, 4):
                for dy in range(-3, 4):
                    if dx * dx + dy * dy <= 9:
                        px, py = x + dx, y + dy
                        if 0 <= px < self.img_size and 0 <= py < self.img_size:
                            for c in range(3):
                                image[c, py, px] = colors[i][c]
        return image

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        return self.images[idx], self.positions[idx]
class bFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 64x64x3 for Ball Agent
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # 64x64x32
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # 32x32x64
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 16x16x128
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)

        # Correct flattened size for 64x64 input
        self.flat_size = 128 * 8 * 8  # 8192
        self.fc1 = nn.Linear(self.flat_size, 512)
        self.fc2 = nn.Linear(512, 8)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
class bClassifier(nn.Module):
    def __init__(self, n_environments):
        super().__init__()
        self.phi = bFeatureExtractor()  # Use bFeatureExtractor instead of FeatureExtractor
        self.beta = nn.Parameter(torch.zeros(8, 8))  # Change output size to match position dimensions
        self.etas = nn.ParameterList([nn.Parameter(torch.zeros(8, 8)) for _ in range(n_environments)])

    def forward(self, x, env_id):
        phi = self.phi(x)
        f_beta = phi @ self.beta
        f_eta = phi @ self.etas[env_id]
        return f_beta + f_eta, f_beta, f_eta, phi
def btrain_model(model, train_loaders, test_loader, device, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for env_id, loader in enumerate(train_loaders):
            for batch_idx, (x, y) in enumerate(loader):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                out, f_beta, f_eta, phi = model(x, env_id)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Env {env_id}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / (num_batches if num_batches > 0 else 1)
        test_loss = bevaluate(model, test_loader, device)
        print(f'Epoch {epoch}: Avg Loss = {avg_loss:.4f}, Test Loss = {test_loss:.4f}')
        accuracy, env_ind, low_level, interv_rob = bcalculate_metrics(model, train_loaders, test_loader, args.device)
        print(f"\nMetrics:")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Environment Independence: {env_ind:.2f}")
        print(f"Low-level Invariance: {low_level:.2f}")
        print(f"Intervention Robustness: {interv_rob:.2f}")
def bevaluate(model, test_loader, device):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out, _, _, _ = model(x, 0)
            total_loss += criterion(out, y).item()

    return total_loss / len(test_loader)
def bcompute_low_level_invariance(model, train_loaders, device):
    # Variance of representations across environments
    all_reps = []

    with torch.no_grad():
        for loader in train_loaders:
            env_reps = []
            for x, _ in loader:
                x = x.to(device)
                _, _, _, phi = model(x, 0)
                env_reps.append(phi)
            all_reps.append(torch.cat(env_reps).mean(dim=0))

    return torch.var(torch.stack(all_reps), dim=0).mean().item()
def bcalculate_metrics(model, train_loaders, test_loader, device):
    # Accuracy/MSE
    test_loss = bevaluate(model, test_loader, device)
    accuracy = 100 * (1 - test_loss)  # Convert to percentage

    # Environment Independence
    env_ind = bcompute_env_independence(model, train_loaders, device)

    # Low-level Invariance (R1)
    low_level = bcompute_representation_variance(model, train_loaders, device)

    # Intervention Robustness (R2)
    interv_rob = bcompute_intervention_robustness(model, train_loaders[0], test_loader, device)

    return accuracy, env_ind, low_level, interv_rob
def bcompute_environment_independence(model, train_loaders, device):
    reps_by_env = []
    labels_by_env = []

    with torch.no_grad():
        for env_id, loader in enumerate(train_loaders):
            env_reps = []
            env_labels = []
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                _, _, _, phi = model(x, env_id)
                env_reps.append(phi)
                env_labels.append(y)
            reps_by_env.append(torch.cat(env_reps))
            labels_by_env.append(torch.cat(env_labels))

    # Calculate MI between representations and environments conditioned on labels
    mi = 0
    for y_val in torch.unique(torch.cat(labels_by_env)):
        for env_id in range(len(train_loaders)):
            mask = labels_by_env[env_id] == y_val
            if mask.sum() > 0:
                p_rep_env = bcompute_density(reps_by_env[env_id][mask])
                p_rep = bcompute_density(torch.cat([r[l == y_val] for r, l in zip(reps_by_env, labels_by_env)]))
                mi += torch.mean(torch.log(p_rep_env / p_rep))

    return mi.item()
def bcompute_representation_variance(model, train_loaders, device):
    all_reps = []

    with torch.no_grad():
        for loader in train_loaders:
            env_reps = []
            for x, _ in loader:
                x = x.to(device)
                _, _, _, phi = model(x, 0)
                env_reps.append(phi)
            all_reps.append(torch.cat(env_reps).mean(dim=0))

    return torch.var(torch.stack(all_reps), dim=0).mean().item()
def bcompute_intervention_robustness(model, obs_loader, int_loader, device):
    obs_reps = []
    int_reps = []

    with torch.no_grad():
        for (x_obs, _), (x_int, _) in zip(obs_loader, int_loader):
            x_obs, x_int = x_obs.to(device), x_int.to(device)
            _, _, _, phi_obs = model(x_obs, 0)
            _, _, _, phi_int = model(x_int, 0)
            obs_reps.append(phi_obs)
            int_reps.append(phi_int)

    obs_reps = torch.cat(obs_reps)
    int_reps = torch.cat(int_reps)

    # Normalize the representations
    obs_reps = (obs_reps - obs_reps.mean()) / (obs_reps.std() + 1e-6)
    int_reps = (int_reps - int_reps.mean()) / (int_reps.std() + 1e-6)

    # Calculate normalized difference and convert to robustness score
    diff = torch.mean(torch.abs(obs_reps - int_reps))
    robustness = 1.0 / (1.0 + diff)  # Convert to [0,1] range where 1 is most robust

    return robustness.item()
def bcompute_density(x):
    if x.dim() == 1:
        x = x.unsqueeze(0)  # Add batch dimension
    sigma = torch.std(x) + 1e-6  # Add small constant to avoid division by zero
    dists = torch.cdist(x, x)
    return torch.mean(torch.exp(-dists / (2 * sigma ** 2)), dim=1)
def bcompute_env_independence(model, train_loaders, device):
    """Compute environment independence using mean feature differences"""
    all_means = []

    with torch.no_grad():
        for loader in train_loaders:
            env_features = []
            for x, _ in loader:
                x = x.to(device)
                _, _, _, phi = model(x, 0)
                env_features.append(phi)
            env_features = torch.cat(env_features)
            env_mean = env_features.mean(dim=0)
            all_means.append(env_mean)

    all_means = torch.stack(all_means)
    # Compute pairwise differences between environment means
    differences = []
    for i in range(len(all_means)):
        for j in range(i + 1, len(all_means)):
            diff = torch.abs(all_means[i] - all_means[j]).mean()
            differences.append(diff)

    # Convert to independence score (0 = completely dependent, 1 = completely independent)
    mean_diff = torch.stack(differences).mean()
    independence = 1.0 / (1.0 + mean_diff)

    return independence.item()
def ballagent():
    args = Args()
    model = bClassifier(n_environments=4).to(args.device)

    # Create train loaders for environments
    train_loaders = [
        DataLoader(BallAgentDataset(env_id=i),
                   batch_size=args.batch_size,
                   shuffle=True)
        for i in range(4)
    ]

    # Test environment
    test_loader = DataLoader(
        BallAgentDataset(env_id=4),
        batch_size=args.batch_size
    )

    btrain_model(model, train_loaders, test_loader, args.device, args)


if __name__ == '__main__':
    # cmnist()
    # rmnist()
    # camelyon17()
    ballagent()