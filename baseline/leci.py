from torch_geometric.data import Data, Batch
from torch.optim import Adam
from scipy.stats import entropy
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from tqdm import tqdm
import torchvision
from munch import munchify
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Dataset, Data
from PIL import Image
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

class LECICMNIST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.lc_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

        self.la_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

        self.ea_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

        self.ef_filter = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1)
        )

        self.LA = config.ood.LA
        self.EA = config.ood.EA
        self.EF = config.ood.EF

    def forward(self, x, training=True):
        if self.EF:
            x = self.ef_filter(x)

        features = self.feature_extractor(x)

        lc_logits = self.lc_classifier(features)

        if training:
            la_logits = self.la_classifier(cgradient_reverse(features, self.LA)) if self.LA else None
            ea_logits = self.ea_classifier(cgradient_reverse(features, self.EA)) if self.EA else None
            return lc_logits, la_logits, None, ea_logits

        return lc_logits

    def compute_loss(self, data, targets, env_ids):
        lc_logits, la_logits, _, ea_logits = self(data, training=True)

        lc_loss = F.cross_entropy(lc_logits, targets)
        total_loss = lc_loss

        if self.LA and la_logits is not None:
            la_loss = F.cross_entropy(la_logits, targets)
            total_loss += self.LA * la_loss

        if self.EA and ea_logits is not None:
            ea_loss = F.cross_entropy(ea_logits, env_ids)
            total_loss += self.EA * ea_loss

        return total_loss
class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None
def cgradient_reverse(x, alpha):
    return GradientReverse.apply(x, alpha)
class ColoredMNIST(torch.utils.data.Dataset):
    def __init__(self, root='./data', environment='e1', train=True):
        super().__init__()
        self.mnist = datasets.MNIST(root=root, train=train, download=True)
        self.environment = environment

        self.data = np.zeros((len(self.mnist), 28, 28, 3), dtype=np.float32)
        self.targets = []

        for idx, (img, label) in enumerate(self.mnist):
            if environment == 'e1':
                p_red = 0.75 if label % 2 == 0 else 0.25
            else:
                p_red = 0.25 if label % 2 == 0 else 0.75

            img_np = np.array(img)
            if np.random.random() < p_red:
                self.data[idx, :, :, 0] = img_np  # Red channel
            else:
                self.data[idx, :, :, 1] = img_np  # Green channel

            self.targets.append(label)

        self.data = torch.from_numpy(self.data).permute(0, 3, 1, 2)
        self.targets = torch.tensor(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], 0 if self.environment == 'e1' else 1
def cget_cmnist_loaders(config):
    train_e1 = ColoredMNIST(root=config.dataset.root, environment='e1', train=True)
    val_e2 = ColoredMNIST(root=config.dataset.root, environment='e2', train=False)

    train_loader = torch.utils.data.DataLoader(train_e1, batch_size=config.train.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_e2, batch_size=config.train.batch_size)

    return train_loader, val_loader
def ccalculate_metrics(model, loader, config):
    model.eval()
    correct = 0
    total = 0

    representations = []
    labels = []
    env_ids = []

    with torch.no_grad():
        for data, target, env_id in loader:
            data, target = data.to(config.device), target.to(config.device)
            features = model.feature_extractor(data)
            output = model.lc_classifier(features)

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            representations.append(features.cpu().numpy())
            labels.append(target.cpu().numpy())
            env_ids.append(env_id.numpy())

    accuracy = 100. * correct / total
    representations = np.concatenate(representations)
    labels = np.concatenate(labels)
    env_ids = np.concatenate(env_ids)

    env_independence = ccalculate_conditional_mi(representations, env_ids, labels)
    low_level_inv = ccalculate_representation_variance(representations, env_ids)
    interv_robustness = ccalculate_intervention_robustness(model, loader, config)

    return accuracy, env_independence, low_level_inv, interv_robustness
def ccalculate_conditional_mi(representations, env_ids, labels):
    mi = 0
    for label in np.unique(labels):
        mask = labels == label
        if mask.sum() > 0:
            p_z_y = cestimate_density(representations[mask])
            p_z_e_y = cestimate_density(representations[mask], env_ids[mask])
            mi += np.mean(np.log(p_z_e_y / (p_z_y + 1e-10) + 1e-10))
    return mi
def ccalculate_representation_variance(representations, env_ids):
    env_means = []
    for env_id in np.unique(env_ids):
        env_mask = env_ids == env_id
        env_means.append(np.mean(representations[env_mask], axis=0))
    return np.mean(np.var(env_means, axis=0))
def ccalculate_intervention_robustness(model, loader, config):
    original_preds = []
    intervened_preds = []

    with torch.no_grad():
        for data, _, _ in loader:
            data = data.to(config.device)
            orig_out = model(data, training=False)
            original_preds.append(orig_out.cpu().numpy())

            intervened_data = data.clone()
            intervened_data[:, [0, 1]] = intervened_data[:, [1, 0]]
            int_out = model(intervened_data, training=False)
            intervened_preds.append(int_out.cpu().numpy())

    original_preds = np.concatenate(original_preds)
    intervened_preds = np.concatenate(intervened_preds)

    return np.mean(np.abs(original_preds - intervened_preds))
def cestimate_density(x, condition=None):
    if condition is None:
        return np.histogram(x, bins='auto', density=True)[0] + 1e-10
    else:
        densities = []
        for c in np.unique(condition):
            mask = condition == c
            densities.append(cestimate_density(x[mask]))
        return np.mean(densities, axis=0)
def ctrain_epoch(model, train_loader, optimizer, config):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data, target, env_id in tqdm(train_loader):
        data = data.to(config.device)
        target = target.to(config.device)
        env_id = env_id.to(config.device)

        optimizer.zero_grad()
        loss = model.compute_loss(data, target, env_id)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            pred = model(data, training=False).argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    return total_loss / len(train_loader), 100. * correct / total
def ctrain_leci(config):
    train_loader, val_loader = cget_cmnist_loaders(config)

    model = LECICMNIST(config).to(config.device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.train.lr,
        momentum=config.optim.momentum,
        weight_decay=config.train.weight_decay,
        nesterov=config.optim.nesterov
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.epochs)

    best_metrics = {
        'acc': 0,
        'env_ind': float('inf'),
        'low_level': float('inf'),
        'int_rob': float('inf')
    }

    for epoch in range(config.train.epochs):
        train_loss, train_acc = ctrain_epoch(model, train_loader, optimizer, config)
        accuracy, env_independence, low_level_inv, interv_robustness = ccalculate_metrics(model, val_loader, config)

        if accuracy > best_metrics['acc']:
            best_metrics = {
                'acc': accuracy,
                'env_ind': env_independence,
                'low_level': low_level_inv,
                'int_rob': interv_robustness
            }
            if config.save_model:
                torch.save(model.state_dict(), config.save_path)

        scheduler.step()

        print(f'Epoch: {epoch}')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Metrics - Acc: {accuracy:.2f}%, Env Ind: {env_independence:.4f}, '
              f'Low-level Inv: {low_level_inv:.4f}, Int Rob: {interv_robustness:.4f}')


    return best_metrics
def cmnist():
    config = munchify({
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'ood': {
            'LA': 0.1,
            'EA': 0.01,
            'EF': 0.001
        },
        'train': {
            'batch_size': 128,
            'lr': 0.0001,
            'epochs': 2,
            'weight_decay': 1e-4
        },
        'optim': {
            'momentum': 0.9,
            'nesterov': True
        },
        'dataset': {
            'root': './data'
        },
        'save_model': True,
        'save_path': 'leci_cmnist.pt'
    })
    print(f"Using device: {config.device}")

    best_metrics = ctrain_leci(config)
    print(f'Final metrics:')
    print(f'Accuracy: {best_metrics["acc"]:.2f}%')
    print(f'Environment Independence: {best_metrics["env_ind"]:.4f}')
    print(f'Low-level Invariance: {best_metrics["low_level"]:.4f}')
    print(f'Intervention Robustness: {best_metrics["int_rob"]:.4f}')


class RotatedMNIST(Dataset):
    def __init__(self, root, rotation_angles, train=True):
        super().__init__()
        self.mnist = torchvision.datasets.MNIST(root=root, train=train, download=True)
        self.rotation_angles = rotation_angles
        self.train = train

        self.data = []
        self.targets = []
        self.env_ids = []

        for img, label in self.mnist:
            if train:
                angle_prob = 0.75 if label % 2 == 0 else 0.25
            else:
                angle_prob = 0.5

            angle = rotation_angles[0] if torch.rand(1) < angle_prob else rotation_angles[-1]

            # Convert PIL Image to tensor and normalize
            img_tensor = transforms.ToTensor()(img)

            # Apply rotation
            rotated = transforms.functional.rotate(img_tensor, angle)

            # Flatten and store
            self.data.append(rotated.view(-1))  # Flatten to 784 dimensions
            self.targets.append(label)
            self.env_ids.append(self.rotation_angles.index(angle))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return Data(
            x=self.data[idx],  # Already flattened during initialization
            y=torch.tensor(self.targets[idx]),
            edge_index=None,
            env_id=torch.tensor(self.env_ids[idx])
        )
class SimpleLECI(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 300),  # 28*28 = 784 input dimensions
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.classifier = nn.Linear(300, 10)

    def forward(self, data):
        # Add batch dimension handling if needed
        if data.x.dim() == 1:
            x = data.x.unsqueeze(0)
        else:
            x = data.x
        x = self.encoder(x)
        return self.classifier(x)
def rcompute_mutual_information(representations, env_ids, labels):
    """Compute mutual information between representations and environments conditioned on labels"""
    # Simplified estimation using correlation
    mi = 0
    for label in torch.unique(labels):
        mask = labels == label
        if mask.sum() > 0:
            label_repr = representations[mask]
            label_envs = env_ids[mask]
            # Use correlation as a proxy for MI
            corr = torch.corrcoef(torch.stack([label_repr.mean(1), label_envs.float()]))[0, 1]
            mi += corr.abs().item()
    return mi / len(torch.unique(labels))
def rcompute_r1_score(representations, env_ids):
    """Compute low-level invariance score"""
    # Measure variance of representations across environments
    r1 = 0
    for env_id in torch.unique(env_ids):
        env_mask = env_ids == env_id
        if env_mask.sum() > 0:
            env_repr = representations[env_mask]
            other_repr = representations[~env_mask]
            if other_repr.size(0) > 0:
                # Compute distance between environment means
                r1 += torch.norm(env_repr.mean(0) - other_repr.mean(0)).item()
    return r1 / len(torch.unique(env_ids))
def rcompute_r2_score(representations, env_ids, labels):
    """Compute intervention robustness score"""
    # Measure consistency under interventions
    r2 = 0
    for label in torch.unique(labels):
        label_mask = labels == label
        if label_mask.sum() > 0:
            label_repr = representations[label_mask]
            label_envs = env_ids[label_mask]
            # Compute variance across environments for same label
            r2 += torch.var(label_repr.mean(1)).item()
    return r2 / len(torch.unique(labels))
def revaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model and datasets
    model = SimpleLECI().to(device)
    train_angles = [15, 30, 45]
    test_angles = [60, 75]

    test_dataset = RotatedMNIST('./data', rotation_angles=test_angles, train=False)
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=collate_fn)

    # Train model (your existing training code)

    # Compute metrics
    accuracy, env_ind, r1, r2 = rcompute_metrics(model, test_loader, device)

    print(f"Results for RMNIST:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Environment Independence: {env_ind:.3f}")
    print(f"Low-level Invariance (R1): {r1:.3f}")
    print(f"Intervention Robustness (R2): {r2:.3f}")

    return accuracy, env_ind, r1, r2
def collate_fn(data_list):
    batch = Batch.from_data_list(data_list)
    # Ensure correct shape for batched tensor
    batch.x = batch.x.view(-1, 784)  # Reshape to (batch_size, 784)
    return batch
def rplot_metrics(history):
    """Plot training metrics over time"""
    plt.figure(figsize=(15, 10))

    # Accuracy plot
    plt.subplot(2, 2, 1)
    plt.plot(history['epoch'], history['train_acc'], label='Train Acc')
    plt.plot(history['epoch'], history['test_acc'], label='Test Acc')
    plt.title('Accuracy over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Environment Independence plot
    plt.subplot(2, 2, 2)
    plt.plot(history['epoch'], history['env_ind'])
    plt.title('Environment Independence')
    plt.xlabel('Epoch')
    plt.ylabel('MI Score')

    # R1 Score plot
    plt.subplot(2, 2, 3)
    plt.plot(history['epoch'], history['r1'])
    plt.title('Low-level Invariance (R1)')
    plt.xlabel('Epoch')
    plt.ylabel('R1 Score')

    # R2 Score plot
    plt.subplot(2, 2, 4)
    plt.plot(history['epoch'], history['r2'])
    plt.title('Intervention Robustness (R2)')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')

    plt.tight_layout()
    plt.savefig('rmnist_metrics.png')
    plt.close()
def rplot_confusion_matrix(model, loader, device):
    """Plot confusion matrix for predictions"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    cm = np.zeros((10, 10))
    for p, t in zip(all_preds, all_labels):
        cm[t][p] += 1

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('rmnist_confusion.png')
    plt.close()
def rcompute_metrics(model, loader, device):
    """Compute evaluation metrics"""
    model.eval()
    correct = 0
    total = 0

    representations = []
    labels = []
    env_ids = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            hidden = model.encoder(batch.x)

            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

            representations.append(hidden)
            labels.append(batch.y)
            env_ids.append(batch.env_id)

    accuracy = correct / total * 100

    # Concatenate stored tensors
    representations = torch.cat(representations)
    labels = torch.cat(labels)
    env_ids = torch.cat(env_ids)

    # Compute other metrics
    env_ind = rcompute_mutual_information(representations, env_ids, labels)
    r1 = rcompute_r1_score(representations, env_ids)
    r2 = rcompute_r2_score(representations, env_ids, labels)

    return accuracy, env_ind, r1, r2
def rmnist():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleLECI().to(device)

    train_angles = [15, 30, 45]
    test_angles = [60, 75]

    train_dataset = RotatedMNIST('./data', rotation_angles=train_angles)
    test_dataset = RotatedMNIST('./data', rotation_angles=test_angles, train=False)

    def collate_fn(data_list):
        batch = Batch.from_data_list(data_list)
        batch.x = batch.x.view(-1, 784)
        return batch

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 50  # Increased epochs for better convergence

    # For tracking metrics
    history = defaultdict(list)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch)
            loss = F.cross_entropy(out, batch.y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)

        # Compute metrics every 5 epochs
        if epoch % 5 == 0:
            accuracy, env_ind, r1, r2 = rcompute_metrics(model, test_loader, device)

            history['epoch'].append(epoch)
            history['train_acc'].append(train_correct / train_total * 100)
            history['test_acc'].append(accuracy)
            history['env_ind'].append(env_ind)
            history['r1'].append(r1)
            history['r2'].append(r2)

            print(f'Epoch {epoch}:')
            print(f'Train Acc: {train_correct / train_total * 100:.2f}%')
            print(f'Test Acc: {accuracy:.2f}%')
            print(f'Env Ind: {env_ind:.3f}')
            print(f'R1: {r1:.3f}')
            print(f'R2: {r2:.3f}\n')

    # Final evaluation
    final_acc, final_env_ind, final_r1, final_r2 = rcompute_metrics(model, test_loader, device)

    # Visualize Results
    rplot_metrics(history)
    rplot_confusion_matrix(model, test_loader, device)

    return {
        'accuracy': final_acc,
        'env_independence': final_env_ind,
        'r1': final_r1,
        'r2': final_r2,
        'history': history
    }


class Camelyon17Dataset(Dataset):
    def __init__(self, root, transform=None, train=True):
        super().__init__()
        self.root = root
        self.transform = transform or transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.train = train

        # Read metadata
        self.metadata = pd.read_csv(os.path.join(root, 'metadata.csv'))

        # Convert hospital_id to environment index (0-4)
        self.metadata['env_id'] = self.metadata['center'].astype('category').cat.codes

        # Split train/test (use first 4 hospitals for training, last for testing)
        if train:
            self.metadata = self.metadata[self.metadata['env_id'] < 4]
        else:
            self.metadata = self.metadata[self.metadata['env_id'] == 4]

        self.samples = []
        self.process_data()

    def process_data(self):
        """Process data and create samples list"""
        patches_dir = os.path.join(self.root, 'patches')

        for idx, row in self.metadata.iterrows():
            patch_path = os.path.join(patches_dir, row['patch_path'])
            if os.path.exists(patch_path):
                self.samples.append({
                    'path': patch_path,
                    'tumor': row['tumor'],
                    'env_id': row['env_id']
                })

    def len(self):
        return len(self.samples)

    def get(self, idx):
        sample = self.samples[idx]

        # Load and transform image
        image = Image.open(sample['path']).convert('RGB')
        image = self.transform(image)

        # Create graph structure (you can modify this based on your needs)
        # Here we're treating each pixel as a node with 8-neighbor connectivity
        edge_index = acreate_grid_graph(96, 96)

        return Data(
            x=image.reshape(-1, 3),  # Reshape to (N_nodes, 3)
            edge_index=edge_index,
            y=torch.tensor(sample['tumor']),
            env_id=torch.tensor(sample['env_id'])
        )
def acreate_grid_graph(h, w):
    """Create grid-like graph structure for image"""
    edge_index = []

    # 8-neighbor connectivity
    for i in range(h):
        for j in range(w):
            node = i * w + j

            # Add edges to neighbors
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        neighbor = ni * w + nj
                        edge_index.append([node, neighbor])

    return torch.tensor(edge_index).t()
def camelyon17():
    """Evaluate LECI on Camelyon17 dataset"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Configure paths
    data_root = "C:/University/Spring2025/Research/Session9/baseline/data/camelyon17_v1.0"

    # Create datasets
    train_dataset = Camelyon17Dataset(data_root, train=True)
    test_dataset = Camelyon17Dataset(data_root, train=False)

    def collate_fn(data_list):
        batch = Batch.from_data_list(data_list)
        return batch

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)

    # Initialize model (adapt input dimensions for Camelyon17)
    model = LECIGIN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training and evaluation
    num_epochs = 50
    history = defaultdict(list)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(batch)
            loss = F.cross_entropy(out, batch.y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)

        # Compute metrics
        if epoch % 5 == 0:
            accuracy, env_ind, r1, r2 = acompute_metrics(model, test_loader, device)

            print(f'Epoch {epoch}:')
            print(f'Train Acc: {train_correct / train_total * 100:.2f}%')
            print(f'Test Acc: {accuracy:.2f}%')
            print(f'Env Ind: {env_ind:.3f}')
            print(f'R1: {r1:.3f}')
            print(f'R2: {r2:.3f}\n')

            # Store metrics
            history['epoch'].append(epoch)
            history['train_acc'].append(train_correct / train_total * 100)
            history['test_acc'].append(accuracy)
            history['env_ind'].append(env_ind)
            history['r1'].append(r1)
            history['r2'].append(r2)

    # Visualize results
    aplot_metrics(history)
    avisualize_hospital_patterns(model, test_loader, device)

    return history
def avisualize_hospital_patterns(model, loader, device):
    """Visualize learned patterns for each hospital"""
    model.eval()
    hospital_features = {i: [] for i in range(5)}

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            features = model.encoder(batch.x)

            for i, env_id in enumerate(batch.env_id):
                hospital_features[env_id.item()].append(features[i].cpu())

    # Plot t-SNE visualization
    plt.figure(figsize=(10, 8))
    for hospital_id, features in hospital_features.items():
        if features:
            features = torch.stack(features)
            features_2d = TSNE(n_components=2).fit_transform(features)
            plt.scatter(features_2d[:, 0], features_2d[:, 1], label=f'Hospital {hospital_id + 1}')

    plt.title('t-SNE Visualization of Hospital-specific Features')
    plt.legend()
    plt.savefig('camelyon17_features.png')
    plt.close()

    camelyon17()

if __name__ == '__main__':
    # cmnist()
    # results = rmnist()
    # print("\nFinal Results:")
    # print(f"Accuracy: {results['accuracy']:.2f}%")
    # print(f"Environment Independence: {results['env_independence']:.3f}")
    # print(f"Low-level Invariance (R1): {results['r1']:.3f}")
    # print(f"Intervention Robustness (R2): {results['r2']:.3f}")
    camelyon17()
    # ballagent()