
from torchvision import datasets, transforms
from scipy.ndimage import rotate
from sklearn.decomposition import PCA
from scipy.stats import entropy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import mutual_info_score
import math
import matplotlib.pyplot as plt

class ColoredMNIST:
    def __init__(self, root='./data'):
        # Load original MNIST
        self.mnist_train = datasets.MNIST(root, train=True, download=True)
        self.mnist_test = datasets.MNIST(root, train=False, download=True)

    def create_environments(self):
        """Create E1 (training) and E2 (test) environments"""
        # Environment 1 (60,000 images)
        train_images = self.mnist_train.data.numpy()
        train_labels = self.mnist_train.targets.numpy()

        # Environment 2 (10,000 images)
        test_images = self.mnist_test.data.numpy()
        test_labels = self.mnist_test.targets.numpy()

        # Create colored versions
        e1_images, e1_colors = self._colorize_images(train_images, train_labels, env=1)
        e2_images, e2_colors = self._colorize_images(test_images, test_labels, env=2)

        return {
            'e1': {
                'images': torch.FloatTensor(e1_images),
                'labels': torch.LongTensor(train_labels),
                'colors': torch.LongTensor(e1_colors)
            },
            'e2': {
                'images': torch.FloatTensor(e2_images),
                'labels': torch.LongTensor(test_labels),
                'colors': torch.LongTensor(e2_colors)
            }
        }

    def _colorize_images(self, images, labels, env):
        """Add color to MNIST images based on environment and label"""
        n_images = len(images)
        colored_images = np.zeros((n_images, 28, 28, 3))
        colors = np.zeros(n_images)  # 0 for Red, 1 for Green

        for i in range(n_images):
            is_even = labels[i] % 2 == 0
            # Probability of being red
            p_red = 0.75 if (is_even and env == 1) or (not is_even and env == 2) else 0.25

            # Assign color
            is_red = np.random.random() < p_red
            colors[i] = 0 if is_red else 1

            # Create colored image
            if is_red:
                colored_images[i, :, :, 0] = images[i] / 255.0  # Red channel
            else:
                colored_images[i, :, :, 1] = images[i] / 255.0  # Green channel

        return colored_images, colors
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)  # Changed from Dropout2d to Dropout
        self.dropout2 = nn.Dropout(0.5)   # Changed from Dropout2d to Dropout
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
def irm_penalty(logits, y):
    scale = torch.tensor(1., requires_grad=True)  # Removed .cuda()
    loss = nn.functional.cross_entropy(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)
def train_model(model, environments, device, epochs=2, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_samples = len(environments['e1']['images'])
    n_batches = n_samples // batch_size

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_penalty = 0

        # Create random permutation for batching
        perm = torch.randperm(n_samples)

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_indices = perm[start_idx:end_idx]

            # Get batch data
            images = environments['e1']['images'][batch_indices].permute(0, 3, 1, 2).to(device)
            labels = environments['e1']['labels'][batch_indices].to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            penalty = irm_penalty(logits, labels)

            # IRM penalty weight (λ)
            penalty_weight = 10000.0 if epoch >= 10 else 1.0
            total_loss = loss + penalty_weight * penalty

            total_loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            # Evaluate on both environments
            model.eval()
            with torch.no_grad():
                # E1 accuracy
                images_e1 = environments['e1']['images'].permute(0, 3, 1, 2).to(device)
                labels_e1 = environments['e1']['labels'].to(device)
                pred_e1 = model(images_e1).argmax(dim=1)
                acc_e1 = (pred_e1 == labels_e1).float().mean().item()

                # E2 accuracy
                images_e2 = environments['e2']['images'].permute(0, 3, 1, 2).to(device)
                labels_e2 = environments['e2']['labels'].to(device)
                pred_e2 = model(images_e2).argmax(dim=1)
                acc_e2 = (pred_e2 == labels_e2).float().mean().item()

                print(
                    f'Epoch {epoch}: E1 acc = {acc_e1:.3f}, E2 acc = {acc_e2:.3f}, Loss = {total_loss:.3f}, Penalty = {penalty:.3f}')
class MetricsCalculator:
    def __init__(self, model, environments, device):
        self.model = model
        self.environments = environments
        self.device = device

    def compute_accuracy(self, env_key='e2'):
        """Compute classification accuracy on specified environment"""
        self.model.eval()
        with torch.no_grad():
            images = self.environments[env_key]['images'].permute(0, 3, 1, 2).to(self.device)
            labels = self.environments[env_key]['labels'].to(self.device)
            pred = self.model(images).argmax(dim=1)
            acc = (pred == labels).float().mean().item()
        return acc * 100

    def get_representations(self, env_key):
        """Extract intermediate representations from the model"""
        self.model.eval()
        representations = []

        def get_activation(name):
            def hook(model, input, output):
                representations.append(output.detach().cpu().numpy())

            return hook

        # Register hooks
        handles = []
        handles.append(self.model.conv1.register_forward_hook(get_activation('conv1')))
        handles.append(self.model.conv2.register_forward_hook(get_activation('conv2')))

        with torch.no_grad():
            images = self.environments[env_key]['images'].permute(0, 3, 1, 2).to(self.device)
            self.model(images)

        # Remove hooks
        for handle in handles:
            handle.remove()

        return representations

    def compute_env_independence(self):
        """Compute mutual information between representations and environments conditioned on labels"""
        # Get representations for both environments
        e1_reps = self.get_representations('e1')[-1]  # Use last layer
        e2_reps = self.get_representations('e2')[-1]

        # Flatten and reduce dimensionality
        e1_flat = e1_reps.reshape(e1_reps.shape[0], -1)
        e2_flat = e2_reps.reshape(e2_reps.shape[0], -1)

        # Use PCA to reduce dimensionality to 1D for MI calculation
        pca = PCA(n_components=1)
        combined_reps = np.vstack([e1_flat, e2_flat])
        reduced_reps = pca.fit_transform(combined_reps)

        # Create environment labels (0 for e1, 1 for e2)
        env_labels = np.concatenate([
            np.zeros(len(e1_reps)),
            np.ones(len(e2_reps))
        ])

        # Get true labels
        labels = np.concatenate([
            self.environments['e1']['labels'].numpy(),
            self.environments['e2']['labels'].numpy()
        ])

        # Compute conditional mutual information
        mi_sum = 0
        for label in np.unique(labels):
            mask = labels == label
            if np.sum(mask) > 0:
                mi = mutual_info_score(
                    env_labels[mask],
                    reduced_reps[mask].ravel()
                )
                mi_sum += mi * np.mean(mask)

        return mi_sum

    def compute_low_level_invariance(self):
        """Compute R1 metric (stability of low-level representations)"""
        e1_reps = self.get_representations('e1')[0]  # Use first conv layer
        e2_reps = self.get_representations('e2')[0]

        # Compute mean activation difference
        e1_mean = np.mean(e1_reps, axis=0)
        e2_mean = np.mean(e2_reps, axis=0)

        # Compute normalized difference
        diff = e1_mean - e2_mean
        r1 = np.mean(diff ** 2)

        return r1

    def compute_intervention_robustness(self):
        """Compute R2 metric (consistency under interventions)"""
        self.model.eval()
        with torch.no_grad():
            # E1 predictions
            images_e1 = self.environments['e1']['images'].permute(0, 3, 1, 2).to(self.device)
            pred_e1 = torch.softmax(self.model(images_e1), dim=1).cpu().numpy()

            # E2 predictions (intervention)
            images_e2 = self.environments['e2']['images'].permute(0, 3, 1, 2).to(self.device)
            pred_e2 = torch.softmax(self.model(images_e2), dim=1).cpu().numpy()

        # Compute average KL divergence
        kl_divs = []
        for p_e1, p_e2 in zip(pred_e1, pred_e2):
            # Add small epsilon to avoid division by zero
            eps = 1e-10
            p_e1 = np.clip(p_e1, eps, 1 - eps)
            p_e2 = np.clip(p_e2, eps, 1 - eps)
            kl_div = entropy(p_e1, p_e2)
            kl_divs.append(kl_div)

        r2 = np.mean(kl_divs)
        return r2

    def compute_all_metrics(self):
        """Compute all evaluation metrics"""
        metrics = {
            'accuracy': self.compute_accuracy(),
            'env_independence': self.compute_env_independence(),
            'low_level_invariance': self.compute_low_level_invariance(),
            'intervention_robustness': self.compute_intervention_robustness()
        }
        return metrics
def cevaluate_model(model, environments, device):
    calculator = MetricsCalculator(model, environments, device)
    metrics = calculator.compute_all_metrics()

    print("\nFinal Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Environment Independence: {metrics['env_independence']:.3f}")
    print(f"Low-level Invariance (R1): {metrics['low_level_invariance']:.3f}")
    print(f"Intervention Robustness (R2): {metrics['intervention_robustness']:.3f}")

    return metrics
def cmnist():
    device = torch.device("cpu")
    print(f"Using device: {device}")

    print("Creating Colored MNIST dataset...")
    cmnistt = ColoredMNIST()
    environments = cmnistt.create_environments()

    print("Creating model...")
    model = ConvNet().to(device)

    print("Starting training...")
    train_model(model, environments, device)

    print("\nComputing final metrics...")
    metrics = cevaluate_model(model, environments, device)
    return metrics



class RotatedMNIST:
    def __init__(self, root='./data'):
        # Load original MNIST
        self.mnist_train = datasets.MNIST(root, train=True, download=True)
        self.mnist_test = datasets.MNIST(root, train=False, download=True)

        # Define rotation angles
        self.angles = [0, 15, 30, 45, 60, 75]

    def rotate_image(self, image, angle):
        # Rotate image using scipy.ndimage
        return rotate(image, angle, reshape=False)

    def create_environments(self):
        """Create environments E1 (training) and E2 (test) based on the paper's distribution"""
        # Environment 1 (60,000 images)
        train_images = self.mnist_train.data.numpy()
        train_labels = self.mnist_train.targets.numpy()

        # Environment 2 (10,000 images)
        test_images = self.mnist_test.data.numpy()
        test_labels = self.mnist_test.targets.numpy()

        # Initialize environment data structures
        e1_images = []
        e1_labels = []
        e1_angles = []
        e2_images = []
        e2_labels = []
        e2_angles = []

        # Process each digit according to the distribution in the paper
        for digit in range(10):
            # Environment 1
            digit_mask_train = train_labels == digit
            digit_images_train = train_images[digit_mask_train]
            n_train = len(digit_images_train)

            # Apply rotations according to the paper's distribution
            if digit % 2 == 0:  # Even digits
                n_15deg = int(0.75 * n_train)
                n_75deg = n_train - n_15deg

                rotated_15 = np.array([self.rotate_image(img, 15) for img in digit_images_train[:n_15deg]])
                rotated_75 = np.array([self.rotate_image(img, 75) for img in digit_images_train[n_15deg:]])

                e1_images.extend(rotated_15)
                e1_images.extend(rotated_75)
                e1_labels.extend([digit] * n_train)
                e1_angles.extend([15] * n_15deg + [75] * n_75deg)
            else:  # Odd digits
                n_15deg = int(0.25 * n_train)
                n_75deg = n_train - n_15deg

                rotated_15 = np.array([self.rotate_image(img, 15) for img in digit_images_train[:n_15deg]])
                rotated_75 = np.array([self.rotate_image(img, 75) for img in digit_images_train[n_15deg:]])

                e1_images.extend(rotated_15)
                e1_images.extend(rotated_75)
                e1_labels.extend([digit] * n_train)
                e1_angles.extend([15] * n_15deg + [75] * n_75deg)

            # Environment 2 (reverse probabilities)
            digit_mask_test = test_labels == digit
            digit_images_test = test_images[digit_mask_test]
            n_test = len(digit_images_test)

            if digit % 2 == 0:  # Even digits
                n_15deg = int(0.25 * n_test)
                n_75deg = n_test - n_15deg
            else:  # Odd digits
                n_15deg = int(0.75 * n_test)
                n_75deg = n_test - n_15deg

            rotated_15 = np.array([self.rotate_image(img, 15) for img in digit_images_test[:n_15deg]])
            rotated_75 = np.array([self.rotate_image(img, 75) for img in digit_images_test[n_15deg:]])

            e2_images.extend(rotated_15)
            e2_images.extend(rotated_75)
            e2_labels.extend([digit] * n_test)
            e2_angles.extend([15] * n_15deg + [75] * n_75deg)

        return {
            'e1': {
                'images': torch.FloatTensor(np.array(e1_images)[:, None, :, :] / 255.0),
                'labels': torch.LongTensor(e1_labels),
                'angles': torch.LongTensor(e1_angles)
            },
            'e2': {
                'images': torch.FloatTensor(np.array(e2_images)[:, None, :, :] / 255.0),
                'labels': torch.LongTensor(e2_labels),
                'angles': torch.LongTensor(e2_angles)
            }
        }
class ConvNetr(nn.Module):
    def __init__(self):
        super(ConvNetr, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
class MetricsCalculatorr:
    def __init__(self, model, environments, device):
        self.model = model
        self.environments = environments
        self.device = device

    def compute_accuracy(self, env_key='e2'):
        """Compute classification accuracy on specified environment"""
        self.model.eval()
        with torch.no_grad():
            images = self.environments[env_key]['images'].to(self.device)
            labels = self.environments[env_key]['labels'].to(self.device)
            pred = self.model(images).argmax(dim=1)
            acc = (pred == labels).float().mean().item()
        return acc * 100

    def get_representations(self, env_key):
        """Extract intermediate representations from the model"""
        self.model.eval()
        representations = []

        def get_activation(name):
            def hook(model, input, output):
                representations.append(output.detach().cpu().numpy())

            return hook

        handles = []
        handles.append(self.model.conv1.register_forward_hook(get_activation('conv1')))
        handles.append(self.model.conv2.register_forward_hook(get_activation('conv2')))

        with torch.no_grad():
            images = self.environments[env_key]['images'].to(self.device)
            self.model(images)

        for handle in handles:
            handle.remove()

        return representations

    def compute_env_independence(self):
        """Compute mutual information between representations and environments (rotation angles)"""
        e1_reps = self.get_representations('e1')[-1]
        e2_reps = self.get_representations('e2')[-1]

        e1_flat = e1_reps.reshape(e1_reps.shape[0], -1)
        e2_flat = e2_reps.reshape(e2_reps.shape[0], -1)

        pca = PCA(n_components=1)
        combined_reps = np.vstack([e1_flat, e2_flat])
        reduced_reps = pca.fit_transform(combined_reps)

        angles = torch.cat([
            self.environments['e1']['angles'],
            self.environments['e2']['angles']
        ]).numpy()

        labels = torch.cat([
            self.environments['e1']['labels'],
            self.environments['e2']['labels']
        ]).numpy()

        mi_sum = 0
        for label in np.unique(labels):
            mask = labels == label
            if np.sum(mask) > 0:
                mi = mutual_info_score(
                    angles[mask],
                    reduced_reps[mask].ravel()
                )
                mi_sum += mi * np.mean(mask)

        return mi_sum

    def compute_low_level_invariance(self):
        """Compute R1 metric (stability of low-level representations across rotations)"""
        e1_reps = self.get_representations('e1')[0]
        e2_reps = self.get_representations('e2')[0]

        e1_15deg_mask = self.environments['e1']['angles'] == 15
        e1_75deg_mask = self.environments['e1']['angles'] == 75

        e1_15deg_mean = np.mean(e1_reps[e1_15deg_mask], axis=0)
        e1_75deg_mean = np.mean(e1_reps[e1_75deg_mask], axis=0)

        diff = e1_15deg_mean - e1_75deg_mean
        r1 = np.mean(diff ** 2)

        return r1

    def compute_intervention_robustness(self):
        """Compute R2 metric (consistency under rotation interventions)"""
        self.model.eval()
        with torch.no_grad():
            images_e1 = self.environments['e1']['images'].to(self.device)
            pred_e1 = torch.softmax(self.model(images_e1), dim=1).cpu().numpy()

            images_e2 = self.environments['e2']['images'].to(self.device)
            pred_e2 = torch.softmax(self.model(images_e2), dim=1).cpu().numpy()

        kl_divs = []
        for p_e1, p_e2 in zip(pred_e1, pred_e2):
            eps = 1e-10
            p_e1 = np.clip(p_e1, eps, 1 - eps)
            p_e2 = np.clip(p_e2, eps, 1 - eps)
            kl_div = entropy(p_e1, p_e2)
            kl_divs.append(kl_div)

        r2 = np.mean(kl_divs)
        return r2

    def compute_all_metrics(self):
        """Compute all evaluation metrics"""
        metrics = {
            'accuracy': self.compute_accuracy(),
            'env_independence': self.compute_env_independence(),
            'low_level_invariance': self.compute_low_level_invariance(),
            'intervention_robustness': self.compute_intervention_robustness()
        }
        return metrics
def rtrain_model(model, environments, device, epochs=50, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_samples = len(environments['e1']['images'])
    n_batches = n_samples // batch_size

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_penalty = 0

        perm = torch.randperm(n_samples)

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_indices = perm[start_idx:end_idx]

            images = environments['e1']['images'][batch_indices].to(device)
            labels = environments['e1']['labels'][batch_indices].to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)

            # Compute IRM penalty
            scale = torch.tensor(1., requires_grad=True, device=device)
            scaled_logits = logits * scale
            scaled_loss = nn.functional.cross_entropy(scaled_logits, labels)
            penalty = torch.autograd.grad(scaled_loss, [scale], create_graph=True)[0] ** 2

            penalty_weight = 10000.0 if epoch >= 10 else 1.0
            total_loss = loss + penalty_weight * penalty

            total_loss.backward()
            optimizer.step()

        if epoch % 5 == 0:
            model.eval()
            metrics = MetricsCalculatorr(model, environments, device).compute_all_metrics()
            print(f'Epoch {epoch}:')
            print(f'Accuracy: {metrics["accuracy"]:.2f}%')
            print(f'Environment Independence: {metrics["env_independence"]:.3f}')
            print(f'Low-level Invariance (R1): {metrics["low_level_invariance"]:.3f}')
            print(f'Intervention Robustness (R2): {metrics["intervention_robustness"]:.3f}\n')
def revaluate_model(model, environments, device):
    calculator = MetricsCalculatorr(model, environments, device)
    metrics = calculator.compute_all_metrics()

    print("\nFinal Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Environment Independence: {metrics['env_independence']:.3f}")
    print(f"Low-level Invariance (R1): {metrics['low_level_invariance']:.3f}")
    print(f"Intervention Robustness (R2): {metrics['intervention_robustness']:.3f}")

    return metrics
def rmnist():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Creating Rotated MNIST dataset...")
    rrmnist = RotatedMNIST()
    environments = rrmnist.create_environments()

    print("Creating model...")
    model = ConvNetr().to(device)

    print("Starting training...")
    rtrain_model(model, environments, device)

    print("\nComputing final metrics...")
    metrics = revaluate_model(model, environments, device)
    return metrics


class BallAgentEnv:
    def __init__(self, n_balls=4, image_size=64, num_samples=10000):
        self.n_balls = n_balls
        self.image_size = image_size
        self.num_samples = num_samples
        self.ball_radius = int(image_size * 0.05)
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        self.min_distance = 0.2

    def _check_distance(self, positions):
        for i in range(0, len(positions), 2):
            for j in range(i + 2, len(positions), 2):
                x1, y1 = positions[i:i + 2]
                x2, y2 = positions[j:j + 2]
                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if dist < self.min_distance:
                    return False
        return True

    def _generate_positions(self):
        while True:
            positions = np.random.uniform(0.1, 0.9, size=2 * self.n_balls)
            if self._check_distance(positions):
                return positions

    def _render_image(self, positions):
        img = Image.new('RGB', (self.image_size, self.image_size), 'white')
        draw = ImageDraw.Draw(img)

        for i in range(self.n_balls):
            x, y = positions[2 * i:2 * i + 2]
            x_px = int(x * self.image_size)
            y_px = int(y * self.image_size)
            bbox = [(x_px - self.ball_radius, y_px - self.ball_radius),
                    (x_px + self.ball_radius, y_px + self.ball_radius)]
            draw.ellipse(bbox, fill=self.colors[i])

        return np.array(img)

    def create_environments(self):
        e1_images, e1_positions, e1_interventions = [], [], []
        e2_images, e2_positions, e2_interventions = [], [], []

        n_train = int(0.8 * self.num_samples)

        # Environment 1: Random interventions
        for _ in range(n_train):
            pos = self._generate_positions()
            n_interventions = np.random.randint(1, 3)  # Force at least one intervention
            intervention_idx = np.random.choice(self.n_balls, n_interventions, replace=False)

            # Modified intervention strength
            interventions = np.zeros(self.n_balls)
            interventions[intervention_idx] = np.random.uniform(0.5, 1.0, size=len(intervention_idx))

            intervened_pos = pos.copy()
            for idx in intervention_idx:
                intervened_pos[2 * idx:2 * idx + 2] = self._generate_positions()[2 * idx:2 * idx + 2]

            img = self._render_image(pos)
            e1_images.append(img)
            e1_positions.append(pos)
            e1_interventions.append(interventions)

        for _ in range(self.num_samples - n_train):
            pos = self._generate_positions()
            n_interventions = np.random.randint(0, self.n_balls + 1)
            intervention_idx = np.random.choice(self.n_balls, n_interventions, replace=False)
            interventions = np.zeros(self.n_balls)
            interventions[intervention_idx] = 1

            intervened_pos = pos.copy()
            for idx in intervention_idx:
                intervened_pos[2 * idx:2 * idx + 2] = self._generate_positions()[2 * idx:2 * idx + 2]

            img = self._render_image(intervened_pos)
            e2_images.append(img)
            e2_positions.append(intervened_pos)
            e2_interventions.append(interventions)

        return {
            'e1': {
                'images': torch.FloatTensor(np.array(e1_images)).permute(0, 3, 1, 2) / 255.0,
                'positions': torch.FloatTensor(np.array(e1_positions)) / self.image_size,
                'interventions': torch.FloatTensor(np.array(e1_interventions))
            },
            'e2': {
                'images': torch.FloatTensor(np.array(e2_images)).permute(0, 3, 1, 2) / 255.0,
                'positions': torch.FloatTensor(np.array(e2_positions)) / self.image_size,  # Added normalization
                'interventions': torch.FloatTensor(np.array(e2_interventions))
            }
        }
class BallPredictor(nn.Module):
    def __init__(self, n_balls=4):
        super(BallPredictor, self).__init__()
        self.n_balls = n_balls

        # Deeper CNN architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
        )

        # Calculate fc_size dynamically
        with torch.no_grad():
            x = torch.randn(1, 3, 64, 64)
            x = self.features(x)
            self.fc_size = x.numel()

        self.regressor = nn.Sequential(
            nn.Linear(self.fc_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 2 * n_balls)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)  # Changed from view to reshape
        x = self.regressor(x)
        return x
def btrain_model(model, environments, device, epochs=30, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    criterion = nn.MSELoss()

    n_samples = len(environments['e1']['images'])
    n_batches = n_samples // batch_size
    best_loss = float('inf')

    def get_penalty_weight(epoch):
        if epoch < 5:
            return 0.1
        elif epoch < 15:
            return 10.0
        else:
            return 100.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        perm = torch.randperm(n_samples)

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_indices = perm[start_idx:end_idx]

            images = environments['e1']['images'][batch_indices].to(device)
            positions = environments['e1']['positions'][batch_indices].to(device)

            optimizer.zero_grad()
            pred_positions = model(images)
            loss = criterion(pred_positions, positions)

            scale = torch.tensor(1., requires_grad=True, device=device)
            scaled_preds = pred_positions * scale
            penalty = torch.autograd.grad(criterion(scaled_preds, positions), [scale],
                                        create_graph=True)[0] ** 2

            penalty_weight = get_penalty_weight(epoch)
            total_loss = loss + penalty_weight * penalty + 0.01 * torch.norm(pred_positions, p=2)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / n_batches
        scheduler.step(avg_loss)

        if epoch % 5 == 0:
            print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
            metrics = MetricsCalculatorb(model, environments, device).compute_all_metrics()
            print(f'Accuracy: {metrics["accuracy"]:.2f}%')
            print(f'Environment Independence: {metrics["env_independence"]:.3f}')
            print(f'Low-level Invariance (R1): {metrics["low_level_invariance"]:.3f}')
            print(f'Intervention Robustness (R2): {metrics["intervention_robustness"]:.3f}\n')

            if avg_loss < best_loss:
                best_loss = avg_loss
class MetricsCalculatorb:
    def __init__(self, model, environments, device):
        self.model = model
        self.environments = environments
        self.device = device
        self.threshold = 0.15

    def compute_accuracy(self, env_key='e2'):
        self.model.eval()
        with torch.no_grad():
            images = self.environments[env_key]['images'].to(self.device)
            true_pos = self.environments[env_key]['positions'].to(self.device)
            pred_pos = self.model(images)
            errors = torch.abs(pred_pos - true_pos)
            correct = (errors < self.threshold).all(dim=1)
            acc = correct.float().mean().item()
        return acc * 100

    def compute_intervention_robustness(self):
        with torch.no_grad():
            min_len = min(len(self.environments['e1']['images']),
                          len(self.environments['e2']['images']))

            e1_images = self.environments['e1']['images'][:min_len].to(self.device)
            e2_images = self.environments['e2']['images'][:min_len].to(self.device)
            e1_interv = self.environments['e1']['interventions'][:min_len].to(self.device)

            e1_pred = self.model(e1_images)
            e2_pred = self.model(e2_images)

            mask = (e1_interv == 0).float()
            pred_diff = ((e1_pred - e2_pred) ** 2) * mask.repeat_interleave(2, dim=1)

            return pred_diff.mean().item()

    def compute_env_independence(self):
        with torch.no_grad():
            e1_images = self.environments['e1']['images'].to(self.device)
            e1_pred = self.model(e1_images)
            e1_interv = self.environments['e1']['interventions']

            independence_score = 0
            for i in range(self.model.n_balls):
                pred = e1_pred[:, 2 * i:2 * i + 2]
                other_balls_pred = torch.cat([e1_pred[:, :2 * i], e1_pred[:, 2 * i + 2:]], dim=1)

                # Compute correlation between predictions
                correlation = torch.corrcoef(torch.cat([pred, other_balls_pred], dim=1).T)[0:2, 2:]
                independence_score += torch.abs(correlation).mean()

            return independence_score.item() / self.model.n_balls

    def compute_low_level_invariance(self):
        self.model.eval()
        with torch.no_grad():
            min_len = min(len(self.environments['e1']['images']),
                          len(self.environments['e2']['images']))

            e1_images = self.environments['e1']['images'][:min_len].to(self.device)
            e2_images = self.environments['e2']['images'][:min_len].to(self.device)

            def get_conv1_output(images):
                return self.model.features[0](images)

            e1_rep = get_conv1_output(e1_images).cpu().numpy()
            e2_rep = get_conv1_output(e2_images).cpu().numpy()

            diff = np.mean(e1_rep, axis=0) - np.mean(e2_rep, axis=0)
            r1 = np.mean(diff ** 2)
        return r1

    def compute_all_metrics(self):
        return {
            'accuracy': self.compute_accuracy(),
            'env_independence': self.compute_env_independence(),
            'low_level_invariance': self.compute_low_level_invariance(),
            'intervention_robustness': self.compute_intervention_robustness()
        }
def ball_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Creating Ball Agent environment...")
    env = BallAgentEnv(n_balls=4, image_size=64, num_samples=2000)
    environments = env.create_environments()

    print("Creating model...")
    model = BallPredictor().to(device)

    print("Starting training...")
    btrain_model(model, environments, device, epochs=30, batch_size=32)

    print("\nComputing final metrics...")
    metrics = MetricsCalculatorb(model, environments, device).compute_all_metrics()
    return metrics


def plot_comparative_metrics(ball_metrics):
    metrics = ['Accuracy', 'Env Independence', 'Low-level Invariance', 'Intervention Robustness']

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract values
    ball_values = [ball_metrics['accuracy'],
                   ball_metrics['env_independence'],
                   ball_metrics['low_level_invariance'],
                   ball_metrics['intervention_robustness']]

    # rotated_values = [rotated_metrics['accuracy'],
    #                   rotated_metrics['env_independence'],
    #                   rotated_metrics['low_level_invariance'],
    #                   rotated_metrics['intervention_robustness']]
    #
    # colored_values = [colored_metrics['accuracy'],
    #                   colored_metrics['env_independence'],
    #                   colored_metrics['low_level_invariance'],
    #                   colored_metrics['intervention_robustness']]

    # Create bars
    ax.bar(x - width, ball_values, width, label='BallAgent')
    # ax.bar(x, rotated_values, width, label='RotatedMNIST')
    # ax.bar(x + width, colored_values, width, label='ColoredMNIST')

    ax.set_ylabel('Values')
    ax.set_title('Comparison of Metrics Across Implementations')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # cmnist()
    # rmnist()
    # camelyon17()
    # metrics = ball_agent()
    ball_metrics = ball_agent()
    plot_comparative_metrics(ball_metrics)
