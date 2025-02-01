"""
dataset.py
"""
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import torch
from torch.utils.data import TensorDataset
from scipy.ndimage import rotate


class ColoredMNIST:
    def __init__(self, env, root='./data'):
        self.env = env
        self.root = root
        mnist = datasets.MNIST(root, train=(env == 'e1'), download=True)
        self.images, self.labels = mnist.data.numpy(), mnist.targets.numpy()
        self.images = self.color_images()

    def color_images(self):
        colored_images = np.zeros((len(self.images), 28, 28, 3), dtype=np.uint8)
        for i, img in enumerate(self.images):
            color = self.get_color(self.labels[i])
            colored_images[i, :, :, 0] = (img * color[0]).astype(np.uint8)
            colored_images[i, :, :, 1] = (img * color[1]).astype(np.uint8)
        return colored_images

    def get_color(self, label):
        if self.env == 'e1':
            # In e1, P(red|even) = 0.75, P(red|odd) = 0.25
            p_red = 0.75 if label % 2 == 0 else 0.25
        else:
            # In e2, P(red|even) = 0.25, P(red|odd) = 0.75
            p_red = 0.25 if label % 2 == 0 else 0.75
        return (1, 0) if np.random.random() < p_red else (0, 1)  # (R, G)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
def calculate_label_color_stats(env):
    stats = {i: {'red': 0, 'green': 0} for i in range(10)}
    for img, label in zip(env.images, env.labels):
        if img[:, :, 0].max() > 0:  # Red
            stats[label]['red'] += 1
        else:
            stats[label]['green'] += 1
    return stats
def display_examples(env, num_examples=5):
    fig, axes = plt.subplots(2, num_examples, figsize=(20, 8))
    fig.suptitle(f"Examples from {env.env.upper()}")
    label_color_stats = calculate_label_color_stats(env)
    for i in range(num_examples):
        idx = np.random.randint(len(env))
        img, label = env[idx]
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Label: {label}")
        axes[0, i].axis('off')
        red_count = label_color_stats[label]['red']
        green_count = label_color_stats[label]['green']
        x = np.arange(2)
        width = 0.35
        axes[1, i].bar(x[0], red_count, width, color='red', label='Red')
        axes[1, i].bar(x[1], green_count, width, color='green', label='Green')
        axes[1, i].set_ylabel('Count')
        axes[1, i].set_title(f"Label {label} Color Distribution")
        axes[1, i].set_xticks(x)
        axes[1, i].set_xticklabels(['Red', 'Green'])
        axes[1, i].text(x[0], red_count, str(red_count), ha='center', va='bottom')
        axes[1, i].text(x[1], green_count, str(green_count), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
def calculate_detailed_statistics(env):
    total = len(env)
    stats = {
        "Total images": total,
        "Digit counts": {},
        "Color counts": {"Red": 0, "Green": 0},
        "Digit-Color counts": {}
    }
    for i in range(10):
        stats["Digit counts"][i] = np.sum(env.labels == i)
        stats["Digit-Color counts"][i] = {"Red": 0, "Green": 0}
    for img, label in zip(env.images, env.labels):
        if img[:, :, 0].max() > 0:  # Red
            stats["Color counts"]["Red"] += 1
            stats["Digit-Color counts"][label]["Red"] += 1
        else:
            stats["Color counts"]["Green"] += 1
            stats["Digit-Color counts"][label]["Green"] += 1
    return stats
def visualize_cmnist():
    e1 = ColoredMNIST('e1')
    e2 = ColoredMNIST('e2')
    for env in [e1, e2]:
        print(f"\nEnvironment: {env.env.upper()}")
        display_examples(env)
        stats = calculate_detailed_statistics(env)
        print(f"Total images: {stats['Total images']}")
        print("\nDigit counts:")
        for digit, count in stats["Digit counts"].items():
            print(f"  {digit}: {count}")
        print("\nColor counts:")
        for color, count in stats["Color counts"].items():
            print(f"  {color}: {count}")
        print("\nDigit-Color counts:")
        for digit in range(10):
            red_count = stats["Digit-Color counts"][digit]["Red"]
            green_count = stats["Digit-Color counts"][digit]["Green"]
            total_digit = red_count + green_count
            red_percentage = (red_count / total_digit) * 100 if total_digit > 0 else 0
            print(f"  {digit}: Red = {red_count} ({red_percentage:.2f}%), Green = {green_count}")
        print("\n" + "=" * 50 + "\n")


class RotatedMNIST:
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']
    def __init__(self, root='./data'):
        self.root = root
        self.envs = {}
        mnist = datasets.MNIST(root, train=True, download=True)
        self.images, self.labels = mnist.data.numpy(), mnist.targets.numpy()
        for angle in [0, 15, 30, 45, 60, 75]:
            self.envs[str(angle)] = self.rotate_dataset(self.images, self.labels, angle)
    def rotate_dataset(self, images, labels, angle):
        rotated_images = np.zeros_like(images)
        for i, img in enumerate(images):
            rotated_images[i] = rotate(img, angle, reshape=False)
        return TensorDataset(torch.tensor(rotated_images), torch.tensor(labels))

    def __getitem__(self, env):
        return self.envs[env]
def rdisplay_examples(env_data, angle, num_examples=5):
    fig, axes = plt.subplots(1, num_examples, figsize=(15, 3))
    fig.suptitle(f"Examples from Rotated MNIST (Angle: {angle}°)")
    for i in range(num_examples):
        idx = np.random.randint(len(env_data))
        img, label = env_data[idx]
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
def calculate_statistics(env_data):
    labels = [item[1].item() for item in env_data]
    unique, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique, counts))
def visualize_rmnist():
    rotated_mnist = RotatedMNIST()
    for angle in RotatedMNIST.ENVIRONMENTS:
        print(f"\nEnvironment: Rotation {angle}°")
        env_data = rotated_mnist[angle]
        rdisplay_examples(env_data, angle)
        stats = calculate_statistics(env_data)
        print("Digit counts:")
        for digit, count in stats.items():
            print(f"  {digit}: {count}")
        print("\n" + "=" * 50 + "\n")


class Camelyon17Dataset:
    def __init__(self, hospital='h1'):
        self.hospital = hospital
        self.distribution = {
            'h1': {'total': 17934, 'tumor': 7786, 'normal': 10148},
            'h2': {'total': 15987, 'tumor': 6446, 'normal': 9541},
            'h3': {'total': 16828, 'tumor': 7212, 'normal': 9616},
            'h4': {'total': 17155, 'tumor': 7502, 'normal': 9653},
            'h5': {'total': 16960, 'tumor': 7089, 'normal': 9871}
        }
        self.hospital_params = {
            'h1': {'background': (0.98, 0.95, 0.95), 'tumor': (0.6, 0.2, 0.2)},
            'h2': {'background': (0.97, 0.95, 0.98), 'tumor': (0.5, 0.15, 0.4)},
            'h3': {'background': (0.98, 0.98, 0.92), 'tumor': (0.5, 0.4, 0.15)},
            'h4': {'background': (0.97, 0.95, 0.93), 'tumor': (0.5, 0.3, 0.2)},
            'h5': {'background': (0.95, 0.97, 0.97), 'tumor': (0.3, 0.4, 0.4)}
        }
        self.images = {'normal': self._generate_tissue_pattern(False),
                       'tumor': self._generate_tissue_pattern(True)}

    def _generate_noise_pattern(self, size, density=0.15, scale=1.0):
        noise = np.random.rand(size, size) < density
        noise = gaussian_filter(noise.astype(float), sigma=0.5) * scale
        return noise

    def _generate_tissue_pattern(self, has_tumor=False):
        size = 96
        img = np.ones((size, size, 3)) * np.array(self.hospital_params[self.hospital]['background'])
        white_spots = self._generate_noise_pattern(size, density=0.1, scale=1.2)
        for c in range(3):
            img[:, :, c] = np.clip(img[:, :, c] + white_spots, 0, 1)

        if has_tumor:
            center = size // 2
            radius = size // 5
            y, x = np.ogrid[-center:size - center, -center:size - center]
            tumor_base = x * x + y * y <= radius * radius
            tumor_texture = np.zeros((size, size))
            for _ in range(3):  # Layer multiple noise patterns
                noise = self._generate_noise_pattern(size, density=0.3, scale=0.5)
                tumor_texture += noise
            tumor_texture = tumor_texture / tumor_texture.max()
            tumor_mask = tumor_base & (tumor_texture > 0.3)
            tumor_color = np.array(self.hospital_params[self.hospital]['tumor'])
            for c in range(3):
                img[:, :, c][tumor_mask] = tumor_color[c]
        return img
def visualize_camelyon17():
    plt.style.use('default')

    # Create main figure for examples
    fig_cam = plt.figure(figsize=(12, 15))
    hospitals = [f'h{i}' for i in range(1, 6)]
    datasets = [Camelyon17Dataset(hospital=h) for h in hospitals]

    # Plot examples from each hospital
    for i, dataset in enumerate(datasets):
        # Normal tissue
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(dataset.images['normal'])
        plt.axis('off')
        plt.title(f'Hospital {hospitals[i]} - Normal')

        # Tumor tissue
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(dataset.images['tumor'])
        plt.axis('off')
        plt.title(f'Hospital {hospitals[i]} - Tumor')

    plt.suptitle('Camelyon17 Examples (by hospital)')

    # Create figure for distribution
    fig_dist = plt.figure(figsize=(10, 6))

    # Plot distribution using actual numbers from the paper
    tumor_counts = [d.distribution[d.hospital]['tumor'] for d in datasets]
    normal_counts = [d.distribution[d.hospital]['normal'] for d in datasets]

    x = np.arange(len(hospitals))
    width = 0.35

    plt.bar(x - width / 2, tumor_counts, width, label='Tumor', color='#e74c3c')
    plt.bar(x + width / 2, normal_counts, width, label='Normal', color='#2ecc71')

    plt.title('Camelyon17 Distribution')
    plt.xlabel('Hospital')
    plt.ylabel('Number of Images')
    plt.xticks(x, hospitals)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print distribution statistics
    print("\nCamelyon17 Distribution:")
    print("Hospital\tTotal\t\tTumor\t\tNormal")
    print("-" * 50)
    for dataset in datasets:
        dist = dataset.distribution[dataset.hospital]
        print(
            f"{dataset.hospital}\t\t{dist['total']}\t\t{dist['tumor']} ({dist['tumor'] / dist['total'] * 100:.2f}%)\t{dist['normal']} ({dist['normal'] / dist['total'] * 100:.2f}%)")


class BallAgentDataset:
    def __init__(self, n_balls=3, n_samples=10000):
        self.n_balls = n_balls
        self.n_samples = n_samples
        self.colors = [(1, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0.7, 0)]  # Added yellow for 4th ball
        self.size = 64
        self.positions, self.images, self.interventions = self._generate_data()

    def _generate_data(self):
        positions = []
        images = []
        interventions = []

        while len(positions) < self.n_samples:
            # Generate positions with minimum distance constraint
            pos = np.random.uniform(0.1, 0.9, (self.n_balls, 2))
            valid = True

            # Check distances between all pairs
            for i in range(self.n_balls):
                for j in range(i + 1, self.n_balls):
                    dist = np.sqrt(np.sum((pos[i] - pos[j]) ** 2))
                    if dist < 0.2:  # Minimum distance constraint
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                continue

            # Generate interventions
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
def visualize_ball_agent(n_balls):
    dataset = BallAgentDataset(n_balls=n_balls, n_samples=10000)
    intervention_counts = np.sum(dataset.interventions, axis=0)
    simultaneous_interventions = np.sum(dataset.interventions, axis=1)
    pattern_counts = np.bincount(simultaneous_interventions)
    cooccurrence = np.zeros((n_balls, n_balls))
    for intervention in dataset.interventions:
        cooccurrence += np.outer(intervention, intervention)
    for i in range(n_balls):
        cooccurrence[i] = cooccurrence[i] / cooccurrence[i, i]
    distances = []
    for pos in dataset.positions:
        pos_reshaped = pos.reshape(-1, 2)
        for i in range(n_balls):
            for j in range(i + 1, n_balls):
                dist = np.sqrt(np.sum((pos_reshaped[i] - pos_reshaped[j]) ** 2))
                distances.append(dist)
    fig = plt.figure(figsize=(15, 12))
    for i in range(4):
        ax = plt.subplot(2, 2, i + 1)
        plt.imshow(dataset.images[i])
        plt.axis('off')
        plt.title(f'Example {i + 1}')
    plt.tight_layout()

    fig = plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, color='#2ecc71')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    print(f"\nBall Agent Dataset Statistics ({n_balls} balls):")
    print("-" * 50)
    print(f"Total samples: {dataset.n_samples}")
    print(f"\nPer-Ball Intervention Statistics:")
    for i in range(n_balls):
        print(
            f"Ball {i}: {intervention_counts[i]} interventions ({intervention_counts[i] / dataset.n_samples * 100:.1f}%)")
    print(f"\nSimultaneous Intervention Patterns:")
    for i, count in enumerate(pattern_counts):
        print(f"{i} ball{'s' if i != 1 else ''} intervened: {count} samples ({count / dataset.n_samples * 100:.1f}%)")
    print(f"\nDistance Statistics:")
    print(f"Minimum distance: {min(distances):.3f}")
    print(f"Maximum distance: {max(distances):.3f}")
    print(f"Mean distance: {np.mean(distances):.3f}")
    print(f"Median distance: {np.median(distances):.3f}")
    pattern_probs = pattern_counts / np.sum(pattern_counts)
    intervention_entropy = entropy(pattern_probs)
    print(f"\nIntervention Pattern Entropy: {intervention_entropy:.3f} bits")
    plt.show()

if __name__ == "__main__":
    # visualize_cmnist()
    # visualize_rmnist()
    # visualize_camelyon17()
    visualize_ball_agent(n_balls = 4)
