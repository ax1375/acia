import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from typing import List, Tuple, Set
import seaborn as sns

class BallEnvironment:
    def __init__(self, num_balls: int = 4, image_size: int = 64, ball_radius: float = 0.05,
                 min_distance: float = 0.2):
        self.num_balls = num_balls
        self.image_size = image_size
        self.ball_radius = ball_radius
        self.min_distance = min_distance
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # R,G,B,Y

    def _check_distance_constraint(self, positions: np.ndarray) -> bool:
        """Check if balls maintain minimum distance"""
        for i in range(self.num_balls):
            for j in range(i + 1, self.num_balls):
                pos1 = positions[2 * i:2 * i + 2]
                pos2 = positions[2 * j:2 * j + 2]
                if np.linalg.norm(pos1 - pos2) < self.min_distance:
                    return False
        return True

    def _sample_valid_positions(self) -> np.ndarray:
        """Sample valid ball positions maintaining minimum distance"""
        while True:
            positions = np.random.uniform(0.1, 0.9, size=2 * self.num_balls)
            if self._check_distance_constraint(positions):
                return positions

    def render_image(self, positions: np.ndarray) -> np.ndarray:
        """Render balls on image given positions"""
        img = Image.new('RGB', (self.image_size, self.image_size), 'white')
        draw = ImageDraw.Draw(img)

        radius_px = int(self.ball_radius * self.image_size)
        for i in range(self.num_balls):
            x = int(positions[2 * i] * self.image_size)
            y = int(positions[2 * i + 1] * self.image_size)
            draw.ellipse([x - radius_px, y - radius_px, x + radius_px, y + radius_px],
                         fill=self.colors[i])

        return np.array(img)

    def generate_dataset(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray, List[Set[int]]]:
        """Generate dataset of images with positions and interventions"""
        all_positions = []
        all_images = []
        all_interventions = []

        for _ in range(num_samples):
            # Sample base positions
            positions = self._sample_valid_positions()

            # Sample interventions
            num_interventions = np.random.randint(0, self.num_balls + 1)
            intervention_coords = set(np.random.choice(2 * self.num_balls,
                                                       size=num_interventions,
                                                       replace=False))

            # Apply interventions
            intervened_positions = positions.copy()
            for coord in intervention_coords:
                intervened_positions[coord] = np.random.uniform(0.1, 0.9)

            # Ensure minimum distance is maintained after interventions
            while not self._check_distance_constraint(intervened_positions):
                for coord in intervention_coords:
                    intervened_positions[coord] = np.random.uniform(0.1, 0.9)

            # Render image
            image = self.render_image(intervened_positions)

            all_positions.append(intervened_positions)
            all_images.append(image)
            all_interventions.append(intervention_coords)

        return (np.stack(all_positions),
                np.stack(all_images),
                all_interventions)
class ResNetBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return x + self.layers(x)
class DANNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2)
        )

        self.label_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 5)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 8)
        )

    def forward(self, x, alpha=1.0):
        features = self.feature_extractor(x)
        class_pred = self.label_classifier(features)

        if self.training:
            # Gradient reversal only for domain classification
            reversed_features = GradientReversal.apply(features, alpha)
            domain_pred = self.domain_classifier(reversed_features)
            return class_pred, domain_pred
        return class_pred
def dann_loss(class_pred, domain_pred, labels, domain_labels):
    lambda_param = 0.3  # Reduced domain importance
    classification_loss = F.cross_entropy(class_pred, labels)
    domain_loss = F.cross_entropy(domain_pred, domain_labels)
    return classification_loss + lambda_param * domain_loss
def train_dann(model, optimizer, x, labels, domain_labels, epoch):
    model.train()
    optimizer.zero_grad()

    alpha = 2.0 / (1.0 + np.exp(-10 * epoch / 200)) - 1.0
    class_pred, domain_pred = model(x, alpha)

    loss = dann_loss(class_pred, domain_pred, labels, domain_labels)
    loss.backward()
    optimizer.step()
    return loss.item()
class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None
def evaluate_dann(model, x, labels, env_labels):
    model.eval()
    with torch.no_grad():
        # Get predictions
        pred = model(x).argmax(dim=1)
        features = model.feature_extractor(x)

        accuracy = (pred == labels).float().mean().item() * 100
        env_ind = bcalculate_mutual_information(features, env_labels, labels)
        low_level = bcalculate_representation_variance(features, env_labels)
        interv_rob = bcalculate_intervention_robustness(model, x, labels)

    return accuracy, env_ind, low_level, interv_rob
def print_metrics(epoch, loss, metrics):
    print(f"Epoch {epoch}:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {metrics[0]:.2f}%")
    print(f"Env Independence: {metrics[1]:.3f}")
    print(f"Low-level Inv: {metrics[2]:.3f}")
    print(f"Interv Robustness: {metrics[3]:.3f}\n")
def bcalculate_mutual_information(features, env_labels, class_labels):
    return torch.corrcoef(torch.stack([features.mean(1), env_labels.float()]))[0, 1].abs().item()
def bcalculate_representation_variance(features, env_labels):
    return torch.var(features, dim=0).mean().item()
def bcalculate_intervention_robustness(model, x, labels):
    perturbed_x = x + torch.randn_like(x) * 0.1
    with torch.no_grad():
        orig_preds = model(x).argmax(dim=1)
        pert_preds = model(perturbed_x).argmax(dim=1)
    return (orig_preds != pert_preds).float().mean().item()
def ballagent(ball_env, num_epochs=200):
    positions, images, interventions = ball_env.generate_dataset(10000)
    train_size = int(0.8 * len(positions))
    indices = torch.randperm(len(positions))

    train_x = torch.FloatTensor(positions[indices[:train_size]])
    test_x = torch.FloatTensor(positions[indices[train_size:]])
    train_labels = torch.LongTensor([len(i) for i in interventions])[indices[:train_size]]
    test_labels = torch.LongTensor([len(i) for i in interventions])[indices[train_size:]]
    train_env = torch.LongTensor([min(7, list(i)[0]) if len(i) > 0 else 0
                                  for i in interventions])[indices[:train_size]]
    test_env = torch.LongTensor([min(7, list(i)[0]) if len(i) > 0 else 0
                                 for i in interventions])[indices[train_size:]]

    model = DANNModel(input_dim=positions.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    metrics_history = []
    for epoch in range(num_epochs):
        batch_size = 256
        for i in range(0, len(train_x), batch_size):
            batch_x = train_x[i:i + batch_size]
            batch_labels = train_labels[i:i + batch_size]
            batch_env = train_env[i:i + batch_size]
            loss = train_dann(model, optimizer, batch_x, batch_labels, batch_env, epoch)

        if epoch % 20 == 0:
            metrics = evaluate_dann(model, test_x, test_labels, test_env)
            metrics_history.append(metrics)
            print_metrics(epoch, loss, metrics)

    return model, metrics_history



if __name__ == '__main__':
    # cmnist()
    # rmnist()
    # camelyon17()
    env = BallEnvironment(num_balls=4)
    model, history = ballagent(env)