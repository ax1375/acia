import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
from cauca_model import NonlinearCauCAModel, LinearCauCAModel
from configs import DGP

class CauCACMNIST(Dataset):
    def __init__(self, env, transform=None):
        """
        Colored MNIST dataset for CauCA
        Args:
            env: 'e1' or 'e2' for environment
            transform: Optional transform to be applied on images
        """
        self.cmnist = ColoredMNIST(env)
        self.transform = transform

        # Convert numpy arrays to torch tensors
        self.images = torch.FloatTensor(self.cmnist.images) / 255.0
        self.labels = torch.LongTensor(self.cmnist.labels)

        # Create environment tensor (0 for e1, 1 for e2)
        self.env_id = torch.zeros(len(self.images)) if env == 'e1' else torch.ones(len(self.images))

        # Create intervention targets tensor based on environment and digit parity
        self.intervention_targets = torch.zeros((len(self.images), 2))  # 2 for color channels
        for i, label in enumerate(self.labels):
            if env == 'e1':
                self.intervention_targets[i, 0] = 1 if label % 2 == 0 else 0
            else:
                self.intervention_targets[i, 0] = 0 if label % 2 == 0 else 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        return {
            'x': image,
            'y': self.labels[idx],
            'e': self.env_id[idx],
            'int_target': self.intervention_targets[idx]
        }


class CauCARMNIST(Dataset):
    def __init__(self, angles, transform=None):
        """
        Rotated MNIST dataset for CauCA
        Args:
            angles: List of rotation angles
            transform: Optional transform to be applied on images
        """
        self.angles = angles
        mnist = datasets.MNIST(root='./data', train=True, download=True)
        self.images = mnist.data.float() / 255.0
        self.labels = mnist.targets

        # Create rotated versions for each angle
        self.rotated_images = []
        self.env_ids = []
        self.intervention_targets = []

        for env_id, angle in enumerate(angles):
            rotated = transforms.functional.rotate(self.images, angle)
            self.rotated_images.append(rotated)
            self.env_ids.extend([env_id] * len(self.images))

            # Create intervention targets
            targets = torch.zeros((len(self.images), len(angles)))
            targets[:, env_id] = 1
            self.intervention_targets.append(targets)

        self.rotated_images = torch.cat(self.rotated_images)
        self.env_ids = torch.tensor(self.env_ids)
        self.intervention_targets = torch.cat(self.intervention_targets)
        self.labels = self.labels.repeat(len(angles))

        self.transform = transform

    def __len__(self):
        return len(self.rotated_images)

    def __getitem__(self, idx):
        image = self.rotated_images[idx]
        if self.transform:
            image = self.transform(image)

        return {
            'x': image,
            'y': self.labels[idx],
            'e': self.env_ids[idx],
            'int_target': self.intervention_targets[idx]
        }


class CauCACamelyon17(Dataset):
    def __init__(self, hospital_ids, transform=None):
        """
        Camelyon17 dataset for CauCA
        Args:
            hospital_ids: List of hospital IDs to include
            transform: Optional transform to be applied on images
        """
        self.transform = transform
        self.datasets = [Camelyon17Dataset(f'h{h}') for h in hospital_ids]

        # Combine data from all hospitals
        self.images = []
        self.labels = []
        self.env_ids = []
        self.intervention_targets = []

        for env_id, dataset in enumerate(self.datasets):
            # Get normal and tumor images
            normal_images = dataset.images['normal']
            tumor_images = dataset.images['tumor']

            self.images.extend([normal_images, tumor_images])
            self.labels.extend([0] * len(normal_images) + [1] * len(tumor_images))
            self.env_ids.extend([env_id] * (len(normal_images) + len(tumor_images)))

            # Create intervention targets
            targets = torch.zeros((len(normal_images) + len(tumor_images), len(hospital_ids)))
            targets[:, env_id] = 1
            self.intervention_targets.append(targets)

        self.images = torch.tensor(np.array(self.images))
        self.labels = torch.tensor(self.labels)
        self.env_ids = torch.tensor(self.env_ids)
        self.intervention_targets = torch.cat(self.intervention_targets)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        return {
            'x': image,
            'y': self.labels[idx],
            'e': self.env_ids[idx],
            'int_target': self.intervention_targets[idx]
        }


class CauCABallAgent(Dataset):
    def __init__(self, n_balls=4, n_samples=10000, transform=None):
        """
        Ball Agent dataset for CauCA
        Args:
            n_balls: Number of balls
            n_samples: Number of samples
            transform: Optional transform to be applied on images
        """
        self.dataset = BallAgentDataset(n_balls=n_balls, n_samples=n_samples)
        self.transform = transform

        # Convert numpy arrays to torch tensors
        self.images = torch.FloatTensor(self.dataset.images)
        self.positions = torch.FloatTensor(self.dataset.positions)
        self.interventions = torch.FloatTensor(self.dataset.interventions)

        # Create environment IDs based on intervention patterns
        self.env_ids = torch.argmax(self.interventions, dim=1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)

        return {
            'x': image,
            'y': self.positions[idx],
            'e': self.env_ids[idx],
            'int_target': self.interventions[idx]
        }


class CauCADataModule(pl.LightningDataModule):
    def __init__(self, dataset_name, batch_size=32, num_workers=4, **dataset_kwargs):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_kwargs = dataset_kwargs

        # Define dataset-specific parameters
        self.dataset_configs = {
            'cmnist': {
                'class': CauCACMNIST,
                'envs': ['e1', 'e2']
            },
            'rmnist': {
                'class': CauCARMNIST,
                'angles': [0, 15, 30, 45, 60, 75]
            },
            'camelyon17': {
                'class': CauCACamelyon17,
                'hospital_ids': range(1, 6)
            },
            'ball_agent': {
                'class': CauCABallAgent,
                'n_balls': 4,
                'n_samples': 10000
            }
        }

    def setup(self, stage=None):
        config = self.dataset_configs[self.dataset_name]

        if self.dataset_name == 'cmnist':
            self.train_dataset = config['class'](env='e1', **self.dataset_kwargs)
            self.val_dataset = config['class'](env='e2', **self.dataset_kwargs)
        elif self.dataset_name == 'rmnist':
            angles = config['angles']
            self.train_dataset = config['class'](angles=angles[:-1], **self.dataset_kwargs)
            self.val_dataset = config['class'](angles=angles[-1:], **self.dataset_kwargs)
        else:
            # For Camelyon17 and Ball Agent, split the dataset
            full_dataset = config['class'](**self.dataset_kwargs)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size]
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


def train_cauca(args):
    # Set up data module
    data_module = CauCADataModule(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    data_module.setup()

    # Get dataset-specific configurations
    if args.dataset == 'cmnist':
        latent_dim = 2  # For color channels
        adjacency_matrix = np.array([[0, 1], [0, 0]])  # Y → C structure
    elif args.dataset == 'rmnist':
        latent_dim = 2  # For rotation
        adjacency_matrix = np.array([[0, 1], [0, 0]])  # Y → R structure
    elif args.dataset == 'camelyon17':
        latent_dim = 3  # RGB channels
        adjacency_matrix = np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])  # Y → X structure
    else:  # ball_agent
        latent_dim = args.n_balls * 2  # x,y coordinates for each ball
        adjacency_matrix = DGP['graph-4-0']['adj_matrix']  # Using predefined graph

    # Create model
    model_class = NonlinearCauCAModel if args.nonlinear else LinearCauCAModel
    model = model_class(
        latent_dim=latent_dim,
        adjacency_matrix=adjacency_matrix,
        intervention_targets_per_env=data_module.train_dataset.intervention_targets,
        lr=args.learning_rate,
        k_flows=args.k_flows,
        net_hidden_dim=args.hidden_dim,
        net_hidden_layers=args.hidden_layers,
        fix_mechanisms=True
    )

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/{args.dataset}/',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if args.gpu else 'cpu',
        callbacks=[checkpoint_callback],
        deterministic=True
    )

    # Train model
    trainer.fit(model, data_module)

    # Test model
    trainer.test(model, data_module)


if __name__ == "__main__":
    parser = ArgumentParser()

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='cmnist',
                        choices=['cmnist', 'rmnist', 'camelyon17', 'ball_agent'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)

    # Model arguments
    parser.add_argument('--nonlinear', action='store_true')
    parser.add_argument('--k_flows', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--hidden_layers', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-3)

    # Training arguments
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--gpu', action='store_true')

    # Ball Agent specific arguments
    parser.add_argument('--n_balls', type=int, default=4)

    args = parser.parse_args()

    train_cauca(args)