from pathlib import Path
import torch
from torchvision import datasets
import torchvision.transforms.v2.functional as TVF


def load_MNIST(train=True, transform=None):
    """Downloads and loads the MNIST dataset as float Tensors normalized to [0, 1]."""

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    dataset = datasets.MNIST(root=str(data_dir), train=train, download=True)
    data = dataset.data.float() / 255.0
    targets = dataset.targets

    if transform is not None:
        data, targets = transform(data, targets)

    print(f"data loaded")
    print(f"data: {data.shape}")
    print(f"targets {targets.shape}")
    return data, targets


def load_data(full_dataset=True):
    """Wrapper function, loads both train and test sets
    combined into one Tensor if full_dataset=True."""

    if full_dataset:
        data_tr, targets_tr = load_MNIST(train=True)
        data_te, targets_te = load_MNIST(train=False)
        data = torch.cat([data_tr, data_te], dim=0)
        targets = torch.cat([targets_tr, targets_te], dim=0)
        return data, targets
    else:
        return load_MNIST(train=True)


def rotate_dataset(data, targets, max_angle=180, seed=None):
    """Rotates the dataset by random angles within [-max_angle, max_angle].
    Use `seed` to ensure the test set rotation is reproducible.
    """

    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        angles = torch.FloatTensor(data.size(0)).uniform_(
            -max_angle, max_angle, generator=generator
        )
    else:
        angles = torch.FloatTensor(data.size(0)).uniform_(-max_angle, max_angle)

    rotated_data = torch.stack([
        TVF.rotate(img.unsqueeze(0), angle.item()).squeeze(0)
        for img, angle in zip(data, angles)
    ])

    return rotated_data, targets