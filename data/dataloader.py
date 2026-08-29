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

    split_name = "train" if train else "test"
    print(f"data loaded ({split_name})")
    print(f"data: {data.shape}")
    print(f"targets {targets.shape}")
    return data, targets


def load_data():
    """Loads MNIST's official split.

    Returns a 4-tuple (train_data, train_targets, test_data, test_targets).
    The two splits are kept separate so that the test set is never seen
    during training.
    """

    data_tr, targets_tr = load_MNIST(train=True)
    data_te, targets_te = load_MNIST(train=False)
    return data_tr, targets_tr, data_te, targets_te


def rotate_dataset(data, targets, max_angle=180, seed=None,
                   interpolation=TVF.InterpolationMode.BILINEAR):
    """Rotates each image by an independent random angle drawn uniformly
    from [-max_angle, max_angle).

    Bilinear interpolation is used by default. The torchvision default is
    NEAREST, which aliases the rotated digits and introduces a train/test
    domain shift unrelated to rotation itself.

    Use `seed` to make the test set rotation reproducible.
    """

    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)

    angles = torch.empty(data.size(0)).uniform_(-max_angle, max_angle,
                                                generator=generator)

    print(f"rotating {data.size(0)} images | "
          f"angle range sampled: [{angles.min():.1f}, {angles.max():.1f}] deg")

    rotated_data = torch.stack([
        TVF.rotate(img.unsqueeze(0), angle.item(),
                   interpolation=interpolation).squeeze(0)
        for img, angle in zip(data, angles)
    ])

    return rotated_data, targets