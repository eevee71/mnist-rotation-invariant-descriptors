from pathlib import Path
import torch
import torchvision.transforms.v2.functional as TVF
import numpy as np


def _load_amat_pair(train_filename, test_filename):
    """Helper to parse a pair of Larochelle .amat files into combined Tensors."""

    data_dir = Path(__file__).resolve().parents[1] / "data"

    train_val = np.loadtxt(data_dir / train_filename)
    test = np.loadtxt(data_dir / test_filename)

    full_data = np.vstack((train_val, test))
    images = full_data[:, :-1].reshape(-1, 28, 28)
    targets = full_data[:, -1]

    return torch.tensor(images, dtype=torch.float32), torch.tensor(targets, dtype=torch.long)


def load_data(threshold=0.05):
    """Loads the unrotated Larochelle MNIST-12k dataset."""

    print("Loading official MNIST-12k dataset...")
    data, targets = _load_amat_pair(
        "mnist_train.amat",
        "mnist_test.amat"
    )
    print(f"data: {data.shape}, targets: {targets.shape}")
    return data, targets


def load_mnist_rot(threshold=0.05):
    """Loads the official Larochelle MNIST-Rot dataset."""

    print("Loading MNIST-Rot dataset...")
    data_rot, targets_rot = _load_amat_pair(
        "mnist_all_rotation_normalized_float_train_valid.amat",
        "mnist_all_rotation_normalized_float_test.amat"
    )
    print(f"data_rot: {data_rot.shape}, targets_rot: {targets_rot.shape}")
    return data_rot, targets_rot


def rotate_dataset(data, targets, max_angle=180, seed=None):
    """Rotates the dataset by random angles within [-max_angle, max_angle].
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