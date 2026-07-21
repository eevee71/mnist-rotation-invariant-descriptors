from pathlib import Path
import torch
from torchvision import datasets
from torchvision.transforms.functional import rotate


def load_MNIST(train=True, transform=None):

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


def augment_with_rotations(data, targets, max_angle=180):

    angles = torch.FloatTensor(data.size(0)).uniform_(-max_angle, max_angle)

    rotated_data = torch.stack([
        rotate(img.unsqueeze(0), angle.item()).squeeze(0)
        for img, angle in zip(data, angles)
    ])

    return rotated_data, targets