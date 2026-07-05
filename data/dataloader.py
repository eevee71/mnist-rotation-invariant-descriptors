from pathlib import Path
from torchvision import datasets

def load_MNIST():

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    dataset = datasets.MNIST(root=str(data_dir), train=True, download=True)
    data = dataset.data.float() / 255.0
    targets = dataset.targets

    print(f"data loaded")
    print(f"data: {data.shape}")
    print(f"targets {targets.shape}")
    return data, targets