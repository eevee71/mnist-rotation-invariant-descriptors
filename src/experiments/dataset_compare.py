import torch
import torchvision.transforms.v2.functional as TVF
import matplotlib.pyplot as plt
from data.dataloader import load_data, load_mnist_rot


def check_compatibility():
    d_12k, y_12k = load_data()
    d_rot, y_rot = load_mnist_rot()

    print(f"12k shape: {d_12k.shape} | Rot shape: {d_rot.shape}")
    print(f"12k mass:  {d_12k.sum((1, 2)).mean():.2f} | Rot mass:  {d_rot.sum((1, 2)).mean():.2f}")
    print(f"12k mean:  {d_12k.mean():.4f} | Rot mean:  {d_rot.mean():.4f}")
    print(f"12k zeros: {(d_12k == 0).float().mean() * 100:.2f}% | Rot zeros: {(d_rot == 0).float().mean() * 100:.2f}%")

    sub = d_12k[:2000]
    sub_blur = torch.stack([
        TVF.rotate(img.unsqueeze(0), 0.0, interpolation=TVF.InterpolationMode.BILINEAR).squeeze(0)
        for img in sub
    ])
    print(f"Identity resampling diff: {(sub - sub_blur).abs().mean().item():.6f}")

    fig, axes = plt.subplots(3, 2, figsize=(5, 7))
    for i in range(3):
        axes[i, 0].imshow(d_12k[i].numpy(), cmap='gray')
        axes[i, 0].set_title(f"12k: {y_12k[i].item()}")
        axes[i, 1].imshow(d_rot[i].numpy(), cmap='gray')
        axes[i, 1].set_title(f"Rot: {y_rot[i].item()}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    check_compatibility()