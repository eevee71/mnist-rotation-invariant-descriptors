
"""This pipeline (run_reconstruct_pipeline(single_image)) normalizes a single image for scale and translation,
extracts its Gaussian-Hermite moments and reconstructs it to verify the captured structural information."""

import torch
from matplotlib import pyplot as plt
from data.dataloader import load_MNIST, augment_with_rotations
from src.moment_transforms import MomentTransform


def reconstruct_images(transform_instance, features, x_norm, y_norm, original_sum, trace):
    B, H, W = x_norm.shape
    f_x = transform_instance.get_1d_basis(x_norm)
    f_y = transform_instance.get_1d_basis(y_norm)

    reconstruction = torch.zeros((B, H, W))

    idx = 0
    for j1 in range(transform_instance.max_degree + 1):
        for j2 in range(transform_instance.max_degree + 1):
            if j1 + j2 <= transform_instance.max_degree:
                moment = features[..., idx].view(-1, 1, 1)
                reconstruction += moment * (f_x[j1] * f_y[j2])
                idx += 1

    scale_sq = (trace / transform_instance.spatial_dimensions).view(-1, 1, 1)
    reconstruction = reconstruction / scale_sq
    reconstruction = torch.clamp(reconstruction * original_sum.view(-1, 1, 1), 0, 1)

    return reconstruction


def plot_debug_comparison(data, reconstructed_data):

    orig = data[0].squeeze(0)
    recon = reconstructed_data[0].squeeze(0)

    comparison = torch.cat([orig, recon], dim=-1)

    plt.imshow(comparison, cmap='gray')
    plt.title("Original (Left) vs Reconstructed (Right)")
    plt.axis('off')
    plt.show()


def run_reconstruct_pipeline(single_image):

    data = single_image.unsqueeze(0)

    density_map, x_centered, y_centered = transform.prepare_density_center(data)
    invariants = transform.covariance_invariants(density_map, x_centered, y_centered)
    x_norm, y_norm = transform.normalization(x_centered, y_centered, invariants[:, 0])
    final_representations = transform.gaussian_polynomial_moments(density_map, x_norm, y_norm)

    original_sums = data.sum(dim=(-2, -1))

    reconstructed_data = reconstruct_images(
        transform,
        features=final_representations,
        x_norm=x_norm,
        y_norm=y_norm,
        original_sum=original_sums,
        trace=invariants[:, 0]
    )

    print("Reconstructed images shape:", reconstructed_data.shape)
    plot_debug_comparison(data, reconstructed_data)

if __name__ == "__main__":
    transform = MomentTransform()
    all_data, _ = load_MNIST(transform=augment_with_rotations)
    img = all_data[2137].squeeze(0)
    run_reconstruct_pipeline(img)