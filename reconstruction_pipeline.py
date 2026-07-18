import torch
from matplotlib import pyplot as plt
from data.dataloader import load_MNIST
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import affine
from src import invariants as inv

if __name__ == '__main__':

    data, targets = load_MNIST()
    data, targets = data[4:6], targets[4:6]

    data[1] = affine(
        data[0].unsqueeze(0),
        angle=45,
        translate=[5, 2],
        scale=0.8,
        shear=0.0,  # skewing
        interpolation=InterpolationMode.BILINEAR
    ).squeeze(0)
    targets[1] = targets[0]

    # Main pipeline
    density_map, x_centered, y_centered = inv.prepare_density_center(data)
    x_aligned, y_aligned = inv.align_coordinates(density_map, x_centered, y_centered)
    invariants = inv.covariance_invariants(density_map, x_aligned, y_aligned)
    x_norm_aligned, y_norm_aligned = inv.normalization(x_aligned, y_aligned, invariants[:, 0], inv.SPATIAL_DIMENSIONS)
    final_representations = inv.gaussian_polynomial_moments(density_map, x_norm_aligned, y_norm_aligned,
                                                        inv.MAX_POLYNOMIAL_DEGREE)

    # For reconstruction
    original_sums = data.sum(dim=(-2, -1))
    shared_trace = invariants[:, 0].mean(dim=0, keepdim=True).expand_as(invariants[:, 0])
    shared_sum = original_sums.mean(dim=0, keepdim=True).expand_as(original_sums)
    shared_x_centered = x_centered.mean(dim=0, keepdim=True).expand_as(x_centered)
    shared_y_centered = y_centered.mean(dim=0, keepdim=True).expand_as(y_centered)

    x_norm_straight, y_norm_straight = inv.normalization(shared_x_centered, shared_y_centered, shared_trace,
                                                     inv.SPATIAL_DIMENSIONS)

    # print(final_representations.shape)
    # print(final_representations[0])

    reconstructed_data = inv.reconstruct_images(
        features=final_representations,
        x_norm=x_norm_straight,
        y_norm=y_norm_straight,
        max_degree=inv.MAX_POLYNOMIAL_DEGREE,
        original_sum=shared_sum,
        trace=shared_trace,
        d=inv.SPATIAL_DIMENSIONS
    )

    print("Reconstructed images shape:", reconstructed_data.shape)

    orig_0 = data[0].squeeze(0)
    recon_0 = reconstructed_data[0].squeeze(0)
    comparison_0 = torch.cat([orig_0, recon_0], dim=-1)  # Stitch side-by-side

    orig_1 = data[1].squeeze(0)
    recon_1 = reconstructed_data[1].squeeze(0)
    comparison_1 = torch.cat([orig_1, recon_1], dim=-1)  # Stitch side-by-side

    final_comparison = torch.cat([comparison_0, comparison_1], dim=0)

    plt.imshow(final_comparison, cmap='gray')
    plt.title("Top: Upright (Original / Reconstructed)\nBottom: Transformed (Original / Reconstructed)")
    plt.axis('off')
    plt.show()