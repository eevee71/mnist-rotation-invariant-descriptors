
"""
This tool performs linear interpolation between two image embeddings in the latent
space of Gaussian-Hermite moments and provides a slider to interactively visualize
the reconstructed intermediate states.
"""

import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from data.dataloader import load_MNIST
from src.moment_transforms import MomentTransform


def interpolate_and_visualize(img1, img2, steps=20, max_degree=8, spatial_dims=2):
    transform = MomentTransform(max_degree=max_degree, spatial_dimensions=spatial_dims)

    batch = torch.stack([img1.squeeze(0), img2.squeeze(0)])
    original_sums = batch.sum(dim=(-2, -1))

    density_map, x_centered, y_centered = transform.prepare_density_center(batch)
    invariants = transform.covariance_invariants(density_map, x_centered, y_centered)

    trace = invariants[:, 0]
    x_norm, y_norm = transform.normalization(x_centered, y_centered, trace)
    features = transform.gaussian_polynomial_moments(density_map, x_norm, y_norm)

    alphas = torch.linspace(0, 1, steps)

    interp_features = (1 - alphas[:, None]) * features[0] + alphas[:, None] * features[1]
    interp_sums = (1 - alphas) * original_sums[0] + alphas * original_sums[1]
    interp_trace = (1 - alphas) * trace[0] + alphas * trace[1]

    interp_xc = (1 - alphas[:, None, None]) * x_centered[0] + alphas[:, None, None] * x_centered[1]
    interp_yc = (1 - alphas[:, None, None]) * y_centered[0] + alphas[:, None, None] * y_centered[1]

    interp_x_norm, interp_y_norm = transform.normalization(interp_xc, interp_yc, interp_trace)

    B, H, W = interp_x_norm.shape
    f_x = transform.get_1d_basis(interp_x_norm)
    f_y = transform.get_1d_basis(interp_y_norm)

    reconstructed = torch.zeros((B, H, W))
    idx = 0
    for j1 in range(transform.max_degree + 1):
        for j2 in range(transform.max_degree + 1):
            if j1 + j2 <= transform.max_degree:
                moment = interp_features[..., idx].view(-1, 1, 1)
                reconstruction = moment * (f_x[j1] * f_y[j2])
                reconstructed += reconstruction
                idx += 1

    scale_sq = (interp_trace / transform.spatial_dimensions).view(-1, 1, 1)
    reconstructed = reconstructed / scale_sq
    reconstructed = torch.clamp(reconstructed * interp_sums.view(-1, 1, 1), 0, 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
    plt.subplots_adjust(bottom=0.25, wspace=0.3)

    ax1.imshow(img1.squeeze().numpy(), cmap='gray')
    ax1.set_title("Image A (Start)")
    ax1.axis('off')

    im = ax2.imshow(reconstructed[0].numpy(), cmap='gray', vmin=0, vmax=1)
    ax2.set_title("Reconstruction Link")
    ax2.axis('off')

    ax3.imshow(img2.squeeze().numpy(), cmap='gray')
    ax3.set_title("Image B (End)")
    ax3.axis('off')

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Alpha Step',
        valmin=0,
        valmax=steps - 1,
        valinit=0,
        valstep=1
    )

    def update():
        step_idx = int(slider.val)
        im.set_data(reconstructed[step_idx].numpy())
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    train_data, _ = load_MNIST()
    img_A = train_data[5]
    img_B = train_data[6]

    interpolate_and_visualize(img_A, img_B, steps=30, max_degree=12)