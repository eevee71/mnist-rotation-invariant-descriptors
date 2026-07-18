from src.invariants import prepare_density_center, align_coordinates, covariance_invariants, normalization, gaussian_polynomial_moments, reconstruct_images

import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def visualize_base_representation(image, label):

    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label.item()}")
    plt.axis('off')
    plt.show()


def interpolate_and_visualize(img1, img2, steps=20, max_degree=8, spatial_dims=2):
    """
    Takes two images, computes their embeddings, linearly interpolates 
    between them in the latent space, and plots the reconstruction with a slider.
    Displays Image A on the left, interpolation in the middle, and Image B on the right.
    """
    
    batch = torch.stack([img1, img2]) # Shape: (2, H, W)
    original_sums = batch.sum(dim=(-2, -1)) # Shape: (2,)
    
    density_map, x_centered, y_centered = prepare_density_center(batch)
    x_aligned, y_aligned = align_coordinates(density_map, x_centered, y_centered)
    invariants = covariance_invariants(density_map, x_aligned, y_aligned)
    
    trace = invariants[:, 0] # Shape: (2,)
    
    x_norm_aligned, y_norm_aligned = normalization(x_aligned, y_aligned, trace, spatial_dims)
    features = gaussian_polynomial_moments(density_map, x_norm_aligned, y_norm_aligned, max_degree)
    
    alphas = torch.linspace(0, 1, steps)
    
    # Linearly interpolate features, mass, scale, and centered grids
    interp_features = (1 - alphas[:, None]) * features[0] + alphas[:, None] * features[1]
    interp_sums = (1 - alphas) * original_sums[0] + alphas * original_sums[1]
    interp_trace = (1 - alphas) * trace[0] + alphas * trace[1]
    
    interp_xc = (1 - alphas[:, None, None]) * x_centered[0] + alphas[:, None, None] * x_centered[1]
    interp_yc = (1 - alphas[:, None, None]) * y_centered[0] + alphas[:, None, None] * y_centered[1]
    
    interp_x_norm, interp_y_norm = normalization(interp_xc, interp_yc, interp_trace, spatial_dims)
    
    reconstructed = reconstruct_images(
        features=interp_features,
        x_norm=interp_x_norm,
        y_norm=interp_y_norm,
        max_degree=max_degree,
        original_sum=interp_sums,
        trace=interp_trace,
        d=spatial_dims
    )
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.5))
    plt.subplots_adjust(bottom=0.25, wspace=0.3)
    
    # Left: Image A (Original)
    ax1.imshow(img1.numpy(), cmap='gray')
    ax1.set_title("Image A")
    ax1.axis('off')
    
    # Middle: Interpolation
    im = ax2.imshow(reconstructed[0].numpy(), cmap='gray')
    ax2.set_title("Reconstructed")
    ax2.axis('off')
    
    # Right: Image B (Original)
    ax3.imshow(img2.numpy(), cmap='gray')
    ax3.set_title("Image B")
    ax3.axis('off')
    
    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(
        ax=ax_slider,
        label='Interpolation Step',
        valmin=0,
        valmax=steps - 1,
        valinit=0,
        valstep=1
    )
    
    # Update function for the slider
    def update(val):
        idx = int(slider.val)
        im.set_data(reconstructed[idx].numpy())
        im.set_clim(vmin=0, vmax=reconstructed[idx].max().item())
        fig.canvas.draw_idle()
        
    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    from data.dataloader import load_MNIST
    train_data, train_targets = load_MNIST()
    img_A = train_data[0] 
    img_B = train_data[1] 
    interpolate_and_visualize(img_A, img_B, steps=30, max_degree=15)