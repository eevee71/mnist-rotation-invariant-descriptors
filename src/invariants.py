from data.dataloader import load_MNIST
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import rotate

MAX_POLYNOMIAL_DEGREE = 8
SPATIAL_DIMENSIONS = 2

def prepare_density_center(images):

    normalized = images / (images.sum(dim=(-2, -1), keepdim=True) + 1e-9) # shape (B, H, W)

    rows, cols = images.shape[-2], images.shape[-1]
    y_grid, x_grid = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='ij') # shapes (H, W)
    x_grid = x_grid.float()
    y_grid = y_grid.float()
    x_centered = x_grid - (normalized * x_grid).sum(dim=(-2, -1), keepdim=True)   # x - mean_x 
    y_centered = y_grid - (normalized * y_grid).sum(dim=(-2, -1), keepdim=True)   # shape (B, H, W)

    return normalized, x_centered, y_centered


def covariance_invariants(density_map, x, y):

    p_xx = (density_map * x * x).sum(dim=(-2, -1)) # shape (B,)
    p_yy = (density_map * y * y).sum(dim=(-2, -1)) # shape (B,)
    p_xy = (density_map * x * y).sum(dim=(-2, -1)) # shape (B,)

    p = torch.stack([torch.stack([p_xx, p_xy], dim=-1), # shape (B, 2, 2)
                     torch.stack([p_xy, p_yy], dim=-1)], dim=-2)

    trace_p1 = torch.diagonal(p, dim1=-2, dim2=-1).sum(dim=-1)                  # shape: (B,)
    trace_p2 = torch.diagonal(torch.matmul(p, p), dim1=-2, dim2=-1).sum(dim=-1) # shape: (B,)

    return torch.stack([trace_p1, trace_p2], dim=-1) # shape (B, 2)


def normalization(x, y, trace, d):

    scale = torch.sqrt(trace / d).view(-1, 1, 1) # shape (B, 1, 1)
    return x / scale, y / scale # shapes (B, H, W)


def get_1d_basis(coords_normalized, max_degree):

    basis_functions = []
    fact = 1
    pi_sqrt = torch.pi ** 0.5
    gaussian_window = torch.exp(-(coords_normalized ** 2) / 2) # shape (B, H, W)

    H = [torch.ones_like(coords_normalized), 2 * coords_normalized] # list of tensors with shape (B, H, W)
    for j in range(2, max_degree + 1):
        next = 2 * coords_normalized * H[j - 1] - 2 * (j - 1) * H[j - 2]
        H.append(next)

    for j in range(max_degree + 1): # base
        norm_factor = 1 / ((2**j * fact * pi_sqrt) ** 0.5)
        f_j = norm_factor * H[j] * gaussian_window # shape (B, H, W)
        basis_functions.append(f_j)
        fact *= (j + 1)

    return torch.stack(basis_functions, dim=0) # shape (max_degree+1, B, H, W)


def gaussian_polynomial_moments(density_map, x_norm, y_norm, max_degree):

    f_x = get_1d_basis(x_norm, max_degree) # shape (max_degree+1, B, H, W)
    f_y = get_1d_basis(y_norm, max_degree) # shape (max_degree+1, B, H, W)

    features = []
    for j1 in range(max_degree + 1):
        for j2 in range(max_degree + 1):

            if j1 + j2 <= max_degree:
                f_product = f_x[j1] * f_y[j2] # shape (B, H, W)

                u_j = (density_map * f_product).sum(dim=(-2, -1)) # expected value, shape (B,)
                features.append(u_j)

    return torch.stack(features, dim=-1) # shape (B, K)


def align_coordinates(density_map, x, y):
    
    # This probably should be combined with covariance_invariants()
    p_xx = (density_map * x * x).sum(dim=(-2, -1)) # shape (B,)
    p_yy = (density_map * y * y).sum(dim=(-2, -1)) # shape (B,)
    p_xy = (density_map * x * y).sum(dim=(-2, -1)) # shape (B,)
    
    # Theta = angle of the principal axis (ellipsoid approximation)
    theta = 0.5 * torch.atan2(2 * p_xy, p_xx - p_yy) # shape (B,)
    theta = theta.view(-1, 1, 1)                     # shape (B, 1, 1)
    
    cos_t = torch.cos(theta) # shape (B, 1, 1)
    sin_t = torch.sin(theta) # shape (B, 1, 1)
    
    x_rotated = x * cos_t + y * sin_t  # shape (B, H, W)
    y_rotated = -x * sin_t + y * cos_t # shape (B, H, W)
    
    # 180 degree ambiguity fix (which side holds more mass)
    skew_x = (density_map * (x_rotated ** 3)).sum(dim=(-2, -1), keepdim=True) # shape (B, 1, 1)
    
    flip_mask = torch.sign(skew_x) # shape (B, 1, 1)
    flip_mask[flip_mask == 0] = 1  # shape (B, 1, 1)

    x_rotated = x_rotated * flip_mask # shape (B, H, W)
    y_rotated = y_rotated * flip_mask # shape (B, H, W)
    
    return x_rotated, y_rotated


def reconstruct_images(features, x_norm, y_norm, max_degree, original_sum, trace, d):
    """
    Reconstructs the original images from their Gaussian-Hermite moments,
    accounting for the normalized coordinate scale.
    """
    B, H, W = x_norm.shape
    
    f_x = get_1d_basis(x_norm, max_degree) # shape (max_degree+1, B, H, W)
    f_y = get_1d_basis(y_norm, max_degree) # shape (max_degree+1, B, H, W)
    
    reconstruction = torch.zeros((B, H, W)) # shape: (B, H, W)
    
    idx = 0
    for j1 in range(max_degree + 1):
        for j2 in range(max_degree + 1):
            
            if j1 + j2 <= max_degree:
                moment = features[..., idx].view(-1, 1, 1) # shape (B, 1, 1)
                reconstruction += moment * (f_x[j1] * f_y[j2]) # shape (B, H, W)
                idx += 1
                
    scale_sq = (trace / d).view(-1, 1, 1) # shape (B, 1, 1)
    reconstruction = reconstruction / scale_sq # shape (B, H, W)
    reconstruction = torch.clamp(reconstruction * original_sum.view(-1, 1, 1), 0, 1)
    return rotate(reconstruction, angle=-90) # This is a little stupid but otherwise the digits are sideways


if __name__ == "__main__":

    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.functional import affine

    data, targets = load_MNIST()
    data, targets = data[4:6], targets[4:6]

    data[1] = affine(
        data[0].unsqueeze(0), 
        angle=45, 
        translate=[5, 2],
        scale=0.8,
        shear=0.0,         # skewing
        interpolation=InterpolationMode.BILINEAR
    ).squeeze(0)
    targets[1] = targets[0]

    # Main pipeline
    density_map, x_centered, y_centered = prepare_density_center(data)
    x_aligned, y_aligned = align_coordinates(density_map, x_centered, y_centered)
    invariants = covariance_invariants(density_map, x_aligned, y_aligned)
    x_norm_aligned, y_norm_aligned = normalization(x_aligned, y_aligned, invariants[:, 0], SPATIAL_DIMENSIONS)
    final_representations = gaussian_polynomial_moments(density_map, x_norm_aligned, y_norm_aligned, MAX_POLYNOMIAL_DEGREE)

    # For reconstruction
    original_sums = data.sum(dim=(-2, -1))
    shared_trace = invariants[:, 0].mean(dim=0, keepdim=True).expand_as(invariants[:, 0])
    shared_sum = original_sums.mean(dim=0, keepdim=True).expand_as(original_sums)
    shared_x_centered = x_centered.mean(dim=0, keepdim=True).expand_as(x_centered)
    shared_y_centered = y_centered.mean(dim=0, keepdim=True).expand_as(y_centered)

    x_norm_straight, y_norm_straight = normalization(shared_x_centered, shared_y_centered, shared_trace, SPATIAL_DIMENSIONS)

    # print(final_representations.shape)
    # print(final_representations[0])

    reconstructed_data = reconstruct_images(
        features=final_representations, 
        x_norm=x_norm_straight, 
        y_norm=y_norm_straight, 
        max_degree=MAX_POLYNOMIAL_DEGREE, 
        original_sum=shared_sum,
        trace=shared_trace,
        d=SPATIAL_DIMENSIONS
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