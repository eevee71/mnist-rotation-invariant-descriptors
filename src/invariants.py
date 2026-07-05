from data.dataloader import load_MNIST
import torch

MAX_POLYNOMIAL_DEGREE = 8
SPATIAL_DIMENSIONS = 2

def prepare_density_center(images):

    normalized = images / (images.sum(dim=(-2, -1), keepdim=True) + 1e-9)

    rows, cols = images.shape[-2], images.shape[-1]
    y_grid, x_grid = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='ij')
    x_grid = x_grid.float()
    y_grid = y_grid.float()
    x_centered = x_grid - (normalized * x_grid).sum(dim=(-2, -1), keepdim=True)   # x - mean_x
    y_centered = y_grid - (normalized * y_grid).sum(dim=(-2, -1), keepdim=True)

    return normalized, x_centered, y_centered


def covariance_invariants(density_map, x, y):

    p_xx = (density_map * x * x).sum(dim=(-2, -1))
    p_yy = (density_map * y * y).sum(dim=(-2, -1))
    p_xy = (density_map * x * y).sum(dim=(-2, -1))

    p = torch.stack([torch.stack([p_xx, p_xy], dim=-1), # shape [x, 2, 2]
                     torch.stack([p_xy, p_yy], dim=-1)], dim=-2)

    trace_p1 = torch.diagonal(p, dim1=-2, dim2=-1).sum(dim=-1)
    trace_p2 = torch.diagonal(torch.matmul(p, p), dim1=-2, dim2=-1).sum(dim=-1)

    return torch.stack([trace_p1, trace_p2], dim=-1)


def normalization(x, y, trace, d):

    scale = torch.sqrt(trace / d).view(-1, 1, 1)
    return x / scale, y / scale


def get_1d_basis(coords_normalized, max_degree):

    basis_functions = []
    fact = 1
    pi_sqrt = torch.pi ** 0.5
    gaussian_window = torch.exp(-(coords_normalized ** 2) / 2)

    H = [torch.ones_like(coords_normalized), 2 * coords_normalized] # Hermite
    for j in range(2, max_degree + 1):
        next = 2 * coords_normalized * H[j - 1] - 2 * (j - 1) * H[j - 2]
        H.append(next)

    for j in range(max_degree + 1): # base
        norm_factor = 1 / ((2**j * fact * pi_sqrt) ** 0.5)
        f_j = norm_factor * H[j] * gaussian_window
        basis_functions.append(f_j)
        fact *= (j + 1)

    return torch.stack(basis_functions, dim=0)


def gaussian_polynomial_moments(density_map, x_norm, y_norm, max_degree):

    f_x = get_1d_basis(x_norm, max_degree)
    f_y = get_1d_basis(y_norm, max_degree)

    features = []
    for j1 in range(max_degree + 1):
        for j2 in range(max_degree + 1):

            if j1 + j2 <= max_degree:
                f_product = f_x[j1] * f_y[j2]

                u_j = (density_map * f_product).sum(dim=(-2, -1)) # expected value
                features.append(u_j)

    return torch.stack(features, dim=-1)


data, targets = load_MNIST()
density_map, x_centered, y_centered = prepare_density_center(data)
invariants = covariance_invariants(density_map, x_centered, y_centered)
x_normalized, y_normalized = normalization(x_centered, y_centered, invariants[:, 0], SPATIAL_DIMENSIONS)

final_representations = gaussian_polynomial_moments(density_map, x_normalized, y_normalized, MAX_POLYNOMIAL_DEGREE)
print(final_representations.shape)
print(final_representations[0])

