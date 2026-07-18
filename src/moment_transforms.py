import torch
from torchvision.transforms.functional import rotate


class MomentTransform:

    def __init__(self, max_degree=6, spatial_dimensions=2):

        self.max_degree = max_degree
        self.spatial_dimensions = spatial_dimensions


    def prepare_density_center(self, images):

        normalized = images / (images.sum(dim=(-2, -1), keepdim=True) + 1e-9)  # shape (B, H, W)

        rows, cols = images.shape[-2], images.shape[-1]
        y_grid, x_grid = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='ij')  # shapes (H, W)
        x_grid = x_grid.float()
        y_grid = y_grid.float()
        x_centered = x_grid - (normalized * x_grid).sum(dim=(-2, -1), keepdim=True)  # x - mean_x
        y_centered = y_grid - (normalized * y_grid).sum(dim=(-2, -1), keepdim=True)  # shape (B, H, W)

        return normalized, x_centered, y_centered


    def covariance_invariants(self, density_map, x, y):

        p_xx = (density_map * x * x).sum(dim=(-2, -1))  # shape (B,)
        p_yy = (density_map * y * y).sum(dim=(-2, -1))  # shape (B,)
        p_xy = (density_map * x * y).sum(dim=(-2, -1))  # shape (B,)

        p = torch.stack([torch.stack([p_xx, p_xy], dim=-1),  # shape (B, 2, 2)
                         torch.stack([p_xy, p_yy], dim=-1)], dim=-2)

        trace_p1 = torch.diagonal(p, dim1=-2, dim2=-1).sum(dim=-1)  # shape: (B,)
        trace_p2 = torch.diagonal(torch.matmul(p, p), dim1=-2, dim2=-1).sum(dim=-1)  # shape: (B,)

        return torch.stack([trace_p1, trace_p2], dim=-1)  # shape (B, 2)


    def normalization(self, x, y, trace):

        scale = torch.sqrt(trace / self.spatial_dimensions).view(-1, 1, 1)  # shape (B, 1, 1)
        return x / scale, y / scale  # shapes (B, H, W)


    def get_1d_basis(self, coords_normalized):

        basis_functions = []
        fact = 1
        pi_sqrt = torch.pi ** 0.5
        gaussian_window = torch.exp(-(coords_normalized ** 2) / 2)  # shape (B, H, W)

        H = [torch.ones_like(coords_normalized), 2 * coords_normalized]  # list of tensors with shape (B, H, W)
        for j in range(2, self.max_degree + 1):
            next = 2 * coords_normalized * H[j - 1] - 2 * (j - 1) * H[j - 2]
            H.append(next)

        for j in range(self.max_degree + 1):  # base
            norm_factor = 1 / ((2 ** j * fact * pi_sqrt) ** 0.5)
            f_j = norm_factor * H[j] * gaussian_window  # shape (B, H, W)
            basis_functions.append(f_j)
            fact *= (j + 1)

        return torch.stack(basis_functions, dim=0)  # shape (max_degree+1, B, H, W)


    def gaussian_polynomial_moments(self, density_map, x_norm, y_norm):

        f_x = self.get_1d_basis(x_norm)  # shape (max_degree+1, B, H, W)
        f_y = self.get_1d_basis(y_norm)  # shape (max_degree+1, B, H, W)

        features = []
        for j1 in range(self.max_degree + 1):
            for j2 in range(self.max_degree + 1):

                if j1 + j2 <= self.max_degree:
                    f_product = f_x[j1] * f_y[j2]  # shape (B, H, W)

                    u_j = (density_map * f_product).sum(dim=(-2, -1))  # expected value, shape (B,)
                    features.append(u_j)

        return torch.stack(features, dim=-1)  # shape (B, K)


    def gaussian_polynomial_moment_matrix(self, density_map, x_norm, y_norm):
        """Same as gaussian_polynomial_moments but returns a square matrix"""

        f_x = self.get_1d_basis(x_norm)  # shape (max_degree+1, B, H, W)
        f_y = self.get_1d_basis(y_norm)

        d, B = self.max_degree, density_map.shape[0]
        U = torch.zeros(B, d + 1, d + 1)

        for j1 in range(d + 1):
            for j2 in range(d + 1):

                if j1 + j2 <= d:  # rest is 0 -> degree <= d
                    U[:, j1, j2] = (density_map * f_x[j1] * f_y[j2]).sum(dim=(-2, -1))

        return U  # (B, d+1, d+1)


    def hermite_to_monomial_matrix(self):
        """T[n, m] = N_n * (coeff of x^m in H_n), same norm as get_1d_basis."""

        d = self.max_degree
        h = torch.zeros(d + 1, d + 1)  # h[n, m] = coeff x^m in H_n
        h[0, 0] = 1.0

        if d >= 1:
            h[1, 1] = 2.0

        for n in range(2, d + 1):  # H_n = 2x H_{n-1} - 2(n-1) H_{n-2}
            h[n, 1:] += 2.0 * h[n - 1, :-1]  # *2x
            h[n, :] -= 2.0 * (n - 1) * h[n - 2, :]

        fact = 1.0
        N = torch.zeros(d + 1)
        pi_sqrt = torch.pi ** 0.5

        for n in range(d + 1):
            N[n] = 1.0 / ((2 ** n * fact * pi_sqrt) ** 0.5)
            fact *= (n + 1)

        return N.view(-1, 1) * h  # (d+1, d+1)


    def homogeneous_coefficients(self, U, T):
        """Returns c_{ijk} where (i+j+k=d) with shape (B, K) + list of indices (i, j, k)"""

        A = torch.einsum('im,bij,jn->bmn', T, U, T)  # change basis A[b,i,j] = coeff for x^i y^j
        d = self.max_degree

        coeffs, index = [], []

        for i in range(d + 1):
            for j in range(d + 1):

                if i + j <= d:
                    coeffs.append(A[:, i, j])
                    index.append((i, j, d - i - j))
        return torch.stack(coeffs, dim=-1), index  # (B, K)


    def reconstruct_images(self, features, x_norm, y_norm, original_sum, trace):
        """
        Reconstructs the original images from their Gaussian-Hermite moments,
        accounting for the normalized coordinate scale.
        """
        B, H, W = x_norm.shape

        f_x = self.get_1d_basis(x_norm)  # shape (max_degree+1, B, H, W)
        f_y = self.get_1d_basis(y_norm)  # shape (max_degree+1, B, H, W)

        reconstruction = torch.zeros((B, H, W))  # shape: (B, H, W)

        idx = 0
        for j1 in range(self.max_degree + 1):
            for j2 in range(self.max_degree + 1):

                if j1 + j2 <= self.max_degree:
                    moment = features[..., idx].view(-1, 1, 1)  # shape (B, 1, 1)
                    reconstruction += moment * (f_x[j1] * f_y[j2])  # shape (B, H, W)
                    idx += 1

        scale_sq = (trace / self.spatial_dimensions).view(-1, 1, 1)  # shape (B, 1, 1)
        reconstruction = reconstruction / scale_sq  # shape (B, H, W)
        reconstruction = torch.clamp(reconstruction * original_sum.view(-1, 1, 1), 0, 1)
        return rotate(reconstruction, angle=-90)  # This is a little stupid but otherwise the digits are sideways


# image rotation -> without using invariants
def align_coordinates(density_map, x, y):

    # This probably should be combined with covariance_invariants()
    p_xx = (density_map * x * x).sum(dim=(-2, -1))  # shape (B,)
    p_yy = (density_map * y * y).sum(dim=(-2, -1))  # shape (B,)
    p_xy = (density_map * x * y).sum(dim=(-2, -1))  # shape (B,)

    # Theta = angle of the principal axis (ellipsoid approximation)
    theta = 0.5 * torch.atan2(2 * p_xy, p_xx - p_yy)  # shape (B,)
    theta = theta.view(-1, 1, 1)  # shape (B, 1, 1)

    cos_t = torch.cos(theta)  # shape (B, 1, 1)
    sin_t = torch.sin(theta)  # shape (B, 1, 1)

    x_rotated = x * cos_t + y * sin_t  # shape (B, H, W)
    y_rotated = -x * sin_t + y * cos_t  # shape (B, H, W)

    # 180 degree ambiguity fix (which side holds more mass)
    skew_x = (density_map * (x_rotated ** 3)).sum(dim=(-2, -1), keepdim=True)  # shape (B, 1, 1)

    flip_mask = torch.sign(skew_x)  # shape (B, 1, 1)
    flip_mask[flip_mask == 0] = 1  # shape (B, 1, 1)

    x_rotated = x_rotated * flip_mask  # shape (B, H, W)
    y_rotated = y_rotated * flip_mask  # shape (B, H, W)

    return x_rotated, y_rotated