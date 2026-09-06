import math
import torch
class MomentTransform:
    def __init__(self, max_degree=6, spatial_dimensions=2):
        self.max_degree = max_degree
        self.spatial_dimensions = spatial_dimensions
    def prepare_density_center(self, images):
        normalized = images / (images.sum(dim=(-2, -1), keepdim=True) + 1e-9)
        rows, cols = images.shape[-2], images.shape[-1]
        y_grid, x_grid = torch.meshgrid(
            torch.arange(rows, device=images.device),
            torch.arange(cols, device=images.device),
            indexing='ij',
        )
        x_grid = x_grid.float()
        y_grid = y_grid.float()
        x_centered = x_grid - (normalized * x_grid).sum(dim=(-2, -1), keepdim=True)
        y_centered = y_grid - (normalized * y_grid).sum(dim=(-2, -1), keepdim=True)
        return normalized, x_centered, y_centered
    def covariance_invariants(self, density_map, x, y):
        p_xx = (density_map * x * x).sum(dim=(-2, -1))
        p_yy = (density_map * y * y).sum(dim=(-2, -1))
        p_xy = (density_map * x * y).sum(dim=(-2, -1))
        p = torch.stack(
            [
                torch.stack([p_xx, p_xy], dim=-1),
                torch.stack([p_xy, p_yy], dim=-1),
            ],
            dim=-2,
        )
        trace_p1 = torch.diagonal(p, dim1=-2, dim2=-1).sum(dim=-1)
        trace_p2 = torch.diagonal(torch.matmul(p, p), dim1=-2, dim2=-1).sum(dim=-1)
        return torch.stack([trace_p1, trace_p2], dim=-1)
    def normalization(self, x, y, trace):
        scale = torch.sqrt(trace / 3).view(-1, 1, 1)
        return x / scale, y / scale
    def complex_coefficients(self, images):
        """Extracts complex Hermite coefficients row-by-row via the three-term recurrences
            H_{0,n} = conj(z)**n,   H_{m,n} = z * H_{m-1,n} - n * H_{m-1,n-1}
        """
        dm, xc, yc = self.prepare_density_center(images)
        trace = self.covariance_invariants(dm, xc, yc)[:, 0]
        xn, yn = self.normalization(xc, yc, trace)
        z = torch.complex(xn, yn)
        z_bar = torch.conj(z)
        z_abs2 = xn**2 + yn**2
        gaussian_window = torch.exp(-0.5 * z_abs2)
       # r = torch.sqrt(z_abs2 + 1e-9)
       # gaussian_window = torch.exp(-(r ** 1.5))
        coeffs_list = []
        index = []
        d = self.max_degree
        prev_row = None
        for m in range(d + 1):
            if m == 0:
                row = [torch.ones_like(z)]
                for _ in range(d):
                    row.append(z_bar * row[-1])
            else:
                row = []
                for n in range(d - m + 1):
                    H_mn = z * prev_row[n]
                    if n > 0:
                        H_mn = H_mn - n * prev_row[n - 1]
                    row.append(H_mn)
            for n in range(m + 1):
                if m + n <= d:
                    norm = 1.0 / math.sqrt(
                        math.pi * math.factorial(m) * math.factorial(n)
                    )
                    psi_mn = norm * row[n] * gaussian_window
                    c_mn = (dm * torch.conj(psi_mn)).sum(dim=(-2, -1))
                    coeffs_list.append(c_mn)
                    index.append((m, n))
            prev_row = row
        coeffs = torch.stack(coeffs_list, dim=-1)  # shape (B, K)
        return coeffs, index