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
        """Extracts complex Hermite coefficients mode-by-mode to keep memory footprint minimal."""

        dm, xc, yc = self.prepare_density_center(images)
        trace = self.covariance_invariants(dm, xc, yc)[:, 0]
        xn, yn = self.normalization(xc, yc, trace)

        z = torch.complex(xn, yn)
        z_bar = torch.conj(z)
        z_abs2 = xn**2 + yn**2
       # gaussian_window = torch.exp(-0.5 * z_abs2)
        r = torch.sqrt(z_abs2 + 1e-9)
        gaussian_window = torch.exp(-(r ** 1.5))

        coeffs_list = []
        index = []
        d = self.max_degree

        for n in range(d + 1):
            for m in range(n + 1):
                if n + m <= d:
                    H_nm = torch.zeros_like(z)
                    for k in range(min(n, m) + 1):
                        coeff = (
                            ((-1) ** k)
                            * math.factorial(n)
                            * math.factorial(m)
                            / (math.factorial(k) * math.factorial(n - k) * math.factorial(m - k))
                        )
                        z_pow = z ** (n - k) if (n - k) > 0 else torch.ones_like(z)
                        z_bar_pow = z_bar ** (m - k) if (m - k) > 0 else torch.ones_like(z)
                        H_nm = H_nm + coeff * z_pow * z_bar_pow

                    norm = 1.0 / math.sqrt(
                        math.pi * (2 ** (n + m)) * math.factorial(n) * math.factorial(m)
                    )
                    psi_nm = norm * H_nm * gaussian_window

                    c_nm = (dm * torch.conj(psi_nm)).sum(dim=(-2, -1))
                    coeffs_list.append(c_nm)
                    index.append((n, m))

        coeffs = torch.stack(coeffs_list, dim=-1)  # shape (B, K)
        return coeffs, index