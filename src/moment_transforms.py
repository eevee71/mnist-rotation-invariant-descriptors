import math
import torch


def gaussian_window(r):
    """Standard Gaussian radial window, exp(-r**2 / 2).

    This is the window the Hermite normalisation below assumes, so it is the
    principled default. Alternatives (see ``window_sweep.WINDOWS``) still yield
    usable features because any per-mode constant is absorbed downstream by the
    StandardScaler, but they are not orthonormal.
    """
    return torch.exp(-(r ** 2) / 2.0)


class MomentTransform:
    """Image -> complex Hermite coefficients (Gaussian-weighted).

    Each image is treated as a density map, recentred on its centroid and
    rescaled to unit second moment, then projected onto a complex Hermite
    basis. Under a rotation the resulting coefficients c_{n,m} only pick up a
    phase, which ``SO2Invariants`` turns into rotation-invariant features.

    ``window`` is a callable ``w(r) -> weights`` over the radius r = ||x||;
    swap it to probe other radial profiles.
    """

    def __init__(self, max_degree=6, spatial_dimensions=2, window=gaussian_window):
        self.max_degree = max_degree
        self.spatial_dimensions = spatial_dimensions
        self.window = window

    def prepare_density_center(self, images):
        """Normalise to a probability density and recentre on the centroid."""
        density = images / (images.sum(dim=(-2, -1), keepdim=True) + 1e-9)

        rows, cols = images.shape[-2], images.shape[-1]
        y_grid, x_grid = torch.meshgrid(
            torch.arange(rows, device=images.device),
            torch.arange(cols, device=images.device),
            indexing='ij',
        )
        x_grid = x_grid.float()
        y_grid = y_grid.float()

        x = x_grid - (density * x_grid).sum(dim=(-2, -1), keepdim=True)
        y = y_grid - (density * y_grid).sum(dim=(-2, -1), keepdim=True)
        return density, x, y

    def second_moment_trace(self, density, x, y):
        """Trace of the covariance matrix (the mean squared radius)."""
        p_xx = (density * x * x).sum(dim=(-2, -1))
        p_yy = (density * y * y).sum(dim=(-2, -1))
        return p_xx + p_yy

    def normalize_scale(self, x, y, trace):
        """Rescale coordinates to unit second moment (scale invariance)."""
        scale = torch.sqrt(trace / self.spatial_dimensions).view(-1, 1, 1)
        return x / scale, y / scale

    def complex_coefficients(self, images):
        """Project each image onto the complex Hermite basis, mode by mode.

        Returns ``(coeffs, index)`` where ``coeffs`` has shape ``(B, K)`` and
        ``index`` lists the ``(n, m)`` degree pair for each column.
        """
        density, xc, yc = self.prepare_density_center(images)
        trace = self.second_moment_trace(density, xc, yc)
        xn, yn = self.normalize_scale(xc, yc, trace)

        z = torch.complex(xn, yn)
        z_bar = torch.conj(z)
        r = torch.sqrt(xn ** 2 + yn ** 2 + 1e-9)
        window = self.window(r)

        coeffs, index = [], []
        d = self.max_degree
        for n in range(d + 1):
            for m in range(n + 1):
                if n + m > d:
                    continue

                H = torch.zeros_like(z)
                for k in range(min(n, m) + 1):
                    coeff = (
                        ((-1) ** k)
                        * math.factorial(n) * math.factorial(m)
                        / (math.factorial(k) * math.factorial(n - k) * math.factorial(m - k))
                    )
                    z_pow = z ** (n - k) if (n - k) > 0 else torch.ones_like(z)
                    z_bar_pow = z_bar ** (m - k) if (m - k) > 0 else torch.ones_like(z)
                    H = H + coeff * z_pow * z_bar_pow

                norm = 1.0 / math.sqrt(
                    math.pi * (2 ** (n + m)) * math.factorial(n) * math.factorial(m)
                )
                psi = norm * H * window

                # Project the density onto this mode; keeps memory at O(B) per
                # mode instead of materialising the full 4D basis tensor.
                coeffs.append((density * torch.conj(psi)).sum(dim=(-2, -1)))
                index.append((n, m))

        return torch.stack(coeffs, dim=-1), index