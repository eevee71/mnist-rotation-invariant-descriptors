"""Pixel-square variant of ``MomentTransform``.

The original transform evaluates the basis at pixel centres:

    c_{a,b} = sum_i rho_i * psi_{a,b}(z_i)

which is the *midpoint quadrature rule* for the coefficient integral

    c_{a,b} = integral rho(z) psi_{a,b}(z) dA(z)

that you get if you treat each pixel as a uniform square of side h rather than
a point mass. This module computes that integral exactly (to machine precision)
with tensor Gauss-Legendre quadrature on each pixel, so the two conventions can
be compared under otherwise identical conditions.
"""

import math
import numpy as np
import torch

from src.moment_transforms import MomentTransform


def gaussian_window(r):
    return torch.exp(-0.5 * r ** 2)


class QuadMomentTransform(MomentTransform):
    """``MomentTransform`` with a selectable pixel model.

    Parameters
    ----------
    n_gauss : int
        1 -> point masses (identical to the parent class).
        k>1 -> exact integral over each pixel square, k*k Gauss-Legendre nodes.
    exact_scale_moment : bool
        Add the h^2/6 pixel-variance correction to the second moment used for
        scale normalisation. See module docstring.
    dtype : torch.dtype
        Working precision. float64 is strongly recommended for the sweep.
    """

    def __init__(self, max_degree=9, moment_scale=3.0, window=gaussian_window,
                 n_gauss=1, exact_scale_moment=False, dtype=torch.float64):
        super().__init__(max_degree=max_degree, spatial_dimensions=2)
        self.moment_scale = moment_scale
        self.window = window
        self.n_gauss = int(n_gauss)
        self.exact_scale_moment = bool(exact_scale_moment)
        self.dtype = dtype

        if self.n_gauss <= 1:
            self._nodes = np.array([0.0])
            self._weights = np.array([1.0])
        else:
            t, w = np.polynomial.legendre.leggauss(self.n_gauss)
            self._nodes = t                 # on [-1, 1]
            self._weights = w / 2.0         # sum to 1, i.e. an average

    def second_moment_trace(self, density, xc, yc):
        """Helper to calculate second moment trace for scaling."""
        p_xx = (density * xc * xc).sum(dim=(-2, -1))
        p_yy = (density * yc * yc).sum(dim=(-2, -1))
        return p_xx + p_yy

    # -- geometry ----------------------------------------------------------

    def normalised_frame(self, images):
        """Return ``(density, xn, yn, h)``."""
        images = images.to(self.dtype)
        density, xc, yc = self.prepare_density_center(images)
        trace = self.second_moment_trace(density, xc, yc)
        if self.exact_scale_moment:
            trace = trace + 1.0 / 6.0

        scale = torch.sqrt(trace / self.moment_scale + 1e-12)[..., None, None]
        return density, xc / scale, yc / scale, 1.0 / scale

    # -- projection --------------------------------------------------------

    def _project_at(self, density, x, y):
        """Point projection of ``density`` onto the basis, sampled at (x, y)."""
        z = torch.complex(x, y)
        z_bar = torch.conj(z)
        r = torch.sqrt(x ** 2 + y ** 2 + 1e-12)
        weight = (density * self.window(r)).to(z.dtype)

        coeffs = []
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
                    psi_nm = norm * H_nm * weight
                    c_nm = psi_nm.sum(dim=(-2, -1))
                    coeffs.append(c_nm)

        return torch.stack(coeffs, dim=-1)

    def complex_coefficients(self, images):
        """Project onto the complex Hermite basis under the chosen pixel model."""
        density, xn, yn, h = self.normalised_frame(images)

        total = None
        for tx, wx in zip(self._nodes, self._weights):
            for ty, wy in zip(self._nodes, self._weights):
                c = self._project_at(density,
                                     xn + float(tx) * h / 2.0,
                                     yn + float(ty) * h / 2.0)
                c = c * float(wx * wy)
                total = c if total is None else total + c

        # Generate index structure matching MomentTransform
        index = []
        d = self.max_degree
        for n in range(d + 1):
            for m in range(n + 1):
                if n + m <= d:
                    index.append((n, m))

        return total, index

    def mean_pixel_width(self, images):
        """Mean h over the batch, in normalised units (a scalar float)."""
        _, _, _, h = self.normalised_frame(images)
        return float(h.mean())


# -- diagnostics -----------------------------------------------------------

def check_quadrature(images, max_degree=9, moment_scale=3.0,
                     window=gaussian_window, orders=(2, 4, 6, 10)):
    ref = None
    out = {}
    for k in sorted(orders, reverse=True):
        mt = QuadMomentTransform(max_degree=max_degree, moment_scale=moment_scale,
                                 window=window, n_gauss=k)
        c, _ = mt.complex_coefficients(images)
        if ref is None:
            ref = c
            out[k] = 0.0
        else:
            out[k] = float(torch.linalg.norm(c - ref) / torch.linalg.norm(ref))
    return dict(sorted(out.items()))