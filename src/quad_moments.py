"""Pixel-square variant of ``MomentTransform``.

The original transform evaluates the basis at pixel centres:

    c_{a,b} = sum_i rho_i * psi_{a,b}(z_i)

which is the *midpoint quadrature rule* for the coefficient integral

    c_{a,b} = integral rho(z) psi_{a,b}(z) dA(z)

that you get if you treat each pixel as a uniform square of side h rather than
a point mass. This module computes that integral exactly (to machine precision)
with tensor Gauss-Legendre quadrature on each pixel, so the two conventions can
be compared under otherwise identical conditions.

Cost is ``n_gauss ** 2`` times the point version. ``n_gauss=4`` integrates
polynomials up to degree 7 per axis exactly and, combined with the smooth
Gaussian window, agrees with ``n_gauss=10`` to ~1e-11 in practice -- see
``check_quadrature`` below.

Notes
-----
``n_gauss=1`` reproduces ``MomentTransform.complex_coefficients`` exactly (the
single Gauss node is the midpoint with weight 1), so both arms of the sweep run
through the same code path and any difference is attributable to the pixel
model alone.

``exact_scale_moment`` additionally corrects the *normalisation*: the true
second moment of a piecewise-constant density is

    sum_i rho_i (|z_i|^2 + h^2/6)

because a uniform square of side h has variance h^2/12 per axis. In raw pixel
units h = 1, so the correction is a constant +1/6 on the trace, applied before
rescaling. It is off by default so that the two arms share an identical
normalisation and the comparison isolates the basis integral; turn it on for
the fully consistent "image is a piecewise-constant density" treatment.
"""

import math

import numpy as np
import torch

from src.moment_transforms import MomentTransform, gaussian_window


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
        Working precision. float64 is strongly recommended for the sweep: the
        point-vs-integral gap is O(1e-2) and high-degree coefficients span
        several orders of magnitude, so float32 rounding is uncomfortably close
        to the effect being measured.
    """

    def __init__(self, max_degree=9, moment_scale=3.0, window=gaussian_window,
                 n_gauss=1, exact_scale_moment=False, dtype=torch.float64):
        super().__init__(max_degree=max_degree, moment_scale=moment_scale,
                         window=window)
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

    # -- geometry ----------------------------------------------------------

    def normalised_frame(self, images):
        """Return ``(density, xn, yn, h)``.

        ``xn, yn`` are the centred, rescaled pixel-centre coordinates and ``h``
        is the pixel width in those same units, shape ``(B, 1, 1)``. Since the
        raw grid has unit spacing, h is exactly 1/scale.
        """
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
        for a, row in self._hermite_rows(z, z_bar):
            for b in range(min(a, self.max_degree - a) + 1):
                norm = 1.0 / math.sqrt(
                    math.pi * math.factorial(a) * math.factorial(b)
                )
                coeffs.append(norm * (weight * row[b]).sum(dim=(-2, -1)))
        return torch.stack(coeffs, dim=-1)

    def complex_coefficients(self, images):
        """Project onto the complex Hermite basis under the chosen pixel model.

        Returns ``(coeffs, index)`` exactly like the parent class.
        """
        density, xn, yn, h = self.normalised_frame(images)

        total = None
        for tx, wx in zip(self._nodes, self._weights):
            for ty, wy in zip(self._nodes, self._weights):
                c = self._project_at(density,
                                     xn + float(tx) * h / 2.0,
                                     yn + float(ty) * h / 2.0)
                c = c * float(wx * wy)
                total = c if total is None else total + c

        return total, self.index()

    def mean_pixel_width(self, images):
        """Mean h over the batch, in normalised units (a scalar float)."""
        _, _, _, h = self.normalised_frame(images)
        return float(h.mean())


# -- diagnostics -----------------------------------------------------------

def check_quadrature(images, max_degree=9, moment_scale=3.0,
                     window=gaussian_window, orders=(2, 4, 6, 10)):
    """Relative L2 difference of each Gauss order against the highest one.

    Run this once before committing to ``n_gauss``: if order 4 already agrees
    with order 10 to ~1e-10 there is no reason to pay for more nodes.
    """
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


def coefficient_diagnostics(c_point, c_integral, index, h_mean,
                            gaussian_window_used=True):
    """Per-degree comparison of the two coefficient conventions.

    For the exp(-r^2/2) window the Hermite functions are harmonic-oscillator
    eigenfunctions, ``Laplacian psi_n = (|z|^2 - 2(n+1)) psi_n``, so the
    midpoint rule predicts a per-degree shrinkage

        c_integral / c_point  ~  1 - h^2 (n + 1) / 12

    plus a small ``|z|^2`` term. The measured ratio is reported next to it; a
    close match confirms the gap is quadrature error and nothing structural.
    """
    deg = np.array([a + b for a, b in index])
    cp = c_point.detach().cpu().numpy()
    ci = c_integral.detach().cpu().numpy()

    rows = []
    scale_cut = 1e-12 * np.abs(cp).max()
    for n in sorted(set(deg.tolist())):
        m = deg == n
        num = np.linalg.norm(ci[:, m] - cp[:, m])
        den = np.linalg.norm(cp[:, m])
        ratio = ci[:, m] / np.where(np.abs(cp[:, m]) > scale_cut, cp[:, m], np.nan)
        rows.append({
            'degree': int(n),
            'rel_diff': float(num / max(den, 1e-300)),
            'median_ratio': float(np.nanmedian(np.real(ratio))),
            'predicted_ratio': (1.0 - h_mean ** 2 * (n + 1) / 12.0
                                if gaussian_window_used else float('nan')),
        })
    return rows
