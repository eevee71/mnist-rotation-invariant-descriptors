import torch
import src.so2_invariants as so2
from src.moment_transforms import MomentTransform


class Embedder:
    """Pixels -> rotation-invariant, rescaled vector, one frozen path for
    both cluster centers and queries"""

    def __init__(self, max_degree=6, dtype=torch.float32):
        self.mt = MomentTransform(max_degree=max_degree)
        self.inv = None
        self.scale = None
        self.dtype = dtype

    def _coeffs(self, images):
        coeffs, index = self.mt.complex_coefficients(images)

        if self.inv is None:
            self.inv = so2.SO2Invariants(index, degree=self.mt.max_degree, dtype=self.dtype)
        return coeffs

    def fit(self, images, alpha=0.0):
        """Calibrate the rescaler on a reference set, call once"""
        coeffs = self._coeffs(images)
        raw = self.inv(coeffs)
        self.scale = so2.fit_scale(raw, alpha=alpha)
        return self

    def embed(self, images):
        """Pixels -> (B, 38) comparison-ready vector, same path every time"""
        raw = self.inv(self._coeffs(images))
        return so2.apply_scale(raw, self.scale) if self.scale is not None else raw