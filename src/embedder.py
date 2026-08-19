import torch

from src.moment_transforms import MomentTransform, gaussian_window
from src.so2_invariants import SO2Invariants


class Embedder:

    def __init__(self, max_degree=9, dtype=torch.float32, window=gaussian_window):
        self.mt = MomentTransform(max_degree=max_degree, window=window)
        self.inv = None
        self.dtype = dtype

    def _coeffs(self, images):
        coeffs, index = self.mt.complex_coefficients(images)
        if self.inv is None:
            self.inv = SO2Invariants(index, degree=self.mt.max_degree, dtype=self.dtype)
        return coeffs

    def embed(self, images):
        coeffs = self._coeffs(images)
        return self.inv(coeffs)