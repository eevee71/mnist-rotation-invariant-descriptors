from src.moment_transforms import MomentTransform
import src.so2_invariants as so2
import torch

class Embedder:
    """Pixels -> rotation-invariant, rescaled vector, one frozen path for
    both cluster centers and queries"""

    def __init__(self, max_degree=6, dtype=torch.float32):

        self.mt = MomentTransform(max_degree=max_degree)
        self.T  = self.mt.hermite_to_monomial_matrix()
        self.inv = None
        self.scale = None
        self.dtype = dtype


    def _coeffs(self, images):

        dm, xc, yc = self.mt.prepare_density_center(images)
        trace = self.mt.covariance_invariants(dm, xc, yc)[:, 0]
        xn, yn = self.mt.normalization(xc, yc, trace)
        U = self.mt.gaussian_polynomial_moment_matrix(dm, xn, yn)
        coeffs, index = self.mt.homogeneous_coefficients(U, self.T)

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