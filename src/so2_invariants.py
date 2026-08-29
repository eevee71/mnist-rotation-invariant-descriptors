import torch


class SO2Invariants:

    def __init__(self, index, degree=6, ref=None, dtype=torch.float32):
        self.degree = degree
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.index = index

        # Partition coefficient indices into invariant linear modes (n=m) and positive modes (n>m)
        self.lin_idx = [i for i, (n, m) in enumerate(index) if n == m]
        self.pos_idx = [i for i, (n, m) in enumerate(index) if n > m]
        self.pos_s = [n - m for (n, m) in [index[i] for i in self.pos_idx]]

        # Select the reference mode used for phase alignment
        if ref is None:
            s1_modes = [i for i, s in enumerate(self.pos_s) if s == 1]
            self.ref_pos = s1_modes[0] if s1_modes else 0
        elif isinstance(ref, tuple):
            pos_modes = [index[i] for i in self.pos_idx]
            self.ref_pos = pos_modes.index(ref)
        else:
            self.ref_pos = ref

        # Construct index mappings for achiral components (real parts and magnitudes)
        # and chiral components (imaginary parts)
        n_lin = len(self.lin_idx)
        n_mag = len(self.pos_idx)
        base = n_lin + n_mag

        re_offsets = []
        im_offsets = []
        phase_count = 0

        for i in range(len(self.pos_idx)):
            if i == self.ref_pos:
                continue
            re_offsets.append(base + phase_count)
            im_offsets.append(base + phase_count + 1)
            phase_count += 2

        self.independent_idx = list(range(base)) + re_offsets
        self.achiral_idx = self.independent_idx
        self.chiral_idx = im_offsets

        # diagnostic, filled in on each __call__
        self.last_ref_abs = None

    def amplitudes(self, coeffs):
        return coeffs.to(self.cdtype)

    def __call__(self, coeffs):
        """coeffs (B, K) complex -> (B, 38) real rotation invariants"""

        coeffs = coeffs.to(self.cdtype)
        # Compute real linear components and squared magnitudes of positive modes
        lin = coeffs[:, self.lin_idx].real  # (B, n_lin)
        Zp = coeffs[:, self.pos_idx]  # (B, n_pos)
        mags = Zp.abs() ** 2  # (B, n_pos)

        # Phase alignment relative to the chosen reference mode.
        # The reference is normalized to unit modulus so that only its PHASE
        # is used. Using the raw coefficient would scale each feature by
        # |g|^s (s up to `degree`), which injects a large nuisance magnitude
        # unrelated to shape and destabilizes downstream covariance estimates.
        g = Zp[:, self.ref_pos]
        g_abs = g.abs()
        self.last_ref_abs = g_abs.detach()
        g_hat = g / g_abs.clamp(min=1e-12).to(self.cdtype)
        g_conj = torch.conj(g_hat)

        phase = []
        for c, s in enumerate(self.pos_s):
            if c == self.ref_pos:
                continue
            # Neutralize the phase shift induced by spatial rotation
            W = Zp[:, c] * (g_conj ** s)
            phase.append(W.real)
            phase.append(W.imag)
        # Concatenate all derived invariant descriptors
        parts = [lin, mags]
        if phase:
            parts.append(torch.stack(phase, dim=-1))

        return torch.cat(parts, dim=-1).to(self.dtype)

    def ref_magnitude_report(self, eps=1e-6):
        """Diagnostic: how often the reference coefficient is near zero.

        When |g| is small the phase estimate is unstable and the aligned
        features become noise. Report this in the paper.
        """
        if self.last_ref_abs is None:
            return None
        a = self.last_ref_abs
        return {
            "min": a.min().item(),
            "median": a.median().item(),
            "frac_below_eps": (a < eps).float().mean().item(),
        }

    def independent(self, coeffs):
        """Extract algebraically independent invariants by omitting redundant imaginary components."""

        return self.__call__(coeffs)[:, self.independent_idx]

    def labels(self):

        labs = [f"lin(n={n},m={m})" for (n, m) in [self.index[i] for i in self.lin_idx]]
        labs += [f"|c[{n},{m}]|^2" for (n, m) in [self.index[i] for i in self.pos_idx]]

        for c, (n, m) in enumerate([self.index[i] for i in self.pos_idx]):
            if c == self.ref_pos:
                continue
            s = n - m
            labs += [f"Re c[{n},{m}]*conj(g_hat)^{s}", f"Im c[{n},{m}]*conj(g_hat)^{s}"]
        return labs


def rotate_coeffs(coeffs, index, theta):
    """Simulate spatial rotation by angle theta via complex phase modulation: exp(-i * (n-m) * theta)."""

    frequencies = torch.tensor(
        [n - m for (n, m) in index], dtype=coeffs.real.dtype, device=coeffs.device
    )
    phases = torch.exp(-1j * frequencies * theta)
    return coeffs * phases


def fit_scale(features, alpha=1.0, eps=1e-8):
    """Calculate the feature-wise standard deviation vector exponentiated
    by parameter alpha for feature normalization.

    Currently unused: the pipeline calls `embedder.inv(...)` directly and
    standardization is handled by StandardScaler in the classifiers.
    """

    std = features.std(dim=0, unbiased=False)
    return std.clamp(min=eps) ** alpha


def apply_scale(features, scale):
    """Normalize features by scaling with the precomputed standard deviation vector."""

    return features / scale