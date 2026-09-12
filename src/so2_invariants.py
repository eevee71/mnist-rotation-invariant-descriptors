import torch


class SO2Invariants:

    def __init__(self, index, degree=6, ref=None, dtype=torch.float32):
        self.degree = degree
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.index = index

        # Separate indices into linear (n=m) and positive (n>m) modes
        self.lin_idx = [i for i, (n, m) in enumerate(index) if n == m]
        self.pos_idx = [i for i, (n, m) in enumerate(index) if n > m]
        self.pos_s = [n - m for (n, m) in [index[i] for i in self.pos_idx]]

        # Set reference mode for phase alignment
        if ref is None:
            s1_modes = [i for i, s in enumerate(self.pos_s) if s == 1]
            self.ref_pos = s1_modes[0] if s1_modes else 0
        elif isinstance(ref, tuple):
            pos_modes = [index[i] for i in self.pos_idx]
            self.ref_pos = pos_modes.index(ref)
        else:
            self.ref_pos = ref

    def __call__(self, coeffs):
        """Transforms complex coefficients into real SO(2) rotation invariants."""

        coeffs = coeffs.to(self.cdtype)

        # Get real linear components and squared magnitudes of positive modes
        lin = coeffs[:, self.lin_idx].real
        Zp = coeffs[:, self.pos_idx]
        mags = Zp.abs() ** 2

        # Perform phase alignment relative to the reference mode
        g = Zp[:, self.ref_pos]
        g_conj = torch.conj(g)

        phase = []
        for c, s in enumerate(self.pos_s):
            if c == self.ref_pos:
                continue
            # Neutralize phase shift
            W = Zp[:, c] * (g_conj ** s)
            phase.append(W.real)
            phase.append(W.imag)

        # Concatenate all invariants
        parts = [lin, mags]
        if phase:
            parts.append(torch.stack(phase, dim=-1))

        return torch.cat(parts, dim=-1).to(self.dtype)


def fit_scale(features, alpha=1.0, eps=1e-8):
    """Calculates standard deviation for feature normalization."""

    std = features.std(dim=0, unbiased=False)
    return std.clamp(min=eps) ** alpha


def apply_scale(features, scale):
    """Normalizes features using the precomputed scale."""

    return features / scale