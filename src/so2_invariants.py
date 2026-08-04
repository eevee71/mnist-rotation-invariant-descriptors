import torch


class SO2Invariants:

    def __init__(self, index, degree=6, ref=None, dtype=torch.float32):
        self.degree = degree
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.index = index

        self.lin_idx = [i for i, (n, m) in enumerate(index) if n == m]
        self.pos_idx = [i for i, (n, m) in enumerate(index) if n > m]
        self.pos_s = [n - m for (n, m) in [index[i] for i in self.pos_idx]]

        # Reference mode for phase alignment: defaults to first s=1 mode
        if ref is None:
            s1_modes = [i for i, s in enumerate(self.pos_s) if s == 1]
            self.ref_pos = s1_modes[0] if s1_modes else 0
        elif isinstance(ref, tuple):
            pos_modes = [index[i] for i in self.pos_idx]
            self.ref_pos = pos_modes.index(ref)
        else:
            self.ref_pos = ref

        # Index sets for achiral / chiral invariants
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

    def amplitudes(self, coeffs):
        return coeffs.to(self.cdtype)

    def __call__(self, coeffs):
        """coeffs (B, K) complex -> (B, 38) real rotation invariants"""

        coeffs = coeffs.to(self.cdtype)

        lin = coeffs[:, self.lin_idx].real  # (B, n_lin)
        Zp = coeffs[:, self.pos_idx]  # (B, n_pos)
        mags = Zp.abs() ** 2  # (B, n_pos)

        g = Zp[:, self.ref_pos]  # reference phase
        g_conj = torch.conj(g)

        phase = []
        for c, s in enumerate(self.pos_s):
            if c == self.ref_pos:
                continue
            W = Zp[:, c] * (g_conj**s)
            phase.append(W.real)
            phase.append(W.imag)

        parts = [lin, mags]
        if phase:
            parts.append(torch.stack(phase, dim=-1))

        return torch.cat(parts, dim=-1).to(self.dtype)

    def independent(self, coeffs):
        """coeffs (B, K) -> algebraically-independent invariants"""

        return self.__call__(coeffs)[:, self.independent_idx]

    def labels(self):
        labs = [f"lin(n={n},m={m})" for (n, m) in [self.index[i] for i in self.lin_idx]]
        labs += [f"|c[{n},{m}]|^2" for (n, m) in [self.index[i] for i in self.pos_idx]]

        for c, (n, m) in enumerate([self.index[i] for i in self.pos_idx]):
            if c == self.ref_pos:
                continue
            s = n - m
            labs += [f"Re c[{n},{m}]*conj(g)^{s}", f"Im c[{n},{m}]*conj(g)^{s}"]
        return labs


def rotate_coeffs(coeffs, index, theta):
    """Rotates complex Hermite coefficients: c_{n,m} -> c_{n,m} * exp(-i (n-m) theta)"""

    frequencies = torch.tensor(
        [n - m for (n, m) in index], dtype=coeffs.real.dtype, device=coeffs.device
    )
    phases = torch.exp(-1j * frequencies * theta)
    return coeffs * phases


def fit_scale(features, alpha=1.0, eps=1e-8):
    std = features.std(dim=0, unbiased=False)
    return std.clamp(min=eps) ** alpha


def apply_scale(features, scale):
    return features / scale