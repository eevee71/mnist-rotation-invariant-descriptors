import torch
from math import comb
from collections import defaultdict


def _monomial_to_z_coeff(x_pow, y_pow, z_pow):
    """Calculates the transition coefficient to the complex basis"""

    coeff = 0j
    for p in range(max(0, z_pow - y_pow), min(x_pow, z_pow) + 1):
        coeff += comb(x_pow, p) * comb(y_pow, z_pow - p) * ((-1) ** (y_pow - z_pow + p))

    return coeff * (2.0 ** (-(x_pow + y_pow))) * (1j ** (-y_pow))


class SO2Invariants:

    def __init__(self, index, degree=6, ref=None, dtype=torch.float64):

        self.degree = degree
        self.dtype = dtype
        self.cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.col = {tuple(t): p for p, t in enumerate(index)}   # (x_pow,y_pow,z_pow) -> column in the input coefficient matrix
        self.n_monomials = len(index)

        rows, amp_km = [], []
        for k in range(degree + 1):
            n = degree - k
            for m in range(n, -1, -2):  # frequencies present in this block
                a = (n + m) // 2
                row = torch.zeros(self.n_monomials, dtype=self.cdtype)
                for i in range(n + 1):
                    row[self.col[(i, n - i, k)]] = complex(_monomial_to_z_coeff(i, n - i, a))
                rows.append(row)
                amp_km.append((k, m))
        self.amp_coeff = torch.stack(rows)  # (n_amp_all, K) complex
        self.amp_km = amp_km

        # split m==0 (linear) vs m>0 (complex)
        self.lin_idx = [c for c, (k, m) in enumerate(amp_km) if m == 0]
        self.pos_idx = [c for c, (k, m) in enumerate(amp_km) if m > 0]
        self.pos_km = [amp_km[c] for c in self.pos_idx]

        # default = smallest-block frequency-1 amplitude (k=d-1, m=1)
        ref = ref if ref is not None else (degree - 1, 1)
        self.ref_pos = self.pos_km.index(ref)
        self.ref = ref
        self.pos_m = torch.tensor([m for (_, m) in self.pos_km])

        # length-27 independent subset: linear + magnitudes + Re-parts of phases
        n_lin, n_mag = len(self.lin_idx), len(self.pos_idx)
        base = n_lin + n_mag
        re_offsets = list(range(base, base + 2 * (len(self.pos_idx) - 1), 2))
        self.independent_idx = list(range(base)) + re_offsets

        full_dim = base + 2 * (len(self.pos_idx) - 1)
        self.achiral_idx = self.independent_idx
        _ach = set(self.achiral_idx)
        self.chiral_idx = [i for i in range(full_dim) if i not in _ach]


    def amplitudes(self, coeffs):
        """coeffs (B, K) real -> (B, n_amp_all) complex"""

        c = coeffs.to(self.cdtype)
        return c @ self.amp_coeff.transpose(0, 1)


    def __call__(self, coeffs):

        Z = self.amplitudes(coeffs) # (B, n_amp_all)
        lin = Z[:, self.lin_idx].real   # (B, 4)
        Zp = Z[:, self.pos_idx] # (B, 12)
        mags = Zp.abs() ** 2    # (B, 12)
        g = Zp[:, self.ref_pos] # (B,)
        parts = [lin, mags]
        phase = []

        for c, (k, m) in enumerate(self.pos_km):
            if c == self.ref_pos:
                continue
            W = Zp[:, c] * torch.conj(g) ** m   # weight m - m*1 = 0 (invariant)
            phase.append(W.real)
            phase.append(W.imag)

        parts.append(torch.stack(phase, dim=-1))    # (B, 22)
        return torch.cat(parts, dim=-1)


    def independent(self, coeffs):
        """coeffs (B, K) real -> (B, 27) algebraically-independent invariants"""

        return self.__call__(coeffs)[:, self.independent_idx]


    def labels(self):

        labs = [f"lin(k={k})" for (k, m) in [self.amp_km[c] for c in self.lin_idx]]
        labs += [f"|Z[k={k},m={m}]|^2" for (k, m) in self.pos_km]

        for c, (k, m) in enumerate(self.pos_km):
            if c == self.ref_pos:
                continue
            labs += [f"Re Z[{k},{m}]*conj(g)^{m}", f"Im Z[{k},{m}]*conj(g)^{m}"]
        return labs


    def cross_spectrum(self, coeffs):
        """Computes (B,D) degree-2 robust invariants, combines linear and
        phase-aligned cross-terms, avoids high-degree noise/instability, rank ~22/27"""

        Z = self.amplitudes(coeffs)
        Zp = Z[:, self.pos_idx]
        parts = [Z[:, self.lin_idx].real, Zp.abs() ** 2]
        byfreq = defaultdict(list)

        for c, (k, m) in enumerate(self.pos_km):
            byfreq[m].append(c)

        cross = []
        for m, cols in byfreq.items():
            for a in range(len(cols)):
                for b in range(a + 1, len(cols)):
                    w = Zp[:, cols[a]] * torch.conj(Zp[:, cols[b]])
                    cross += [w.real, w.imag]
        if cross:
            parts.append(torch.stack(cross, dim=-1))

        return torch.cat(parts, dim=-1)


def rotation_matrix(index, theta, dtype=torch.float64):

    c_, s_ = torch.cos(torch.tensor(theta, dtype=dtype)), torch.sin(torch.tensor(theta, dtype=dtype))
    col = {tuple(t): p for p, t in enumerate(index)}
    K = len(index)
    M = torch.zeros(K, K, dtype=dtype)

    for (i, j, k), c in col.items():    # x^i y^j z^k, x->cx-sy, y->sx+cy
        for p in range(i + 1):
            for q in range(j + 1):
                coef = comb(i, p) * c_ ** p * (-s_) ** (i - p) \
                     * comb(j, q) * s_ ** q * c_ ** (j - q)
                M[col[(p + q, (i - p) + (j - q), k)], c] += coef

    return M    # rotated_coeffs = coeffs @ M.T


def rotate_coeffs(coeffs, index, theta):

    M = rotation_matrix(index, theta, dtype=coeffs.dtype)
    return coeffs @ M.transpose(0, 1)


def fit_scale(features, alpha=1.0, eps=1e-8):
    """features (N, D) -> scale (D,). Store this and reuse for both sides"""

    std = features.std(dim=0, unbiased=False)
    return std.clamp(min=eps) ** alpha


def apply_scale(features, scale):
    """features (B, D) / scale (D,) -> rescaled features"""

    return features / scale