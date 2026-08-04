"""Ablation over radial windows, scored by QDA and MLP on rotated test digits.

Features are extracted once per window (via the ``window`` parameter of the
pipeline, no monkey-patching) and fed to both classifiers, so the two accuracy
columns are directly comparable. Run as a script, or import ``run_window_sweep``.
"""
import torch
from sklearn.metrics import accuracy_score

from src.pipeline import prepare_pipeline
from src.qda import fit_qda
from src.mlp import fit_mlp


# Rotation-invariant radial windows w(r), r = ||x||. "gaussian_r2" is the
# principled default (consistent with the Hermite normalisation); the rest are
# the alternatives probed in the paper's ablation.
WINDOWS = {
    "gaussian_r2":           lambda r: torch.exp(-(r ** 2) / 2.0),
    "gaussian_narrow":       lambda r: torch.exp(-(r ** 2)),
    "gaussian_wide":         lambda r: torch.exp(-(r ** 2) / 4.0),
    "ring_r0.5":             lambda r: torch.exp(-((r - 0.5) ** 2) / 2.0),
    "ring_r1.0":             lambda r: torch.exp(-((r - 1.0) ** 2) / 2.0),
    "ring_r1.5":             lambda r: torch.exp(-((r - 1.5) ** 2) / 2.0),
    "ring_r2.0":             lambda r: torch.exp(-((r - 2.0) ** 2) / 2.0),
    "laplace_r1":            lambda r: torch.exp(-r),
    "laplace_r1_scale0.5":   lambda r: torch.exp(-2.0 * r),
    "exppower_r1.5":         lambda r: torch.exp(-(r ** 1.5)),
    "exppower_r2.5":         lambda r: torch.exp(-(r ** 2.5)),
    "exppower_r3.0":         lambda r: torch.exp(-(r ** 3.0)),
    "hybrid_center_ring1.0": lambda r: 0.5 * torch.exp(-(r ** 2) / 2.0)
                                       + 0.5 * torch.exp(-((r - 1.0) ** 2) / 2.0),
    "hybrid_center_ring1.5": lambda r: 0.5 * torch.exp(-(r ** 2) / 2.0)
                                       + 0.5 * torch.exp(-((r - 1.5) ** 2) / 2.0),
    "cauchy_r2":             lambda r: 1.0 / (1.0 + r ** 2),
    "heavytail_r4":          lambda r: 1.0 / (1.0 + r ** 4),
    "compact_r1.5":          lambda r: torch.where(r <= 1.5, torch.exp(-(r ** 2) / 2.0),
                                                   torch.zeros_like(r)),
    "compact_r2.0":          lambda r: torch.where(r <= 2.0, torch.exp(-(r ** 2) / 2.0),
                                                   torch.zeros_like(r)),
    "cosine_r2.0":           lambda r: torch.where(r <= 2.0,
                                                   torch.cos((torch.pi * r) / 4.0) ** 2,
                                                   torch.zeros_like(r)),
    "logistic":              lambda r: 1.0 / (1.0 + torch.exp(2.0 * (r - 1.0))),
}


def run_window_sweep(data, targets, degree=9, K=9, epochs=40, eval_n=5000,
                     reg_param=0.01, seed=0):
    """Score every window in ``WINDOWS`` with both QDA and MLP; print a table."""
    rows = []
    for i, (name, window) in enumerate(WINDOWS.items(), 1):
        print(f"[{i:02d}/{len(WINDOWS)}] {name}")
        Xtr, Xte, ytr, yte, _ = prepare_pipeline(
            data, targets, degree=degree, K=K, window=window, eval_n=eval_n, seed=seed
        )

        _, _, qda_pred = fit_qda(Xtr, ytr, Xte, yte, reg_param=reg_param)
        _, _, mlp_pred = fit_mlp(Xtr, ytr, Xte, yte, K, epochs=epochs,
                                 seed=seed, log_interval=0)

        qda_acc = accuracy_score(yte, qda_pred)
        mlp_acc = accuracy_score(yte, mlp_pred)
        rows.append((name, qda_acc, mlp_acc))
        print(f"    QDA {qda_acc * 100:5.2f}% | MLP {mlp_acc * 100:5.2f}%")

    _print_table(rows)
    return rows


def _print_table(rows):
    rows = sorted(rows, key=lambda r: (r[1] + r[2]) / 2, reverse=True)
    print(f"\n{'window':<24} {'QDA':>7} {'MLP':>7} {'avg':>7}")
    for name, q, m in rows:
        print(f"{name:<24} {q * 100:6.2f}% {m * 100:6.2f}% {(q + m) / 2 * 100:6.2f}%")
    print(f"\nbest (avg): {rows[0][0]}")


if __name__ == "__main__":
    from data.dataloader import load_data

    data, targets = load_data(full_dataset=True)
    run_window_sweep(data, targets, degree=9, K=9, epochs=60, eval_n=5000)