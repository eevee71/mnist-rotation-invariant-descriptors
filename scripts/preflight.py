"""Preflight checks. Run from the repo root before committing to a long sweep.

    python scripts/preflight.py

Checks, in order:
  1. imports resolve and the dataloader signatures match what the sweep expects
  2. n_gauss=1 reproduces MomentTransform.complex_coefficients exactly
  3. both pixel models are rotation-equivariant
  4. the invariant + classifier path runs end to end

Any FAIL means stop and fix before launching -- in particular a non-zero
difference in check 2 means the two sweep arms are not comparable and every
downstream number is meaningless.
"""

import os
import sys
import traceback

sys.path.insert(0, os.getcwd())

import numpy as np
import torch

OK, FAIL = "  [ok]  ", "  [FAIL]"
problems = []


def check(name, fn):
    print(f"{name}")
    try:
        msg = fn()
        print(f"{OK} {msg}")
        return True
    except Exception as exc:
        print(f"{FAIL} {type(exc).__name__}: {exc}")
        traceback.print_exc()
        problems.append(name)
        return False


# -- 1. wiring -------------------------------------------------------------

state = {}


def _wiring():
    from data.dataloader import load_data, rotate_dataset
    data, targets = load_data(full_dataset=True)
    assert data.ndim == 3, f"expected (N, H, W), got {tuple(data.shape)}"
    rot, lab = rotate_dataset(data[:16], targets[:16], max_angle=180, seed=42)
    assert rot.shape == data[:16].shape, f"rotate_dataset changed shape: {tuple(rot.shape)}"
    state['data'], state['targets'] = data, targets
    return (f"load_data -> {tuple(data.shape)} {data.dtype}, "
            f"targets {tuple(targets.shape)}; rotate_dataset ok")


# -- 2. the arms are the same code path --------------------------------------

def _identical():
    from src.moment_transforms import MomentTransform
    from src.quad_moments import QuadMomentTransform
    sub = state['data'][:64]
    c0, _ = MomentTransform(max_degree=9, moment_scale=3.0).complex_coefficients(sub)
    qt = QuadMomentTransform(max_degree=9, moment_scale=3.0, n_gauss=1,
                             dtype=sub.dtype if sub.dtype.is_floating_point else torch.float32)
    c1, _ = qt.complex_coefficients(sub)
    rel = float(torch.linalg.norm(c1 - c0) / torch.linalg.norm(c0))
    assert rel == 0.0, f"n_gauss=1 differs from parent by {rel:.3e} (must be exactly 0)"
    return "n_gauss=1 reproduces MomentTransform bit for bit"


# -- 3. equivariance ---------------------------------------------------------

def _equivariance():
    from src.moment_transforms import check_equivariance
    from src.quad_moments import QuadMomentTransform
    sub = state['data'][:64]
    out = []
    for ng in (1, 4):
        qt = QuadMomentTransform(max_degree=9, moment_scale=3.0, n_gauss=ng)
        good, err = check_equivariance(qt, sub, atol=1e-3)
        assert good, f"n_gauss={ng} failed equivariance, max error {err:.3e}"
        out.append(f"n_gauss={ng}: {err:.2e}")
        if ng == 4:
            state['h'] = qt.mean_pixel_width(sub)
    return "rot90 phase law holds  " + "  ".join(out)


# -- 4. quadrature is converged ---------------------------------------------

def _quadrature():
    from src.quad_moments import check_quadrature
    errs = check_quadrature(state['data'][:128], max_degree=12, moment_scale=3.0)
    n4 = errs.get(4, float('inf'))
    assert n4 < 1e-6, f"n_gauss=4 error {n4:.2e} is too close to the effect size"
    return ("  ".join(f"n={k}: {v:.1e}" for k, v in errs.items())
            + f"   (mean pixel width h={state.get('h', float('nan')):.4f})")


# -- 5. end to end -----------------------------------------------------------

def _end_to_end():
    from src.sweep2d import run_sweep
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        run_sweep(state['data'][:800], state['targets'][:800],
                  degrees=[5], scales=[3.0], methods=('point', 'integral'),
                  seeds=(0,), n_gauss=4, epochs=3,
                  out_path=os.path.join(tmp, 'p.jsonl'),
                  cache_dir=os.path.join(tmp, 'cache'))
        with open(os.path.join(tmp, 'p.jsonl')) as f:
            rows = [l for l in f if l.strip()]
    assert len(rows) == 2, f"expected 2 records, got {len(rows)}"
    return "invariants -> QDA -> MLP -> JSONL round-trips"


if __name__ == '__main__':
    print(f"python {sys.version.split()[0]}   torch {torch.__version__}   "
          f"numpy {np.__version__}")
    print(f"cwd {os.getcwd()}\n")

    check("1. repo wiring (load_data / rotate_dataset)", _wiring)
    if state:
        check("2. n_gauss=1 == MomentTransform", _identical)
        check("3. rotation equivariance, both arms", _equivariance)
        check("4. Gauss-Legendre convergence", _quadrature)
        check("5. end-to-end sweep record", _end_to_end)

    print()
    if problems:
        print("PREFLIGHT FAILED: " + ", ".join(problems))
        sys.exit(1)
    print("PREFLIGHT PASSED -- safe to launch the sweep.")
