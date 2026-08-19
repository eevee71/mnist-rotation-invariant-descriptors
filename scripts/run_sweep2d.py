"""Overnight 2D sweep: (max_degree x moment_scale) x {point masses, pixel integrals}.

Run from the repository root, e.g.

    python scripts/run_sweep2d.py --check
    python scripts/run_sweep2d.py --diagnostics-only
    python scripts/run_sweep2d.py 2>&1 | tee results/sweep.log

Resumable: rerunning the same command skips configurations already present in
the JSONL output.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.getcwd())

from data.dataloader import load_data                       # noqa: E402
from src.moment_transforms import gaussian_window           # noqa: E402
from src.quad_moments import check_quadrature               # noqa: E402
from src.sweep2d import run_sweep, run_coefficient_diagnostics  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--degrees', type=int, nargs='+',
                   default=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    p.add_argument('--scales', type=float, nargs='+',
                   default=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--methods', nargs='+', default=['point', 'integral'])
    p.add_argument('--n-gauss', type=int, default=4)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--mlp-batch-size', type=int, default=64)
    p.add_argument('--mlp-lr', type=float, default=1e-3)
    p.add_argument('--reg-param', type=float, default=0.15)
    p.add_argument('--eval-n', type=int, default=None,
                   help='cap the dataset size (None = full)')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--K', type=int, default=9)
    p.add_argument('--no-chirality', action='store_true')
    p.add_argument('--exact-scale-moment', action='store_true',
                   help='also apply the h^2/6 pixel-variance correction to the '
                        'second moment used for scale normalisation')
    p.add_argument('--classifiers', nargs='+', default=['qda', 'mlp'])
    p.add_argument('--out', default='results/sweep2d.jsonl')
    p.add_argument('--cache-dir', default='results/coeff_cache')
    p.add_argument('--check', action='store_true',
                   help='verify Gauss-Legendre convergence and exit')
    p.add_argument('--diagnostics-only', action='store_true',
                   help='run the coefficient-level comparison and exit')
    p.add_argument('--skip-diagnostics', action='store_true',
                   help='go straight to the grid (use if diagnostics already ran)')
    args = p.parse_args()

    torch.set_num_threads(max(1, os.cpu_count() // 2))

    data, targets = load_data(full_dataset=True)
    if args.eval_n:
        data, targets = data[:args.eval_n], targets[:args.eval_n]
    print(f"dataset: {tuple(data.shape)}\n")

    if args.check:
        for ms in args.scales:
            errs = check_quadrature(data[:200], max_degree=max(args.degrees),
                                    moment_scale=ms, window=gaussian_window)
            pretty = '  '.join(f'n={k}: {v:.2e}' for k, v in errs.items())
            print(f"moment_scale={ms:<5}  rel. diff vs finest quadrature   {pretty}")
        print("\npick the smallest n_gauss whose error is orders of magnitude "
              "below the point-vs-integral gap you are measuring.")
        return

    if args.diagnostics_only:
        run_coefficient_diagnostics(
            data, args.scales, max_degree=max(args.degrees),
            window=gaussian_window, n_gauss=args.n_gauss,
            exact_scale_moment=args.exact_scale_moment,
            out_path=os.path.join(os.path.dirname(args.out) or '.',
                                  'coeff_diagnostics.jsonl'))
        return

    if not args.skip_diagnostics:
        run_coefficient_diagnostics(
            data, args.scales, max_degree=max(args.degrees),
            window=gaussian_window, n_gauss=args.n_gauss,
            exact_scale_moment=args.exact_scale_moment,
            out_path=os.path.join(os.path.dirname(args.out) or '.',
                                  'coeff_diagnostics.jsonl'))

    run_sweep(
        data, targets,
        degrees=args.degrees,
        scales=args.scales,
        methods=tuple(args.methods),
        seeds=tuple(args.seeds),
        n_gauss=args.n_gauss,
        window=gaussian_window,
        window_name='gaussian_r2',
        K=args.K,
        use_chirality=not args.no_chirality,
        exact_scale_moment=args.exact_scale_moment,
        epochs=args.epochs,
        mlp_batch_size=args.mlp_batch_size,
        mlp_lr=args.mlp_lr,
        reg_param=args.reg_param,
        eval_n=None,
        batch_size=args.batch_size,
        out_path=args.out,
        cache_dir=args.cache_dir,
        classifiers=tuple(args.classifiers),
    )


if __name__ == '__main__':
    main()