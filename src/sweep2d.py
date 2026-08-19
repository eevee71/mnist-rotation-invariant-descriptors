"""2D sweep over (max_degree, moment_scale) x {point masses, pixel integrals}.

Design notes
------------
Two things make an overnight run of this size affordable.

1. **Degrees are nested.** The emitted mode set is {a >= b, a + b <= d}, so the
   columns for degree d are a *subset* of the columns for degree D > d. We
   extract coefficients once at the largest degree in the grid and slice for
   every smaller one. Extraction count drops from |degrees| x |scales| x
   |methods| to |scales| x |methods|.

2. **Rotation is applied up front.** ``prepare_pipeline`` rotates whichever
   images land in the test split, so coefficients would have to be recomputed
   per seed. Here the whole dataset is rotated once (fixed seed), coefficients
   are extracted for both the upright and the rotated copy, and each seed just
   picks different *indices*: train on upright[train_idx], test on
   rotated[test_idx]. Statistically identical, but extraction no longer scales
   with the number of seeds.

Everything is checkpointed to JSONL after each configuration, so the run is
interruptible and resumable -- rerun the same command and it skips what is
already done.
"""

import itertools
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.moment_transforms import gaussian_window
from src.quad_moments import QuadMomentTransform, coefficient_diagnostics
from src.so2_invariants import SO2Invariants
from src.metrics import merge_labels
from src.qda import fit_qda
from src.mlp import fit_mlp


# -- extraction ------------------------------------------------------------

def extract_coefficients(images, max_degree, moment_scale, n_gauss,
                         window=gaussian_window, exact_scale_moment=False,
                         batch_size=512, dtype=torch.float64, verbose=True):
    """Coefficients for a whole image tensor, in batches.

    Returns ``(coeffs (N, K) complex, index, mean_pixel_width)``.
    """
    mt = QuadMomentTransform(max_degree=max_degree, moment_scale=moment_scale,
                             window=window, n_gauss=n_gauss,
                             exact_scale_moment=exact_scale_moment, dtype=dtype)
    chunks, widths = [], []
    t0 = time.time()
    for s in range(0, len(images), batch_size):
        batch = images[s:s + batch_size]
        c, index = mt.complex_coefficients(batch)
        chunks.append(c)
        widths.append(mt.mean_pixel_width(batch) * len(batch))
        if verbose and s % (batch_size * 10) == 0:
            print(f"      {s + len(batch):>6d}/{len(images)}  "
                  f"({time.time() - t0:5.1f}s)", flush=True)
    return torch.cat(chunks, 0), index, float(sum(widths) / len(images))


def cached_coefficients(tag, images, cache_dir, **kw):
    """``extract_coefficients`` with an on-disk cache keyed by ``tag``."""
    path = os.path.join(cache_dir, f"{tag}.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return (torch.from_numpy(d['coeffs']),
                [tuple(t) for t in d['index']],
                float(d['h_mean']))
    coeffs, index, h_mean = extract_coefficients(images, **kw)
    np.savez_compressed(path,
                        coeffs=coeffs.numpy(),
                        index=np.array(index),
                        h_mean=h_mean)
    return coeffs, index, h_mean


# -- feature construction --------------------------------------------------

def invariants_at_degree(coeffs, index, degree, use_chirality=True,
                         dtype=torch.float64):
    """Slice to ``a + b <= degree`` and build the SO(2) invariants."""
    keep = [i for i, (a, b) in enumerate(index) if a + b <= degree]
    sub_index = [index[i] for i in keep]
    inv = SO2Invariants(sub_index, degree=degree, dtype=dtype)
    feats = inv(coeffs[:, keep])
    if not use_chirality:
        feats = feats[:, inv.achiral_idx]
    return feats.cpu().numpy().astype(np.float64), inv


# -- the sweep -------------------------------------------------------------

def run_sweep(data, targets, degrees, scales,
              methods=('point', 'integral'),
              seeds=(0, 1, 2),
              n_gauss=4,
              window=gaussian_window,
              window_name='gaussian_r2',
              K=9,
              use_chirality=True,
              exact_scale_moment=False,
              epochs=40,
              mlp_batch_size=64,
              mlp_lr=1e-3,
              reg_param=0.15,
              test_size=0.3,
              eval_n=None,
              rotation_seed=42,
              max_angle=180,
              batch_size=512,
              out_path='results/sweep2d.jsonl',
              cache_dir='results/coeff_cache',
              classifiers=('qda', 'mlp')):
    """Run the full grid and append one JSON record per configuration.

    Records carry ``method``, ``degree``, ``moment_scale``, ``seed``,
    ``qda_acc``, ``mlp_acc``, ``dim``, ``h_mean`` and timing.
    """
    from data.dataloader import rotate_dataset

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    if eval_n is not None:
        data, targets = data[:eval_n], targets[:eval_n]

    y_raw = np.asarray(targets.cpu().numpy())
    y_merged, C = merge_labels(y_raw, {6: 9} if K == 9 else {6: 9, 2: 5})
    assert C == K, f"expected {K} classes, got {C}"

    # rotate the whole set once; each seed selects test indices from this copy
    data_rot, _ = rotate_dataset(data, torch.tensor(y_raw),
                                 max_angle=max_angle, seed=rotation_seed)

    D = max(degrees)
    done = _load_done(out_path)
    total = len(methods) * len(scales) * len(degrees) * len(seeds)
    print(f"grid: {len(methods)} methods x {len(scales)} scales x "
          f"{len(degrees)} degrees x {len(seeds)} seeds = {total} configs "
          f"({len(done)} already done)\n", flush=True)

    counter = 0
    for method, ms in itertools.product(methods, scales):
        ng = 1 if method == 'point' else n_gauss
        print(f"[extract] method={method} moment_scale={ms} "
              f"degree={D} n_gauss={ng}", flush=True)
        kw = dict(max_degree=D, moment_scale=ms, n_gauss=ng, window=window,
                  exact_scale_moment=exact_scale_moment, batch_size=batch_size)
        base = f"{window_name}_ms{ms}_{method}_d{D}_esm{int(exact_scale_moment)}"
        c_up, index, h_mean = cached_coefficients(base + "_upright", data,
                                                  cache_dir, **kw)
        c_rot, _, _ = cached_coefficients(base + "_rot", data_rot,
                                          cache_dir, **kw)

        for degree, seed in itertools.product(degrees, seeds):
            counter += 1
            key = _key(method, degree, ms, seed)
            if key in done:
                continue

            t0 = time.time()
            idx = np.arange(len(y_merged))
            tr, te = train_test_split(idx, test_size=test_size,
                                      random_state=seed, stratify=y_merged)

            Fup, inv = invariants_at_degree(c_up, index, degree, use_chirality)
            Frot, _ = invariants_at_degree(c_rot, index, degree, use_chirality)
            Xtr, ytr = Fup[tr], y_merged[tr]
            Xte, yte = Frot[te], y_merged[te]

            rec = {**_fields(method, degree, ms, seed),
                   'dim': int(Xtr.shape[1]), 'h_mean': h_mean,
                   'n_train': int(len(tr)), 'n_test': int(len(te))}

            if 'qda' in classifiers:
                _, _, pred = fit_qda(Xtr, ytr, Xte, yte, reg_param=reg_param)
                rec['qda_acc'] = float(accuracy_score(yte, pred))
            if 'mlp' in classifiers:
                _, _, pred = fit_mlp(Xtr, ytr, Xte, yte, K, epochs=epochs,
                                     lr=mlp_lr, batch_size=mlp_batch_size,
                                     seed=seed, log_interval=0)
                rec['mlp_acc'] = float(accuracy_score(yte, pred))

            rec['seconds'] = round(time.time() - t0, 1)
            _append(out_path, rec)
            done.add(key)
            print(f"  [{counter:>4d}/{total}] {method:<8s} d={degree:<3d} "
                  f"ms={ms:<5} seed={seed} | "
                  f"QDA {rec.get('qda_acc', float('nan')) * 100:5.2f}% "
                  f"MLP {rec.get('mlp_acc', float('nan')) * 100:5.2f}% "
                  f"({rec['seconds']}s)", flush=True)

    print(f"\ndone -> {out_path}")


def run_coefficient_diagnostics(data, scales, max_degree,
                                window=gaussian_window,
                                window_name='gaussian_r2',
                                n_gauss=4, n_images=500,
                                exact_scale_moment=False,
                                out_path='results/coeff_diagnostics.jsonl'):
    """Per-degree point-vs-integral coefficient gap, independent of any classifier.

    Cheap, and the most direct measurement of what the pixel model actually
    changes. Run it first.
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    sub = data[:n_images]
    with open(out_path, 'a') as f:
        for ms in scales:
            kw = dict(max_degree=max_degree, moment_scale=ms, window=window,
                      exact_scale_moment=exact_scale_moment, verbose=False)
            cp, index, h_mean = extract_coefficients(sub, n_gauss=1, **kw)
            ci, _, _ = extract_coefficients(sub, n_gauss=n_gauss, **kw)
            overall = float(torch.linalg.norm(ci - cp) / torch.linalg.norm(cp))
            print(f"moment_scale={ms}  h={h_mean:.4f}  "
                  f"||int-pt||/||pt|| = {overall:.3e}")
            print("   deg   rel_diff    med_ratio    predicted")
            for row in coefficient_diagnostics(
                    cp, ci, index, h_mean,
                    gaussian_window_used=(window_name == 'gaussian_r2')):
                print(f"   {row['degree']:>3d}   {row['rel_diff']:.3e}   "
                      f"{row['median_ratio']:.6f}    {row['predicted_ratio']:.6f}")
                f.write(json.dumps({'moment_scale': ms, 'h_mean': h_mean,
                                    'overall_rel_diff': overall, **row}) + '\n')
            print()


# -- checkpoint helpers ----------------------------------------------------

def _fields(method, degree, moment_scale, seed):
    return {'method': method, 'degree': int(degree),
            'moment_scale': float(moment_scale), 'seed': int(seed)}


def _key(method, degree, moment_scale, seed):
    return (method, int(degree), float(moment_scale), int(seed))


def _load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done.add(_key(r['method'], r['degree'],
                              r['moment_scale'], r['seed']))
    return done


def _append(path, record):
    with open(path, 'a') as f:
        f.write(json.dumps(record) + '\n')
        f.flush()
        os.fsync(f.fileno())