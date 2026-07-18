import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from src.embedder import Embedder
from src.metrics import merge_labels, cluster_accuracy


def run_sweep(data, targets, degrees=range(5, 12), alphas=None,
              class_counts=(8, 9), fit_n=5000, eval_n=None, n_init=5, seed=0):

    if alphas is None:
        alphas = np.round(np.arange(0.0, 2.001, 0.2), 3)          # [0,0.2,...,2.0]
    y_full = np.asarray(targets.cpu().numpy() if torch.is_tensor(targets) else targets)

    results = []
    for d in degrees:
        embedder = Embedder(max_degree=d)
        coeffs = embedder._coeffs(data)
        raw = embedder.inv(coeffs)
        y = y_full

        if eval_n is not None:
            raw, y = raw[:eval_n], y_full[:eval_n]
        std = raw[:fit_n].std(dim=0, unbiased=False).clamp(min=1e-8)

        for K in class_counts:
            merges = {6: 9} if K == 9 else {6: 9, 2: 5}
            y_merged, C = merge_labels(y, merges)
            assert C == K, f"K={K} != merged classes={C}"

            for a in alphas:
                X = (raw / std ** float(a)).cpu().numpy()
                pred = KMeans(n_clusters=K, n_init=n_init,
                              random_state=seed).fit_predict(X)
                results.append(dict(deg=d, K=K, alpha=float(a),
                                    dim=raw.shape[1],
                                    acc=cluster_accuracy(y_merged, pred),
                                    ari=adjusted_rand_score(y_merged, pred),
                                    nmi=normalized_mutual_info_score(y_merged, pred)))
        print(f"  deg {d} done (dim={raw.shape[1]})")

    return results


def print_report(results, top=10):
    fmt = "{:>3} {:>3} {:>6.2f} {:>4} {:>7.4f} {:>7.4f} {:>7.4f}"

    print(f"\n{'deg':>3} {'K':>3} {'alpha':>6} {'dim':>4} {'acc':>7} {'ari':>7} {'nmi':>7}")
    for r in sorted(results, key=lambda x: x['acc'], reverse=True)[:top]:
        print(fmt.format(r['deg'], r['K'], r['alpha'], r['dim'], r['acc'], r['ari'], r['nmi']))

    best = max(results, key=lambda x: x['acc'])
    print(f"\nBEST: deg={best['deg']}, K={best['K']}, alpha={best['alpha']:.2f} -> "
          f"acc={best['acc']:.4f}, ari={best['ari']:.4f}, nmi={best['nmi']:.4f}")

    return best
