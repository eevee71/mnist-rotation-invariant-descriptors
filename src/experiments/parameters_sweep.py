import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from src.embedder import Embedder
from src.metrics import merge_labels, cluster_accuracy


def run_sweep(data, targets, degrees=range(5, 12), alphas=None, class_counts=(8, 9), fit_n=5000, eval_n=None, n_init=5,
              seed=0):
    """Performs grid search to find optimal degree and scaling factor"""

    if alphas is None:
        alphas = np.round(np.arange(0.0, 2.001, 0.2), 3)

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
                pred = KMeans(n_clusters=K, n_init=n_init, random_state=seed).fit_predict(X)

                results.append(dict(deg=d, K=K, alpha=float(a), dim=raw.shape[1],
                                    acc=cluster_accuracy(y_merged, pred),
                                    ari=adjusted_rand_score(y_merged, pred),
                                    nmi=normalized_mutual_info_score(y_merged, pred)))
        print(f"  deg {d} done (dim={raw.shape[1]})")

    return results