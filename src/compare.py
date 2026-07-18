import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from src.embedder import Embedder
from src.metrics import merge_labels, cluster_accuracy
from src.whitening import BlockWhitener


def _score(y, pred):

    return dict(
        acc=cluster_accuracy(y, pred),
        ari=adjusted_rand_score(y, pred),
        nmi=normalized_mutual_info_score(y, pred),
    )


def compare_methods(data, targets, degree=9, K=9, alpha=0.8, var_keep=0.99, eval_n=None, seed=0):

    y = np.asarray(targets.cpu().numpy())
    embedder = Embedder(max_degree=degree)
    coeffs = embedder._coeffs(data)
    raw = (embedder.inv(coeffs).cpu().numpy())  # raw invariants (with chirality)

    if eval_n is not None:
        raw, y = raw[:eval_n], y[:eval_n]

    merges = {6: 9} if K == 9 else {6: 9, 2: 5}
    y_m, C = merge_labels(y, merges)
    assert C == K

    out = {}

    # 1 diagonal scaling std^alpha + KMeans
    std = raw.std(0)
    std[std < 1e-8] = 1e-8
    Xa = raw / std**alpha
    out["KMeans + std^alpha (current)"] = _score(y_m, KMeans(K, n_init=10, random_state=seed).fit_predict(Xa))

    # 2 PCA-whitening (full Mahalanobis) + KMeans
    Xw = PCA(n_components=var_keep, whiten=True, random_state=seed).fit_transform(raw)
    out[f"KMeans + whitening ({Xw.shape[1]}d)"] = _score(y_m, KMeans(K, n_init=10,
                                                                     random_state=seed).fit_predict(Xw))

    # 3 GMM full covariance on whitened data (local Mahalanobis per cluster)
    gmm = GaussianMixture(K, covariance_type="full", n_init=3, random_state=seed)
    out["GMM full-cov + whitening"] = _score(y_m, gmm.fit(Xw).predict(Xw))

    # 4 spectral RBF on whitened data (non-convex clusters)
    out["Spectral RBF + whitening"] = _score(y_m, SpectralClustering(K, affinity="rbf",
                                                                     assign_labels="kmeans",
                                                                     random_state=seed).fit_predict(Xw),)
    return out


def print_compare(out):

    fmt = "{:<32} {:>7.4f} {:>7.4f} {:>7.4f}"

    print(f"{'method':<32} {'acc':>7} {'ari':>7} {'nmi':>7}")
    for name, s in out.items():
        print(fmt.format(name, s['acc'], s['ari'], s['nmi']))

    best = max(out.items(), key=lambda x: x[1]['acc'])
    print(f"\nBEST: {best[0]} -> acc={best[1]['acc']:.4f}")


def feature_ceiling(data, targets, degree=9, K=9, eval_n=None, seed=0):
    """Benchmarks feature signal via kNN and logistic regression
    to isolate model-specific failures from poor feature quality"""

    y = np.asarray(targets.cpu().numpy())
    embedder = Embedder(max_degree=degree)
    coeffs = embedder._coeffs(data)
    raw = embedder.inv(coeffs).cpu().numpy()

    if eval_n is not None:
        raw, y = raw[:eval_n], y[:eval_n]

    merges = {6: 9} if K == 9 else {6: 9, 2: 5}
    y_m, C = merge_labels(y, merges)
    assert C == K

    Xtr, Xte, ytr, yte = train_test_split(raw, y_m, test_size=0.3, random_state=seed, stratify=y_m)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    knn = KNeighborsClassifier(n_neighbors=15).fit(Xtr, ytr)
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    acc_knn = (knn.predict(Xte) == yte).mean()
    acc_lr = (lr.predict(Xte) == yte).mean()
    print(f"FEATURE CEILING (supervised, {C} classes):  kNN={acc_knn:.4f}   logreg={acc_lr:.4f}")

    return dict(knn=acc_knn, logreg=acc_lr)


def compare_blockwhiten(data, targets, degree=9, K=9, chiral_weights=(0.0, 0.3, 1.0), eval_n=None, seed=0):
    """Block whitening (non-diagonal) with chirality separated into its own block,
    sweeps a few chirality weights; 0.0 = achiral block only
    """

    y = np.asarray(targets.cpu().numpy())
    embedder = Embedder(max_degree=degree)
    coeffs = embedder._coeffs(data)
    raw = embedder.inv(coeffs)
    inv = embedder.inv
    raw = raw.cpu().numpy()

    if eval_n is not None:
        raw, y = raw[:eval_n], y[:eval_n]

    merges = {6: 9} if K == 9 else {6: 9, 2: 5}
    y_m, C = merge_labels(y, merges)
    assert C == K

    out = {}
    for w in chiral_weights:
        Xw = BlockWhitener(inv, chiral_weight=w).fit_transform(raw)
        out[f"KMeans blockwhiten (chi_w={w})"] = _score(y_m, KMeans(K,n_init=10, random_state=seed).fit_predict(Xw))
        gmm = GaussianMixture(K, covariance_type="full", n_init=3, random_state=seed)
        out[f"GMM   blockwhiten (chi_w={w})"] = _score(y_m, gmm.fit(Xw).predict(Xw))

    return out


def compare_supervised_projection(data, targets, degree=9, K=9, use_chirality=True, eval_n=None,
                                  seed=0, test_size=0.3):
    """LDA extracts discriminative subspaces to reveal if clustering misses low-variance signals,
    using labeled training prevents leakage;
    comparing chiral vs. achiral blocks isolates chirality's performance impact."""


    y = np.asarray(targets.cpu().numpy())
    embedder = Embedder(max_degree=degree)
    coeffs = embedder._coeffs(data)
    raw = embedder.inv(coeffs).cpu().numpy()

    if not use_chirality:  # only achiral columns
        raw = raw[:, embedder.inv.achiral_idx]

    if eval_n is not None:
        raw, y = raw[:eval_n], y[:eval_n]

    merges = {6: 9} if K == 9 else {6: 9, 2: 5}
    y_m, C = merge_labels(y, merges)
    assert C == K

    Xtr, Xte, ytr, yte = train_test_split(raw, y_m, test_size=test_size, random_state=seed, stratify=y_m)
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    lda = LinearDiscriminantAnalysis(n_components=C - 1).fit(Xtr, ytr)
    Ztr, Zte = lda.transform(Xtr), lda.transform(Xte)

    tag = "with chi" if use_chirality else "no chi"
    out = {}
    nc = NearestCentroid().fit(Ztr, ytr)

    out[f"NearestCentroid in LDA ({tag})"] = _score(yte, nc.predict(Zte))
    pred = KMeans(C, n_init=10, random_state=seed).fit_predict(Zte)
    out[f"KMeans in LDA subspace ({tag})"] = _score(yte, pred)

    return out