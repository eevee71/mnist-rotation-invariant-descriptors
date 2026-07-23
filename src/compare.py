import numpy as np
import torch
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import StandardScaler

from data.dataloader import rotate_dataset
from src.embedder import Embedder
from src.metrics import cluster_accuracy, merge_labels
from src.whitening import BlockWhitener


def _score(y, pred):
    return dict(
        acc=cluster_accuracy(y, pred),
        ari=adjusted_rand_score(y, pred),
        nmi=normalized_mutual_info_score(y, pred),
    )


def _prepare_pipeline(data, targets, degree=9, K=9, use_chirality=True, eval_n=None, test_size=0.3, seed=0):
    """Splits raw images, rotates test set, and extracts invariants."""

    if eval_n is not None:
        data, targets = data[:eval_n], targets[:eval_n]

    # 1. Train/test split on raw image tensors
    y_raw = targets.cpu().numpy()
    img_tr, img_te, ytr_raw, yte_raw = train_test_split(
        data, y_raw, test_size=test_size, random_state=seed, stratify=y_raw
    )

    # 2. Rotate test images only
    img_te_rot, yte_tensor = rotate_dataset(
        img_te, torch.tensor(yte_raw), max_angle=180, seed=42
    )

    # 3. Extract features (invariants)
    embedder = Embedder(max_degree=degree)

    coeffs_tr = embedder._coeffs(img_tr)
    Xtr = embedder.inv(coeffs_tr).cpu().numpy()

    coeffs_te = embedder._coeffs(img_te_rot)
    Xte = embedder.inv(coeffs_te).cpu().numpy()

    if not use_chirality:
        achiral = embedder.inv.achiral_idx
        Xtr, Xte = Xtr[:, achiral], Xte[:, achiral]

    # 4. Merge labels
    merges = {6: 9} if K == 9 else {6: 9, 2: 5}
    ytr, _ = merge_labels(ytr_raw, merges)
    yte, C = merge_labels(yte_tensor.numpy(), merges)
    assert C == K

    return Xtr, Xte, ytr, yte, embedder


def print_compare(out):
    fmt = "{:<32} {:>7.4f} {:>7.4f} {:>7.4f}"
    print(f"{'method':<32} {'acc':>7} {'ari':>7} {'nmi':>7}")
    for name, s in out.items():
        print(fmt.format(name, s['acc'], s['ari'], s['nmi']))

    best = max(out.items(), key=lambda x: x[1]['acc'])
    print(f"\nBEST: {best[0]} -> acc={best[1]['acc']:.4f}")


def compare_methods(data, targets, degree=9, K=9, alpha=0.8, var_keep=0.99, eval_n=None, seed=0):
    Xtr, Xte, ytr, yte, _ = _prepare_pipeline(data, targets, degree, K, eval_n=eval_n, seed=seed)
    out = {}

    # Fit scaling/transforms on train, predict on rotated test
    std = Xtr.std(0)
    std[std < 1e-8] = 1e-8
    Xte_a = Xte / std ** alpha
    out["KMeans + std^alpha (current)"] = _score(yte, KMeans(K, n_init=10, random_state=seed).fit_predict(Xte_a))

    pca = PCA(n_components=var_keep, whiten=True, random_state=seed).fit(Xtr)
    Xte_w = pca.transform(Xte)
    out[f"KMeans + whitening ({Xte_w.shape[1]}d)"] = _score(yte,
                                                            KMeans(K, n_init=10, random_state=seed).fit_predict(Xte_w))

    gmm = GaussianMixture(K, covariance_type="full", n_init=3, random_state=seed)
    out["GMM full-cov + whitening"] = _score(yte, gmm.fit_predict(Xte_w))

    out["Spectral RBF + whitening"] = _score(yte, SpectralClustering(K, affinity="rbf", assign_labels="kmeans",
                                                                     random_state=seed).fit_predict(Xte_w))
    return out


def feature_ceiling(data, targets, degree=9, K=9, eval_n=None, seed=0):
    """Benchmarks feature signals via supervised classifiers."""

    Xtr, Xte, ytr, yte,_ = _prepare_pipeline(data, targets, degree, K, eval_n=eval_n, seed=seed)

    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    knn = KNeighborsClassifier(n_neighbors=15).fit(Xtr, ytr)
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    qda = QuadraticDiscriminantAnalysis(reg_param=0.01).fit(Xtr, ytr)

    acc_knn = (knn.predict(Xte) == yte).mean()
    acc_lr = (lr.predict(Xte) == yte).mean()
    acc_qda = (qda.predict(Xte) == yte).mean()

    print(f"FEATURE CEILING (supervised, {K} classes):  kNN={acc_knn:.4f}   logreg={acc_lr:.4f} QDA={acc_qda:.4f}")
    return dict(knn=acc_knn, logreg=acc_lr, qda=acc_qda)


def compare_blockwhiten(data, targets, degree=9, K=9, chiral_weights=(0.0, 0.3, 1.0), eval_n=None, seed=0):
    """Evaluates non-diagonal whitening with isolated chirality blocks."""

    Xtr, Xte, ytr, yte, embedder = _prepare_pipeline(data, targets, degree, K, eval_n=eval_n, seed=seed)

    out = {}
    for w in chiral_weights:
        whitener = BlockWhitener(embedder.inv, chiral_weight=w).fit(Xtr)
        Xte_w = whitener.transform(Xte)

        out[f"KMeans blockwhiten (chi_w={w})"] = _score(yte, KMeans(K, n_init=10, random_state=seed).fit_predict(Xte_w))
        gmm = GaussianMixture(K, covariance_type="full", n_init=3, random_state=seed)
        out[f"GMM   blockwhiten (chi_w={w})"] = _score(yte, gmm.fit_predict(Xte_w))

    return out


def compare_supervised_projection(data, targets, degree=9, K=9, use_chirality=True, eval_n=None, seed=0, test_size=0.3):
    """Evaluates discriminative LDA subspaces."""

    Xtr, Xte, ytr, yte, _ = _prepare_pipeline(data, targets, degree, K, use_chirality, eval_n, test_size, seed)

    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    lda = LinearDiscriminantAnalysis(n_components=K - 1).fit(Xtr, ytr)
    Ztr, Zte = lda.transform(Xtr), lda.transform(Xte)

    tag = "with chi" if use_chirality else "no chi"
    out = {}

    nc = NearestCentroid().fit(Ztr, ytr)
    out[f"NearestCentroid in LDA ({tag})"] = _score(yte, nc.predict(Zte))

    pred = KMeans(K, n_init=10, random_state=seed).fit_predict(Zte)
    out[f"KMeans in LDA subspace ({tag})"] = _score(yte, pred)

    qda = QuadraticDiscriminantAnalysis().fit(Ztr, ytr)
    out[f"QDA in LDA subspace ({tag})"] = _score(yte, qda.predict(Zte))

    return out