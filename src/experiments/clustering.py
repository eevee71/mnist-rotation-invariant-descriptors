from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from src.dataset_preparation import prepare_pipeline
from src.whitening import BlockWhitener
from src.experiments.utils import _score
from src.visualization import visualize_cluster_prototypes
from sklearn.cluster import KMeans


def compare_methods(data, targets, degree=9, k=9, alpha=0.8, var_keep=0.99, eval_n=None, seed=0):
    """Compares standard unsupervised clustering (KMeans, GMM, Spectral) using PCA whitening and scaling."""

    Xtr, Xte, ytr, yte, _ = prepare_pipeline(data, targets, degree, k, eval_n=eval_n, seed=seed)
    out = {}

    std = Xtr.std(0)
    std[std < 1e-8] = 1e-8
    Xte_a = Xte / std ** alpha
    out["KMeans + std^alpha (current)"] = _score(yte, KMeans(k, n_init=10, random_state=seed).fit_predict(Xte_a))

    pca = PCA(n_components=var_keep, whiten=True, random_state=seed).fit(Xtr)
    Xte_w = pca.transform(Xte)
    out[f"KMeans + whitening ({Xte_w.shape[1]}d)"] = _score(yte, KMeans(k, n_init=10, random_state=seed).fit_predict(Xte_w))

    gmm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=seed)
    out["GMM full-cov + whitening"] = _score(yte, gmm.fit_predict(Xte_w))

    out["Spectral RBF + whitening"] = _score(yte, SpectralClustering(k, affinity="rbf", assign_labels="kmeans", random_state=seed).fit_predict(Xte_w))
    return out


def compare_blockwhiten(data, targets, degree=9, k=9, chiral_weights=(0.0, 0.3, 1.0), eval_n=None, seed=0):
    """Evaluates KMeans and GMM clustering after applying custom block whitening with varying chiral weights."""

    Xtr, Xte, ytr, yte, embedder = prepare_pipeline(data, targets, degree, k, eval_n=eval_n, seed=seed)

    out = {}
    for w in chiral_weights:
        whitener = BlockWhitener(embedder.inv, chiral_weight=w).fit(Xtr)
        Xte_w = whitener.transform(Xte)

        out[f"KMeans blockwhiten (chi_w={w})"] = _score(yte, KMeans(k, n_init=10, random_state=seed).fit_predict(Xte_w))
        gmm = GaussianMixture(k, covariance_type="full", n_init=3, random_state=seed)
        out[f"GMM   blockwhiten (chi_w={w})"] = _score(yte, gmm.fit_predict(Xte_w))

    return out


def run_and_plot_best_clustering(data, targets, degree=9, k=9, save_path="results/cluster_prototypes.png"):
    """Helper function to run KMeans and plot cluster prototypes directly from the clustering module."""

    Xtr, _, _, _, _ = prepare_pipeline(data, targets, degree=degree, k=k, seed=0)

    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    kmeans.fit(Xtr)

    visualize_cluster_prototypes(
        X=Xtr,
        raw_images=data,
        true_labels=targets,
        kmeans_model=kmeans,
        title=f"KMeans Cluster Prototypes (Degree={degree}, K={k})",
        save_path=save_path
    )