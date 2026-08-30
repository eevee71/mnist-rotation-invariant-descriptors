from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from src.dataset_preparation import prepare_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


def visualize_cluster_prototypes(X, raw_images, true_labels, kmeans_model, title="Cluster Prototypes", save_path=None):
    """
    Visualizes cluster representatives in a compact, publication-ready grid
    and pops up the window on the screen.
    """

    centers = kmeans_model.cluster_centers_
    cluster_preds = kmeans_model.labels_

    n_clusters = len(centers)

    if n_clusters <= 6:
        cols = 3
    elif n_clusters <= 12:
        cols = 4
    else:
        cols = 5

    rows = int(np.ceil(n_clusters / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.3), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_clusters):
        cluster_indices = np.where(cluster_preds == i)[0]

        if len(cluster_indices) == 0:
            axes[i].axis('off')
            continue

        cluster_vectors = X[cluster_indices]
        center = centers[i]
        distances = np.linalg.norm(cluster_vectors - center, axis=1)

        closest_local_idx = np.argmin(distances)
        global_idx = cluster_indices[closest_local_idx]

        proto_image = raw_images[global_idx]
        t_label = true_labels[global_idx]

        axes[i].imshow(proto_image, cmap='gray')
        axes[i].set_title(f"Cl. {i} (Label: {t_label})", fontsize=10, fontweight='medium')
        axes[i].axis('off')

    for j in range(n_clusters, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=13, fontweight='bold')

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] Cluster visualization saved to: {save_path}")

    plt.show()


def visualize_cluster_sweep(data, targets, k_values=[5, 8, 10, 15, 20], degree=9, save_dir="results"):
    """
    Loops through different values of K, fits KMeans, saves each image,
    and pops up the plot window on the screen for each K sequentially.
    """

    Xtr, _, _, _, _ = prepare_pipeline(data, targets, degree=degree, k=9)
    os.makedirs(save_dir, exist_ok=True)

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        kmeans.fit(Xtr)

        save_path = os.path.join(save_dir, f"cluster_prototypes_K{k}.png")

        visualize_cluster_prototypes(
            X=Xtr,
            raw_images=data,
            true_labels=targets,
            kmeans_model=kmeans,
            title=f"KMeans Cluster Prototypes (Degree = {degree}, K = {k})",
            save_path=save_path
        )

    print(f"[Completed] All K-sweep figures saved and displayed")


def visualize_lda_projection(data, targets, degree=9, k=9, save_path=None):
    """
    Projects invariant features into a 2D LDA subspace and visualizes class separation.
    """

    print("[Pipeline] Preparing features for LDA projection...")
    Xtr, _, ytr, _, _ = prepare_pipeline(data, targets, degree=degree, k=k)

    print("[LDA] Fitting and projecting data...")
    lda = LDA(n_components=2)
    X_lda = lda.fit_transform(Xtr, ytr)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    scatter = ax.scatter(X_lda[:, 0], X_lda[:, 1], c=ytr, cmap='tab10', alpha=0.5, s=8)

    cbar = fig.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label("Digit Classes", fontsize=11)

    ax.set_title(f"LDA Subspace Projection (Degree = {degree}, K = {k})", fontsize=13, fontweight='bold')
    ax.set_xlabel("LDA Component 1", fontsize=11)
    ax.set_ylabel("LDA Component 2", fontsize=11)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Saved] LDA projection visualization saved to: {save_path}")

    plt.show()


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """Plots a confusion matrix directly from true and predicted labels."""

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')

    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion Matrix saved to: {save_path}")

    plt.show()