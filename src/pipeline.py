import numpy as np
import torch
from sklearn.model_selection import train_test_split

from data.dataloader import rotate_dataset
from src.embedder import Embedder
from src.metrics import merge_labels


def get_features(data, targets, degree=9, K=9, use_chirality=True, eval_n=None):
    """Path for extracting invariants and merging labels directly."""

    y = np.asarray(targets.cpu().numpy())
    embedder = Embedder(max_degree=degree)
    coeffs = embedder._coeffs(data)
    raw = embedder.inv(coeffs).cpu().numpy()

    if not use_chirality:
        raw = raw[:, embedder.inv.achiral_idx]

    if eval_n is not None:
        raw, y = raw[:eval_n], y[:eval_n]

    merges = {6: 9} if K == 9 else {6: 9, 2: 5}
    y_m, C = merge_labels(y, merges)
    assert C == K

    return raw, y_m, C, embedder


def prepare_pipeline(data, targets, degree=9, K=9, use_chirality=True, eval_n=None, test_size=0.3, seed=0):
    """Splits raw images, rotates test set, and extracts invariants."""

    if eval_n is not None:
        data, targets = data[:eval_n], targets[:eval_n]

    # Train/test split on raw image tensors
    y_raw = targets.cpu().numpy()
    img_tr, img_te, ytr_raw, yte_raw = train_test_split(
        data, y_raw, test_size=test_size, random_state=seed, stratify=y_raw
    )

    # Rotate test images
    img_te_rot, yte_tensor = rotate_dataset(
        img_te, torch.tensor(yte_raw), max_angle=180, seed=42
    )

    # Extract features (invariants)
    Xtr, ytr, _, embedder = get_features(img_tr, torch.tensor(ytr_raw), degree, K, use_chirality, eval_n=None)
    Xte, yte, _, _ = get_features(img_te_rot, yte_tensor, degree, K, use_chirality, eval_n=None)

    return Xtr, Xte, ytr, yte, embedder