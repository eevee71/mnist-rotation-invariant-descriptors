import numpy as np
import torch
from data.dataloader import rotate_dataset
from src.embedder import Embedder
from src.metrics import merge_labels


def get_features(data, targets, degree=9, K=9, eval_n=None):
    """Path for extracting invariants and merging labels directly."""

    y = np.asarray(targets.cpu().numpy())
    embedder = Embedder(max_degree=degree)
    coeffs = embedder._coeffs(data)
    raw = embedder.inv(coeffs).cpu().numpy()

    if eval_n is not None:
        raw, y = raw[:eval_n], y[:eval_n]

    merges = {} if K == 10 else ({6: 9} if K == 9 else {6: 9, 2: 5})
    y_m, C = merge_labels(y, merges)
    assert C == K

    return raw, y_m, C, embedder


def prepare_pipeline(data, targets, degree=9, k=9, eval_n=None):
    """Splits raw images, rotates test set, and extracts invariants."""

    if eval_n is not None:
        data, targets = data[:eval_n], targets[:eval_n]

    y_raw = targets.cpu().numpy()
    img_tr, img_te = data[:60000], data[60000:]
    ytr_raw, yte_raw = y_raw[:60000], y_raw[60000:]

    # rotate test images
    img_te_rot, yte_tensor = rotate_dataset(
        img_te, torch.tensor(yte_raw), max_angle=180, seed=42
    )

    # extract features (invariants)
    Xtr, ytr, _, embedder = get_features(img_tr, torch.tensor(ytr_raw), degree, k, eval_n=None)
    Xte, yte, _, _ = get_features(img_te_rot, yte_tensor, degree, k, eval_n=None)

    return Xtr, Xte, ytr, yte, embedder