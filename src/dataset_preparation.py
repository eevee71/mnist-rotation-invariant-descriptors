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


def prepare_pipeline(
        data,
        targets,
        degree=9,
        k=9,
        eval_n=None,
        split_seed=42,
        rot_seed=42,
        train_rotated=False
):
    """Splits raw images, rotates test set, and extracts invariants."""

    if eval_n is not None:
        data, targets = data[:eval_n], targets[:eval_n]

    y_raw = targets.cpu().numpy()
    np.random.seed(split_seed)
    idx = np.random.permutation(len(data))

    # 10k train, 2k val, 50k test
    train_idx = idx[:10000]
    val_idx = idx[10000:12000]
    test_idx = idx[12000:62000]

    img_tr, img_val, img_te = data[train_idx], data[val_idx], data[test_idx]
    ytr_raw, yval_raw, yte_raw = y_raw[train_idx], y_raw[val_idx], y_raw[test_idx]
    ytr_tensor = torch.tensor(ytr_raw)
    yval_tensor = torch.tensor(yval_raw)

    # rotate test images
    img_te_rot, yte_tensor = rotate_dataset(img_te, torch.tensor(yte_raw), max_angle=180, seed=rot_seed)

    #with rotated train and valid
    if train_rotated:
        img_tr, ytr_tensor = rotate_dataset(img_tr, ytr_tensor, max_angle=180, seed=rot_seed)
        img_val, yval_tensor = rotate_dataset(img_val, yval_tensor, max_angle=180, seed=rot_seed)

    # extract features (invariants)
    Xtr, ytr, _, embedder = get_features(img_tr, ytr_tensor, degree, k)
    Xval, yval, _, _ = get_features(img_val, yval_tensor, degree, k)
    Xte, yte, _, _ = get_features(img_te_rot, yte_tensor, degree, k)

    return Xtr, Xval, Xte, ytr, yval, yte, embedder