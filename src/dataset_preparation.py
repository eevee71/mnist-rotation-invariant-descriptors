import numpy as np
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
        official_rot=False,
        data_rot=None,
        targets_rot=None
):
    """Splits raw images, applies dataset rotation, and extracts invariants.

    - (official_rot=True): Uses the official pre-rotated MNIST-Rot dataset for all splits
                            (train, val, and test are completely pre-rotated).
    - (official_rot=False): Uses unrotated MNIST-12k for train/val.
                            If data_rot is provided, uses official pre-rotated test set;
                            otherwise, dynamically rotates the test set using rot_seed.
    """

    if eval_n is not None:
        data, targets = data[:eval_n], targets[:eval_n]
        if data_rot is not None:
            data_rot, targets_rot = data_rot[:eval_n], targets_rot[:eval_n]

    np.random.seed(split_seed)

    perm_train_val = np.random.permutation(12000)
    perm_test = 12000 + np.random.permutation(50000)

    train_idx = perm_train_val[:10000]
    val_idx = perm_train_val[10000:12000]
    test_idx = perm_test

    if official_rot:
        img_tr, ytr_tensor = data_rot[train_idx], targets_rot[train_idx]
        img_val, yval_tensor = data_rot[val_idx], targets_rot[val_idx]
        img_te_rot, yte_tensor = data_rot[test_idx], targets_rot[test_idx]
    else:
        img_tr, ytr_tensor = data[train_idx], targets[train_idx]
        img_val, yval_tensor = data[val_idx], targets[val_idx]

        # Uses official rotated test set if data_rot is provided, falls back to dynamic rotation otherwise
        if data_rot is not None:
            img_te_rot, yte_tensor = data_rot[test_idx], targets_rot[test_idx]
        else:
            img_te, yte_tensor = data[test_idx], targets[test_idx]
            img_te_rot, yte_tensor = rotate_dataset(img_te, yte_tensor, max_angle=180, seed=rot_seed)

    Xtr, ytr, _, embedder = get_features(img_tr, ytr_tensor, degree, k)
    Xval, yval, _, _ = get_features(img_val, yval_tensor, degree, k)
    Xte, yte, _, _ = get_features(img_te_rot, yte_tensor, degree, k)

    return Xtr, Xval, Xte, ytr, yval, yte, embedder