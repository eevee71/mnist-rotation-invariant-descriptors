import numpy as np
import torch
from data.dataloader import rotate_dataset
from src.embedder import Embedder
from src.metrics import merge_labels

# Rotation-equivalent digit pairs.
#
# 6 and 9 differ by a 180 degree rotation, so NO rotation-invariant
# representation can separate them: they are related by an element of the
# group being quotiented out. Merging them is a property of the task under
# SO(2), not a convenience.
#
# K=9 (default) merges them. K=10 keeps all ten digits, which is only useful
# for measuring how much the merge is worth when comparing against 10-class
# baselines from the literature.
ROTATION_MERGES = {6: 9}


def _merges_for(K):
    if K == 9:
        return ROTATION_MERGES
    if K == 10:
        return {}
    raise ValueError(
        f"K must be 9 (6/9 merged, default) or 10 (no merge); got {K}. "
        "Merging non-rotation-equivalent classes is not supported."
    )


def get_features(data, targets, degree=9, K=9, eval_n=None, embedder=None):
    """Extract rotation invariants and merge rotation-equivalent labels.

    Pass an existing `embedder` to guarantee that train and test features
    are produced by the identical frozen path (same feature ordering).
    """

    y = np.asarray(torch.as_tensor(targets).cpu().numpy())

    if embedder is None:
        embedder = Embedder(max_degree=degree)

    coeffs = embedder._coeffs(data)
    raw = embedder.inv(coeffs).cpu().numpy()

    if eval_n is not None:
        raw, y = raw[:eval_n], y[:eval_n]

    y_m, C = merge_labels(y, _merges_for(K))
    assert C == K, f"expected {K} classes after merging, got {C}"

    return raw, y_m, C, embedder


def prepare_pipeline(splits, degree=9, k=9, eval_n=None, rot_seed=42):
    """Extract invariants using MNIST's official 60k/10k split.

    The test set is rotated; the training set is left upright. This is the
    zero-rotation-supervision regime: the model never sees a rotated digit
    during training.

    splits : (train_data, train_targets, test_data, test_targets)
    k      : 9 merges 6 and 9 (default); 10 keeps them separate
    eval_n : optional cap on the TRAINING set only, for fast debug runs.
             The test set always stays at its full 10,000 images.
    """

    train_data, train_targets, test_data, test_targets = splits

    if eval_n is not None:
        train_data = train_data[:eval_n]
        train_targets = train_targets[:eval_n]

    ytr_tensor = torch.as_tensor(train_targets)
    yte_tensor = torch.as_tensor(test_targets)

    # rotate the held-out test images
    img_te_rot, yte_tensor = rotate_dataset(
        test_data, yte_tensor, max_angle=180, seed=rot_seed
    )

    # extract features; the test set reuses the embedder fitted on train
    Xtr, ytr, _, embedder = get_features(train_data, ytr_tensor, degree, k)
    Xte, yte, _, _ = get_features(img_te_rot, yte_tensor, degree, k,
                                  embedder=embedder)

    merge_note = "6/9 merged" if k == 9 else "no merge"
    print(f"train features: {Xtr.shape} | test features: {Xte.shape} "
          f"| {k} classes ({merge_note})")

    return Xtr, Xte, ytr, yte, embedder