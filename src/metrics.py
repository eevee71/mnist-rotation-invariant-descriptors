import numpy as np


def merge_labels(y, merges):
    """Merge label ``b`` into ``a`` for each ``{a: b}``, then remap to ``0..C-1``.

    Used to collapse digit pairs that are rotations of one another (e.g. 6<->9),
    which are indistinguishable to a rotation-invariant model.
    """
    y = np.asarray(y).copy()
    for a, b in merges.items():
        y[y == b] = a

    uniq = np.unique(y)
    remap = {v: i for i, v in enumerate(uniq)}
    y_new = np.array([remap[v] for v in y], dtype=np.int64)

    return y_new, len(uniq)