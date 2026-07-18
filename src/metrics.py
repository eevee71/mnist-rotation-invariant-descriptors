import numpy as np


def merge_labels(y, merges):

    y = np.asarray(y).copy()
    for a, b in merges.items():
        y[y == a] = b

    uniq = np.unique(y)
    remap = {v: i for i, v in enumerate(uniq)}
    y_new = np.array([remap[v] for v in y], dtype=np.int64)

    return y_new, len(uniq)


def cluster_accuracy(y_true, y_pred):

    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    total = 0

    for c in np.unique(y_pred):
        vals = y_true[y_pred == c]
        maj = np.bincount(vals).argmax()
        total += (vals == maj).sum()

    return total / len(y_true)