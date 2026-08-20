import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


def classification_metrics(y_true, y_pred, class_names=None, digits=4):
    """Print accuracy / macro P-R-F1 and a per-class report. Return the summary.
    """

    acc = accuracy_score(y_true, y_pred)
    summary = {
        'accuracy': acc,
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }

    print(f"Accuracy       : {acc * 100:.2f}%")
    print(f"Precision (M)  : {summary['precision_macro']:.4f}")
    print(f"Recall (M)     : {summary['recall_macro']:.4f}")
    print(f"F1 (macro)     : {summary['f1_macro']:.4f}")
    print(f"F1 (weighted)  : {summary['f1_weighted']:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=class_names,
                                digits=digits, zero_division=0))

    return summary


def merge_labels(y, merges):

    y = np.asarray(y).copy()
    for a, b in merges.items():
        y[y == b] = a

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