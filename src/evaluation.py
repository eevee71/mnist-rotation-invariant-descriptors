import os

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)


def classification_metrics(y_true, y_pred, class_names=None, digits=4):
    """Print accuracy / macro P-R-F1 and a per-class report; return the summary.

    Classifier-agnostic: pass the ``(y_true, y_pred)`` returned by either
    ``train_mlp`` or ``train_qda``.
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


def plot_confusion_matrix(y_true, y_pred, class_names=None,
                          title="Confusion Matrix", save_path=None):
    """Plot (and optionally save) a confusion matrix for any classifier."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[saved] {save_path}")

    plt.show()
