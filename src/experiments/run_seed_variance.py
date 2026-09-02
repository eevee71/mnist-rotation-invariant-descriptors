import numpy as np
from data.dataloader import load_data
from src.experiments.supervised import evaluate_qda_in_lda
from src.models.mlp import train_mlp


def run_experiment(seeds=None):
    """
    Runs a seed experiment comparing QDA and MLP classifiers
    on rotation-invariant descriptors, printing accuracy and variance summary.
    """

    if seeds is None:
        seeds = [42, 100, 2026, 7, 999]

    qda_accuracies = []
    mlp_accuracies = []

    print("=== Loading Dataset ===")
    data, targets = load_data(full_dataset=True)
    CLASS_NUMBER = 10

    print(f"\n=== Running Multi-Seed Experiment ({len(seeds)} runs) ===")
    for s in seeds:
        print(f"\n--- Seed: {s} ---")

        score_qda, y_true_qda, y_pred_qda = evaluate_qda_in_lda(
            data, targets, degree=9, k=CLASS_NUMBER, split_seed=s, rot_seed=s
        )
        qda_acc = np.mean(y_pred_qda == y_true_qda)
        qda_accuracies.append(qda_acc)
        print(f"QDA Test Acc: {qda_acc * 100:.2f}%")

        _, y_true_mlp, y_pred_mlp = train_mlp(
            data, targets, degree=9, k=CLASS_NUMBER, epochs=30,
            seed=s, split_seed=s, rot_seed=s
        )
        mlp_acc = np.mean(y_pred_mlp == y_true_mlp)
        mlp_accuracies.append(mlp_acc)
        print(f"MLP Test Acc: {mlp_acc * 100:.2f}%")

    print("\n" + "=" * 50)
    print(" EXPERIMENT RESULTS SUMMARY")
    print("=" * 50)

    print(f"Seeds used: {seeds}")
    print("-" * 50)

    print(f"QDA Accuracies: {[f'{acc * 100:.2f}%' for acc in qda_accuracies]}")
    print(f"QDA Mean      : {np.mean(qda_accuracies) * 100:.2f}%")
    print(f"QDA Variance  : {np.var(qda_accuracies):.6f}")

    print("-" * 50)

    print(f"MLP Accuracies: {[f'{acc * 100:.2f}%' for acc in mlp_accuracies]}")
    print(f"MLP Mean      : {np.mean(mlp_accuracies) * 100:.2f}%")
    print(f"MLP Variance  : {np.var(mlp_accuracies):.6f}")
    print("=" * 50)


run_experiment()