import numpy as np
from sklearn.metrics import accuracy_score
from data.dataloader import load_data, load_mnist_rot
from src.experiments.supervised import evaluate_qda_in_lda
from src.models.mlp import train_mlp

"""
1.Unrotated Train + Official Rotated Test: Replicates Section 5.2 of Sangalli et al. by testing `MNIST-12k`-trained models against 
    official pre-rotated `.amat` test files to assess robustness to interpolation artifacts.
2.Full Official MNIST-Rot: Standard benchmark evaluation where both training and test sets use official pre-rotated `.amat` files.
3.Dynamic Test Rotation: Evaluates exact SO(2) invariance by training on unrotated `MNIST-12k` 
    and applying dynamic in-code rotations to the test set.
"""


def run_seed_variance(
        data,
        targets,
        official_rot=False,
        data_rot=None,
        targets_rot=None,
        use_base_mnist=False
):
    """Evaluates QDA and MLP across multiple random seeds."""
    seeds = [42, 100, 2026, 7, 999, 13, 88, 123, 456, 789]
    qda_accs = []
    mlp_accs = []
    class_num = 10

    if use_base_mnist:
        mode_name = "Base MNIST Train (10k unrotated) + Official MNIST-Rot Test"
    elif official_rot:
        mode_name = "MNIST-Rot (fully rotated train and test)"
    elif data_rot is not None:
        mode_name = "MNIST-12k (unrotated train, official MNIST-Rot test)"
    else:
        mode_name = "MNIST-12k (unrotated train, dynamic test rot)"

    for i, s in enumerate(seeds, 1):
        print(f"\n--- Run [{i}/10] (Seed: {s}, Mode: {mode_name}) ---")

        # QDA evaluation
        _, y_true_qda, y_pred_qda = evaluate_qda_in_lda(
            data, targets, degree=9, k=class_num,
            split_seed=s, rot_seed=s,
            official_rot=official_rot, data_rot=data_rot, targets_rot=targets_rot
        )
        qda_acc = accuracy_score(y_true_qda, y_pred_qda) * 100
        qda_accs.append(qda_acc)

        # MLP evaluation
        _, y_true_mlp, y_pred_mlp = train_mlp(
            data, targets, degree=9, k=class_num, epochs=30,
            seed=s, split_seed=s, rot_seed=s,
            official_rot=official_rot, data_rot=data_rot, targets_rot=targets_rot
        )
        mlp_acc = accuracy_score(y_true_mlp, y_pred_mlp) * 100
        mlp_accs.append(mlp_acc)

    print("\n" + "=" * 50)
    print(f"SUMMARY (Mode: {mode_name})")
    print("=" * 50)
    print(f"QDA Mean Accuracy: {np.mean(qda_accs):.2f}% (± {np.std(qda_accs):.2f}%)")
    print(f"MLP Mean Accuracy: {np.mean(mlp_accs):.2f}% (± {np.std(mlp_accs):.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    data, targets = load_data()
    data_rot, targets_rot = load_mnist_rot()

    # Unrotated Train (MNIST-12k) + Official Rotated Test (MNIST-Rot)
    # Relies on data_rot internally being transposed (via dataset_preparation.prepare_pipeline)
    # to align spatial axes between the unrotated MNIST-12k grid and the pre-rotated MNIST-Rot dataset.
    run_seed_variance(
        data, targets,
        official_rot=False, data_rot=data_rot, targets_rot=targets_rot
    )

    # MNIST-Rot (Train and Test fully rotated)
    run_seed_variance(
        data, targets,
        official_rot=True, data_rot=data_rot, targets_rot=targets_rot
    )

    # Dynamic test rotation (Train (MNIST-12k) + Dynamic Rotated Test ((MNIST-12k)) ---
    run_seed_variance(
        data, targets,
        official_rot=False, data_rot=None, targets_rot=None
    )