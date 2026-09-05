import numpy as np
from sklearn.metrics import accuracy_score
from data.dataloader import load_data
from src.experiments.supervised import evaluate_qda_in_lda
from src.models.mlp import train_mlp


def run_seed_variance(train_rotated=True):
    seeds = [42, 100, 2026, 7, 999, 13, 88, 123, 456, 789]
    qda_accs = []
    mlp_accs = []

    data, targets = load_data(full_dataset=True)
    class_num = 10

    for i, s in enumerate(seeds, 1):
        print(f"\n--- Run [{i}/10] (Seed: {s}, Train Rotated: {train_rotated}) ---")

        # QDA evaluation
        _, y_true_qda, y_pred_qda = evaluate_qda_in_lda(
            data, targets, degree=9, k=class_num,
            split_seed=s, rot_seed=s, train_rotated=train_rotated
        )
        qda_acc = accuracy_score(y_true_qda, y_pred_qda) * 100
        qda_accs.append(qda_acc)

        # MLP evaluation
        _, y_true_mlp, y_pred_mlp = train_mlp(
            data, targets, degree=9, k=class_num, epochs=30,
            seed=s, split_seed=s, rot_seed=s, train_rotated=train_rotated
        )
        mlp_acc = accuracy_score(y_true_mlp, y_pred_mlp) * 100
        mlp_accs.append(mlp_acc)

    print("\n" + "=" * 50)
    print(f"SUMMARY (Train Rotated: {train_rotated})")
    print("=" * 50)
    print(f"QDA Mean Accuracy: {np.mean(qda_accs):.2f}% (± {np.std(qda_accs):.2f}%)")
    print(f"MLP Mean Accuracy: {np.mean(mlp_accs):.2f}% (± {np.std(mlp_accs):.2f}%)")
    print("=" * 50)


if __name__ == "__main__":
    run_seed_variance(train_rotated=False)
    run_seed_variance(train_rotated=True)