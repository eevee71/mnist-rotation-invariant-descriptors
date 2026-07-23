import torch

from data.dataloader import load_MNIST, rotate_dataset
import src.compare as cmp
from models.mlp import train_mlp

DEGREE = 9
N_CLUSTERS = 9
EVAL_N = None


def main():
    """
    data, targets = load_MNIST(train=True, transform=None)

    # is the class signal present in the invariants
    print("feature ceiling (supervised)")
    cmp.feature_ceiling(data, targets, degree=DEGREE, K=N_CLUSTERS, eval_n=EVAL_N)

    # block whitening (non-diagonal) + chirality weighting
    print("\nblock whitening (unsupervised)")
    cmp.print_compare(cmp.compare_blockwhiten(
        data, targets, degree=DEGREE, K=N_CLUSTERS,eval_n=EVAL_N))

    # LDA (QDA)
    print("\nLDA discriminative subspace")
    cmp.print_compare(cmp.compare_supervised_projection(
        data, targets, degree=DEGREE, K=N_CLUSTERS, eval_n=EVAL_N))

    cmp.print_compare(cmp.compare_supervised_projection(
        data, targets, degree=9, K=9, use_chirality=False))  # no chi
    """
    # MLP
    train_data, train_targets = load_MNIST()
    train_mlp(train_data, train_targets, degree=9, K=9, use_chirality=True)


if __name__ == '__main__':
    main()