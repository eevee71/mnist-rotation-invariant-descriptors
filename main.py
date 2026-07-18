from data.dataloader import load_MNIST
import src.compare as cmp

DEGREE = 9
N_CLUSTERS = 9
EVAL_N = None


def main():

    data, targets = load_MNIST()

    # is the class signal present in the invariants
    print("feature ceiling (supervised)")
    cmp.feature_ceiling(data, targets, degree=DEGREE, K=N_CLUSTERS, eval_n=EVAL_N)

    # block whitening (non-diagonal) + chirality weighting
    print("\nblock whitening (unsupervised)")
    cmp.print_compare(cmp.compare_blockwhiten(
        data, targets, degree=DEGREE, K=N_CLUSTERS,eval_n=EVAL_N))

    # LDA
    print("\nLDA discriminative subspace")
    cmp.print_compare(cmp.compare_supervised_projection(
        data, targets, degree=DEGREE, K=N_CLUSTERS, eval_n=EVAL_N))

    cmp.print_compare(cmp.compare_supervised_projection(
        data, targets, degree=9, K=9, use_chirality=False))  # no chi


if __name__ == '__main__':
    main()