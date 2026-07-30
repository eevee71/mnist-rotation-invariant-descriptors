from data.dataloader import load_data
from models.mlp import train_mlp
from src.experiments.clustering import compare_methods, compare_blockwhiten
from src.experiments.supervised import feature_ceiling, compare_supervised_projection
from src.experiments.sweep import run_sweep
from src.experiments.utils import print_compare

def main():
    print("=== Loading Dataset ===")
    data, targets = load_data(full_dataset=True)

    print("\n=== Supervised Benchmarks (Feature Ceiling) ===")
    feature_ceiling(data, targets, degree=9, K=9)

    print("\n=== Unsupervised Clustering Comparison ===")
    clustering_results = compare_methods(data, targets, degree=9, K=9, eval_n=5000)
    print_compare(clustering_results)

    print("\n=== Block Whitening Comparison ===")
    blockwhiten_results = compare_blockwhiten(data, targets, degree=9, K=9, eval_n=5000)
    print_compare(blockwhiten_results)

    print("\n=== Supervised Projection (LDA Subspace) ===")
    lda_results = compare_supervised_projection(data, targets, degree=9, K=9)
    print_compare(lda_results)

    print("\n=== Training Invariant MLP Classifier ===")
    train_mlp(data, targets, degree=9, K=9, epochs=30)

#parameter grid search
# print("\n=== 7. Running Parameter Sweep (Grid Search) ===")
# sweep_results = run_sweep(data, targets, degrees=range(6, 10))
# print_report(sweep_results)

if __name__ == "__main__":
    main()