from data.dataloader import load_data
from models.mlp import train_mlp
from src.metrics import classification_metrics
from src.experiments.clustering import compare_methods, compare_blockwhiten
from src.experiments.supervised import feature_ceiling, evaluate_all_in_lda_space
from src.experiments.parameters_sweep import run_sweep
from src.experiments.utils import print_compare
from src.visualization import visualize_cluster_sweep, visualize_lda_projection, plot_confusion_matrix
from src.experiments.window_sweep import run_window_sweep


"""Full experiments pipeline"""


def run_full_experiments():
    print("=== Loading Dataset ===")
    data, targets = load_data(full_dataset=True)

    print("\n=== Supervised Benchmarks (Feature Ceiling) ===")
    feature_ceiling(data, targets, degree=9, k=9)

    print("\n=== Unsupervised Clustering Comparison ===")
    clustering_results = compare_methods(data, targets, degree=9, k=9, eval_n=5000)
    print_compare(clustering_results)

    print("\n=== Block Whitening Comparison ===")
    blockwhiten_results = compare_blockwhiten(data, targets, degree=9, k=9, eval_n=5000)
    print_compare(blockwhiten_results)

    print("\n=== Generating Cluster Prototypes Sweep (Different K) ===")
    visualize_cluster_sweep(data=data, targets=targets, k_values=[9], degree=9)

    print("\n=== Supervised Projection (LDA Subspace) ===")
    lda_results, y_true_qda, y_pred_qda = evaluate_all_in_lda_space(data, targets, degree=9, k=9)
    print_compare(lda_results)

    print("\n=== QDA Detailed Metrics ===")
    classification_metrics(y_true_qda, y_pred_qda)

    print("\n=== Generating LDA 2D Projection Visualization ===")
    visualize_lda_projection(data, targets, degree=9, k=9, save_path="results/lda_projection2d.png")

    print("\n=== Training Invariant MLP Classifier ===")
    mlp_model, y_true_mlp, y_pred_mlp = train_mlp(data, targets, epochs=60, degree=9, k=9)

    print("\n=== MLP Detailed Metrics ===")
    classification_metrics(y_true_mlp, y_pred_mlp)

    print("\n=== Generating MLP Confusion Matrix ===")
    plot_confusion_matrix(y_true=y_true_mlp, y_pred=y_pred_mlp,
                          title="MLP Classifier Confusion Matrix",
                          save_path="results/mlp_confusion_matrix.png")

    print("\n=== Running Parameter Sweep (Grid Search) ===")
    sweep_results = run_sweep(data, targets, degrees=range(6, 10))
    print(sweep_results)

    print("\n=== Running 20 Window Functions Sweep ===")
    run_window_sweep(data, targets, degree=9, k=9, epochs=15, eval_n=5000)

run_full_experiments()