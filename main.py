from data.dataloader import load_data
from src.qda import train_qda
from src.mlp import train_mlp
from src.evaluation import classification_metrics, plot_confusion_matrix
# from src.window_sweep import run_window_sweep


def main():
    print("=== Loading dataset ===")
    data, targets = load_data(full_dataset=True)

    print("\n=== QDA ===")
    _, y_true, y_pred = train_qda(data, targets, degree=9, K=9, reg_param=0.15)
    classification_metrics(y_true, y_pred)

    print("\n=== MLP ===")
    _, y_true, y_pred = train_mlp(data, targets, degree=9, K=9, epochs=30)
    classification_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, save_path="results/mlp_confusion_matrix.png")

    # Radial-window ablation (QDA + MLP over the windows in WINDOWS):
    # run_window_sweep(data, targets, degree=11, K=9, epochs=30, eval_n=5000)


if __name__ == "__main__":
    main()