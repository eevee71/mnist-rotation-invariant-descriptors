from data.dataloader import load_data
from src.models.mlp import train_mlp
from src.metrics import classification_metrics
from src.experiments.supervised import evaluate_qda_in_lda
from src.visualization import plot_confusion_matrix

CLASS_NUMBER = 10

def main():

    print("=== Loading Dataset ===")
    data, targets = load_data(full_dataset=True)

    print("\n=== Supervised Projection (QDA) ===")
    score_qda, y_true_qda, y_pred_qda = evaluate_qda_in_lda(data, targets, k=CLASS_NUMBER)
    classification_metrics(y_true_qda, y_pred_qda)

    print("\n=== Generating QDA Confusion Matrix ===")
    plot_confusion_matrix(y_true=y_true_qda, y_pred=y_pred_qda,
                          title="QDA Classifier Confusion Matrix",
                          save_path="results/qda_confusion_matrix.png")

    print("\n=== Training Invariant MLP Classifier ===")
    mlp_model, y_true_mlp, y_pred_mlp = train_mlp(data, targets, epochs=60, degree=9, k=CLASS_NUMBER)
    classification_metrics(y_true_mlp, y_pred_mlp)

    print("\n=== Generating MLP Confusion Matrix ===")
    plot_confusion_matrix(y_true=y_true_mlp, y_pred=y_pred_mlp,
                          title="MLP Classifier Confusion Matrix",
                          save_path="results/mlp_confusion_matrix.png")

if __name__ == "__main__":
    main()