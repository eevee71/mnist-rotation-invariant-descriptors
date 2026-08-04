from data.dataloader import load_data
from models.mlp import train_mlp
from src.evaluation_metrics import classification_metrics
from src.experiments.supervised import evaluate_qda_in_lda
from src.visualization import visualize_mlp_confusion_matrix


def main():

    print("=== Loading Dataset ===")
    data, targets = load_data(full_dataset=True)

    print("\n=== Supervised Projection (LDA/QDA) ===")
    score_qda, y_true_qda, y_pred_qda = evaluate_qda_in_lda(data, targets)
    classification_metrics(y_true_qda, y_pred_qda)

    print("\n=== Training Invariant MLP Classifier ===")
    mlp_model, y_true_mlp, y_pred_mlp = train_mlp(data, targets, epochs=45, degree=9, K=9)
    classification_metrics(y_true_mlp, y_pred_mlp)

    print("\n=== Generating MLP Confusion Matrix ===")
    visualize_mlp_confusion_matrix(mlp_model, data, targets, degree=9, K=9,
                                   save_path="results/mlp_confusion_matrix.png")

if __name__ == "__main__":
    main()