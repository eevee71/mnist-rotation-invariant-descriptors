import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from src.pipeline import prepare_pipeline


def evaluate_model_metrics(model, data, targets, degree=9, K=9, seed=0, scaler=None, class_names=None):
    """
    Oblicza i wypisuje pełną analizę metryk klasyfikacji:
    Accuracy, Precision, Recall, F1-Score (per-klasa oraz zagregowane).

    Returns:
        dict: Słownik zawierający zagregowane wartości metryk i surowe predykcje.
    """
    print("=" * 60)
    print(f"        RAPORT EVALUACJI MODELU (Degree={degree}, K={K})")
    print("=" * 60)

    # 1. Przygotowanie danych testowych i standaryzacja
    Xtr, Xte, _, yte, _ = prepare_pipeline(data, targets, degree=degree, K=K, seed=seed)

    if scaler is not None:
        Xte = scaler.transform(Xte)
    else:
        scaler_tmp = StandardScaler()
        scaler_tmp.fit(Xtr)
        Xte = scaler_tmp.transform(Xte)

    # 2. Ewaluacja modelu w PyTorch
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(Xte, dtype=torch.float32)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    # 3. Nazwy klas (jeśli brak, tworzy domyślne np. Cyfra 0, Cyfra 1...)
    if class_names is None:
        class_names = [f"Klasa {i}" for i in range(K)]

    # 4. Obliczenie zagregowanych metryk
    acc = accuracy_score(yte, preds)
    prec_macro = precision_score(yte, preds, average='macro', zero_division=0)
    rec_macro = recall_score(yte, preds, average='macro', zero_division=0)
    f1_macro = f1_score(yte, preds, average='macro', zero_division=0)
    f1_weighted = f1_score(yte, preds, average='weighted', zero_division=0)

    # 5. Generowanie szczegółowego raportu dla każdej klasy
    report_str = classification_report(
        yte,
        preds,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    # 6. Wypisanie czytelnego podsumowania w konsoli
    print("\n--- ZAGREGOWANE METRYKI GŁÓWNE ---")
    print(f"  • Accuracy:         {acc * 100:.2f}%  ({acc:.4f})")
    print(f"  • Precision (Macro): {prec_macro:.4f}")
    print(f"  • Recall (Macro):    {rec_macro:.4f}")
    print(f"  • F1-Score (Macro):  {f1_macro:.4f}")
    print(f"  • F1-Score (Weight): {f1_weighted:.4f}")
    print("\n--- SZCZEGÓŁOWY RAPORT KLASYFIKACJI (PER-KLASA) ---")
    print(report_str)
    print("=" * 60)

    return {
        'accuracy': acc,
        'precision_macro': prec_macro,
        'recall_macro': rec_macro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'predictions': preds,
        'true_labels': yte
    }