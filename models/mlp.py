import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from src.dataset_preparation import prepare_pipeline


class InvariantMLP(nn.Module):
    """Simple MLP Classifier for extracted invariant features."""

    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            # Input normalization layer for added stability
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(data, targets, degree=9, K=9, epochs=30, lr=1e-3, batch_size=64, seed=0, log_interval=5):
    """Trains an MLP model on invariants and prints detailed epoch-by-epoch progress."""

    torch.manual_seed(seed)

    print(f"\n--- Preparing Data for MLP (degree={degree}, K={K}) ---")
    Xtr, Xte, ytr, yte, _ = prepare_pipeline(data, targets, degree=degree, K=K, seed=seed)

    # 1. Standardize invariant features (Fit ONLY on training set)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    Xtr_t, ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)
    Xte_t, yte_t = torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte, dtype=torch.long)

    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)

    model = InvariantMLP(input_dim=Xtr.shape[1], num_classes=K)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    history = {'train_loss': [], 'test_acc': []}

    print(f"--- Starting Training ({epochs} epochs, batch_size={batch_size}) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * bx.size(0)

        epoch_train_loss = running_loss / len(Xtr_t)
        history['train_loss'].append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            test_preds = model(Xte_t).argmax(dim=1)
            test_acc = (test_preds == yte_t).float().mean().item()
            history['test_acc'].append(test_acc)

        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {epoch_train_loss:.4f} | Test Acc: {test_acc * 100:.2f}%"
            )

    final_acc = history['test_acc'][-1]
    print(f"--- Training Finished. Final Test Accuracy: {final_acc:.4f} ---")

    model.eval()
    with torch.no_grad():
        test_preds = model(Xte_t).argmax(dim=1).cpu().numpy()
        yte_np = yte_t.cpu().numpy()

    final_acc = history['test_acc'][-1]
    print(f"--- Training Finished. Final Test Accuracy: {final_acc:.4f} ---")

    return model, yte_np, test_preds