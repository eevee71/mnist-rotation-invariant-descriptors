import copy
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


def train_mlp(
        data,
        targets,
        degree: int = 9,
        k: int = 9,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 64,
        seed: int = 0,
        split_seed: int = 42,
        rot_seed: int = 42,
        log_interval: int = 5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        train_rotated = False
):
    """Trains an MLP model on invariants and prints progress."""

    torch.manual_seed(seed)

    print(f"\n--- Preparing Data for MLP (degree={degree}, K={k}) ---")
    Xtr, Xval, Xte, ytr, yval, yte, _ = prepare_pipeline(
        data=data,
        targets=targets,
        degree=degree,
        k=k,
        split_seed=split_seed,
        rot_seed=rot_seed,
        train_rotated=train_rotated
    )

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xval = scaler.transform(Xval)
    Xte = scaler.transform(Xte)

    Xtr_t = torch.from_numpy(Xtr).float()
    ytr_t = torch.from_numpy(ytr).long()

    Xval_t = torch.from_numpy(Xval).float().to(device)
    yval_t = torch.from_numpy(yval).long().to(device)

    Xte_t = torch.from_numpy(Xte).float().to(device)
    yte_t = torch.from_numpy(yte).long().to(device)

    loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    model = InvariantMLP(input_dim=Xtr.shape[1], num_classes=k).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    history = {'train_loss': [], 'val_acc': []}

    best_val_acc = 0.0
    best_epoch = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    print(f"--- Starting Training ({epochs} epochs, batch_size={batch_size}, device={device}) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * bx.size(0)

        epoch_train_loss = running_loss / len(loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            val_preds_tensor = model(Xval_t).argmax(dim=1)
            val_acc = (val_preds_tensor == yval_t).float().mean().item()
            history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_model_weights = copy.deepcopy(model.state_dict())

        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {epoch_train_loss:.4f} | Val Acc: {val_acc * 100:.2f}%"
            )
    print(f"--- Training Finished. Best Val Accuracy: {best_val_acc * 100:.2f}% (Achieved at Epoch {best_epoch}) ---")
    model.load_state_dict(best_model_weights)
    model.eval()
    with torch.no_grad():
        best_preds = model(Xte_t).argmax(dim=1).cpu().numpy()

    yte_np = yte_t.cpu().numpy()

    return model, yte_np, best_preds