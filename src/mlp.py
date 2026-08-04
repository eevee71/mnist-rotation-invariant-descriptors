import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.moment_transforms import gaussian_window
from src.pipeline import prepare_pipeline


class InvariantMLP(nn.Module):
    """Small MLP classifier over the rotation-invariant features."""

    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def fit_mlp(Xtr, ytr, Xte, yte, num_classes, epochs=30, lr=1e-3,
            batch_size=64, seed=0, log_interval=5):
    """Train ``InvariantMLP`` on already-extracted invariants.

    Standardises internally (scaler fit on the train split). Returns
    ``(model, y_true, y_pred)`` on the test set. Set ``log_interval=0`` to run
    silently.
    """
    torch.manual_seed(seed)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    Xte_t = torch.tensor(Xte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.long)

    loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)

    model = InvariantMLP(input_dim=Xtr.shape[1], num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            running += loss.item() * bx.size(0)

        if log_interval and (epoch == 1 or epoch % log_interval == 0 or epoch == epochs):
            model.eval()
            with torch.no_grad():
                acc = (model(Xte_t).argmax(1) == yte_t).float().mean().item()
            print(f"Epoch [{epoch:02d}/{epochs}] | loss {running / len(Xtr_t):.4f} "
                  f"| test acc {acc * 100:.2f}%")

    model.eval()
    with torch.no_grad():
        y_pred = model(Xte_t).argmax(1).cpu().numpy()

    return model, yte, y_pred


def train_mlp(data, targets, degree=9, K=9, window=gaussian_window,
              epochs=30, lr=1e-3, batch_size=64, seed=0, log_interval=5):
    """Extract invariants and train ``InvariantMLP``.

    Returns ``(model, y_true, y_pred)`` on the rotated test set.
    """
    Xtr, Xte, ytr, yte, _ = prepare_pipeline(
        data, targets, degree=degree, K=K, window=window, seed=seed
    )
    return fit_mlp(Xtr, ytr, Xte, yte, num_classes=K, epochs=epochs, lr=lr,
                   batch_size=batch_size, seed=seed, log_interval=log_interval)