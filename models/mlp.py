import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from src.compare import _prepare_pipeline
from src.embedder import Embedder
from src.metrics import merge_labels


class MLP(nn.Module):

    def __init__(self, in_dim, n_classes, hidden=(256, 128), p_drop=0.2):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(p_drop)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _features(data, targets, degree=9, K=9, use_chirality=True, eval_n=None):
    """Same frozen path as compare.py: pixels -> raw invariants + merged labels"""

    y = np.asarray(targets.cpu().numpy())
    embedder = Embedder(max_degree=degree)
    coeffs = embedder._coeffs(data)
    raw = embedder.inv(coeffs).cpu().numpy()

    if not use_chirality:
        raw = raw[:, embedder.inv.achiral_idx]

    if eval_n is not None:
        raw, y = raw[:eval_n], y[:eval_n]

    merges = {6: 9} if K == 9 else {6: 9, 2: 5}
    y_m, C = merge_labels(y, merges)
    assert C == K
    return raw, y_m, C


def train_mlp(data, targets, degree=9, K=9, use_chirality=True, eval_n=None,
              hidden=(256, 128), p_drop=0.05, epochs=80, batch_size=128,
              lr=0.2, weight_decay=1e-4, test_size=0.3, seed=0, patience=20, device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    raw, y_m, C = _features(data, targets, degree, K, use_chirality, eval_n)

    Xtr, Xte, ytr, yte, _ = _prepare_pipeline(
        data, targets, degree=degree, K=K,
        use_chirality=use_chirality, eval_n=eval_n, test_size=test_size, seed=seed
    )

    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    Xtr = torch.tensor(Xtr, dtype=torch.float32, device=device)
    Xte = torch.tensor(Xte, dtype=torch.float32, device=device)
    ytr = torch.tensor(ytr, dtype=torch.long, device=device)
    yte = torch.tensor(yte, dtype=torch.long, device=device)

    model = MLP(Xtr.shape[1], K, hidden=hidden, p_drop=p_drop).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = (Xtr.shape[0] + batch_size - 1) // batch_size  # Pamiętaj o podzieleniu przez batch_size!
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs
    )
    loss_fn = nn.CrossEntropyLoss()

    n = Xtr.shape[0]
    best_acc = 0.0
    best_state = None
    patience_counter = 0

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        running_train_loss = 0.0

        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            running_train_loss += loss.item() * idx.size(0)

        sched.step()

        train_loss = running_train_loss / n

        model.eval()
        with torch.no_grad():
            test_out = model(Xte)
            test_loss = loss_fn(test_out, yte).item()
            test_acc = (test_out.argmax(1) == yte).float().mean().item()

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"epoch {ep + 1:3d}   train_loss={train_loss:.4f}   test_loss={test_loss:.4f}   test_acc={test_acc:.4f}   (best={best_acc:.4f})"
            )

        if patience_counter >= patience:
            print(
                f"\n[Early Stopping] No improvement for {patience} consecutive epochs. Stopping early at epoch {ep + 1}."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model checkpoint with test_acc={best_acc:.4f}\n")

    return model, best_acc

