from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from src.moment_transforms import gaussian_window
from src.pipeline import prepare_pipeline


def fit_qda(Xtr, ytr, Xte, yte, reg_param=0.15):
    """Fit QDA on already-extracted invariants.

    Standardises internally (scaler fit on the train split). Returns
    ``(model, y_true, y_pred)`` on the test set. ``reg_param`` must be > 0:
    the invariants are collinear, so each class covariance is rank-deficient
    and pure QDA (reg_param=0) fails to invert it.
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr)
    Xte = scaler.transform(Xte)

    model = QuadraticDiscriminantAnalysis(reg_param=reg_param).fit(Xtr, ytr)
    y_pred = model.predict(Xte)

    return model, yte, y_pred


def train_qda(data, targets, degree=9, K=9, window=gaussian_window,
              reg_param=0.15, seed=0):
    """Extract invariants and fit QDA.

    QDA fits one Gaussian (with its own covariance) per class, which pairs
    naturally with a compact, already rotation-invariant feature vector.
    ``reg_param`` shrinks each class covariance toward the identity; it must be
    > 0 because the invariants are collinear (their covariances are singular
    otherwise). Returns ``(model, y_true, y_pred)`` on the rotated test set.
    """
    Xtr, Xte, ytr, yte, _ = prepare_pipeline(
        data, targets, degree=degree, K=K, window=window, seed=seed
    )
    model, yte, y_pred = fit_qda(Xtr, ytr, Xte, yte, reg_param=reg_param)
    acc = (y_pred == yte).mean()
    print(f"QDA (degree={degree}, K={K}, reg_param={reg_param}) | test acc {acc * 100:.2f}%")

    return model, yte, y_pred