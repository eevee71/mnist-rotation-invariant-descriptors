from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import StandardScaler
from src.dataset_preparation import prepare_pipeline
from src.experiments.utils import _score


def feature_ceiling(data, targets, degree=9, k=9, eval_n=None):
    """Benchmarks supervised classifiers (kNN, LogReg, QDA) to establish an accuracy ceiling."""

    Xtr, Xval, Xte, ytr, yval, yte, _ = prepare_pipeline(data, targets, degree=degree, k=k, eval_n=eval_n)

    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    knn = KNeighborsClassifier(n_neighbors=15).fit(Xtr, ytr)
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    qda = QuadraticDiscriminantAnalysis(reg_param=0.15).fit(Xtr, ytr)

    acc_knn = (knn.predict(Xte) == yte).mean()
    acc_lr = (lr.predict(Xte) == yte).mean()
    acc_qda = (qda.predict(Xte) == yte).mean()

    print(f"FEATURE CEILING (supervised, {k} classes): \n kNN={acc_knn:.4f}  \n logreg={acc_lr:.4f} \n QDA={acc_qda:.4f}")
    return dict(knn=acc_knn, logreg=acc_lr, qda=acc_qda)


def prepare_lda_subspace(data, targets, degree=9, k=9, eval_n=None, split_seed=42, rot_seed=42):
    """Prepares data, applies feature standardization, and fits the LDA transformation."""

    Xtr, Xval, Xte, ytr, yval, yte, _ = prepare_pipeline(
        data, targets, degree=degree, k=k, eval_n=eval_n, split_seed=split_seed, rot_seed=rot_seed
    )
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    lda = LinearDiscriminantAnalysis(n_components=k - 1).fit(Xtr, ytr)
    Ztr, Zte = lda.transform(Xtr), lda.transform(Xte)

    return Ztr, Zte, ytr, yte


def evaluate_qda_in_lda(data_or_Ztr, targets_or_Zte, ytr=None, yte=None, degree=9, k=9, eval_n=None,  split_seed=42, rot_seed=42):
    """
    Trains and evaluates QDA in LDA space.
    Accepts either (data, targets) OR precomputed (Ztr, Zte, ytr, yte).
    """

    if ytr is not None and yte is not None:
        Ztr, Zte = data_or_Ztr, targets_or_Zte
    else:
        Ztr, Zte, ytr, yte = prepare_lda_subspace(
            data_or_Ztr, targets_or_Zte, degree=degree, k=k, eval_n=eval_n, split_seed=42, rot_seed=42
        )

    qda = QuadraticDiscriminantAnalysis().fit(Ztr, ytr)
    y_pred = qda.predict(Zte)
    score = _score(yte, y_pred)

    acc = (y_pred == yte).mean()
    print(f"QDA Accuracy: {acc * 100:.2f}%\n")
    return score, yte, y_pred


def evaluate_nearest_centroid_in_lda(Ztr, Zte, ytr, yte):
    """Trains and evaluates a NearestCentroid classifier in the reduced LDA space."""

    nc = NearestCentroid().fit(Ztr, ytr)
    y_pred = nc.predict(Zte)
    score = _score(yte, y_pred)
    return score, yte, y_pred


def evaluate_kmeans_in_lda(Zte, yte, k=9, seed=0):
    """Performs KMeans clustering in the reduced LDA space."""

    y_pred = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Zte)
    score = _score(yte, y_pred)
    return score, yte, y_pred


def evaluate_all_in_lda_space(data, targets, degree=9, k=9, eval_n=None, seed=0):
    """Aggregates all LDA-subspace evaluations without redundant LDA re-computations."""

    Ztr, Zte, ytr, yte = prepare_lda_subspace(data, targets, degree=degree, k=k, eval_n=eval_n)

    score_nc, _, _ = evaluate_nearest_centroid_in_lda(Ztr, Zte, ytr, yte)
    score_km, _, _ = evaluate_kmeans_in_lda(Zte, yte, k=k, seed=seed)

    score_qda, y_true, qda_preds = evaluate_qda_in_lda(Ztr, Zte, ytr, yte)

    out = {
        f"NearestCentroid in LDA": score_nc,
        f"KMeans in LDA subspace": score_km,
        f"QDA in LDA subspace": score_qda,
    }

    return out, y_true, qda_preds