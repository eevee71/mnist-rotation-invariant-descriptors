from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from src.dataset_preparation import prepare_pipeline
from src.experiments.utils import _score


def feature_ceiling(data, targets, degree=9, K=9, eval_n=None, seed=0):
    """Benchmarks supervised classifiers (kNN, LogReg, QDA) to establish an accuracy ceiling."""

    Xtr, Xte, ytr, yte, _ = prepare_pipeline(data, targets, degree, K, eval_n=eval_n, seed=seed)

    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    knn = KNeighborsClassifier(n_neighbors=15).fit(Xtr, ytr)
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    qda = QuadraticDiscriminantAnalysis(reg_param=0.15).fit(Xtr, ytr)

    acc_knn = (knn.predict(Xte) == yte).mean()
    acc_lr = (lr.predict(Xte) == yte).mean()
    acc_qda = (qda.predict(Xte) == yte).mean()

    print(f"FEATURE CEILING (supervised, {K} classes): \n kNN={acc_knn:.4f}  \n logreg={acc_lr:.4f} \n QDA={acc_qda:.4f}")
    return dict(knn=acc_knn, logreg=acc_lr, qda=acc_qda)


def prepare_lda_subspace(data, targets, degree=9, K=9, use_chirality=True, eval_n=None, seed=0, test_size=0.3):
    """Prepares data, applies feature standardization, and fits the LDA transformation."""
    Xtr, Xte, ytr, yte, _ = prepare_pipeline(data, targets, degree, K, use_chirality, eval_n, test_size, seed)

    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    lda = LinearDiscriminantAnalysis(n_components=K - 1).fit(Xtr, ytr)
    Ztr, Zte = lda.transform(Xtr), lda.transform(Xte)

    return Ztr, Zte, ytr, yte


def evaluate_qda_in_lda(data, targets, degree=9, K=9, use_chirality=True, eval_n=None, seed=0, test_size=0.3):
    """Trains and evaluates a QDA classifier in the reduced LDA space, handling LDA subspace preparation internally."""

    Ztr, Zte, ytr, yte = prepare_lda_subspace(
        data, targets, degree=degree, K=K,
        use_chirality=use_chirality, eval_n=eval_n, seed=seed, test_size=test_size
    )

    qda = QuadraticDiscriminantAnalysis().fit(Ztr, ytr)
    y_pred = qda.predict(Zte)
    score = _score(yte, y_pred)

    print(f"QDA Accuracy: {accuracy_score(yte, y_pred) * 100:.2f}%\n")
    return score, yte, y_pred


def evaluate_nearest_centroid_in_lda(Ztr, Zte, ytr, yte):
    """Trains and evaluates a NearestCentroid classifier in the reduced LDA space."""

    nc = NearestCentroid().fit(Ztr, ytr)
    y_pred = nc.predict(Zte)
    score = _score(yte, y_pred)
    return score, yte, y_pred


def evaluate_kmeans_in_lda(Zte, yte, K=9, seed=0):
    """Performs KMeans clustering in the reduced LDA space."""

    y_pred = KMeans(K, n_init=10, random_state=seed).fit_predict(Zte)
    score = _score(yte, y_pred)
    return score, yte, y_pred


def evaluate_all_in_lda_space(data, targets, degree=9, K=9, use_chirality=True, eval_n=None, seed=0, test_size=0.3):
    """Aggregates all LDA-subspace evaluations (NearestCentroid, KMeans, QDA) for convenience."""

    Ztr, Zte, ytr, yte = prepare_lda_subspace(data, targets, degree, K, use_chirality, eval_n, seed, test_size)
    tag = "with chi" if use_chirality else "no chi"

    score_nc, _, _ = evaluate_nearest_centroid_in_lda(Ztr, Zte, ytr, yte)
    score_km, _, _ = evaluate_kmeans_in_lda(Zte, yte, K=K, seed=seed)
    score_qda, y_true, qda_preds = evaluate_qda_in_lda(Ztr, Zte, ytr, yte)

    out = {
        f"NearestCentroid in LDA ({tag})": score_nc,
        f"KMeans in LDA subspace ({tag})": score_km,
        f"QDA in LDA subspace ({tag})": score_qda,
    }

    return out, y_true, qda_preds