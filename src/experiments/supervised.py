from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.preprocessing import StandardScaler

from src.pipeline import prepare_pipeline
from src.experiments.utils import _score


def feature_ceiling(data, targets, degree=9, K=9, eval_n=None, seed=0):
    """Benchmarks supervised classifiers (kNN, LogReg, QDA) to establish an accuracy ceiling."""

    Xtr, Xte, ytr, yte, _ = prepare_pipeline(data, targets, degree, K, eval_n=eval_n, seed=seed)

    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    knn = KNeighborsClassifier(n_neighbors=15).fit(Xtr, ytr)
    lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
    qda = QuadraticDiscriminantAnalysis(reg_param=0.01).fit(Xtr, ytr)

    acc_knn = (knn.predict(Xte) == yte).mean()
    acc_lr = (lr.predict(Xte) == yte).mean()
    acc_qda = (qda.predict(Xte) == yte).mean()

    print(f"FEATURE CEILING (supervised, {K} classes): \n kNN={acc_knn:.4f}  \n logreg={acc_lr:.4f} \n QDA={acc_qda:.4f}")
    return dict(knn=acc_knn, logreg=acc_lr, qda=acc_qda)


def compare_supervised_projection(data, targets, degree=9, K=9, use_chirality=True, eval_n=None, seed=0, test_size=0.3):
    """Evaluates clustering and classification performance in a discriminative LDA subspace."""

    Xtr, Xte, ytr, yte, _ = prepare_pipeline(data, targets, degree, K, use_chirality, eval_n, test_size, seed)

    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    lda = LinearDiscriminantAnalysis(n_components=K - 1).fit(Xtr, ytr)
    Ztr, Zte = lda.transform(Xtr), lda.transform(Xte)

    tag = "with chi" if use_chirality else "no chi"
    out = {}

    nc = NearestCentroid().fit(Ztr, ytr)
    out[f"NearestCentroid in LDA ({tag})"] = _score(yte, nc.predict(Zte))

    pred = KMeans(K, n_init=10, random_state=seed).fit_predict(Zte)
    out[f"KMeans in LDA subspace ({tag})"] = _score(yte, pred)

    qda = QuadraticDiscriminantAnalysis().fit(Ztr, ytr)
    out[f"QDA in LDA subspace ({tag})"] = _score(yte, qda.predict(Zte))

    return out