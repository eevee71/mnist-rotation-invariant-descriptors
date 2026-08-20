This folder contains the core experiments for exploring Rotation-Invariant MNIST Classification Using Complex Hermite Polynomials, along with auxiliary helper modules.

* **`clustering.py`**  
  Compares unsupervised clustering methods (KMeans, GMM, Spectral) using custom scaling, standard PCA whitening and block whitening techniques.

* **`supervised.py`**  
 Evaluates supervised baseline classifiers (kNN, Logistic Regression, QDA) and measures performance in reduced LDA subspaces.


* **`parameters_sweep.py`**  
 Executes a grid search across polynomial degrees and feature scaling factors to optimize clustering accuracy.


* **`window_sweep.py`**  
 Evaluates 20 different radial window functions to measure their impact on feature extraction performance in QDA and MLP classifiers.


* **`utils.py`**  
 Provides helper functions for calculating evaluation metrics (Accuracy, ARI, NMI) and formatting experimental results.