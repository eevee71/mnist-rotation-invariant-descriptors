This folder contains the core experiments for exploring Rotation-Invariant MNIST Classification Using Complex Hermite Polynomials, along with auxiliary helper modules.

* **`supervised.py`**  
 Evaluates supervised baseline classifiers (kNN, Logistic Regression, QDA) and measures performance in reduced LDA subspaces.

* **`window_sweep.py`**  
 Evaluates 20 different radial window functions to measure their impact on feature extraction performance in QDA and MLP classifiers.

* **`full_pipeline.py`**  
 Provides full experiments pipeline.

* **`utils.py`**  
 Provides helper functions for calculating evaluation metrics (Accuracy, ARI, NMI) and formatting experimental results.