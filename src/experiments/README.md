# Experiments

This folder contains the core experiments and auxiliary helper modules for exploring Rotation-Invariant MNIST Classification.

* **`dataset_compare.py`**  
Checks compatibility and compares statistics such as shapes, mass, means, and zeros between the MNIST-12k and MNIST-Rot datasets.

* **`full_pipeline.py`**  
Provides the full experiments pipeline, including supervised benchmarks, MLP classifier training and the generation of visualizations like LDA projections and confusion matrices.

* **`quad_moments.py`**  
Implements a `QuadMomentTransform` that computes exact integrals over pixel squares using tensor Gauss-Legendre quadrature.

* **`run_seed_variance.py`**  
Evaluates QDA and MLP classifiers across multiple random seeds under various official and dynamic dataset rotation configurations to assess invariance and robustness.

* **`supervised.py`**  
Benchmarks supervised baseline classifiers to establish an accuracy ceiling and measures performance within reduced LDA subspaces.

* **`utils.py`**  
Provides helper functions for calculating evaluation metrics and formatting experimental results.

* **`window_sweep.py`**  
Evaluates 20 different radial window functions to measure their impact on feature extraction performance in QDA and MLP classifiers.