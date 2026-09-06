# Source

This directory contains the core implementation of the rotation-invariant feature extraction pipeline, machine learning models, experimental scripts, and visualization utilities.


* **`embedder.py`**  
Provides the `Embedder` class that coordinates the full pipeline: transforming raw image pixels into complex moments, computing $SO(2)$ rotation invariants, and applying feature scaling.

* **`so2_invariants.py`**  
Implements algebraic $SO(2)$ rotation invariants from complex moment coefficients.

* **`moment_transforms.py`**  
Computes density centers, normalized coordinates, and complex Hermite polynomial coefficients mode-by-mode.

* **`dataset_preparation.py`**  
Handles data splitting (train, validation, test).

* **`metrics.py`**  
allProvides standard classification evaluation metrics (accuracy, precision, rec, F1-score), label remapping/merging utilities and cluster accuracy functions.

* **`visualization.py`**  
Generates and saves visual analyses.

* **`models/`**  
Contains the implementation of the Invariant MLP classifier (`mlp.py`).

* **`experiments/`**  
Contains dedicated scripts and setups for running systematic evaluations, seed variance tests and supervised benchmarks. ([experiments README](experiments/README.md))