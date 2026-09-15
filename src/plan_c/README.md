# Plan C main stream

This package is the canonical production path. It organizes existing signal
processing and feature code without changing its calculations.

| File | Responsibility |
|---|---|
| `config.py` | Classes, 7 base features, 5 delta features, XGBoost parameters |
| `data.py` | Build and validate the model-ready feature table |
| `model.py` | Construct the configured `SafeXGBClassifier` |
| `evaluation.py` | Leave-one-recording-out evaluation and reports |
| `training.py` | Fit on all recordings and save production artifacts |
| `__main__.py` | Orchestrate `prepare`, `evaluate`, `train`, and `run-all` |

`run-all` prepares the feature table once. The same in-memory table and model
configuration are then used for evaluation and final training, preventing the
two paths from silently using different feature lists or parameters.
