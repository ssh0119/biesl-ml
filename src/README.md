# Source-code map

## Production main stream

The production model is the 12-feature Plan C XGBoost workflow. Its public
entry point is `python -m src.plan_c`.

```text
src/plan_c/__main__.py
│
├── prepare_dataset()                         src/plan_c/data.py
│   ├── list_recordings(), load_recording()  src/io.py
│   ├── preprocess_recording()               src/features.py
│   │   └── signal filters and detectors     src/preprocess.py
│   ├── windows_for_recording()              src/features.py
│   ├── filter_feature_df()                  src/exclusions.py
│   └── add_delta_features()                 src/pipeline.py
│
├── evaluate_loro()                          src/plan_c/evaluation.py
│   ├── create_model()                       src/plan_c/model.py
│   ├── SafeXGBClassifier                    src/pipeline.py
│   └── metrics + confusion matrix           src/plan_c/evaluation.py
│
└── train_production_model()                 src/plan_c/training.py
    ├── create_model()                       src/plan_c/model.py
    └── model and metadata serialization     src/plan_c/training.py
```

The single source of truth for the 12 features and XGBoost parameters is
`src/plan_c/config.py`.

## Commands

```powershell
# Full main stream: prepare once, evaluate, then train the production model.
python -m src.plan_c run-all --output-dir outputs/plan_c

# Individual main-stream stages.
python -m src.plan_c prepare  --output-dir outputs/plan_c
python -m src.plan_c evaluate --output-dir outputs/plan_c
python -m src.plan_c train    --output-dir outputs/plan_c
```

## Supporting source modules

| Module | Main-stream responsibility |
|---|---|
| `io.py` | Read recording files and protocol metadata |
| `preprocess.py` | Filter ECG, respiration, and microphone signals; detect peaks |
| `features.py` | Create 2-second anchors and calculate physiological features |
| `exclusions.py` | Apply recording/window exclusion rules |
| `pipeline.py` | Legacy shared implementations used by Plan C: feature table, deltas, safe XGBoost wrapper |

## Outside the production main stream

| Module | Purpose |
|---|---|
| `raw_windows.py`, `dl_models.py`, `dl_train.py` | 1D-CNN research path |
| `transfer.py`, `wesad_io.py` | WESAD/transfer experiments |
| `local_eval.py` | Legacy classical-model evaluation |
| `plots.py` | Legacy visualization helpers |

These modules remain available for historical experiments but are not called
by `python -m src.plan_c run-all`.
