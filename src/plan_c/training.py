"""Train and persist the production Plan C model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .config import CLASSES, FEATURES, XGB_PARAMS
from .model import create_model


def train_production_model(df: pd.DataFrame, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = create_model()
    model.fit(df[list(FEATURES)].to_numpy(float), df["activity"].to_numpy())

    joblib_path = output_dir / "xgboost_model.joblib"
    native_path = output_dir / "xgboost_model.json"
    metadata_path = output_dir / "model_metadata.json"
    feature_names_path = output_dir / "feature_names.joblib"
    joblib.dump(model, joblib_path)
    joblib.dump({"feature_names": list(FEATURES), "classes": list(CLASSES)}, feature_names_path)
    model.clf_.save_model(str(native_path))
    metadata = {
        "model": "SafeXGBClassifier",
        "classes": list(CLASSES),
        "features": list(FEATURES),
        "n_recordings": int(df["rec_name"].nunique()),
        "n_windows": int(len(df)),
        "parameters": XGB_PARAMS,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "joblib": str(joblib_path),
        "native_xgboost": str(native_path),
        "metadata": str(metadata_path),
        "feature_names": str(feature_names_path),
    }
