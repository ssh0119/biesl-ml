"""Build the model-ready Plan C feature table."""
from __future__ import annotations

import pandas as pd

from ..exclusions import filter_feature_df
from ..pipeline import add_delta_features, build_feature_table
from .config import CLASSES, FEATURES


def prepare_dataset(data_dir: str = "data") -> pd.DataFrame:
    """Load recordings, extract features, apply exclusions, and add deltas."""
    df = build_feature_table(data_dir=data_dir, include_temp=False)
    df = filter_feature_df(df)
    df = df[df["activity"].isin(CLASSES)].copy()
    df = add_delta_features(df)

    missing = sorted(set(FEATURES) - set(df.columns))
    if missing:
        raise ValueError(f"Plan C feature table is missing columns: {missing}")
    return df
