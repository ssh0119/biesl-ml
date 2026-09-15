"""Plan C XGBoost model factory."""

from ..pipeline import SafeXGBClassifier
from .config import XGB_PARAMS


def create_model() -> SafeXGBClassifier:
    return SafeXGBClassifier(**XGB_PARAMS)
