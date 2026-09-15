"""Single source of truth for the production Plan C model."""

CLASSES = ("rest", "mental1", "mental2", "physical1", "physical2")

BASE_FEATURES = (
    "hr",
    "hrv_rmssd",
    "mic_rms",
    "resp_amp",
    "rr",
    "rrv",
    "ss",
)

DELTA_FEATURES = (
    "delta_hr",
    "delta_hrv_rmssd",
    "delta_ss",
    "delta_rr",
    "delta_resp_amp",
)

FEATURES = BASE_FEATURES + DELTA_FEATURES

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "random_state": 0,
    "n_jobs": -1,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "device": "cpu",
    "class_weight": "balanced",
}
