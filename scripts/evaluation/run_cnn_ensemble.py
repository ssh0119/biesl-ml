"""Compare saved CNN folds and Plan C using inner-validation-selected fusion.

Run after python -m src.cnn evaluate. No outer labels select fusion parameters.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cnn.data import load_cache
from src.plan_c.config import CLASSES, FEATURES
from src.plan_c.data import prepare_dataset
from src.plan_c.model import create_model


def fuse(cnn, xgb, weight, geometric=False):
    if geometric:
        p = np.exp(weight * np.log(np.clip(cnn, 1e-8, 1)) +
                   (1 - weight) * np.log(np.clip(xgb, 1e-8, 1)))
        return p / p.sum(axis=1, keepdims=True)
    return weight * cnn + (1 - weight) * xgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-dir", default="output_new/cnn_phase2")
    parser.add_argument("--output-dir", default="output_new/cnn_xgb")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    feature_path = out / "features.csv"
    if not feature_path.exists():
        prepare_dataset().to_csv(feature_path, index=False)
    if args.prepare_only:
        return
    df = pd.read_csv(feature_path)
    root = Path(args.cnn_dir)
    data = load_cache(root / "raw_windows_250hz.npz")
    keys = pd.DataFrame({"rec_name": data["recordings"], "t_start": data["anchors"]})
    aligned = keys.merge(df, on=["rec_name", "t_start"], how="left", validate="one_to_one", indicator=True)
    if not aligned["_merge"].eq("both").all():
        raise ValueError("CNN and feature windows do not match exactly")
    y = data["y"]
    assert np.array_equal(aligned.activity.to_numpy(), np.asarray(CLASSES)[y])
    X = aligned[list(FEATURES)].to_numpy(float)
    X[~np.isfinite(X)] = np.nan
    recordings = data["recordings"]
    methods = ("cnn", "xgboost", "mean", "selected_fusion")
    probabilities = {m: np.full((len(y), len(CLASSES)), np.nan) for m in methods}
    folds = []
    # Include endpoints so validation may retain a standalone model if fusion hurts.
    candidates = [(g, w) for g in (False, True) for w in (0.5, 0.25, 0.75, 0.1, 0.9, 0.0, 1.0)]
    for number, rec in enumerate(sorted(set(recordings.tolist()))):
        fold_dir = root / "folds" / f"{number:02d}"
        with np.load(fold_dir / "probabilities.npz") as saved:
            ti, vi = saved["test_indices"], saved["validation_indices"]
            ct, cv = saved["test_probability"], saved["validation_probability"]
        assert np.array_equal(ti, np.flatnonzero(recordings == rec))
        mask = np.ones(len(y), dtype=bool)
        mask[np.r_[ti, vi]] = False
        assert not set(recordings[mask]) & set(recordings[np.r_[ti, vi]])
        model = create_model()
        model.fit(X[mask], np.asarray(CLASSES)[y[mask]])
        order = [list(model.classes_).index(c) for c in CLASSES]
        xv = model.predict_proba(X[vi])[:, order]
        xt = model.predict_proba(X[ti])[:, order]
        scores = [float(f1_score(y[vi], fuse(cv, xv, w, g).argmax(1),
                                labels=range(5), average="macro", zero_division=0))
                  for g, w in candidates]
        best = int(np.argmax(scores))
        geometric, weight = candidates[best]
        for name, p in zip(methods, (ct, xt, (ct + xt) / 2, fuse(ct, xt, weight, geometric))):
            probabilities[name][ti] = p
        folds.append({"test_recording": rec, "validation_recordings": sorted(set(recordings[vi])),
                      "cnn_weight": weight, "geometric": geometric,
                      "validation_macro_f1": scores[best],
                      "candidate_scores": scores})
        model.clf_.save_model(out / f"xgb_fold_{number:02d}.json")
        print(f"[{number + 1}/30] {rec}: CNN weight={weight}, geometric={geometric}", flush=True)
    results = {}
    fig, axes = plt.subplots(1, 4, figsize=(25, 6))
    frame = keys.copy()
    frame["true_label"] = np.asarray(CLASSES)[y]
    for ax, (name, prob) in zip(axes, probabilities.items()):
        assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=1e-5)
        pred = prob.argmax(1)
        cm = confusion_matrix(y, pred, labels=range(5))
        acc = float(accuracy_score(y, pred))
        f1 = float(f1_score(y, pred, labels=range(5), average="macro", zero_division=0))
        results[name] = {"accuracy": acc, "macro_f1": f1, "confusion_matrix": cm.tolist(),
                         "classification_report": classification_report(y, pred, labels=range(5), target_names=CLASSES, output_dict=True, zero_division=0)}
        pct = cm / cm.sum(1, keepdims=True) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm[i,j]}\n{pct[i,j]:.1f}%", ha="center", va="center", color="white" if pct[i,j] > 55 else "black")
        ax.set_xticks(range(5), CLASSES, rotation=35, ha="right")
        ax.set_yticks(range(5), CLASSES)
        ax.set(xlabel="Predicted", ylabel="True", title=f"{name}\nAccuracy {acc:.2%} | Macro F1 {f1:.4f}")
        frame[f"{name}_prediction"] = np.asarray(CLASSES)[pred]
        for k, label in enumerate(CLASSES):
            frame[f"{name}_prob_{label}"] = prob[:, k]
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=180)
    fig.canvas.draw()
    for ax, name in zip(axes, methods):
        extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out / f"confusion_matrix_{name}.png", dpi=180, bbox_inches=extent.expanded(1.04, 1.04))
    plt.close(fig)
    frame.to_csv(out / "predictions.csv", index=False)
    payload = {"protocol": "30-fold LORO; identical 25 train / 4 validation / 1 test recordings",
               "selection": "Inner validation macro F1; arithmetic/geometric pooling and CNN weights; ties prefer simple equal mean",
               "classes": CLASSES, "n_windows": len(y), "results": results, "folds": folds,
               "saved_xgboost_class_order": sorted(CLASSES),
               "features": FEATURES,
               "candidate_order": [{"geometric": g, "cnn_weight": w} for g, w in candidates],
               "limitations": ["Recording-level evaluation is not subject-independent.",
                               "Offline preprocessing uses whole-recording normalization and known initial rest baseline.",
                               "Validation also selects CNN early stopping; final performance uses outer test only."]}
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: {m: v[m] for m in ("accuracy", "macro_f1")} for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
