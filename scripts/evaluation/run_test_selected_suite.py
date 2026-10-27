"""Deliberately test-selected diagnostic experiment, NOT unbiased LORO.

Train on N-1 recordings, select the CNN checkpoint on the remaining recording,
then evaluate that SAME recording. Keep this experiment separate from the
inner-validation/refit pipeline in run_dataset_suite.py.
"""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import logging
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn

from scripts.evaluation.audit_phase_folders import audit, sha256
from scripts.evaluation.run_cnn_embeddings import VARIANTS, extract
from scripts.evaluation.run_cnn_ensemble import fuse
from src.cnn.config import BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, WEIGHT_DECAY
from src.cnn.data import build_raw_dataset, load_cache, save_cache
from src.cnn.model import Phase2CNN1D
from src.cnn.expanded_model import ExpandedPhase2CNN1D
from src.cnn.training import DEVICE, _class_weights, _loader, predict_probabilities, set_seed
from src.plan_c.config import CLASSES, FEATURES, XGB_PARAMS
from src.plan_c.data import prepare_dataset
from src.plan_c.model import create_model

PROTOCOL = "test_selected_diagnostic_v1"
METHODS = ("cnn", "xgboost", "mean", "test_tuned_fusion", *VARIANTS)
CANDIDATES = [(g, w) for g in (False, True) for w in (.5, .25, .75, .1, .9, 0., 1.)]
LOG = logging.getLogger("test_selected")


def make_cnn(architecture="baseline", in_channels=3):
    if architecture not in ("baseline", "expanded"):
        raise ValueError(f"Unknown CNN architecture: {architecture}")
    cls = Phase2CNN1D if architecture == "baseline" else ExpandedPhase2CNN1D
    return cls(in_channels=in_channels, n_classes=len(CLASSES))


def write_json(path, value):
    """A completion marker appears only after its entire content is written."""
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def score_predictions(y, probability):
    prediction = probability.argmax(1)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, labels=range(len(CLASSES)), average="macro", zero_division=0)),
    }


def train_test_selected(X_train, y_train, X_test, y_test, *, max_epochs, selection_metric,
                        batch_size=BATCH_SIZE, stop_on_perfect=True, patience=0, architecture="baseline"):
    """Test labels select a snapshot, never contribute gradients or class weights.

    Stop at a truly perfect prediction by default. Optional patience stops
    after consecutive non-improvements. Ties retain the earliest epoch.
    """
    if max_epochs < 1 or patience < 0 or selection_metric not in ("accuracy", "macro_f1"):
        raise ValueError("Invalid epoch count or selection metric")
    set_seed()
    model = make_cnn(architecture, X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(y_train, len(CLASSES)))
    loader = _loader(X_train, y_train, training=True, batch_size=batch_size)
    best_score, best_epoch, best_state = -1., None, None
    history = []
    stale = 0
    stop_reason = "max_epochs"
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum, count = 0., 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(xb)
            count += len(xb)
        scheduler.step()
        scores = score_predictions(y_test, predict_probabilities(model, X_test, batch_size=batch_size))
        improved = scores[selection_metric] > best_score
        stale = 0 if improved else stale + 1
        if improved:
            best_score, best_epoch = scores[selection_metric], epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "train_loss": loss_sum / count,
                        "test_accuracy": scores["accuracy"], "test_macro_f1": scores["macro_f1"],
                        "selected_so_far": improved})
        LOG.info("  epoch %02d/%02d loss=%.4f TEST accuracy=%.6f macro_f1=%.4f%s",
                 epoch, max_epochs, loss_sum / count, scores["accuracy"], scores["macro_f1"],
                 " [best]" if improved else "")
        # Use unrounded accuracy: a displayed 1.0000 is not sufficient.
        # All-correct is also the attainable maximum of fixed-five-class F1.
        if stop_on_perfect and scores["accuracy"] == 1.0:
            stop_reason = "perfect_test_predictions"
            LOG.info("  STOP: every test window correct; selected epoch=%d", best_epoch)
            break
        if patience and stale >= patience:
            stop_reason = "no_improvement_patience"
            LOG.info("  STOP: no improvement for %d epochs; selected epoch=%d", patience, best_epoch)
            break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"selection_source": "same_test_recording_as_reported_score",
                   "selection_metric": selection_metric, "best_epoch": best_epoch,
                   "best_test_score": best_score, "epochs_scanned": len(history),
                   "max_epochs": max_epochs, "stop_on_perfect": stop_on_perfect,
                   "patience": patience, "stop_reason": stop_reason,
                   "tie_policy": "earliest_epoch", "early_stopping": len(history) < max_epochs, "history": history}


def source_identity(manifest):
    return sorted((m["name"], m["phase2"], m["sha256"]) for m in manifest)


def reusable_cache(cache_dir, checked):
    """Reuse preprocessing only if inputs and preprocessing code still match."""
    required = ("raw_windows_250hz.npz", "features.csv", "run_config.json")
    if not all((cache_dir / p).is_file() for p in required):
        return False
    previous = json.loads((cache_dir / "run_config.json").read_text(encoding="utf-8"))
    if source_identity(previous["manifest"]) != source_identity(checked["manifest"]):
        return False
    if previous["spreadsheet_sha256"] != checked["spreadsheet_sha256"] or previous["aliases"] != checked["aliases"]:
        return False
    for name in ("src/io.py", "src/preprocess.py", "src/features.py", "src/pipeline.py",
                 "src/exclusions.py", "src/raw_windows.py", "src/cnn/data.py", "src/cnn/config.py",
                 "src/plan_c/config.py", "src/plan_c/data.py"):
        if previous["code_hashes"].get(name.replace("/", "\\"), previous["code_hashes"].get(name)) != sha256(name):
            return False
    return True


def load_aligned(raw_path, feature_path, checked):
    data = load_cache(raw_path)
    y, recordings = data["y"], data["recordings"].astype(str)
    expected = {m["name"]: m["phase2"] for m in checked["manifest"]}
    if set(recordings) != set(expected) or not np.isfinite(data["X"]).all():
        raise ValueError("Cached recording set or signals are invalid")
    for rec, label in expected.items():
        if set(y[recordings == rec]) != {0, CLASSES.index(label)}:
            raise ValueError(f"Invalid cached labels for {rec}")
    keys = pd.DataFrame({"rec_name": recordings, "t_start": data["anchors"]})
    features = pd.read_csv(feature_path)
    aligned = keys.merge(features, on=["rec_name", "t_start"], how="left", validate="one_to_one", indicator=True)
    if len(features) != len(keys) or not aligned["_merge"].eq("both").all():
        raise ValueError("Feature and CNN windows do not match exactly")
    if not np.array_equal(aligned.activity.to_numpy(), np.asarray(CLASSES)[y]):
        raise ValueError("Feature and CNN labels disagree")
    X = aligned[list(FEATURES)].to_numpy(float)
    X[~np.isfinite(X)] = np.nan
    return data, X


def proba(model, X):
    return model.predict_proba(X)[:, [list(model.classes_).index(c) for c in CLASSES]]


def save_report(out, data, probabilities, folds, config):
    y = data["y"]
    frame = pd.DataFrame({"recording": data["recordings"], "anchor_seconds": data["anchors"],
                          "true_label": np.asarray(CLASSES)[y]})
    if frame.duplicated(["recording", "anchor_seconds"]).any():
        raise ValueError("Duplicate test windows")
    results = {}
    fig, axes = plt.subplots(3, 3, figsize=(22, 20))
    for ax, (name, p) in zip(axes.flat, probabilities.items()):
        if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=1e-5):
            raise ValueError(f"Incomplete or invalid probabilities: {name}")
        pred = p.argmax(1)
        cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
        results[name] = {**score_predictions(y, p), "confusion_matrix": cm.tolist(),
                         "selection_bias": "none from CNN selection" if name == "xgboost" else "test-selected CNN",
                         "classification_report": classification_report(y, pred, labels=range(5), target_names=CLASSES, output_dict=True, zero_division=0)}
        if name == "test_tuned_fusion":
            results[name]["selection_bias"] = "test-selected CNN AND test-selected fusion weight"
        pct = np.divide(cm * 100., cm.sum(1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(1, keepdims=True) != 0)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm[i,j]}\n{pct[i,j]:.1f}%", ha="center", va="center", color="white" if pct[i,j] > 55 else "black")
        ax.set_xticks(range(5), CLASSES, rotation=30, ha="right")
        ax.set_yticks(range(5), CLASSES)
        tag = "Fixed XGB baseline (no epoch selection)" if name == "xgboost" else "TEST-SELECTED / OPTIMISTIC"
        ax.set(xlabel="Predicted", ylabel="True", title=f"{name}\n{tag}\nAccuracy {results[name]['accuracy']:.2%} | Macro F1 {results[name]['macro_f1']:.4f}")
        frame[f"{name}_prediction"] = np.asarray(CLASSES)[pred]
        for k, label in enumerate(CLASSES):
            frame[f"{name}_prob_{label}"] = p[:, k]
    for ax in list(axes.flat)[len(probabilities):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=150)
    fig.canvas.draw()
    for ax, name in zip(axes.flat, probabilities):
        box = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out / f"confusion_matrix_{name}.png", dpi=180, bbox_inches=box.expanded(1.04, 1.04))
    plt.close(fig)
    frame.to_csv(out / "predictions.csv", index=False)
    payload = {"protocol": PROTOCOL, "unbiased_test_evaluation": False, "classes": CLASSES,
               "n_folds": len(folds), "n_windows": len(y), "config": config, "folds": folds, "results": results}
    write_json(out / "results.json", payload)
    n = len(folds)
    architecture = config.get("architecture", "baseline")
    text = f"# 테스트 선택 비교 실험 결과\n\n**독립 테스트 성능이 아닙니다.** {n-1}개 학습 / 1개 테스트, {n}-fold. 같은 테스트 녹화로 CNN epoch를 선택하고 점수도 계산했습니다.\n\n"
    text += f"CNN 구조: `{architecture}`.\n\n"
    text += f"선택 기준: `{config['selection_metric']}`, 최대 {config['max_epochs']} epoch, 정확히 100%에서 종료={config.get('stop_on_perfect', False)}, 개선 없음 patience={config.get('patience', 0)}(0은 비활성). 동점은 먼저 나온 epoch. 선택된 가중치를 그대로 사용하며 추가 재학습하지 않습니다.\n\n"
    text += "| 방식 | 정확도 | Macro F1 |\n|---|---:|---:|\n"
    for name, result in results.items():
        text += f"| {name} | {result['accuracy']:.2%} | {result['macro_f1']:.4f} |\n"
    text += "\nCNN에서 고른 epoch를 모든 CNN 기반 결합에 공통 적용했습니다. 중간 특징 XGBoost를 매 epoch 학습해서 최적 epoch를 고르는 실험은 아닙니다. `mean`은 고정 50:50 평균, `test_tuned_fusion`은 같은 테스트 정답으로 비중까지 고른 추가 편향 비교입니다. 단독 XGBoost는 고정 설정으로 학습하므로 테스트 epoch 선택의 영향을 받지 않습니다.\n"
    text += "\n## Confusion matrices\n\n![전체 비교](confusion_matrices.png)\n\n"
    text += "행은 실제 클래스, 열은 예측 클래스입니다. 숫자는 개수와 행별 비율입니다. 전체 점수는 모든 테스트 예측을 합산하며 fold 점수의 단순 평균이 아닙니다.\n"
    text += "\n## 기록\n\n`folds/XX/cnn_selection.json`에 epoch별 train loss/test 정확도/test Macro F1과 선택 epoch를 저장했습니다. `cnn_model.pt`는 선택된 checkpoint이며 마지막 epoch의 모델이 아닙니다. `complete.json`에는 학습 목록, 테스트 녹화, 테스트 선택 비중을 저장했습니다.\n"
    text += "\n## 비교 해석\n\n원래 내부 검증 방식 결과는 `output_data_new`에 그대로 있습니다. 이번 점수가 높아도 모델의 일반화 성능이 개선됐다고 결론내릴 수 없습니다. 선배 실험의 최대 epoch, 정확도 집계 방식, 전처리, 분할이 같아야 방식 차이를 비교할 수 있습니다. 91% 재현을 보장하는 실험이 아닙니다.\n"
    (out / "README.md").write_text(text, encoding="utf-8")
    LOG.info("COMPLETE: %s", json.dumps({k: score_predictions(y, p) for k, p in probabilities.items()}))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Experimental TEST-selected CNN and fusion suite; not unbiased LORO")
    parser.add_argument("--data-dir", default="data_new")
    parser.add_argument("--output-dir", default="output_data_new_test_selected_fast")
    parser.add_argument("--aliases", default="output_data_new/label_aliases.json")
    parser.add_argument("--cache-dir", default="output_data_new", help="Reuse only verified preprocessing, never trained models")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--architecture", choices=("baseline", "expanded"), default="baseline")
    parser.add_argument("--no-stop-on-perfect", action="store_true", help="Continue even after every test window is correct")
    parser.add_argument("--patience", type=int, default=0, help="Stop after this many epochs without strict score improvement; 0 disables")
    parser.add_argument("--selection-metric", choices=("accuracy", "macro_f1"), default="accuracy")
    parser.add_argument("--prepare-only", action="store_true", help="Audit and validate inputs without training")
    args = parser.parse_args(argv)
    if args.max_epochs < 1 or args.patience < 0:
        parser.error("--max-epochs must be positive and --patience nonnegative")
    out = Path(args.output_dir)
    # Protect historical outputs and input directories before writing anything.
    protected = [Path(args.data_dir), Path(args.cache_dir), *(Path(p) for p in ("output_new", "output_new_1", "output_new_2", "output_data_new"))]
    if any(out.resolve() == p.resolve() or p.resolve() in out.resolve().parents for p in protected):
        parser.error("Use a separate experimental output directory, outside data and historical outputs")
    if out.exists() and any(out.iterdir()) and not (out / "run_config.json").exists():
        parser.error("Nonempty output directory has no experiment manifest; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "training.log", encoding="utf-8")], force=True)
    LOG.info("TEST-SELECTED DIAGNOSTIC: same recording chooses epochs and supplies reported scores")
    if DEVICE != "cuda":
        raise RuntimeError("CUDA is required. Run the project .venv Python with CUDA PyTorch.")
    aliases = json.loads(Path(args.aliases).read_text(encoding="utf-8"))
    checked = audit(args.data_dir, out, aliases)
    if checked["errors"]:
        raise ValueError(f"Dataset audit failed: {checked['errors']}")
    cache_dir = Path(args.cache_dir)
    reuse = reusable_cache(cache_dir, checked)
    raw_path, feature_path = (cache_dir if reuse else out) / "raw_windows_250hz.npz", (cache_dir if reuse else out) / "features.csv"
    sources = [Path(__file__), Path("scripts/evaluation/audit_phase_folders.py"), Path("scripts/evaluation/run_cnn_embeddings.py"), Path("scripts/evaluation/run_cnn_ensemble.py"), *Path("src").rglob("*.py")]
    config = {"protocol": PROTOCOL, "unbiased_test_evaluation": False, "selection_metric": args.selection_metric,
              "max_epochs": args.max_epochs, "architecture": args.architecture, "stop_on_perfect": not args.no_stop_on_perfect,
              "patience": args.patience, "tie_policy": "earliest_epoch",
              "manifest": checked["manifest"], "spreadsheet_sha256": checked["spreadsheet_sha256"], "aliases": aliases,
              "torch": torch.__version__, "xgboost": xgboost.__version__, "xgb_params": XGB_PARAMS,
              "code_hashes": {str(p): sha256(p) for p in sources}, "classes": CLASSES, "features": FEATURES,
              "variants": VARIANTS, "fusion_candidates": CANDIDATES, "reuse_preprocessing": reuse,
              "cache_dir": str(raw_path.parent.resolve())}
    # Reject incompatible resumes before rebuilding caches.
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["signature"] = signature
    config_path = out / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing["signature"] != signature:
            raise ValueError("Code/data/settings changed. Use a different --output-dir; historical folds were not overwritten.")
    else:
        write_json(config_path, config)
    LOG.info("Preprocessing cache: %s (%s)", raw_path.parent, "verified reuse" if reuse else "build locally")
    if not raw_path.exists():
        save_cache(build_raw_dataset(args.data_dir), raw_path)
    if not feature_path.exists():
        prepare_dataset(args.data_dir).to_csv(feature_path, index=False)
    data, X = load_aligned(raw_path, feature_path, checked)
    y, recordings = data["y"], data["recordings"].astype(str)
    all_recordings = sorted(set(recordings))
    if len(all_recordings) < 2:
        raise ValueError("At least two recordings required")
    data_hashes = {"raw_sha256": sha256(raw_path), "features_sha256": sha256(feature_path)}
    hashes_path = out / "preprocessing_hashes.json"
    if hashes_path.exists() and json.loads(hashes_path.read_text()) != data_hashes:
        raise ValueError("Preprocessing files changed; refusing to mix predictions")
    write_json(hashes_path, data_hashes)
    LOG.info("READY: %d recordings; %d windows; %s; GPU=%s", len(all_recordings), len(y), np.bincount(y).tolist(), torch.cuda.get_device_name(0))
    if args.prepare_only:
        LOG.info("Preparation verified. No model training performed.")
        return
    predictions = {name: np.full((len(y), len(CLASSES)), np.nan) for name in METHODS}
    folds = []
    for i, rec in enumerate(all_recordings):
        folder = out / "folds" / f"{i:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        train, ti = recordings != rec, np.flatnonzero(recordings == rec)
        train_names = sorted(set(recordings[train]))
        if len(train_names) != len(all_recordings) - 1 or rec in train_names:
            raise ValueError("Invalid split")
        LOG.info("[%02d/%02d] train=%d TEST/selection=%s", i + 1, len(all_recordings), len(train_names), rec)
        done = folder / "complete.json"
        if done.exists():
            metadata = json.loads(done.read_text())
            if metadata["signature"] != signature or metadata["training_recordings"] != train_names or metadata["test_recording"] != rec:
                raise ValueError("Completed fold metadata mismatch")
            with np.load(folder / "probabilities.npz") as saved:
                if not np.array_equal(saved["test_indices"], ti):
                    raise ValueError("Completed fold test indices mismatch")
                for name in METHODS:
                    predictions[name][ti] = saved[name]
            folds.append(metadata)
            LOG.info("  reused completed fold")
            continue
        checkpoint, selection_path = folder / "cnn_model.pt", folder / "cnn_selection.json"
        if checkpoint.exists() and selection_path.exists():
            selection = json.loads(selection_path.read_text())
            if selection["signature"] != signature:
                raise ValueError("CNN selection metadata mismatch")
            cnn = make_cnn(args.architecture).to(DEVICE)
            cnn.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
        else:
            cnn, selection = train_test_selected(data["X"][train], y[train], data["X"][ti], y[ti],
                                                 max_epochs=args.max_epochs, selection_metric=args.selection_metric,
                                                 stop_on_perfect=not args.no_stop_on_perfect, patience=args.patience,
                                                 architecture=args.architecture)
            selection.update(signature=signature, test_recording=rec)
            temp = checkpoint.with_suffix(".tmp")
            torch.save(cnn.state_dict(), temp)
            temp.replace(checkpoint)
            write_json(selection_path, selection)
        tr96, tr256, _ = extract(cnn, data["X"][train])
        te96, te256, ct = extract(cnn, data["X"][ti])
        if not np.isclose(score_predictions(y[ti], ct)[args.selection_metric], selection["best_test_score"], atol=1e-6):
            raise ValueError("Restored CNN does not match selected test score")
        del cnn
        model = create_model()
        model.fit(X[train], np.asarray(CLASSES)[y[train]])
        xt = proba(model, X[ti])
        model.clf_.save_model(folder / "xgboost_model.json")
        scores = [score_predictions(y[ti], fuse(ct, xt, w, g))[args.selection_metric] for g, w in CANDIDATES]
        geometric, weight = CANDIDATES[int(np.argmax(scores))]
        outputs = {"cnn": ct, "xgboost": xt, "mean": (ct + xt) / 2,
                   "test_tuned_fusion": fuse(ct, xt, weight, geometric)}
        for name, (include_features, dimension) in VARIANTS.items():
            tr, te = (tr96, te96) if dimension == 96 else (tr256, te256)
            names = [f"cnn_{dimension}_{j:03d}" for j in range(dimension)]
            if include_features:
                tr, te = np.column_stack((X[train], tr)), np.column_stack((X[ti], te))
                names = list(FEATURES) + names
            model = create_model()
            model.fit(tr, np.asarray(CLASSES)[y[train]])
            outputs[name] = proba(model, te)
            model.clf_.save_model(folder / f"{name}.json")
            write_json(folder / f"{name}_features.json", {"features": names, "classes": model.classes_.tolist(), "importance": model.clf_.feature_importances_.tolist()})
        metadata = {"signature": signature, "protocol": PROTOCOL, "test_recording": rec,
                    "training_recordings": train_names, "n_train_windows": int(train.sum()), "n_test_windows": len(ti),
                    "best_epoch": selection["best_epoch"], "selection_metric": args.selection_metric,
                    "best_test_score": selection["best_test_score"], "fusion_selection_source": "same_test_recording",
                    "cnn_weight": weight, "geometric": geometric, "fusion_candidate_scores": scores}
        np.savez(folder / "probabilities.npz", test_indices=ti, **outputs)
        write_json(done, metadata)
        for name in METHODS:
            predictions[name][ti] = outputs[name]
        folds.append(metadata)
        LOG.info("  saved best epoch=%d, test score=%.4f", selection["best_epoch"], selection["best_test_score"])
        gc.collect()
    save_report(out, data, predictions, folds, config)


if __name__ == "__main__":
    main()
