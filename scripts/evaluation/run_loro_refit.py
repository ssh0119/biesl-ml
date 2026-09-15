"""Refit both models on all 29 outer-training recordings after inner selection."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.cnn.config import MAX_EPOCHS
from src.cnn.data import load_cache, build_raw_dataset, save_cache
from src.cnn.evaluation import select_validation_recordings
from src.cnn.training import DEVICE, predict_probabilities, train_fixed_epochs
from src.plan_c.config import CLASSES, FEATURES, XGB_PARAMS
from src.plan_c.model import create_model
from scripts.evaluation.run_cnn_ensemble import fuse


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def report(out, data, probabilities, folds):
    y = data["y"]
    frame = pd.DataFrame({"recording": data["recordings"], "anchor_seconds": data["anchors"],
                          "true_label": np.asarray(CLASSES)[y]})
    assert not frame.duplicated(["recording", "anchor_seconds"]).any()
    results = {}
    fig, axes = plt.subplots(1, len(probabilities), figsize=(25, 6))
    for ax, (name, prob) in zip(axes, probabilities.items()):
        assert np.isfinite(prob).all() and np.allclose(prob.sum(1), 1, atol=1e-5)
        pred = prob.argmax(1)
        cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
        acc = float(accuracy_score(y, pred))
        f1 = float(f1_score(y, pred, average="macro", labels=range(5), zero_division=0))
        results[name] = {"accuracy": acc, "macro_f1": f1, "confusion_matrix": cm.tolist(),
                         "classification_report": classification_report(y, pred, labels=range(5), target_names=CLASSES, output_dict=True, zero_division=0)}
        pct = cm / cm.sum(1, keepdims=True) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm[i,j]}\n{pct[i,j]:.1f}%", ha="center", va="center", color="white" if pct[i,j] > 55 else "black")
        ax.set_xticks(range(5), CLASSES, rotation=35, ha="right")
        ax.set_yticks(range(5), CLASSES)
        ax.set(xlabel="Predicted", ylabel="True", title=f"{name} | 29 train / 1 test\nAccuracy {acc:.2%} | Macro F1 {f1:.4f}")
        frame[f"{name}_prediction"] = np.asarray(CLASSES)[pred]
        for k, label in enumerate(CLASSES):
            frame[f"{name}_prob_{label}"] = prob[:, k]
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=180)
    fig.canvas.draw()
    for ax, name in zip(axes, probabilities):
        extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out / f"confusion_matrix_{name}.png", dpi=180, bbox_inches=extent.expanded(1.04, 1.04))
    plt.close(fig)
    frame.to_csv(out / "predictions.csv", index=False)
    payload = {"protocol": "30-fold LORO; final CNN and XGBoost each fit on 29 recordings, test on 1",
               "selection": "Reuse each fold's previously computed inner-validation epoch and fusion weight; refit from scratch on all 29 outer-training recordings. No outer-test scores select settings.",
               "classes": CLASSES, "features": FEATURES, "saved_xgboost_class_order": sorted(CLASSES),
               "n_windows": len(y), "n_folds": len(folds), "results": results, "folds": folds,
               "limitations": ["Same historical dataset, not a new independent confirmation set.",
                               "Recording-level, not subject-independent evaluation.",
                               "Offline whole-recording signal normalization and initial rest calibration.",
                               "Inner-selected fusion weights may transfer imperfectly to refitted models."]}
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = "\n".join(f"| {name} | {r['accuracy']:.2%} | {r['macro_f1']:.4f} |" for name, r in results.items())
    text = "# 29개 학습 / 1개 테스트: 30-fold 결과\n\n" + "총 6,059개 윈도우, rest 포함 5개 클래스. 각 녹화를 한 번씩 테스트했습니다.\n\n"
    text += "| 모델 | 정확도 | Macro F1 |\n|---|---:|---:|\n" + rows
    cnn_result, mean_result = results["cnn"], results["mean"]
    text += f"\n\n단순 평균 결합은 CNN 단독 대비 정확도 {100*(mean_result['accuracy']-cnn_result['accuracy']):+.2f}%p, Macro F1 {mean_result['macro_f1']-cnn_result['macro_f1']:+.4f}입니다. 이번 비교에서는 단순 평균이 두 지표에서 가장 높았습니다. 테스트 결과로 고른 후보의 우수성은 새로운 독립 데이터에서 추가 확인해야 합니다.\n"
    text += "\n단순 평균은 `0.5 × CNN 확률 + 0.5 × XGBoost 확률`입니다. selected_fusion은 내부 검증으로 고른 산술/기하 평균과 가중치를 사용하는 별도 방식입니다.\n"
    text += "\n## 이전 25개 학습과 비교\n\n| 모델 | 이전 정확도 / Macro F1 | 현재 정확도 / Macro F1 |\n|---|---|---|\n"
    old_results = json.loads((out / "inner_selection/selection_results.json").read_text())["results"]
    for name, r in results.items():
        old = old_results[name]
        text += f"| {name} | {old['accuracy']:.2%} / {old['macro_f1']:.4f} | {r['accuracy']:.2%} / {r['macro_f1']:.4f} |\n"
    text += "\n## 클래스별 재현율\n\n| 클래스 | CNN | XGBoost | 단순 평균 | 검증 선택 결합 |\n|---|---:|---:|---:|---:|\n"
    for label in CLASSES:
        text += "| " + label + " | " + " | ".join(f"{r['classification_report'][label]['recall']:.1%}" for r in results.values()) + " |\n"
    text += "\n\n## 학습 방식\n\n각 fold의 기존 내부 검증에서 정한 epoch와 결합 가중치를 재사용했습니다. 이후 CNN과 XGBoost 모두 나머지 29개 녹화 전체로 처음부터 재학습했습니다. 내부 검증용 4개도 최종 학습에 포함됩니다. 테스트 녹화는 설정 선택과 학습에서 제외했습니다. CNN 학습률 스케줄은 내부 검증과 같은 24-epoch 주기를 유지합니다.\n"
    text += "\n## Confusion matrices\n\n![전체 비교](confusion_matrices.png)\n\n행은 실제 클래스, 열은 예측 클래스이며 각 셀은 개수와 행별 비율입니다.\n"
    text += "\n### CNN\n\n![CNN](confusion_matrix_cnn.png)\n\n### CNN + XGBoost 단순 평균\n\n![단순 평균](confusion_matrix_mean.png)\n"
    text += "\n## 파일\n\n- `folds/`: fold별 CNN, XGBoost 모델, 학습 파일 목록, 테스트 확률\n- `results.json`: 전체·클래스별 지표 및 fold 설정\n- `predictions.csv`: 모든 out-of-fold 예측과 확률\n- `inner_selection/`: 재사용한 내부 검증 기록\n- `run_config.json`: 입력 해시와 실행 설정\n"
    text += "\n## 재실행\n\n```powershell\n./.venv/Scripts/python.exe -u -m scripts.evaluation.run_loro_refit\n```\n완료된 fold는 동일 설정일 때만 재사용합니다.\n"
    text += "\n원본 데이터는 기존 `data/Mental1`, `Mental2`, `Physical1`, `Physical2` 구조로 준비합니다. 원시 신호 캐시가 없으면 자동 생성합니다. 실제 패키지 버전은 `runtime-packages.txt`에 기록했습니다. 입력 해시 또는 환경이 다르면 기존 체크포인트를 재사용하지 않습니다. 모델의 입력 순서는 `results.json`의 `features`, XGBoost 출력 순서는 `saved_xgboost_class_order`, CNN 출력 순서는 `classes`를 확인합니다.\n"
    text += "\n## 해석 범위\n\n동일 참가자의 다른 녹화가 학습에 포함될 수 있습니다. 전체 녹화 정규화와 초기 rest 보정을 사용하는 오프라인 평가이며, 새로운 독립 데이터에서 확인한 결과는 아닙니다. 결합이 모든 클래스에서 개선된다고 보장하지 않습니다.\n"
    (out / "README.md").write_text(text, encoding="utf-8")
    print(json.dumps({k: {m: v[m] for m in ("accuracy", "macro_f1")} for k, v in results.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output_new_1")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path("output_new/cnn_phase2")
    source_features = Path("output_new/cnn_xgb/features.csv")
    source_selection = Path("output_new/cnn_xgb/results.json")
    cache = root / "raw_windows_250hz.npz"
    # Published artifacts also support reruns from a clean checkout plus raw data.
    if not source_features.exists():
        source_features = out / "features.csv"
    if not source_selection.exists():
        source_selection = out / "inner_selection/selection_results.json"
    if not cache.exists():
        cache = out / "raw_windows_250hz.npz"
        if not cache.exists():
            save_cache(build_raw_dataset("data"), cache)
    config = {"protocol": "29 train / 1 test refit", "cache_sha256": digest(cache),
              "features_sha256": digest(source_features), "inner_selection_sha256": digest(source_selection),
              "torch": torch.__version__, "device": DEVICE, "scheduler_epochs": MAX_EPOCHS,
              "xgb_params": XGB_PARAMS, "classes": CLASSES, "features": FEATURES}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["signature"] = signature
    config_path = out / "run_config.json"
    if config_path.exists() and json.loads(config_path.read_text())["signature"] != signature:
        raise ValueError("Existing output belongs to a different configuration")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    assert DEVICE == "cuda", "CUDA required for this run"
    if source_features.resolve() != (out / "features.csv").resolve():
        shutil.copy2(source_features, out / "features.csv")
    inner = out / "inner_selection"
    inner.mkdir(exist_ok=True)
    if source_selection.resolve() != (inner / "selection_results.json").resolve():
        shutil.copy2(source_selection, inner / "selection_results.json")
    data = load_cache(cache)
    df = pd.read_csv(source_features)
    keys = pd.DataFrame({"rec_name": data["recordings"], "t_start": data["anchors"]})
    aligned = keys.merge(df, on=["rec_name", "t_start"], how="left", validate="one_to_one", indicator=True)
    assert aligned["_merge"].eq("both").all()
    y, recordings = data["y"], data["recordings"]
    assert np.array_equal(aligned.activity.to_numpy(), np.asarray(CLASSES)[y])
    X = aligned[list(FEATURES)].to_numpy(float)
    X[~np.isfinite(X)] = np.nan
    selected = {f["test_recording"]: f for f in json.loads(source_selection.read_text())["folds"]}
    all_recordings = sorted(set(recordings.tolist()))
    assert len(all_recordings) == 30
    probabilities = {m: np.full((len(y), 5), np.nan) for m in ("cnn", "xgboost", "mean", "selected_fusion")}
    folds = []
    for number, rec in enumerate(all_recordings):
        folder = out / "folds" / f"{number:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        training_path = root / "folds" / f"{number:02d}" / "training.json"
        if not training_path.exists():
            training_path = inner / f"cnn_training_{number:02d}.json"
        training = json.loads(training_path.read_text())
        val = select_validation_recordings(recordings, y, rec)
        selection = selected[rec]
        assert training["test_recording"] == rec
        assert set(training["validation_recordings"]) == set(val) == set(selection["validation_recordings"])
        if training_path.resolve() != (inner / f"cnn_training_{number:02d}.json").resolve():
            shutil.copy2(training_path, inner / f"cnn_training_{number:02d}.json")
        ti = np.flatnonzero(recordings == rec)
        train = recordings != rec
        train_names = sorted(set(recordings[train].tolist()))
        assert len(train_names) == 29 and rec not in train_names and set(val) <= set(train_names)
        epochs = int(training["best_epoch"])
        print(f"[{number+1:02d}/30] test={rec} train_recordings=29 refit_epochs={epochs}", flush=True)
        done = folder / "complete.json"
        if done.exists():
            metadata = json.loads(done.read_text())
            assert metadata["signature"] == signature and metadata["test_recording"] == rec
            assert metadata["epochs"] == epochs and metadata["training_recordings"] == train_names
            with np.load(folder / "probabilities.npz") as saved:
                assert np.array_equal(saved["test_indices"], ti)
                ct, xt = saved["cnn"], saved["xgboost"]
        else:
            cnn = train_fixed_epochs(data["X"][train], y[train], n_classes=5,
                                     epochs=epochs, scheduler_epochs=MAX_EPOCHS)
            ct = predict_probabilities(cnn, data["X"][ti])
            torch.save(cnn.state_dict(), folder / "cnn_model.pt")
            del cnn
            model = create_model()
            model.fit(X[train], np.asarray(CLASSES)[y[train]])
            order = [list(model.classes_).index(c) for c in CLASSES]
            xt = model.predict_proba(X[ti])[:, order]
            model.clf_.save_model(folder / "xgboost_model.json")
            np.savez(folder / "probabilities.npz", test_indices=ti, cnn=ct, xgboost=xt)
            metadata = {"signature": signature, "test_recording": rec, "training_recordings": train_names,
                        "n_train_recordings": 29, "n_train_windows": int(train.sum()), "n_test_windows": len(ti),
                        "epochs": epochs, "inner_validation_recordings": val,
                        "cnn_weight": selection["cnn_weight"], "geometric": selection["geometric"],
                        "inner_validation_macro_f1": selection["validation_macro_f1"]}
            done.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        folds.append(metadata)
        combined = fuse(ct, xt, selection["cnn_weight"], selection["geometric"])
        for name, prob in zip(probabilities, (ct, xt, (ct+xt)/2, combined)):
            probabilities[name][ti] = prob
        print(f"  saved fold {number+1:02d}", flush=True)
    report(out, data, probabilities, folds)


if __name__ == "__main__":
    main()
