"""Frozen fold-specific CNN representations -> XGBoost; 29 train / 1 test."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.cnn.data import load_cache
from src.cnn.model import Phase2CNN1D
from src.plan_c.config import CLASSES, FEATURES, XGB_PARAMS
from src.plan_c.model import create_model

VARIANTS = {"embedding96_xgb": (False, 96), "features12_embedding96_xgb": (True, 96),
            "features12_pool256_xgb": (True, 256)}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@torch.inference_mode()
def extract(model, signals, batch_size=32):
    """One shared fold encoder for both training and held-out signals."""
    model.eval()
    pooled, embeddings, probabilities = [], [], []
    for start in range(0, len(signals), batch_size):
        x = torch.from_numpy(signals[start:start+batch_size]).to("cuda")
        z = model.features(x)
        p = torch.cat((model.avg_pool(z), model.max_pool(z)), dim=1).squeeze(-1)
        e = model.classifier[:3](p)  # 256 -> 96 -> ReLU; dropout disabled
        logits = model.classifier[3:](e)
        pooled.append(p.cpu().numpy())
        embeddings.append(e.cpu().numpy())
        probabilities.append(logits.softmax(1).cpu().numpy())
    return np.concatenate(embeddings), np.concatenate(pooled), np.concatenate(probabilities)


def summarize(out, data, probabilities, folds, config):
    y = data["y"]
    frame = pd.DataFrame({"recording": data["recordings"], "anchor_seconds": data["anchors"],
                          "true_label": np.asarray(CLASSES)[y]})
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(21, 13))
    for ax, (name, probability) in zip(axes.flat, probabilities.items()):
        assert np.isfinite(probability).all() and np.allclose(probability.sum(1), 1, atol=1e-5)
        pred = probability.argmax(1)
        cm = confusion_matrix(y, pred, labels=range(5))
        acc = float(accuracy_score(y, pred))
        f1 = float(f1_score(y, pred, labels=range(5), average="macro", zero_division=0))
        results[name] = {"accuracy": acc, "macro_f1": f1, "confusion_matrix": cm.tolist(),
                         "classification_report": classification_report(y, pred, labels=range(5), target_names=CLASSES, output_dict=True, zero_division=0)}
        pct = cm / cm.sum(1, keepdims=True) * 100
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f"{cm[i,j]}\n{pct[i,j]:.1f}%", ha="center", va="center", color="white" if pct[i,j]>55 else "black")
        ax.set_xticks(range(5), CLASSES, rotation=30, ha="right")
        ax.set_yticks(range(5), CLASSES)
        ax.set(xlabel="Predicted", ylabel="True", title=f"{name}\nAccuracy {acc:.2%} | Macro F1 {f1:.4f}")
        frame[f"{name}_prediction"] = np.asarray(CLASSES)[pred]
        for k, label in enumerate(CLASSES):
            frame[f"{name}_prob_{label}"] = probability[:,k]
    fig.tight_layout()
    fig.savefig(out / "confusion_matrices.png", dpi=160)
    fig.canvas.draw()
    for ax, name in zip(axes.flat, probabilities):
        extent = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out / f"confusion_matrix_{name}.png", dpi=180, bbox_inches=extent.expanded(1.04,1.04))
    plt.close(fig)
    frame.to_csv(out / "predictions.csv", index=False)
    payload = {"protocol": "30-fold LORO; 29 training recordings / 1 test recording",
               "classes": CLASSES, "results": results, "folds": folds, "config": config,
               "n_windows": len(y), "n_folds": len(folds),
               "selection": "Three fixed feature representations and unchanged Plan C XGBoost settings; no tuning on outer-test scores. Comparing the best observed result is exploratory."}
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = "\n".join(f"| {n} | {r['accuracy']:.2%} | {r['macro_f1']:.4f} |" for n,r in results.items())
    best = max(results, key=lambda n: results[n]["macro_f1"])
    text = "# CNN 중간 특징 + XGBoost 결과 및 변경 보고서\n\n"
    text += "## 결과\n\n| 모델 | 정확도 | Macro F1 |\n|---|---:|---:|\n" + rows
    text += f"\n\n이번 비교에서 Macro F1이 가장 높은 방식은 **{best}**입니다. 동일 데이터에서 여러 방식을 비교한 탐색 결과이며, 새로운 독립 데이터에서 우수성을 확인한 것은 아닙니다.\n"
    text += "\n## 무엇을 바꿨나\n\n기존 출력 확률 평균 대신 CNN 내부 숫자 표현을 XGBoost의 입력 특징으로 사용했습니다.\n\n"
    text += "| 새 방식 | 입력 | 특징 개수 |\n|---|---|---:|\n| embedding96_xgb | CNN 마지막 은닉층 ReLU 출력 | 96 |\n| features12_embedding96_xgb | Plan C 생체 특징 + 마지막 은닉층 | 108 |\n| features12_pool256_xgb | Plan C 생체 특징 + convolution의 평균/최댓값 pooling | 268 |\n"
    text += "\n```mermaid\nflowchart LR\n A[원시 신호 3채널] --> B[기존 fold CNN]\n B --> C[96차원 은닉층 또는 256차원 pooling]\n D[Plan C 12개 생체 특징] --> E[특징 연결]\n C --> E\n E --> F[XGBoost 재학습]\n F --> G[5개 클래스 예측]\n```\n"
    text += "\n## 평가와 학습\n\n- 30개 녹화, 6,059개 윈도우, rest 포함 5개 클래스입니다.\n- 각 fold는 `output_new_1`에서 29개 녹화로 학습한 CNN을 고정하여 사용합니다. CNN 재학습은 하지 않았습니다.\n- 해당 CNN 한 개로 그 fold의 학습과 테스트 특징을 모두 추출합니다. 서로 다른 fold CNN의 임베딩을 섞지 않습니다.\n- XGBoost는 해당 29개 녹화의 특징과 라벨로 새로 학습합니다. 테스트 녹화는 CNN과 XGBoost 모두 학습에서 제외됐습니다.\n- dropout과 BatchNorm은 평가 모드로 고정합니다. 테스트 추론 중 모델 파라미터를 업데이트하지 않습니다.\n- XGBoost는 기존 Plan C의 400 trees, depth 4, learning rate 0.05 및 class weighting을 유지했습니다.\n- 특징 종류 세 가지는 실행 전에 고정했으며 테스트 점수로 하이퍼파라미터를 재선택하지 않았습니다.\n- 기존 CNN, Plan C XGBoost, 단순 평균 결과는 같은 윈도우의 저장된 out-of-fold 확률을 사용한 기준값입니다.\n"
    text += "\n## 변경 파일\n\n- 새 코드: `scripts/evaluation/run_cnn_embeddings.py` — 중간 특징 추출, fold별 XGBoost 학습, 재시작, 지표·그림·보고서 생성\n- 새 결과 폴더: `output_new_2/`\n- 기존 CNN 구조, Plan C 설정, 과거 결과 파일은 수정하지 않았습니다.\n"
    text += "\n## Confusion matrices\n\n![전체 비교](confusion_matrices.png)\n\n행은 실제 클래스, 열은 예측 클래스입니다. 각 셀에 개수와 실제 클래스 내 비율을 표시합니다.\n"
    for n in VARIANTS:
        text += f"\n### {n}\n\n![{n}](confusion_matrix_{n}.png)\n"
    text += "\n## 재실행\n\n```powershell\n./.venv/Scripts/python.exe -u -m scripts.evaluation.run_cnn_embeddings\n```\n\n`output_new_1`의 모델과 특징 CSV 및 `output_new/cnn_phase2/raw_windows_250hz.npz`가 필요합니다. 동일 입력·코드·환경의 완료 fold는 재사용합니다.\n"
    text += "\n## 출력 파일\n\n- `results.json`: 전체·클래스별 성능, 분할 정보\n- `predictions.csv`: 녹화·시간별 모든 방식의 예측과 확률\n- `folds/`: 새 XGBoost 모델, 입력 특징 순서, CNN 원본 체크포인트 경로, 입력 특징 중요도, 테스트 확률\n- `run_config.json`: 입력·코드·CNN 체크포인트 해시, 환경과 설정\n"
    text += "\n## 해석상 한계\n\nCNN은 같은 29개 학습 녹화의 라벨을 사용해 만든 표현이므로 학습 특징을 지나치게 잘 구분할 수 있습니다. 테스트 녹화는 독립적으로 유지했지만 이 방식은 out-of-fold 확률로 학습하는 stacking과 다릅니다. 녹화 단위 평가이고 참가자 단위 독립 평가는 아닙니다. 기존 전체 녹화 정규화와 초기 rest 보정을 유지합니다.\n"
    (out / "README.md").write_text(text, encoding="utf-8")
    print(json.dumps({n:{k:r[k] for k in ("accuracy","macro_f1")} for n,r in results.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output_new_2")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = Path("output_new_1")
    cache = Path("output_new/cnn_phase2/raw_windows_250hz.npz")
    assert torch.cuda.is_available()
    import xgboost
    config = {"variants": VARIANTS, "xgb_params": XGB_PARAMS, "torch": torch.__version__, "xgboost": xgboost.__version__,
              "code_sha256": digest(__file__), "cache_sha256": digest(cache), "features_sha256": digest(base / "features.csv"),
              "baseline_sha256": digest(base / "predictions.csv"),
              "checkpoint_sha256": {f"{i:02d}": digest(base / "folds" / f"{i:02d}" / "cnn_model.pt") for i in range(30)}}
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config["signature"] = signature
    path = out / "run_config.json"
    if path.exists():
        assert json.loads(path.read_text())["signature"] == signature, "Existing results use different code or inputs"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    data = load_cache(cache)
    y, recs = data["y"], data["recordings"]
    keys = pd.DataFrame({"rec_name": recs, "t_start": data["anchors"]})
    features = keys.merge(pd.read_csv(base / "features.csv"), on=["rec_name","t_start"], how="left", validate="one_to_one")
    assert np.array_equal(features.activity.to_numpy(), np.asarray(CLASSES)[y])
    tabular = features[list(FEATURES)].to_numpy(float)
    tabular[~np.isfinite(tabular)] = np.nan
    old = pd.read_csv(base / "predictions.csv")
    assert np.array_equal(old.recording.to_numpy(), recs) and np.array_equal(old.anchor_seconds.to_numpy(), data["anchors"])
    probabilities = {n: old[[f"{n}_prob_{c}" for c in CLASSES]].to_numpy() for n in ("cnn","xgboost","mean")}
    probabilities.update({n:np.full((len(y),5), np.nan) for n in VARIANTS})
    folds=[]
    assert len(set(recs.tolist())) == 30
    for i, rec in enumerate(sorted(set(recs.tolist()))):
        folder = out / "folds" / f"{i:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        source = base / "folds" / f"{i:02d}"
        metadata = json.loads((source / "complete.json").read_text())
        train, test = recs != rec, np.flatnonzero(recs == rec)
        assert metadata["test_recording"] == rec and set(metadata["training_recordings"]) == set(recs[train].tolist())
        assert len(metadata["training_recordings"]) == 29 and rec not in metadata["training_recordings"]
        print(f"[{i+1:02d}/30] test={rec} train=29", flush=True)
        done = folder / "complete.json"
        if done.exists():
            result = json.loads(done.read_text())
            assert result["signature"] == signature
            with np.load(folder / "probabilities.npz") as saved:
                assert np.array_equal(saved["test_indices"], test)
                for name in VARIANTS:
                    probabilities[name][test] = saved[name]
        else:
            cnn = Phase2CNN1D().cuda()
            cnn.load_state_dict(torch.load(source / "cnn_model.pt", map_location="cuda", weights_only=True))
            train96, train256, _ = extract(cnn, data["X"][train])
            test96, test256, cnn_prob = extract(cnn, data["X"][test])
            e96, p256 = np.empty((len(y),96), np.float32), np.empty((len(y),256), np.float32)
            e96[train], e96[test] = train96, test96
            p256[train], p256[test] = train256, test256
            assert np.isfinite(e96).all() and np.isfinite(p256).all()
            assert np.allclose(cnn_prob, probabilities["cnn"][test], atol=2e-5), "Encoder does not reproduce stored CNN output"
            del cnn
            saved = {"test_indices": test}
            for name, (include_features, dimension) in VARIANTS.items():
                z = e96 if dimension == 96 else p256
                names = [f"cnn_{dimension}_{j:03d}" for j in range(dimension)]
                if include_features:
                    z = np.column_stack((tabular,z))
                    names = list(FEATURES) + names
                model = create_model()
                model.fit(z[train], np.asarray(CLASSES)[y[train]])
                order = [list(model.classes_).index(c) for c in CLASSES]
                saved[name] = model.predict_proba(z[test])[:,order]
                probabilities[name][test] = saved[name]
                model.clf_.save_model(folder / f"{name}.json")
                (folder / f"{name}_features.json").write_text(json.dumps({"features": names, "classes": model.classes_.tolist(),
                    "feature_importances": model.clf_.feature_importances_.tolist(), "cnn_checkpoint": str(source / "cnn_model.pt")}, indent=2), encoding="utf-8")
                print(f"  trained {name}: {z.shape[1]} features", flush=True)
            np.savez(folder / "probabilities.npz", **saved)
            result = {"signature":signature, "test_recording":rec, "training_recordings":metadata["training_recordings"],
                      "n_train_windows":int(train.sum()), "n_test_windows":len(test), "cnn_checkpoint":str(source / "cnn_model.pt")}
            done.write_text(json.dumps(result, indent=2), encoding="utf-8")
        folds.append(result)
    summarize(out, data, probabilities, folds, config)


if __name__ == "__main__":
    main()
