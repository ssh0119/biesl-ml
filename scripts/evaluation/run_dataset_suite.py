"""Fresh audited-dataset LORO: CNN, XGBoost, probability and feature fusion."""
from pathlib import Path
import argparse
import hashlib
import json
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xgboost
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from scripts.evaluation.audit_phase_folders import audit, sha256
from scripts.evaluation.run_cnn_embeddings import VARIANTS, extract
from scripts.evaluation.run_cnn_ensemble import fuse
from src.cnn.config import MAX_EPOCHS, PATIENCE
from src.cnn.data import build_raw_dataset, save_cache, load_cache
from src.cnn.evaluation import select_validation_recordings
from src.cnn.training import DEVICE, train_with_validation, train_fixed_epochs, predict_probabilities
from src.cnn.model import Phase2CNN1D
from src.plan_c.config import CLASSES, FEATURES, XGB_PARAMS
from src.plan_c.data import prepare_dataset
from src.plan_c.model import create_model

METHODS = ("cnn", "xgboost", "mean", "selected_fusion", *VARIANTS)
CANDIDATES = [(g,w) for g in (False,True) for w in (.5,.25,.75,.1,.9,0.,1.)]


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding="utf-8")


def proba(model, X):
    return model.predict_proba(X)[:,[list(model.classes_).index(c) for c in CLASSES]]


def summarize(out, data, predictions, folds, config):
    y=data["y"]
    table=pd.DataFrame({"recording":data["recordings"],"anchor_seconds":data["anchors"],"true_label":np.asarray(CLASSES)[y]})
    assert not table.duplicated(["recording","anchor_seconds"]).any()
    results={}
    fig,axes=plt.subplots(3,3,figsize=(22,20))
    for ax,(name,p) in zip(axes.flat,predictions.items()):
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
        pred=p.argmax(1)
        cm=confusion_matrix(y,pred,labels=range(5))
        acc=float(accuracy_score(y,pred))
        f1=float(f1_score(y,pred,labels=range(5),average="macro",zero_division=0))
        results[name]={"accuracy":acc,"macro_f1":f1,"confusion_matrix":cm.tolist(),
                       "classification_report":classification_report(y,pred,labels=range(5),target_names=CLASSES,output_dict=True,zero_division=0)}
        pct=cm/cm.sum(1,keepdims=True)*100
        ax.imshow(pct,cmap="Blues",vmin=0,vmax=100)
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f"{cm[i,j]}\n{pct[i,j]:.1f}%",ha="center",va="center",color="white" if pct[i,j]>55 else "black")
        ax.set_xticks(range(5),CLASSES,rotation=30,ha="right")
        ax.set_yticks(range(5),CLASSES)
        ax.set(xlabel="Predicted",ylabel="True",title=f"{name}\nAccuracy {acc:.2%} | Macro F1 {f1:.4f}")
        table[f"{name}_prediction"]=np.asarray(CLASSES)[pred]
        for k,label in enumerate(CLASSES): table[f"{name}_prob_{label}"]=p[:,k]
    for ax in list(axes.flat)[len(predictions):]: ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out/"confusion_matrices.png",dpi=150)
    fig.canvas.draw()
    for ax,name in zip(axes.flat,predictions):
        box=ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out/f"confusion_matrix_{name}.png",dpi=180,bbox_inches=box.expanded(1.04,1.04))
    plt.close(fig)
    table.to_csv(out/"predictions.csv",index=False)
    write_json(out/"results.json",{"classes":CLASSES,"n_windows":len(y),"n_folds":len(folds),"config":config,"folds":folds,"results":results})
    n=len(folds)
    text=f"# data_new 전체 평가 결과\n\n{n}개 녹화, {len(y):,}개 윈도우. **{n-1}개 학습 / 1개 테스트, {n}-fold LORO**.\n\n[분류 점검 보고서](DATA_AUDIT.md)\n\n"
    text+="## 전체 결과\n\n| 방식 | 정확도 | Macro F1 |\n|---|---:|---:|\n"
    for name,r in results.items(): text+=f"| {name} | {r['accuracy']:.2%} | {r['macro_f1']:.4f} |\n"
    best=max(results,key=lambda name:results[name]["macro_f1"])
    text+=f"\n이번 비교에서 Macro F1 기준 가장 높은 방식은 **{best}**입니다. 동일 데이터에서 후보를 비교한 탐색 결과로, 독립 데이터에서 우수성을 확인한 것은 아닙니다.\n"
    text+="\n## 라벨과 구간\n\n앞 300초는 rest, 이후는 phase separation 표에 대응하는 폴더 클래스입니다. math는 300초 자극, plank는 실제 파일명의 지속 시간을 사용합니다. recovery는 제외합니다. 경계 제외·40초 CNN·2초 간격·기존 특징별 윈도우를 유지했습니다.\n"
    text+=f"\n## 학습 및 선택\n\n각 outer fold의 테스트 녹화는 모든 학습·선택에서 제외합니다. 나머지 {n-1}개 중 클래스별 1개씩 4개를 내부 검증으로 사용하고 {n-5}개로 CNN과 XGBoost를 학습하여 CNN epoch와 확률 결합 가중치를 정합니다. 이후 CNN과 XGBoost를 **{n-1}개 전체로 처음부터 다시 학습**해 테스트합니다. CNN 중간 특징 XGBoost도 해당 최종 CNN과 같은 {n-1}개 학습 녹화만 사용합니다. 과거 데이터의 epoch·모델·가중치는 재사용하지 않았습니다.\n"
    text+="\n## 방식 설명\n\n- cnn: 3채널 원시 신호 CNN\n- xgboost: Plan C 12개 특징\n- mean: CNN/XGBoost 확률 50:50 평균\n- selected_fusion: 내부 검증에서 산술/기하 평균과 비중 선택\n- embedding96_xgb: CNN 96차원 은닉층 → XGBoost\n- features12_embedding96_xgb: 12개 생체 특징 + 96차원 은닉층 → XGBoost\n- features12_pool256_xgb: 12개 생체 특징 + 256차원 pooling → XGBoost\n"
    text+="\n## 클래스별 재현율\n\n| 클래스 | "+" | ".join(results)+" |\n|---|"+"---:|"*len(results)+"\n"
    for c in CLASSES: text+="| "+c+" | "+" | ".join(f"{r['classification_report'][c]['recall']:.1%}" for r in results.values())+" |\n"
    text+="\n## Confusion matrices\n\n![전체 비교](confusion_matrices.png)\n\n행은 실제 분류, 열은 예측 분류입니다. 셀은 개수와 행별 비율입니다.\n"
    for name in ("cnn","mean","features12_pool256_xgb"):
        text+=f"\n### {name}\n\n![{name}](confusion_matrix_{name}.png)\n"
    text+="\n## 재현 및 변경 사항\n\n새 `scripts/evaluation/audit_phase_folders.py`가 Excel과 폴더·해시를 검사합니다. 새 `scripts/evaluation/run_dataset_suite.py`가 데이터 수에 맞춘 전체 평가를 실행합니다. 기존 모델·특징·전처리 코드는 그대로 사용합니다.\n\n```powershell\n./.venv/Scripts/python.exe -u -m scripts.evaluation.run_dataset_suite --data-dir data_new --output-dir output_data_new --aliases output_data_new/label_aliases.json\n```\n\nfold별 모델/검증 기록/확률은 `folds/`, 전체 예측은 `predictions.csv`, 설정과 입력·코드 해시는 `run_config.json`, 전체 지표는 `results.json`입니다. 동일 설정의 완료 단계는 재사용합니다.\n"
    text+="\n## 해석 범위\n\n기존 30개 평가와 데이터 구성 및 일부 라벨이 달라 점수 차이를 모델 개선만으로 해석할 수 없습니다. 녹화 단위 평가이며 동일 참가자의 다른 녹화가 학습에 들어갈 수 있습니다. 전체 녹화 정규화와 초기 rest 보정을 사용하는 오프라인 평가입니다.\n"
    (out/"README.md").write_text(text,encoding="utf-8")
    print(json.dumps({name:{k:r[k] for k in ("accuracy","macro_f1")} for name,r in results.items()},indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",default="data_new")
    parser.add_argument("--output-dir",default="output_data_new")
    parser.add_argument("--aliases")
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    aliases=json.loads(Path(args.aliases).read_text(encoding="utf-8")) if args.aliases else {}
    checked=audit(args.data_dir,out,aliases)
    if checked["errors"]: raise ValueError("Resolve dataset audit errors before training: "+str(checked["errors"]))
    write_json(out/"label_aliases.json",aliases)
    sources=[Path(__file__),Path("scripts/evaluation/audit_phase_folders.py"),Path("scripts/evaluation/run_cnn_embeddings.py"),Path("scripts/evaluation/run_cnn_ensemble.py")]
    sources+=list(Path("src").rglob("*.py"))
    config={"manifest":checked["manifest"],"spreadsheet_sha256":checked["spreadsheet_sha256"],"aliases":aliases,
            "code_hashes":{str(p):sha256(p) for p in sources},"torch":torch.__version__,"xgboost":xgboost.__version__,
            "max_epochs":MAX_EPOCHS,"patience":PATIENCE,"xgb_params":XGB_PARAMS,"classes":CLASSES,"features":FEATURES,
            "variants":VARIANTS,"candidates":CANDIDATES}
    signature=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(); config["signature"]=signature
    if (out/"run_config.json").exists():
        assert json.loads((out/"run_config.json").read_text())["signature"]==signature,"Inputs/code changed; use a new output directory"
    write_json(out/"run_config.json",config)
    assert DEVICE=="cuda"
    raw_path=out/"raw_windows_250hz.npz"
    if not raw_path.exists(): save_cache(build_raw_dataset(args.data_dir),raw_path)
    feature_path=out/"features.csv"
    if not feature_path.exists(): prepare_dataset(args.data_dir).to_csv(feature_path,index=False)
    data=load_cache(raw_path); y=data["y"]; recs=data["recordings"]
    keys=pd.DataFrame({"rec_name":recs,"t_start":data["anchors"]})
    df=keys.merge(pd.read_csv(feature_path),on=["rec_name","t_start"],how="left",validate="one_to_one",indicator=True)
    assert df["_merge"].eq("both").all() and np.array_equal(df.activity.to_numpy(),np.asarray(CLASSES)[y])
    X=df[list(FEATURES)].to_numpy(float); X[~np.isfinite(X)]=np.nan
    unique=sorted(set(recs.tolist())); assert len(unique)==checked["n_recordings"]
    print(f"AUDITED: {len(unique)} recordings, {len(y)} windows, counts={np.bincount(y).tolist()}, GPU={torch.cuda.get_device_name(0)}",flush=True)
    if args.prepare_only: return
    predictions={name:np.full((len(y),5),np.nan) for name in METHODS}; folds=[]
    for i,rec in enumerate(unique):
        fold=out/"folds"/f"{i:02d}"; fold.mkdir(parents=True,exist_ok=True)
        ti=np.flatnonzero(recs==rec); train=recs!=rec
        val_names=select_validation_recordings(recs,y,rec); vi=np.flatnonzero(np.isin(recs,val_names)); inner=train & ~np.isin(recs,val_names)
        print(f"[{i+1:02d}/{len(unique)}] test={rec}; final train={len(unique)-1}",flush=True)
        if (fold/"complete.json").exists():
            metadata=json.loads((fold/"complete.json").read_text()); assert metadata["signature"]==signature
            with np.load(fold/"probabilities.npz") as saved:
                assert np.array_equal(saved["test_indices"],ti)
                for name in METHODS: predictions[name][ti]=saved[name]
            folds.append(metadata); continue
        if not (fold/"inner_selection.json").exists():
            cnn,history=train_with_validation(data["X"][inner],y[inner],data["X"][vi],y[vi],n_classes=5)
            cv=predict_probabilities(cnn,data["X"][vi]); del cnn; gc.collect()
            xgb=create_model(); xgb.fit(X[inner],np.asarray(CLASSES)[y[inner]]); xv=proba(xgb,X[vi])
            scores=[float(f1_score(y[vi],fuse(cv,xv,w,g).argmax(1),labels=range(5),average="macro",zero_division=0)) for g,w in CANDIDATES]
            g,w=CANDIDATES[int(np.argmax(scores))]
            selection={"history":history,"validation_recordings":val_names,"cnn_weight":w,"geometric":g,"scores":scores}
            np.savez(fold/"inner_probabilities.npz",validation_indices=vi,cnn=cv,xgboost=xv)
            write_json(fold/"inner_selection.json",selection)
        selection=json.loads((fold/"inner_selection.json").read_text())
        assert selection["validation_recordings"]==val_names
        checkpoint=fold/"cnn_model.pt"
        if checkpoint.exists():
            cnn=Phase2CNN1D().to(DEVICE); cnn.load_state_dict(torch.load(checkpoint,map_location=DEVICE,weights_only=True))
        else:
            cnn=train_fixed_epochs(data["X"][train],y[train],n_classes=5,epochs=selection["history"]["best_epoch"],scheduler_epochs=MAX_EPOCHS)
            torch.save(cnn.state_dict(),checkpoint)
        tr96,tr256,_=extract(cnn,data["X"][train]); te96,te256,ct=extract(cnn,data["X"][ti]); del cnn; gc.collect()
        xgb=create_model(); xgb.fit(X[train],np.asarray(CLASSES)[y[train]]); xt=proba(xgb,X[ti]); xgb.clf_.save_model(fold/"xgboost_model.json")
        output={"cnn":ct,"xgboost":xt,"mean":(ct+xt)/2,"selected_fusion":fuse(ct,xt,selection["cnn_weight"],selection["geometric"])}
        for name,(include_features,dim) in VARIANTS.items():
            tr,te=(tr96,te96) if dim==96 else (tr256,te256)
            names=[f"cnn_{dim}_{j:03d}" for j in range(dim)]
            if include_features: tr=np.column_stack((X[train],tr)); te=np.column_stack((X[ti],te)); names=list(FEATURES)+names
            model=create_model(); model.fit(tr,np.asarray(CLASSES)[y[train]]); output[name]=proba(model,te)
            model.clf_.save_model(fold/f"{name}.json")
            write_json(fold/f"{name}_features.json",{"features":names,"classes":model.classes_.tolist(),"importance":model.clf_.feature_importances_.tolist()})
        metadata={"signature":signature,"test_recording":rec,"training_recordings":sorted(set(recs[train].tolist())),"n_train_windows":int(train.sum()),
                  "n_test_windows":len(ti),"inner_validation_recordings":val_names,"epochs":selection["history"]["best_epoch"],
                  "cnn_weight":selection["cnn_weight"],"geometric":selection["geometric"]}
        np.savez(fold/"probabilities.npz",test_indices=ti,**output); write_json(fold/"complete.json",metadata)
        for name in METHODS: predictions[name][ti]=output[name]
        folds.append(metadata)
        print(f"  completed fold {i+1}/{len(unique)}",flush=True)
        gc.collect()
    summarize(out,data,predictions,folds,config)


if __name__=="__main__": main()
