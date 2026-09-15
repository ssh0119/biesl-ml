# 1D-CNN 분류 파이프라인

## 목적

기존 XGBoost 코드는 그대로 유지하고, 원시 생체신호를 직접 입력받는 별도의 1D-CNN을 추가했다. 분류 클래스와 평가 단위는 현재 XGBoost와 동일하다.

- 클래스: `rest`, `mental1`, `mental2`, `physical1`, `physical2`
- 라벨 출처: 파일명 문자열이 아니라 `data/Mental1`, `Mental2`, `Physical1`, `Physical2` 폴더
- 평가: 30개 recording에 대한 Leave-One-Recording-Out(LORO)
- 최종 지표: 모든 LORO test 예측을 합친 pooled accuracy와 macro F1

## 입력 데이터

각 anchor를 중심으로 40초 구간을 사용한다. 기존 특징 추출과 동일한 2초 anchor grid 및 phase 경계 안전구간을 사용하므로 XGBoost 예측과 나중에 결합할 수 있다.

| 채널 | 전처리 | 입력 크기 |
|---|---|---:|
| ECG | 기존 필터링 후 recording 단위 z-score | 10,000 |
| Respiration | baseline 제거와 smoothing 후 recording 단위 z-score | 10,000 |
| Microphone | Shannon-energy envelope 후 recording 단위 z-score | 10,000 |

원본 sampling rate인 250 Hz를 낮추지 않는다. CNN 입력 tensor 하나의 shape은 `(3, 10000)`이다. 전체 raw cache는 크기가 크므로 `output_new/cnn_phase2/raw_windows_250hz.npz`에 한 번만 생성한다.

## 네트워크

구현은 `src/cnn/model.py`의 `Phase2CNN1D`다.

1. Conv1d 3->32, kernel 15, stride 2, BatchNorm, ReLU, MaxPool
2. Conv1d 32->64, kernel 9, stride 2, BatchNorm, ReLU, MaxPool
3. Conv1d 64->128, kernel 5, stride 2, BatchNorm, ReLU, MaxPool
4. Conv1d 128->128, kernel 3, BatchNorm, ReLU, MaxPool
5. Global average pooling과 global max pooling을 연결
6. Fully connected 256->96->5와 dropout

학습에는 class-weighted cross entropy와 AdamW를 쓴다. 학습 데이터에만 최대 0.5초 이동, 채널별 작은 scale 변화, 작은 Gaussian noise augmentation을 적용한다.

## 데이터 분할

각 outer fold에서 recording 하나 전체를 test로 뺀다. 같은 recording의 겹치는 window가 train과 test에 동시에 들어가지 않는다.

Early stopping용 validation도 test recording을 사용하지 않는다. 남은 recording 가운데 `mental1`, `mental2`, `physical1`, `physical2`를 하나씩 선택한다. 각 recording에는 rest 구간도 있으므로 validation에는 다섯 클래스가 모두 존재한다. 나머지 recording만 train에 사용한다.

이 방식은 무작위 window split보다 엄격하다. 다만 같은 참가자가 여러 recording에 존재한다면 참가자 특성이 train과 test에 모두 포함될 수 있다. 완전히 새로운 사람에 대한 일반화를 측정하려면 별도로 Leave-One-Subject-Out 평가가 필요하다.

## 실행

```powershell
# raw window cache만 생성
python -m src.cnn prepare

# 30-fold LORO 평가만 수행
python -m src.cnn evaluate

# LORO 평가 후 전체 데이터로 최종 모델까지 학습
python -m src.cnn run-all
```

기본값은 최대 24 epoch, patience 5다. GPU는 PyTorch CUDA가 사용 가능한 경우 자동 선택된다.

```powershell
python -m src.cnn run-all --max-epochs 40 --patience 7
```

다른 컴퓨터에서 GitHub 저장소를 받은 뒤에는 raw recording을 같은
`data/Mental1`, `data/Mental2`, `data/Physical1`, `data/Physical2` 구조로
준비해야 한다. `data/`와 생성 cache는 용량 때문에 GitHub에 포함되지 않는다.

CUDA GPU 인식 여부는 먼저 다음처럼 확인한다.

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -m src.cnn evaluate
```

첫 실행에서는 약 728 MB의 `raw_windows_250hz.npz`를 만든 다음 30-fold를
수행한다. 최종 모델 재학습까지 필요한 경우에만 `evaluate` 대신 `run-all`을
사용한다.

## 출력

기본 출력 위치는 `output_new/cnn_phase2`다.

- `raw_windows_250hz.npz`: 전처리된 raw window cache
- `loro_results.json`: pooled 지표, class별 지표, fold별 결과와 validation 구성
- `loro_predictions.csv`: recording/anchor별 실제 라벨, 예측 라벨, 다섯 클래스 확률
- `confusion_matrix.png`: pooled LORO confusion matrix
- `cnn_model.pt`: 전체 데이터로 다시 학습한 최종 모델 가중치
- `cnn_model_metadata.json`: 입력 규격, 클래스 순서, epoch 등 로딩 정보

최종 모델 epoch 수는 각 LORO fold에서 선택된 best epoch의 중앙값을 사용한다. 저장된 최종 모델 자체의 성능 수치가 아니라 `loro_results.json`의 out-of-fold 결과를 보고 성능을 판단해야 한다.

## XGBoost와 결합할 때

`loro_predictions.csv`는 `recording + anchor_seconds`를 키로 XGBoost의 out-of-fold 확률과 맞출 수 있게 만들었다. 비교 순서는 다음이 적절하다.

1. 두 모델 확률의 단순 평균
2. validation에서만 정한 가중 평균
3. 두 모델 확률을 입력으로 받는 logistic regression stacking
4. CNN의 96차원 penultimate embedding을 XGBoost 특징으로 추가

결합 가중치나 stacking 모델을 test fold 결과로 고르면 leakage가 생긴다. 반드시 각 outer fold 내부 train/validation 데이터에서만 결합 방법을 정해야 한다. CNN 단독 macro F1과 클래스별 오류가 XGBoost와 충분히 다를 때 ensemble의 개선 가능성이 크다.
