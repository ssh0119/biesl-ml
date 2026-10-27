# 테스트로 epoch를 선택하는 비교 실험

## 결정: 기존 코드를 유지하고 실행 경로를 분리

원래 엄밀한 실행 코드인 `scripts/evaluation/run_dataset_suite.py`와 `src/`는 수정하지 않았습니다. 새 진입점 `train_test_selected.py`와 새 실행 모듈 `scripts/evaluation/run_test_selected_suite.py`를 추가했습니다. 데이터 처리·CNN 구조·Plan C XGBoost 설정은 기존 함수를 재사용하고, checkpoint 선택과 결과 보고만 별도로 구현했습니다. 기존 방식을 되돌리거나 원본 파일을 복원할 필요가 없습니다.

## 바로 실행

저장소 루트 `C:\Users\DSKT\Desktop\Biesl-ml`의 PowerShell에서 아래 한 줄을 실행합니다.

```powershell
.\.venv\Scripts\python.exe -u train_test_selected.py
```

현재 환경의 CUDA PyTorch를 사용합니다. 기본 데이터는 `data_new`, 기본 결과 폴더는 **`output_data_new_test_selected_fast`**입니다. 현재 데이터 44개에 대해 43개 학습·1개 테스트를 반복하고, 일곱 방식의 결과와 confusion matrix까지 생성합니다. 터미널은 실행이 끝날 때까지 열어 두어야 합니다. 콘솔 내용은 결과 폴더의 `training.log`에도 자동 저장됩니다.

현재 데이터 점검과 준비 확인은 완료했습니다. 실제 44-fold 학습은 실행하지 않았습니다.

## 기본 동작

1. Phase separation과 실제 폴더·파일을 점검합니다. 이전에 확인한 파일명 대응 `output_data_new/label_aliases.json`을 사용합니다.
2. 기존 `output_data_new`의 전처리 캐시는 원본 파일 해시·라벨·전처리 코드가 일치할 때만 재사용합니다. **기존 학습 모델이나 선택 epoch는 재사용하지 않습니다.** 캐시가 없거나 조건이 다르면 새 출력 폴더에 전처리 결과를 만듭니다.
3. 한 녹화 전체를 테스트로 남기고 나머지 43개로 CNN을 처음부터 학습합니다. 여기서 1개는 하나의 윈도우가 아니라 **녹화 파일 하나 전체**입니다.
4. 각 epoch가 끝날 때 테스트 녹화의 **정확도**를 계산합니다. 기본 최대 24 epoch이며, 테스트의 모든 윈도우를 맞히면 즉시 해당 fold의 CNN 학습을 종료합니다. 100% 미만의 정체에 따른 종료는 기본적으로 비활성화합니다.
5. 가장 높은 테스트 정확도를 기록한 checkpoint를 선택합니다. 점수가 같으면 **먼저 나온 epoch**를 유지합니다. 테스트 정답은 gradient나 class weight 계산에는 쓰지 않고 checkpoint 선택에 사용합니다.
6. 해당 가중치를 그대로 복원합니다. 선택 후 다시 학습하거나 마지막 epoch의 가중치로 바꾸지 않습니다.
7. 동일한 43개로 Plan C XGBoost를 학습하고, 선택된 CNN으로 중간 특징을 추출해 세 가지 XGBoost를 추가 학습합니다.
8. 테스트 녹화의 예측을 저장하고 44개 fold의 예측을 합산해 최종 정확도·Macro F1·confusion matrix를 생성합니다.

## 출력하는 일곱 방식

| 결과 이름 | 동작 | 테스트 선택의 영향 |
|---|---|---|
| `cnn` | 테스트로 선택한 CNN checkpoint | CNN epoch 선택 |
| `xgboost` | Plan C 12개 특징, 고정 400 trees | CNN epoch 선택과 무관한 기준값 |
| `mean` | CNN과 XGBoost 확률의 50:50 평균 | CNN epoch 선택 |
| `test_tuned_fusion` | 산술/기하 평균과 비중까지 **같은 테스트**로 선택 | CNN epoch + 결합 비중 모두 선택 |
| `embedding96_xgb` | 선택된 CNN 은닉층 96개 → XGBoost | CNN epoch 선택 |
| `features12_embedding96_xgb` | 생체 특징 12개 + CNN 은닉층 96개 → XGBoost | CNN epoch 선택 |
| `features12_pool256_xgb` | 생체 특징 12개 + CNN pooling 256개 → XGBoost | CNN epoch 선택 |

`test_tuned_fusion`은 별도 추가 편향 비교입니다. 단순 평균 `mean`의 비중은 항상 50:50입니다. 결합 후보는 기존과 같은 산술/기하 평균 및 CNN 비중 0, 0.1, 0.25, 0.5, 0.75, 0.9, 1이며 동점은 먼저 나열된 산술 50:50을 우선합니다.

**CNN에서 고른 epoch 하나를 모든 CNN 기반 결합에 공통 적용합니다.** 각 epoch마다 중간 특징 XGBoost까지 학습해 그 결과로 최적 epoch를 고르는 방식은 아닙니다. XGBoost 하이퍼파라미터와 신호 전처리는 기존 설정을 유지합니다.

## 옵션

선배 실험의 epoch 수가 다르면 별도 출력 폴더를 지정합니다. 최대 epoch는 탐색 횟수와 cosine 학습률 주기를 함께 바꿉니다.

```powershell
.\.venv\Scripts\python.exe -u train_test_selected.py --max-epochs 50 --output-dir output_data_new_test_selected_fast_50ep
```

위 명령은 최대 50 epoch까지 학습하되 정확도 100%에 도달하면 바로 다음 단계로 넘어갑니다. 로그의 반올림된 숫자가 아니라 실제 정확도 `1.0`을 확인합니다. Cosine 학습률 주기는 여전히 50으로 유지합니다. 동일한 학습 경로라면 기존의 '첫 최고점 선택'에서 어차피 선택하지 않을 뒤쪽 epoch만 생략합니다. 뒤쪽 동점 모델이 더 좋은 중간 특징을 만든다는 보장은 없으므로 동점 선택은 그대로 유지합니다.

100%에 도달하지 않는 fold도 빨리 종료하려면 아래처럼 선택 지표가 **8 epoch 연속 개선되지 않을 때** 멈추도록 설정할 수 있습니다. 동점도 미개선으로 셉니다. 나중의 개선을 놓칠 수 있으므로 기본값은 `--patience 0`(비활성)입니다.

```powershell
.\.venv\Scripts\python.exe -u train_test_selected.py --max-epochs 50 --patience 8 --output-dir output_data_new_test_selected_fast_patience8
```

100%에서도 끝까지 탐색하려면 `--no-stop-on-perfect`를 추가합니다. `--patience 0`과 함께 사용하면 기존 전체 탐색 방식이며 동점은 최초 epoch를 유지합니다. 종료 사유와 실제 실행 epoch 수는 `cnn_selection.json`에 기록됩니다.

테스트 정확도 대신 테스트 Macro F1로 checkpoint와 결합 비중을 선택하려면:

```powershell
.\.venv\Scripts\python.exe -u train_test_selected.py --selection-metric macro_f1 --output-dir output_data_new_test_selected_f1
```

Macro F1은 모든 계산에서 고정된 5개 클래스 기준이며, 해당 fold에 없는 클래스의 F1은 0으로 계산합니다. 두 클래스만 있는 fold에서는 모두 맞혀도 0.4이며, 이 경우에도 완전 정답 종료가 적용됩니다. 최종 지표는 전체 윈도우를 합산해 다시 계산하므로 최종 Macro F1의 최대는 1입니다.

학습 없이 데이터·GPU·캐시만 점검하려면:

```powershell
.\.venv\Scripts\python.exe -u train_test_selected.py --prepare-only
```

## 중단 후 다시 시작

**수정 전에 시작한 프로세스에는 새 종료 조건이 자동 적용되지 않습니다.** 기존 터미널에서 `Ctrl+C`로 중단한 다음 위의 새 출력 폴더 명령을 실행하세요. 코드 해시가 달라 기존 실행 폴더는 재개할 수 없으며, 새 폴더에서는 첫 fold부터 시작합니다. 기존 결과 파일은 보존됩니다.

같은 명령어를 다시 실행합니다. 동일 데이터·코드·설정이면 완료된 fold를 건너뜁니다. CNN 전체 epoch 탐색을 끝내 checkpoint를 저장한 fold는 그 checkpoint부터 결합 모델 생성을 이어갑니다. **CNN epoch 탐색 도중 중단된 fold는 그 fold의 첫 epoch부터 다시 학습**합니다. epoch 중간 optimizer 상태를 복원하는 방식은 아닙니다.

설정이나 코드·입력 해시가 바뀌면 결과 혼합을 막기 위해 재개를 거부합니다. 그 경우 새 `--output-dir`를 지정합니다. 기존 엄밀한 결과 폴더나 원본 데이터 폴더를 출력 경로로 쓰는 명령은 거부됩니다.

## 결과 파일

```text
output_data_new_test_selected_fast/
  README.md                         # 결과 요약, 테스트 선택 방식 표시
  DATA_AUDIT.md / data_audit.json    # 데이터 대조
  run_config.json                   # 설정과 입력/코드 해시
  preprocessing_hashes.json         # 실제 사용한 캐시 내용 해시
  training.log                      # 실시간 진행 로그
  results.json                      # 전체 지표, 클래스별 점수, fold 정보
  predictions.csv                   # 각 윈도우의 실제 라벨과 일곱 방식 확률
  confusion_matrices.png            # 전체 비교
  confusion_matrix_*.png            # 개별 그림
  folds/00/ ... folds/43/
    cnn_model.pt                    # 테스트 점수가 가장 높았던 checkpoint
    cnn_selection.json              # 모든 epoch 점수와 선택 epoch
    xgboost_model.json
    *_xgb.json / *_xgb_features.json # 중간 특징 모델과 입력 순서
    probabilities.npz
    complete.json                   # train/test 목록, 테스트 선택 기록
```

그림과 JSON에는 `TEST-SELECTED / OPTIMISTIC`, `unbiased_test_evaluation: false`를 표시합니다. 단독 XGBoost 그림은 epoch 선택과 무관한 고정 기준값임을 별도로 표시합니다.

## 원래 방식과의 차이

| 항목 | 원래 방식 | 이번 비교 실험 |
|---|---|---|
| 내부 선택 | 39 train + **4 validation** + 1 test | 43 train + 1 test |
| CNN epoch 기준 | validation Macro F1 | test accuracy 기본값 |
| 최종 학습 | 선택 epoch로 43개 재학습 | 43개 학습 중 선택된 가중치 그대로 사용 |
| 테스트 점수 | 선택에 쓰지 않은 테스트 | 선택에 사용한 테스트의 점수 |
| 결과 경로 | `output_data_new` | `output_data_new_test_selected_fast` |

이번 방식의 점수는 테스트를 보고 고른 만큼 낙관적으로 나올 수 있습니다. 이를 실험적으로 비교하는 목적이며, 일반화 성능이 개선됐다고 해석하지 않습니다. 같은 데이터에 대해 선배 실험의 epoch 범위·전처리·선택 지표·집계 방식까지 맞춰야 91%와의 차이를 설명할 수 있습니다.

## 구현 검증

합성 데이터로 전체 epoch 탐색, 완전 정답 즉시 종료(accuracy 및 Macro F1 선택), 선택적 patience 종료, 첫 동점 유지, 최적 checkpoint 복원, 테스트 입력이 학습 loader에 들어가지 않음, 보고서의 편향 표시와 그림 생성, 기존 출력 경로 차단을 검사했습니다.

```powershell
.\.venv\Scripts\python.exe -m unittest scripts.diagnostics.test_test_selected_suite -v
```
