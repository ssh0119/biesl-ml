# 실행 방법과 저장 파일 안내

## 1. 데이터와 환경

Python 3.12 및 CUDA가 동작하는 PyTorch 환경을 사용했습니다. 이번 실행 장치는 NVIDIA GeForce RTX 3060 Ti이며 정확한 패키지 버전은 [runtime-packages.txt](runtime-packages.txt)에 있습니다. `pyproject.toml`은 이전 프로젝트의 의존성 선언이고, 이번 실행 환경의 정확한 목록은 해당 파일을 기준으로 확인합니다.

원본 데이터는 별도로 다음과 같이 준비합니다. 원시 신호와 약 1.09 GB의 생성 캐시는 Git에 포함하지 않았습니다.

```text
data_new/
  Phase separation.xlsx
  Mental1/    # 11 recordings
  Mental2/    # 10 recordings
  Physical1/  # 11 recordings
  Physical2/  # 12 recordings
```

표의 `mta_9_14_pla_5'`와 실제 `mta_9_14_pla_5'5`는 사용자 확인으로 같은 녹화에 대응하며 자극 시간은 305초입니다. 확인된 대응은 [label_aliases.json](label_aliases.json)에 저장했습니다. 원본 Excel과 파일명은 수정하지 않았습니다.

## 2. 분류 점검

저장소 루트에서 실행합니다. 아래 `.venv/Scripts/python.exe`는 이번 Windows 가상환경 경로이며 다른 환경에서는 해당 Python 실행 파일을 사용합니다.

```powershell
./.venv/Scripts/python.exe -m scripts.evaluation.audit_phase_folders --data-dir data_new --output-dir outputs/data_new_audit --aliases output_data_new/label_aliases.json
```

표와 폴더 대응, Phase 1의 rest 라벨, 중복 파일, 파일명 파싱을 확인합니다. 과거 라벨과의 비교는 저장소에 포함된 `output_new_1/features.csv`를 기준으로 합니다. 입력 파일 해시는 `data_audit.json`에 기록됩니다. 분류 점검이 실패하면 전체 학습은 시작하지 않습니다.

## 3. 전체 평가 재실행

이미 공개된 결과를 보존하면서 새 출력 경로에 재실행하는 명령입니다.

```powershell
./.venv/Scripts/python.exe -u -m scripts.evaluation.run_dataset_suite --data-dir data_new --output-dir outputs/data_new_reproduction --aliases output_data_new/label_aliases.json
```

전처리까지만 확인하려면 같은 명령에 `--prepare-only`를 추가합니다. 기존 실행을 이어갈 때는 같은 출력 경로와 동일 입력·코드·환경을 사용합니다. 코드와 데이터 해시가 다르면 완료된 모델을 섞어 쓰지 않도록 재개가 거부됩니다. 다른 환경이나 checkout에서 재현할 때는 위처럼 새 출력 경로를 사용합니다. GPU 수치 연산 특성상 재실행 결과가 모든 자리까지 동일하다고 보장하지는 않습니다.

## 4. fold 내부의 처리 순서

1. 44개 중 한 녹화를 테스트로 분리합니다.
2. 나머지 43개 중 네 stress 클래스에서 하나씩 총 4개를 내부 검증으로 선택합니다.
3. 39개로 CNN을 학습하고 내부 검증 Macro F1로 epoch를 선택합니다. 최대 24 epoch, patience 5입니다.
4. 같은 39개로 Plan C XGBoost를 학습하고, 내부 검증 확률로 산술/기하 평균과 CNN 비중을 선택합니다.
5. 선택된 epoch로 CNN을 43개 전체에서 처음부터 재학습합니다. 학습률 스케줄은 24 epoch 주기를 유지합니다.
6. Plan C XGBoost도 43개 전체로 다시 학습합니다.
7. 최종 CNN을 eval 모드로 고정해 학습/테스트의 96차원 은닉층 및 256차원 pooling 특징을 추출합니다.
8. 동일한 43개 학습 녹화로 중간 특징 XGBoost 세 가지를 새로 학습합니다.
9. 테스트 녹화의 일곱 방식 예측을 저장합니다. 이를 44개 녹화마다 반복한 뒤 합산합니다.

CNN이나 중간 특징을 추출하는 모델은 fold마다 다릅니다. 서로 다른 fold CNN의 특징을 하나의 XGBoost 입력으로 섞지 않습니다. 테스트 정답은 epoch·결합 비중·모델 학습에 사용하지 않습니다.

## 5. 결과 파일

| 파일 | 내용 |
|---|---|
| `README.md` | 전체 성능, 클래스별 재현율, confusion matrices |
| `VALIDATION_REPORT.md` | 구현 변경, 검증 내역과 결과 해석 |
| `DATA_AUDIT.md`, `data_audit.json` | 표/폴더 대조와 데이터 차이 |
| `features.csv` | 생체 특징 테이블 |
| `predictions.csv` | 9,107개 윈도우의 실제 라벨, 방식별 예측·확률 |
| `results.json` | 전체 지표, confusion matrix, fold 정보 |
| `run_config.json` | 입력·코드 해시와 학습 설정 |
| `folds/00` … `folds/43` | 각 테스트 녹화에 대응하는 모델·검증 기록 |
| `training-transcript.txt`, `runtime-packages.txt` | 실행 로그와 환경 |

각 fold에는 CNN 1개와 최종 XGBoost 4개가 있습니다. `complete.json`의 테스트 파일명과 학습 목록으로 fold를 식별합니다. `inner_selection.json`과 `inner_probabilities.npz`는 선택에 사용한 검증 결과, `probabilities.npz`는 최종 테스트 확률입니다.

CNN 출력과 공개 확률 배열의 순서는 `rest, mental1, mental2, physical1, physical2`입니다. 저장된 XGBoost 자체의 출력 인덱스는 LabelEncoder 순서인 `mental1, mental2, physical1, physical2, rest`입니다. 중간 특징 모델의 `*_features.json`에도 특징 순서와 클래스 순서를 저장했습니다. 단독 Plan C 모델은 `src/plan_c/config.py`의 `FEATURES` 순서로 입력합니다.

중간 특징 XGBoost 추론에는 같은 fold의 `cnn_model.pt`가 필요합니다. 이 모델들은 교차검증 체크포인트이며 44개 전체로 학습한 단일 배포 모델은 아닙니다.

## 6. 결과 해석

가장 높은 방식은 12개 생체 특징 + 256개 CNN pooling 특징의 XGBoost로 정확도 85.63%, Macro F1 0.7715입니다. physical1을 mental1으로 혼동하는 오류는 남아 있습니다. 같은 데이터에서 후보를 비교한 탐색 결과이며, 동일 참가자의 다른 녹화가 학습에 있을 수 있습니다. 초기 rest 보정과 전체 녹화 정규화를 유지하는 오프라인 평가입니다.
