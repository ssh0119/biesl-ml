# CNN 확장 비교

## 변경 내용

사용자가 알려준 합성곱 `64 → 128 → 256 → 256`, 은닉층 `256`으로 확장 모델을 구성했습니다. 나머지 세부 설정은 기존 모델을 유지하므로 선배님 모델의 완전한 재현은 아닙니다. 원본 `src/cnn/model.py`는 유지하고 `src/cnn/expanded_model.py`에 새 모델을 정의했습니다.

| 항목 | 기존 baseline | 확장 expanded |
|---|---:|---:|
| 합성곱 블록 채널 | 32 → 64 → 128 → 128 | 64 → 128 → 256 → 256 |
| 학습 파라미터 | 135,845 | 571,077 |
| FP32 파라미터 크기 | 0.518 MiB | 2.178 MiB |
| state_dict 직렬화 크기 | 555,855 bytes | 2,299,599 bytes |
| pooling 특징 | 256 | 512 |
| 은닉층 특징 | 96 | 256 |

파라미터는 약 4.20배입니다. 모든 합성곱 채널을 두 배로 늘리고 은닉층을 256으로 확대했습니다. 평균·최대 pooling을 연결해 512차원이 되며 분류 헤드는 512 → 256 → 5입니다. XGBoost 입력도 임베딩 256개, 기존 특징 12개+임베딩 256개, 기존 특징 12개+pooling 512개로 자동 변경됩니다. 커널, stride, pooling 방식과 dropout은 유지합니다. 표의 크기는 학습 중 GPU 메모리 사용량이 아닙니다. 활성값·gradient·optimizer 상태 때문에 실제 학습 메모리는 더 큽니다.

## 실행

저장소 루트의 PowerShell에서:

```powershell
.\.venv\Scripts\python.exe -u train_test_selected.py --architecture expanded --max-epochs 50 --output-dir output_data_new_test_selected_expanded256_50ep
```

43 train / 1 test, 44-fold를 수행합니다. 정확도 100% 도달 시 해당 fold의 CNN 학습을 종료하고 나머지 결합 학습을 수행합니다. CNN 단독, XGBoost 단독, 확률 결합 2개, 중간 특징 결합 3개와 confusion matrix를 생성합니다. 테스트로 epoch와 일부 결합 비중을 선택하는 기존 실험 조건을 그대로 사용하므로 독립 테스트 성능은 아닙니다.

기존 모델은 `--architecture baseline` 또는 옵션 생략으로 선택합니다. 구조가 달라진 실행에는 반드시 새로운 출력 폴더를 사용하세요. 코드 해시가 바뀌므로 이전 코드의 중단된 실행도 기존 폴더에서 재개할 수 없습니다. 이미 완료된 결과는 보존됩니다.

확장 모델의 전체 학습은 아직 실행하지 않았습니다. 크기가 크다고 정확도가 반드시 높아지는 것은 아니며 학습 시간과 메모리 사용량은 늘어날 수 있습니다. 모델 크기와 성능을 비교할 때는 동일한 평가 조건의 전체 fold 지표를 사용하세요.

## 검증

```powershell
.\.venv\Scripts\python.exe -m scripts.diagnostics.inspect_cnn_sizes
.\.venv\Scripts\python.exe -m unittest scripts.diagnostics.test_test_selected_suite -v
```

크기 점검은 실제 `(2, 3, 10000)` 입력으로 순전파·역전파, 유한한 gradient, 기존 96/256 및 확장 256/512 특징 차원, 분리한 헤드와 전체 모델 출력 일치, checkpoint 저장·복원 후 출력 일치를 확인합니다.
