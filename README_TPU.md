# TPU Version - README

## TPU 버전 파일들

이 프로젝트의 TPU 버전은 기존 GPU 코드를 그대로 유지하면서 TPU 지원을 위한 별도의 파일들로 구성되어 있습니다.

### 생성된 TPU 파일 목록

1. **test_tpu.py** - 메인 실행 파일 (TPU 버전)
2. **api/model_loader_tpu.py** - TPU용 모델 로딩
3. **api/text_generator_tpu.py** - TPU용 텍스트 생성
4. **api/batch_generator_tpu.py** - TPU용 배치 추론
5. **api/layer_operation_tracker_tpu.py** - TPU용 레이어 연산 추적
6. **run_test_tpu.sh** - TPU 실행 스크립트

### 주요 변경사항

#### 1. TPU 환경 설정
모든 TPU 파일은 상단에 다음 코드가 포함되어 있습니다:
```python
import os
os.environ['PJRT_DEVICE'] = 'TPU'

import torch_xla
import torch_xla.core.xla_model as xm
```

#### 2. 디바이스 변경
- CUDA → TPU (XLA)
- `torch.device("cuda")` → `xm.xla_device()`
- GPU 이름 → TPU 이름

#### 3. XLA Mark Step 추가
TPU에서 효율적인 실행을 위해 `xm.mark_step()` 호출 추가:
- 각 생성 스텝 후
- 메모리 정리 시점
- 배치 처리 완료 시점

#### 4. 메모리 관리
- `torch.cuda.empty_cache()` 제거
- TPU에 최적화된 가비지 컬렉션 유지

#### 5. 결정론적 모드
- `torch.cuda.manual_seed()` → `xm.set_rng_state()`
- CUDA 관련 설정 제거

### 사용 방법

#### TPU에서 실행
```bash
# 실행 권한 부여
chmod +x run_test_tpu.sh

# TPU 버전 실행
./run_test_tpu.sh
```

또는 직접 실행:
```bash
export PJRT_DEVICE=TPU
python3 test_tpu.py
```

#### GPU에서 실행 (기존 방식)
```bash
# 기존 코드는 그대로 유지됨
./run_test.sh
```

### TPU 디바이스 확인

실행 전 TPU가 정상적으로 인식되는지 확인:
```bash
python3 -c "import os; os.environ['PJRT_DEVICE'] = 'TPU'; import torch_xla.core.xla_model as xm; print(f'TPU Device: {xm.xla_device()}')"
```

예상 출력: `TPU Device: xla:0`

### 주의사항

1. **Quantized Models**: 4-bit, 8-bit quantized 모델은 TPU에서 제한적으로 지원될 수 있습니다.
2. **FP8 Models**: FP8 모델도 TPU 지원이 제한적일 수 있습니다.
3. **메모리**: TPU는 GPU와 다른 메모리 구조를 가지므로 배치 크기를 조정해야 할 수 있습니다.
4. **성능**: 첫 실행 시 XLA 컴파일로 인해 초기 속도가 느릴 수 있습니다.

### 파일 구조

```
xpu-character-test/
├── test.py                          # 기존 GPU 버전 (유지)
├── test_tpu.py                      # 새로운 TPU 버전
├── run_test.sh                      # 기존 GPU 실행 스크립트
├── run_test_tpu.sh                  # 새로운 TPU 실행 스크립트
├── api/
│   ├── __init__.py                  # 업데이트 (TPU 함수 export)
│   ├── model_loader.py              # 기존 GPU 버전
│   ├── model_loader_tpu.py          # 새로운 TPU 버전
│   ├── text_generator.py            # 기존 GPU 버전
│   ├── text_generator_tpu.py        # 새로운 TPU 버전
│   ├── batch_generator.py           # 기존 GPU 버전
│   ├── batch_generator_tpu.py       # 새로운 TPU 버전
│   ├── layer_operation_tracker.py   # 기존 GPU 버전
│   └── layer_operation_tracker_tpu.py  # 새로운 TPU 버전
└── ...
```

### 기술적 세부사항

#### XLA (Accelerated Linear Algebra)
- TPU는 XLA를 통해 작동
- `xm.mark_step()`은 XLA 그래프를 실행하는 동기화 포인트
- 너무 자주 호출하면 성능 저하, 너무 적게 호출하면 메모리 증가

#### 병렬 처리
- TPU는 데이터 병렬 처리에 최적화
- 배치 크기를 크게 설정하면 TPU 활용도 향상

#### 디버깅
TPU 관련 디버그 정보 출력:
```bash
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
```
