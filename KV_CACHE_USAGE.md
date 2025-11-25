# KV Cache 기능 사용 가이드

## 개요

KV Cache 기능을 사용하면:
1. **저장 모드**: 한 디바이스(예: GPU)에서 생성 시 KV Cache를 파일로 저장
2. **로드 모드**: 다른 디바이스(예: CPU)에서 저장된 KV Cache를 step-by-step으로 로드하여 재현

이를 통해 디바이스 간 생성 결과 일치 여부를 검증하고, 수치 차이를 분석할 수 있습니다.

---

## 사용 방법

### 1단계: GPU에서 KV Cache 저장

#### hyperparameter.json 설정:
```json
{
  "models": ["RedHatAI/Meta-Llama-3.1-8B-FP8"],
  "batch_size": [16],
  "request_number": 64,
  "max_new_tokens": 128,
  
  "kv_cache": {
    "save": true,                    // ✓ 저장 활성화
    "save_mode": "delta",             // delta 압축 방식 (권장)
    "load": false,                    // 로드는 비활성화
    "base_dir": "outputs/{timestamp}/kv_caches"
  }
}
```

#### 실행:
```bash
python test.py
```

#### 결과:
```
outputs/20251125-143000/kv_caches/
└── NVIDIA_H200/
    └── redhatai_meta_llama_3_1_8b_fp8/
        ├── batch16_input0.pt    (~114 MB)
        ├── batch16_input1.pt
        ├── batch16_input2.pt
        └── ... (64개 파일)
```

각 파일은:
- Prefill KV Cache (전체)
- 각 Decoding Step의 Delta (증분)
- 생성된 토큰 정보
- 메타데이터

를 포함합니다.

---

### 2단계: CPU에서 KV Cache 로드 및 재현

#### 디렉토리 구조 준비:
먼저 GPU에서 생성한 KV Cache를 `ref_cache` 폴더로 복사:
```bash
mkdir -p ref_cache
cp -r outputs/20251125-143000/kv_caches/NVIDIA_H200 ref_cache/
```

결과 구조:
```
ref_cache/
└── NVIDIA_H200/
    └── redhatai_meta_llama_3_1_8b_fp8/
        ├── batch16_input0.pt
        ├── batch16_input1.pt
        └── ...
```

#### hyperparameter.json 설정:
```json
{
  "models": ["RedHatAI/Meta-Llama-3.1-8B-FP8"],  // 동일한 모델 (자동 감지됨)
  "batch_size": [16],                             // 동일한 batch size
  "request_number": 64,                           // 동일한 input 개수
  "max_new_tokens": 128,
  "cpu": "Enable",                                // CPU 사용
  
  "kv_cache": {
    "save": true,                                 // ✓ 현재 디바이스 결과도 저장 (선택)
    "save_mode": "delta",
    "load": true,                                 // ✓ 로드 활성화
    "load_from_device": "NVIDIA_H200",            // GPU 디바이스명
    "load_from_batch_size": 16,                   // 원본 batch size
    "load_base_dir": "ref_cache",                 // Reference cache 폴더
    "base_dir": "outputs/{timestamp}/kv_caches"   // 현재 디바이스 저장 경로
  }
}
```

**중요**: 
- `load_from_model`은 제거되었습니다! 현재 실행 중인 모델명이 자동으로 사용됩니다.
- `load_base_dir`만 지정하면, 자동으로 `{load_base_dir}/{device}/{model}/` 경로를 찾습니다.

#### 실행:
```bash
python test.py
```

#### 동작:
1. `ref_cache/NVIDIA_H200/redhatai_meta_llama_3_1_8b_fp8/batch16_input0.pt` 파일 로드
   - 모델명은 현재 실행 중인 모델에서 자동으로 가져옴
2. Prefill KV Cache 로드 → 첫 토큰 생성
3. Step 0까지의 KV Cache 로드 (Prefill + Delta 0) → 두 번째 토큰 생성
4. Step 1까지의 KV Cache 로드 (Prefill + Delta 0-1) → 세 번째 토큰 생성
5. ...반복
6. 각 스텝마다 GPU에서 생성한 토큰과 비교하여 일치 여부 출력

#### 출력 예시:
```
[KV Cache Load Mode] Reproducing generation from saved KV caches...
  Loading KV caches from:
    Base dir: ref_cache
    Device: NVIDIA_H200
    Model: redhatai_meta_llama_3_1_8b_fp8 (auto-detected from current model)
    Batch size: 16
    Full path: ref_cache/NVIDIA_H200/redhatai_meta_llama_3_1_8b_fp8/

  Processing input 0...
    Input text: Hello, how are you?...
    Input tokens: 10
    Reproducing: I am doing great thanks! How about you?
    Steps: 20, Match: 20/20 (100.0%), TPS: 15.32
```

- `✓` 표시: 토큰 일치
- `[예측≠원본]` 표시: 토큰 불일치 (수치 차이로 인한 다른 선택)

---

## 설정 옵션 상세

### save (boolean)
- `true`: 현재 디바이스의 KV Cache 저장
- `false`: 저장하지 않음

### load (boolean)
- `true`: 다른 디바이스에서 저장한 KV Cache 로드하여 재현
- `false`: 일반 생성 모드

### save_mode (string)
- `"delta"`: Prefill + 각 step의 증분만 저장 (~114 MB per input) **권장**
- `"prefill_only"`: Prefill만 저장 (~50 MB per input)
- `"final_only"`: 최종 상태만 저장 (~72 MB per input)

### load_from_device (string)
- 원본 KV Cache를 저장한 디바이스 이름
- 예: `"NVIDIA_H200"`, `"CPU"`, `"NVIDIA_A100"`
- KV Cache 경로: `{load_base_dir}/{load_from_device}/{model}/`

### load_from_batch_size (integer)
- 원본 KV Cache를 저장할 때 사용한 batch size
- 파일명 생성에 사용: `batch{N}_input{M}.pt`

### load_base_dir (string)
- Reference cache의 base 디렉토리
- 예: `"ref_cache"`, `"outputs/20251125-143000/kv_caches"`
- 자동으로 `{load_base_dir}/{device}/{model}/`로 확장됨
- **중요**: 현재 모델명은 자동 감지되므로 별도 지정 불필요

### base_dir (string)
- 현재 디바이스의 KV Cache를 저장할 디렉토리
- `{timestamp}`는 자동으로 현재 실행 시간으로 치환

---

## 파일 구조

### 저장 시 (GPU):
```
outputs/
└── 20251125-143000/              # GPU 실행
    └── kv_caches/
        └── NVIDIA_H200/
            └── redhatai_meta_llama_3_1_8b_fp8/
                ├── batch16_input0.pt
                ├── batch16_input1.pt
                └── ...
```

### 로드 준비 (Reference cache):
GPU 결과를 `ref_cache` 폴더로 복사:
```bash
mkdir -p ref_cache
cp -r outputs/20251125-143000/kv_caches/NVIDIA_H200 ref_cache/
```

결과:
```
ref_cache/                        # load_base_dir
└── NVIDIA_H200/                  # load_from_device
    └── redhatai_meta_llama_3_1_8b_fp8/  # 현재 모델명 (자동 감지)
        ├── batch16_input0.pt
        ├── batch16_input1.pt
        └── ...
```

### 재현 시 (CPU):
```
outputs/
└── 20251125-150000/              # CPU 실행 (재현)
    └── kv_caches/
        └── CPU/
            └── redhatai_meta_llama_3_1_8b_fp8/
                ├── batch16_input0.pt   # CPU에서 생성한 KV Cache (선택)
                ├── batch16_input1.pt
                └── ...
```

---

## 사용 시나리오

### 시나리오 1: GPU vs CPU 비교
1. GPU에서 생성 + KV Cache 저장 (`save: true, load: false`)
2. GPU 결과를 `ref_cache`로 복사: `cp -r outputs/{timestamp}/kv_caches/NVIDIA_H200 ref_cache/`
3. CPU에서 `ref_cache` 로드 + 재현 (`save: false, load: true, load_base_dir: "ref_cache"`)
4. 토큰 일치율 확인

### 시나리오 2: 다른 GPU 간 비교
1. NVIDIA H200에서 저장
2. H200 결과를 `ref_cache`로 복사
3. NVIDIA A100에서 `ref_cache` 로드 + 재현
4. 하드웨어 차이로 인한 수치 오차 분석

### 시나리오 3: 양방향 저장
1. GPU에서 저장 (`save: true, load: false`)
2. GPU 결과를 `ref_cache`로 복사
3. CPU에서 로드 + CPU 결과도 저장 (`save: true, load: true`)
4. 나중에 두 디바이스의 KV Cache 직접 비교

---

## 주의사항

### 메모리 사용량
- **로드 모드**: 각 스텝마다 KV Cache 재구성 필요 (CPU에서 concat 연산)
- Input 하나당 메모리 사용: ~200-300 MB (일시적)
- Batch 단위가 아닌 Input 단위로 순차 처리하여 메모리 효율성 확보

### 디스크 공간
- Delta 모드: 64 inputs × 114 MB = ~7.3 GB
- 충분한 디스크 공간 확보 필요

### 모델 일치
- 동일한 모델 사용 필수
- Tokenizer도 동일해야 함
- 다른 precision (FP8 vs FP16)도 테스트 가능

### 파일명 규칙
- `batch{batch_size}_input{input_index}.pt`
- `load_from_batch_size`와 원본 batch size가 일치해야 파일을 찾을 수 있음

---

## 문제 해결

### "KV cache not found for input X"
- `load_base_dir` 경로 확인
- `load_from_device`, `load_from_batch_size` 확인
- 현재 실행 중인 모델명 확인 (자동으로 사용됨)
- 파일 존재 여부: `ls ref_cache/NVIDIA_H200/redhatai_meta_llama_3_1_8b_fp8/`
- 경로 구조: `{load_base_dir}/{device}/{current_model}/batch{batch_size}_input{index}.pt`

### Match rate가 낮음 (예: 50%)
- 정상적인 현상일 수 있음 (디바이스 간 수치 정밀도 차이)
- FP8 vs FP16/FP32 사용 시 차이 발생 가능
- Temperature=0 (greedy decoding)에서도 logit 값이 미세하게 달라 다른 토큰 선택 가능

### Out of Memory
- `batch_size`를 줄이기
- `request_number`를 줄이기
- `max_new_tokens`를 줄이기

---

## API 사용 (Python)

```python
from api import KVCacheManager

# KV Cache Manager 초기화
kv_manager = KVCacheManager(
    base_dir="ref_cache",                        # Reference cache directory
    gpu_name="NVIDIA_H200",                      # Original device
    model_specific="redhatai_meta_llama_3_1_8b_fp8",  # Current model name
    batch_size=16,
    save_mode='delta'
)

# 캐시 존재 확인
if kv_manager.cache_exists(input_index=0):
    print("Cache found!")

# Step 5까지의 KV Cache 로드
kv_data = kv_manager.load_kv_cache_for_step(
    input_index=0,
    target_step=5,
    device='cpu'
)

# KV Cache 정보 조회
info = kv_manager.get_cache_info(input_index=0)
print(f"File size: {info['file_size_mb']:.1f} MB")
print(f"Prefill length: {info['prefill_length']}")
print(f"Total steps: {info['num_decoding_steps']}")

# 다음 토큰 생성
outputs = model(
    input_ids=last_token,
    past_key_values=kv_data['past_key_values'],
    use_cache=True
)
```

---

## 요약

| 모드 | save | load | 동작 |
|------|------|------|------|
| **일반 생성** | false | false | 일반 텍스트 생성만 |
| **저장 모드** | true | false | 생성 + KV Cache 저장 |
| **재현 모드** | false | true | 저장된 KV Cache로 재현 |
| **저장+재현** | true | true | 재현 + 현재 디바이스도 저장 |

추가 질문이나 문제가 있으면 `example_kv_cache_usage.py`를 참고하거나 이슈를 등록하세요!
