# KV Cache 간단 사용 가이드

## 핵심 변경사항

✨ **모델명 자동 감지**: `load_from_model` 설정이 제거되었습니다!
- 현재 실행 중인 모델명이 자동으로 사용됩니다
- `load_base_dir`만 지정하면 자동으로 올바른 경로를 찾습니다

## 빠른 시작

### 1단계: GPU에서 저장
```json
{
  "kv_cache": {
    "save": true,
    "load": false
  }
}
```
```bash
python test.py
# 결과: outputs/{timestamp}/kv_caches/NVIDIA_H200/redhatai_meta_llama_3_1_8b_fp8/*.pt
```

### 2단계: Reference cache 준비
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
        └── ...
```

### 3단계: CPU에서 재현
```json
{
  "cpu": "Enable",
  "kv_cache": {
    "save": false,
    "load": true,
    "load_from_device": "NVIDIA_H200",
    "load_from_batch_size": 16,
    "load_base_dir": "ref_cache"
  }
}
```
```bash
python test.py
# 자동으로 ref_cache/NVIDIA_H200/redhatai_meta_llama_3_1_8b_fp8/ 에서 로드
```

## 설정 옵션

| 옵션 | 필수 | 설명 | 예시 |
|------|------|------|------|
| `save` | No | 현재 디바이스 저장 여부 | `true`/`false` |
| `load` | No | Reference cache 로드 여부 | `true`/`false` |
| `load_base_dir` | Yes (load시) | Reference cache 폴더 | `"ref_cache"` |
| `load_from_device` | Yes (load시) | 원본 디바이스명 | `"NVIDIA_H200"` |
| `load_from_batch_size` | Yes (load시) | 원본 batch size | `16` |
| ~~`load_from_model`~~ | ~~삭제됨~~ | ~~자동 감지~~ | - |

## 경로 자동 구성

설정:
```json
{
  "load_base_dir": "ref_cache",
  "load_from_device": "NVIDIA_H200"
}
```

현재 실행 모델: `RedHatAI/Meta-Llama-3.1-8B-FP8`

자동으로 찾는 경로:
```
ref_cache/NVIDIA_H200/redhatai_meta_llama_3_1_8b_fp8/batch16_input*.pt
          ↑            ↑                                 ↑
    load_base_dir  load_from_device              현재 모델명 (자동)
```

## 출력 예시

```
[KV Cache Load Mode] Reproducing generation from saved KV caches...
  Loading KV caches from:
    Base dir: ref_cache
    Device: NVIDIA_H200
    Model: redhatai_meta_llama_3_1_8b_fp8 (auto-detected from current model)
    Batch size: 16
    Full path: ref_cache/NVIDIA_H200/redhatai_meta_llama_3_1_8b_fp8/

  Processing input 0...
    Reproducing: Hello! How can I help you today?
    Steps: 20, Match: 20/20 (100.0%), TPS: 15.32
```

## 문제 해결

### "KV cache not found for input X"

1. 경로 확인:
```bash
ls ref_cache/NVIDIA_H200/redhatai_meta_llama_3_1_8b_fp8/
```

2. 파일명 형식 확인:
```
batch16_input0.pt  # batch{load_from_batch_size}_input{index}.pt
```

3. 모델명 확인:
- 현재 실행: `RedHatAI/Meta-Llama-3.1-8B-FP8`
- 안전한 이름: `redhatai_meta_llama_3_1_8b_fp8` (소문자, `/`와 `-`는 `_`로 변환)

### Match rate가 낮음

정상입니다! 디바이스 간 수치 정밀도 차이로 인해:
- 90-100%: 매우 좋음
- 70-90%: 정상 (FP8 등 낮은 정밀도 사용 시)
- <70%: 문제 가능성

---

더 자세한 내용은 `KV_CACHE_USAGE.md`를 참고하세요!
