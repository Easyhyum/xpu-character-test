# Quantized Model Support - GPU vs TPU

## INT8 Quantized Models (W8A8)

### GPU Version (model_loader.py)
- **전략**: Dequantize on-the-fly
- **이유**: GPU는 FP16/FP32 연산이 더 빠름
- **동작**: INT8 weight를 FP32로 변환하여 연산
- **장점**: 범용성, 안정성
- **단점**: 메모리 사용량 증가

### TPU Version (model_loader_tpu.py)
- **전략**: Native INT8 execution
- **이유**: TPU는 INT8 연산 하드웨어 지원
- **동작**: INT8 weight를 그대로 사용하여 연산
- **장점**: 메모리 효율, 속도 향상 가능
- **단점**: TPU INT8 지원에 의존

## FP8 Models

### Both Versions
- **전략**: Convert to FP32/BF16
- **이유**: FP8은 표준이 아직 불안정
- **GPU**: FP32로 변환
- **TPU**: FP32로 변환

## 4-bit Quantized Models (BNB)

### Both Versions
- **전략**: Keep as-is
- **제한사항**: 
  - GPU: bitsandbytes 라이브러리 필요
  - TPU: 제한적 지원, 성능 문제 가능

## Default Models (FP32/FP16)

### GPU Version
- **기본 dtype**: auto (모델 기본값)
- **device_map**: "auto" (자동 배치)

### TPU Version
- **기본 dtype**: bfloat16
- **device_map**: None → 수동으로 TPU로 이동
- **이유**: TPU는 BF16 최적화됨

## 사용 예시

```python
# INT8 모델 - 자동으로 적절한 전략 사용
model = model_load_function("RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8")  # GPU
model = model_load_function_tpu("RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8")  # TPU

# 일반 모델 - 최적 dtype 자동 선택
model = model_load_function("meta-llama/Llama-3.1-8B-Instruct")  # GPU: auto
model = model_load_function_tpu("meta-llama/Llama-3.1-8B-Instruct")  # TPU: BF16
```
