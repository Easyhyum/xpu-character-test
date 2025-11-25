"""
KV Cache 파일의 입력 토큰을 상세 분석하는 스크립트
"""

import torch
import sys

filepath = "/home/work/easyhyum/xpu-character-test/outputs/20251125-144134/kv_caches/NVIDIA H200/redhatai_meta_llama_3.1_8b_fp8/batch16_input0.pt"

print("="*80)
print("입력 토큰 상세 분석")
print("="*80)

# 파일 로드
data = torch.load(filepath, map_location='cpu')

# Input data
input_data = data['input_data']
token_ids = input_data['token_ids']

print(f"\n📝 저장된 입력 토큰:")
print(f"   토큰 수: {len(token_ids)}")
print(f"   토큰 ID: {token_ids}")

# Prefill KV Cache
prefill = data['prefill_kv']
seq_length = prefill['seq_length']
attention_mask = prefill['attention_mask']

print(f"\n🔄 Prefill KV Cache:")
print(f"   시퀀스 길이: {seq_length}")
print(f"   Attention Mask shape: {attention_mask.shape}")

# Attention mask 분석
mask = attention_mask[0]  # [53]
non_zero = mask.nonzero(as_tuple=True)[0]

print(f"\n🔍 Attention Mask 분석:")
print(f"   전체 길이: {len(mask)}")
print(f"   Non-zero (실제 토큰) 개수: {len(non_zero)}")
print(f"   Zero (패딩) 개수: {len(mask) - len(non_zero)}")

if len(non_zero) > 0:
    first_non_zero = non_zero[0].item()
    last_non_zero = non_zero[-1].item()
    print(f"   첫 번째 non-zero 위치: {first_non_zero}")
    print(f"   마지막 non-zero 위치: {last_non_zero}")
    print(f"   실제 토큰 범위: [{first_non_zero}:{last_non_zero+1}] = {last_non_zero - first_non_zero + 1}개")

# Mask 시각화
print(f"\n📊 Attention Mask 시각화 (1=토큰, 0=패딩):")
mask_str = "".join(["1" if m == 1 else "0" for m in mask.tolist()])
print(f"   {mask_str[:53]}")
print(f"   {'↑'*len(non_zero) + ' '*(len(mask)-len(non_zero))}")
print(f"   패딩: {' '*(first_non_zero if len(non_zero) > 0 else 0)}← {first_non_zero if len(non_zero) > 0 else 0}개")

# 왜 53인가?
print(f"\n❓ 왜 시퀀스 길이가 {seq_length}인가?")
print(f"   입력 토큰: {len(token_ids)}개")
print(f"   패딩: {first_non_zero if len(non_zero) > 0 else 0}개 (left padding)")
print(f"   실제 토큰: {len(non_zero)}개")
print(f"   합계: {first_non_zero + len(non_zero) if len(non_zero) > 0 else 0}개")

# Batch processing 설명
print(f"\n💡 설명:")
print(f"   배치 처리 시 모든 입력의 길이를 맞추기 위해 left padding 추가")
print(f"   배치 내 가장 긴 입력에 맞춰 패딩됨")
print(f"   ")
print(f"   원본 입력: 'Hello, how are you?' = {len(token_ids)}개 토큰")
print(f"   배치 크기: {data['metadata']['batch_size']}")
print(f"   배치 내 최대 길이: {seq_length}개 (다른 input이 더 길었을 수 있음)")
print(f"   ")
print(f"   따라서 {len(token_ids)}개 토큰 → {first_non_zero}개 패딩 추가 → {seq_length}개 시퀀스")

print(f"\n" + "="*80)
