"""
KV Cache 파일 조회 스크립트

저장된 .pt 파일의 내용을 읽어서 상세 정보를 출력합니다.
"""

import torch
import sys
import os


def format_bytes(bytes):
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def format_tensor_info(tensor):
    """Format tensor information"""
    if isinstance(tensor, torch.Tensor):
        return f"shape={list(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"
    return str(type(tensor))


def inspect_kv_cache(filepath):
    """
    KV Cache 파일의 상세 정보를 출력
    
    Args:
        filepath: .pt 파일 경로
    """
    
    print("="*80)
    print("KV Cache 파일 조회")
    print("="*80)
    
    # 파일 존재 확인
    if not os.path.exists(filepath):
        print(f"\n❌ 파일을 찾을 수 없습니다: {filepath}")
        return
    
    # 파일 크기
    file_size = os.path.getsize(filepath)
    print(f"\n📁 파일 정보:")
    print(f"   경로: {filepath}")
    print(f"   크기: {format_bytes(file_size)}")
    
    # 파일 로드
    print(f"\n📥 파일 로딩 중...")
    try:
        data = torch.load(filepath, map_location='cpu')
        print(f"   ✓ 로드 완료")
    except Exception as e:
        print(f"   ❌ 로드 실패: {e}")
        return
    
    # 최상위 키 출력
    print(f"\n📋 최상위 구조:")
    for key in data.keys():
        print(f"   - {key}")
    
    # Metadata 출력
    if 'metadata' in data:
        print(f"\n🏷️  메타데이터:")
        metadata = data['metadata']
        for key, value in metadata.items():
            print(f"   {key:20s}: {value}")
    
    # Input data 출력
    if 'input_data' in data:
        print(f"\n📝 입력 데이터:")
        input_data = data['input_data']
        
        if 'text' in input_data:
            text = input_data['text']
            preview = text[:100] + "..." if len(text) > 100 else text
            print(f"   텍스트: {preview}")
            print(f"   텍스트 길이: {len(text)} 문자")
        
        if 'token_ids' in input_data:
            tokens = input_data['token_ids']
            print(f"   토큰 수: {len(tokens)}")
            print(f"   토큰 ID (처음 20개): {tokens[:20]}")
        
        if 'token_length' in input_data:
            print(f"   토큰 길이: {input_data['token_length']}")
    
    # Prefill KV Cache 정보
    if 'prefill_kv' in data:
        print(f"\n🔄 Prefill KV Cache:")
        prefill = data['prefill_kv']
        
        if 'seq_length' in prefill:
            print(f"   시퀀스 길이: {prefill['seq_length']}")
        
        if 'attention_mask' in prefill:
            mask = prefill['attention_mask']
            print(f"   Attention Mask: {format_tensor_info(mask)}")
        
        if 'past_key_values' in prefill:
            kv = prefill['past_key_values']
            print(f"   KV Cache 구조:")
            print(f"     레이어 수: {len(kv)}")
            
            if len(kv) > 0:
                # 첫 번째 레이어 정보
                first_layer = kv[0]
                keys, values = first_layer
                print(f"     각 레이어:")
                print(f"       Keys:   {format_tensor_info(keys)}")
                print(f"       Values: {format_tensor_info(values)}")
                
                # 메모리 계산
                key_size = keys.element_size() * keys.nelement()
                value_size = values.element_size() * values.nelement()
                total_per_layer = key_size + value_size
                total_all_layers = total_per_layer * len(kv)
                
                print(f"     메모리 사용량:")
                print(f"       레이어당: {format_bytes(total_per_layer)}")
                print(f"       전체:     {format_bytes(total_all_layers)}")
    
    # Decoding deltas 정보
    if 'decoding_deltas' in data:
        deltas = data['decoding_deltas']
        print(f"\n🔢 Decoding Steps:")
        print(f"   총 스텝 수: {len(deltas)}")
        
        if len(deltas) > 0:
            # 처음 5개 스텝 출력
            print(f"\n   처음 5개 스텝:")
            for i, step_data in enumerate(deltas[:5]):
                token_id = step_data.get('token_id', 'N/A')
                token_text = step_data.get('token_text', 'N/A')
                # 개행 문자 표시
                display_text = token_text.replace('\n', '\\n')
                print(f"     Step {i}: token_id={token_id:6d}, text='{display_text}'")
            
            if len(deltas) > 5:
                print(f"     ... ({len(deltas) - 5}개 더)")
            
            # Delta 크기 정보
            first_delta = deltas[0]
            if 'kv_delta' in first_delta:
                kv_delta = first_delta['kv_delta']
                print(f"\n   Delta KV Cache 구조:")
                print(f"     레이어 수: {len(kv_delta)}")
                
                if len(kv_delta) > 0:
                    keys, values = kv_delta[0]
                    print(f"     각 Delta (1 토큰):")
                    print(f"       Keys:   {format_tensor_info(keys)}")
                    print(f"       Values: {format_tensor_info(values)}")
                    
                    # Delta 메모리
                    key_size = keys.element_size() * keys.nelement()
                    value_size = values.element_size() * values.nelement()
                    total_per_delta = (key_size + value_size) * len(kv_delta)
                    
                    print(f"     Delta 메모리 (레이어당):")
                    print(f"       1개 Delta: {format_bytes(total_per_delta)}")
                    print(f"       전체 {len(deltas)}개: {format_bytes(total_per_delta * len(deltas))}")
            
            # 마지막 스텝의 cumulative length
            last_step = deltas[-1]
            if 'cumulative_seq_length' in last_step:
                print(f"\n   최종 시퀀스 길이: {last_step['cumulative_seq_length']}")
    
    # Generation result
    if 'generation_result' in data:
        print(f"\n✨ 생성 결과:")
        result = data['generation_result']
        
        if 'generated_text' in result:
            text = result['generated_text']
            preview = text[:200] + "..." if len(text) > 200 else text
            print(f"   생성된 텍스트: {preview}")
        
        if 'generated_token_ids' in result:
            tokens = result['generated_token_ids']
            print(f"   생성된 토큰 수: {len(tokens)}")
            print(f"   토큰 ID (처음 20개): {tokens[:20]}")
        
        if 'total_steps' in result:
            print(f"   총 스텝 수: {result['total_steps']}")
    
    # 전체 파일 구조 요약
    print(f"\n📊 전체 구조 요약:")
    total_memory = 0
    
    if 'prefill_kv' in data and 'past_key_values' in data['prefill_kv']:
        kv = data['prefill_kv']['past_key_values']
        if len(kv) > 0:
            keys, values = kv[0]
            prefill_size = (keys.element_size() * keys.nelement() + 
                           values.element_size() * values.nelement()) * len(kv)
            total_memory += prefill_size
            print(f"   Prefill KV Cache: {format_bytes(prefill_size)}")
    
    if 'decoding_deltas' in data:
        deltas = data['decoding_deltas']
        if len(deltas) > 0 and 'kv_delta' in deltas[0]:
            kv_delta = deltas[0]['kv_delta']
            if len(kv_delta) > 0:
                keys, values = kv_delta[0]
                delta_size = (keys.element_size() * keys.nelement() + 
                             values.element_size() * values.nelement()) * len(kv_delta)
                total_deltas_size = delta_size * len(deltas)
                total_memory += total_deltas_size
                print(f"   Decoding Deltas ({len(deltas)}개): {format_bytes(total_deltas_size)}")
    
    print(f"   ────────────────────────────")
    print(f"   전체 KV Cache: {format_bytes(total_memory)}")
    print(f"   파일 크기:     {format_bytes(file_size)}")
    overhead = file_size - total_memory
    print(f"   오버헤드:      {format_bytes(overhead)} ({100*overhead/file_size:.1f}%)")
    
    print(f"\n" + "="*80)
    print("조회 완료")
    print("="*80)


if __name__ == "__main__":
    # 기본 파일 경로
    default_path = "/home/work/easyhyum/xpu-character-test/outputs/20251125-144134/kv_caches/NVIDIA H200/redhatai_meta_llama_3.1_8b_fp8/batch16_input0.pt"
    
    # 명령줄 인자로 파일 경로 받기
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = default_path
    
    inspect_kv_cache(filepath)
