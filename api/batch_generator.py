"""
Batch text generation utilities with activation tracking for the XPU character test project.
"""

import torch
import gc
import time
from .text_generator import create_activation_hook


def process_batch_inference(model, tokenizer, batch_prompts, device, max_new_tokens=50, track_activations=False, csv_writer=None, gpu_name=None, model_specific=None, batch_start_idx=0):
    """
    Step 2: Batch로 여러 prompt를 한번에 처리하는 함수 (activation 추적 기능 포함)
    
    Args:
        model: 언어 모델
        tokenizer: 토크나이저
        batch_prompts: 문자열 리스트 (여러 개의 prompt)
        device: 실행할 디바이스
        max_new_tokens: 생성할 최대 토큰 수
        track_activations: activation 추적 여부
        csv_writer: CSV writer object (activation 저장용)
        gpu_name: GPU 이름
        model_specific: 모델명
        batch_start_idx: 배치 시작 인덱스
    
    Returns:
        생성된 텍스트 리스트
    """
    print(f"\n[Step 2] Processing batch of {len(batch_prompts)} prompts...")
    
    if track_activations:
        print(f"  Activation tracking: ENABLED")
        return process_batch_with_activations(model, tokenizer, batch_prompts, device, max_new_tokens, csv_writer, gpu_name, model_specific, batch_start_idx)
    else:
        print(f"  Activation tracking: DISABLED")
        return process_batch_simple(model, tokenizer, batch_prompts, device, max_new_tokens)


def process_batch_simple(model, tokenizer, batch_prompts, device, max_new_tokens):
    """기본 배치 처리 (activation 추적 없음)"""
    # Set pad_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Warning: pad_token was None, set to eos_token ({tokenizer.eos_token})")
    
    # Tokenize all prompts at once (batch processing)
    # padding_side를 'left'로 설정 (생성 시 필요)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    inputs = tokenizer(
        batch_prompts, 
        return_tensors="pt", 
        padding=True,  # 길이를 맞추기 위해 padding
        truncation=True,  # 너무 긴 경우 자르기
        max_length=512  # 최대 입력 길이
    ).to(device)
    
    tokenizer.padding_side = original_padding_side
    
    print(f"  Input shape: {inputs['input_ids'].shape}")
    print(f"  Attention mask shape: {inputs['attention_mask'].shape}")
    
    # Generate outputs for the batch
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],  # 중요: attention_mask 명시적으로 전달
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding (deterministic) - temperature/top_p not needed
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
    
    # outputs는 [입력 + 생성된 텍스트]를 포함하므로, 입력 길이만큼 잘라내기
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[:, input_length:]  # 입력 부분 제거, 생성된 부분만 추출
    
    print(f"  Input length: {input_length}, Output length: {outputs.shape[1]}, Generated length: {generated_tokens.shape[1]}")
    
    # Decode only the generated part
    generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    # Extract token IDs as lists (both input and generated)
    input_token_ids = []
    generated_token_ids = []
    
    # Get actual input tokens for each sample (excluding padding)
    for batch_idx in range(len(batch_prompts)):
        # attention_mask에서 1인 부분만 추출 (왼쪽 padding 제외)
        mask = inputs['attention_mask'][batch_idx]
        # 왼쪽 padding을 건너뛰기 위해 mask가 1인 첫 번째 위치부터 추출
        non_pad_indices = mask.nonzero(as_tuple=True)[0]
        if len(non_pad_indices) > 0:
            start_idx = non_pad_indices[0].item()
            end_idx = non_pad_indices[-1].item() + 1
            input_tokens = inputs['input_ids'][batch_idx][start_idx:end_idx].tolist()
        else:
            input_tokens = []
        
        input_token_ids.append(input_tokens)
        
        # Get generated tokens
        generated_token_ids.append(generated_tokens[batch_idx].tolist())
    
    print(f"  ✓ Generated {len(generated_texts)} responses")
    
    return generated_texts, generated_token_ids, input_token_ids


def process_batch_with_activations(model, tokenizer, batch_prompts, device, max_new_tokens, csv_writer, gpu_name, model_specific, batch_start_idx):
    """Activation 추적을 포함한 배치 처리 - 배치 단위로 디코딩하고 input별로 activation 저장"""
    
    # Set pad_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Warning: pad_token was None, set to eos_token ({tokenizer.eos_token})")
    
    # 모델 정보 가져오기
    transformer_layers = model.model.layers
    model_hidden_size = model.config.hidden_size
    num_layers = len(transformer_layers)
    
    print(f"\n  Processing batch with activation tracking...")
    
    # padding_side를 'left'로 설정 (생성 시 필요)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    # 배치 토크나이징
    inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    tokenizer.padding_side = original_padding_side
    
    batch_size = len(batch_prompts)
    generated_ids = inputs['input_ids'].clone()
    eos_token_id = tokenizer.eos_token_id
    
    # 각 샘플의 완료 여부 추적
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
    
    # 첫 번째 input 텍스트 출력
    print(f"    First input: {batch_prompts[0]}")
    print(f"    Generating (batch_size={batch_size}): ", end='', flush=True)
    init_time = time.time()
    
    # Step-by-step generation with activation tracking
    for step in range(max_new_tokens):
        step_activations = {}
        hooks = []
        
        # Register hooks for each layer
        for layer_idx, layer in enumerate(transformer_layers):
            hook = create_activation_hook(f"layer_{layer_idx}", step_activations)
            hooks.append(layer.register_forward_hook(hook))
        
        # Forward pass (배치 전체) - attention_mask를 동적으로 업데이트
        # 원본 입력의 attention_mask를 확장
        current_attention_mask = torch.cat([
            inputs['attention_mask'],
            torch.ones((batch_size, generated_ids.size(1) - inputs['attention_mask'].size(1)), 
                      dtype=inputs['attention_mask'].dtype, device=device)
        ], dim=1)
        
        with torch.no_grad():
            outputs = model(input_ids=generated_ids, attention_mask=current_attention_mask)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Get next tokens (greedy decoding for batch)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # 첫 번째 샘플의 토큰 출력 (완료 여부에 따라)
        if finished[0]:
            print("\\end", end='', flush=True)
        else:
            new_token_text = tokenizer.decode(next_token_ids[0], skip_special_tokens=False)
            display_text = new_token_text.replace('\n', ' \\n ')
            print(display_text, end='', flush=True)
        
        # Write activations to CSV (배치 내 각 샘플별로)
        if csv_writer is not None:
            for batch_idx in range(batch_size):
                # 이미 완료된 샘플은 건너뛰기
                if finished[batch_idx]:
                    continue
                
                input_index = batch_start_idx + batch_idx
                prompt = batch_prompts[batch_idx]
                token_id = next_token_ids[batch_idx].item()
                token_text = tokenizer.decode(next_token_ids[batch_idx], skip_special_tokens=False)
                token_text = token_text.replace('\n', '\\n').replace(',', '[COMMA]')
                
                # 각 layer의 activation 저장
                for layer_idx in range(num_layers):
                    layer_name = f"layer_{layer_idx}"
                    if layer_name in step_activations:
                        # 해당 배치 인덱스의 activation만 추출
                        activation = step_activations[layer_name][batch_idx:batch_idx+1, -1:, :].cpu().numpy()
                        activation_flat = activation.flatten()
                        
                        row = [gpu_name, model_specific, device.type, batch_size, input_index, prompt[:100], layer_idx, step, token_id, token_text]
                        row.extend([f"{val:.8f}" for val in activation_flat])
                        csv_writer.writerow(row)
                        
                        # 메모리에서 즉시 삭제
                        del activation, activation_flat
        
        # step_activations 삭제
        del step_activations
        step_activations = None
        
        # Add next tokens to sequences
        generated_ids = torch.cat([generated_ids, next_token_ids], dim=-1)
        
        # 완료 상태 업데이트 (EOS 토큰 체크)
        finished = finished | (next_token_ids.squeeze(-1) == eos_token_id)
        
        # 중간 변수 정리
        del next_token_logits, outputs
        
        # 메모리 정리 (5스텝마다)
        if step % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # 모든 샘플이 완료되면 중단
        if finished.all():
            break
    
    end_time = time.time()
    print(f" [steps: {step+1}, tps: {(step+1)/(end_time - init_time):.2f}]")
    
    # 생성된 텍스트 추출 (입력 제외)
    input_lengths = inputs['attention_mask'].sum(dim=1)
    generated_texts = []
    generated_token_ids = []
    input_token_ids = []
    
    for batch_idx in range(batch_size):
        # attention_mask에서 1인 부분만 추출 (왼쪽 padding 제외)
        mask = inputs['attention_mask'][batch_idx]
        # 왼쪽 padding을 건너뛰기 위해 mask가 1인 첫 번째 위치부터 추출
        non_pad_indices = mask.nonzero(as_tuple=True)[0]
        if len(non_pad_indices) > 0:
            start_idx = non_pad_indices[0].item()
            end_idx = non_pad_indices[-1].item() + 1
            input_tokens = inputs['input_ids'][batch_idx][start_idx:end_idx].tolist()
            input_length = end_idx
        else:
            input_tokens = []
            input_length = 0
        
        input_token_ids.append(input_tokens)
        
        # Get generated tokens
        generated_tokens = generated_ids[batch_idx, input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_texts.append(generated_text)
        generated_token_ids.append(generated_tokens.tolist())
    
    # 메모리 정리
    del generated_ids, inputs, next_token_ids, finished
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"  ✓ Generated {len(generated_texts)} responses with activation tracking")
    return generated_texts, generated_token_ids, input_token_ids