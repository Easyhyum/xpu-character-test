"""
Batch text generation utilities with activation tracking for TPU devices.
"""
import os
os.environ['PJRT_DEVICE'] = 'TPU'

print("  [API] Loading batch_generator_tpu module...")

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import gc
import time
from .text_generator_tpu import create_activation_hook
from .layer_operation_tracker_tpu import LayerOperationTrackerTPU

print("  [API] ✓ batch_generator_tpu loaded successfully")


def process_batch_inference_tpu(model, tokenizer, batch_prompts, device, max_new_tokens=50, track_activations=False, csv_writer=None, logit_csv_writer=None, tpu_name=None, model_specific=None, batch_start_idx=0, output_dir=None, start_time=None, in_out_value_checkpointing=False):
    """
    Step 2: Batch로 여러 prompt를 한번에 처리하는 함수 (activation 추적 기능 포함) - TPU version
    
    Args:
        model: 언어 모델
        tokenizer: 토크나이저
        batch_prompts: 문자열 리스트 (여러 개의 prompt)
        device: TPU 디바이스
        max_new_tokens: 생성할 최대 토큰 수
        track_activations: activation 추적 여부
        csv_writer: CSV writer object (activation 저장용)
        logit_csv_writer: CSV writer object (logit 저장용)
        tpu_name: TPU 이름
        model_specific: 모델명
        batch_start_idx: 배치 시작 인덱스
        output_dir: Output directory for layer operation CSVs
        start_time: Run timestamp
    
    Returns:
        생성된 텍스트 리스트
    """
    print(f"\n[Step 2] Processing batch of {len(batch_prompts)} prompts on TPU...")
    
    if track_activations:
        print(f"  Activation tracking: ENABLED")
        return process_batch_with_activations_tpu(model, tokenizer, batch_prompts, device, max_new_tokens, csv_writer, logit_csv_writer, tpu_name, model_specific, batch_start_idx, output_dir, start_time, in_out_value_checkpointing)
    else:
        print(f"  Activation tracking: DISABLED")
        return process_batch_simple_tpu(model, tokenizer, batch_prompts, device, max_new_tokens)


def process_batch_simple_tpu(model, tokenizer, batch_prompts, device, max_new_tokens):
    """기본 배치 처리 (activation 추적 없음) - TPU version"""
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
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    
    tokenizer.padding_side = original_padding_side
    
    print(f"  Input shape: {inputs['input_ids'].shape}")
    print(f"  Attention mask shape: {inputs['attention_mask'].shape}")
    
    # Generate outputs for the batch
    print(f"  Starting generation (first run will take 2-10 minutes for XLA compilation)...")
    print(f"  Please wait - XLA is compiling the model graph for TPU...")
    import time
    gen_start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        )
    
    gen_time = time.time() - gen_start
    print(f"  ✓ Generation completed in {gen_time:.1f}s (subsequent runs will be much faster)")
    
    # Mark step for XLA
    xm.mark_step()
    
    # outputs는 [입력 + 생성된 텍스트]를 포함하므로, 입력 길이만큼 잘라내기
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[:, input_length:]
    
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
            input_tokens = inputs['input_ids'][batch_idx][start_idx:].tolist()
        else:
            input_tokens = inputs['input_ids'][batch_idx].tolist()
        
        input_token_ids.append(input_tokens)
        
        # Get generated tokens
        generated_token_ids.append(generated_tokens[batch_idx].tolist())
    
    print(f"  ✓ Generated {len(generated_texts)} responses")
    
    return generated_texts, generated_token_ids, input_token_ids


def process_batch_with_activations_tpu(model, tokenizer, batch_prompts, device, max_new_tokens, csv_writer, logit_csv_writer, tpu_name, model_specific, batch_start_idx, output_dir=None, start_time=None, in_out_value_checkpointing=False):
    """Activation 추적을 포함한 배치 처리 - TPU version"""
    
    # Set pad_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Warning: pad_token was None, set to eos_token ({tokenizer.eos_token})")
    
    # 모델 정보 가져오기
    transformer_layers = model.model.layers
    model_hidden_size = model.config.hidden_size
    num_layers = len(transformer_layers)
    
    print(f"\n  Processing batch with activation tracking on TPU...")
    
    # Get batch size
    batch_size = len(batch_prompts)
    
    # Initialize LayerOperationTrackerTPU for detailed operation I/O capture
    layer_tracker = None
    if output_dir and start_time:
        if in_out_value_checkpointing:
            layer_tracker = LayerOperationTrackerTPU(
                output_dir=output_dir,
                start_time=start_time,
                tpu_name=tpu_name,
                model_specific=model_specific,
                batch_size=batch_size,
                layer_indices=[0, -1],
                print_order=True
            )
            layer_tracker.register_hooks(model, tokenizer)
    
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
    
    print(f"  Input tensors device: {inputs['input_ids'].device}")
    print(f"  Model device: {next(model.parameters()).device}")
    
    tokenizer.padding_side = original_padding_side
    
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
        
        # Mark step for XLA
        xm.mark_step()
        
        # 첫 번째 샘플의 토큰 출력 (완료 여부에 따라)
        if finished[0]:
            print(".", end='', flush=True)
        else:
            new_token_text = tokenizer.decode(next_token_ids[0], skip_special_tokens=False)
            display_text = new_token_text.replace('\n', ' \\n ')
            print(display_text, end='', flush=True)
        
        # Save detailed layer operation data for each sample
        for batch_idx in range(batch_size):
            if layer_tracker is not None:
                input_text = batch_prompts[batch_idx]
                output_token_id = next_token_ids[batch_idx].item()
                output_text = tokenizer.decode([output_token_id], skip_special_tokens=False)
                
                layer_tracker.save_step_data(
                    input_index=batch_start_idx + batch_idx,
                    decoding_step=step,
                    input_text=input_text,
                    output_token_id=output_token_id,
                    output_text=output_text,
                    device=device,
                    batch_idx=batch_idx
                )
        
        # Clear layer tracker buffer after all batch samples are processed
        if layer_tracker is not None:
            layer_tracker.clear_step_buffer()
        
        # Write activations to CSV (배치 내 각 샘플별로)
        if csv_writer is not None:
            for batch_idx in range(batch_size):
                token_id = next_token_ids[batch_idx].item()
                token_text = tokenizer.decode([token_id], skip_special_tokens=False).replace('\n', '\\n').replace(',', '[COMMA]')
                
                for layer_idx in range(num_layers):
                    layer_name = f"layer_{layer_idx}"
                    if layer_name in step_activations:
                        # step_activations[layer_name]의 shape: (batch_size, 1, hidden_size)
                        # batch_idx에 해당하는 부분만 추출
                        activation = step_activations[layer_name][batch_idx:batch_idx+1].cpu().numpy()
                        activation_flat = activation.flatten()
                        
                        row = [tpu_name, model_specific, device.type, batch_start_idx + batch_idx, batch_prompts[batch_idx], layer_idx, step, token_id, token_text]
                        row.extend([f"{val:.8f}" for val in activation_flat])
                        csv_writer.writerow(row)
                        
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
            xm.mark_step()
            gc.collect()
        
        # 모든 샘플이 완료되면 중단
        if finished.all():
            break
    
    # Final XLA mark step
    xm.mark_step()
    
    end_time = time.time()
    print(f" [steps: {step+1}, tps: {(step+1)/(end_time - init_time):.2f}]")
    
    # Close layer operation tracker if it was initialized
    if layer_tracker is not None:
        layer_tracker.close()
    
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
            input_tokens = inputs['input_ids'][batch_idx][start_idx:].tolist()
        else:
            input_tokens = inputs['input_ids'][batch_idx].tolist()
        
        input_token_ids.append(input_tokens)
        
        # 생성된 부분만 추출
        input_len = input_lengths[batch_idx].item()
        generated_part = generated_ids[batch_idx][input_len:]
        generated_token_ids.append(generated_part.tolist())
        
        # 텍스트 디코딩
        generated_text = tokenizer.decode(generated_part, skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    # 메모리 정리
    del generated_ids, inputs
    gc.collect()
    
    print(f"  ✓ Generated {len(generated_texts)} responses with activation tracking")
    return generated_texts, generated_token_ids, input_token_ids
