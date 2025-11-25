"""
Batch text generation utilities with activation tracking for the XPU character test project.
"""

import torch
import gc
import time
from .text_generator import create_activation_hook
from .layer_operation_tracker import LayerOperationTracker
from .kv_cache_manager import KVCacheManager


def process_batch_inference(model, tokenizer, batch_prompts, device, max_new_tokens=50, track_activations=False, csv_writer=None, logit_csv_writer=None, gpu_name=None, model_specific=None, batch_start_idx=0, output_dir=None, start_time=None, in_out_value_checkpointing=False, save_kv_cache=False, kv_cache_config=None):
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
        logit_csv_writer: CSV writer object (logit 저장용)
        gpu_name: GPU 이름
        model_specific: 모델명
        batch_start_idx: 배치 시작 인덱스
        output_dir: Output directory for layer operation CSVs
        start_time: Run timestamp
        in_out_value_checkpointing: Layer operation tracking
        save_kv_cache: Whether to save KV cache
        kv_cache_config: KV cache configuration dict
    
    Returns:
        생성된 텍스트 리스트
    """
    print(f"\n[Step 2] Processing batch of {len(batch_prompts)} prompts...")
    
    # Check if we should load KV cache instead of normal generation
    load_kv_cache = kv_cache_config.get('load', False) if kv_cache_config else False
    
    if load_kv_cache:
        print(f"  KV Cache loading: ENABLED")
        return process_batch_with_kv_cache_loading(
            model, tokenizer, batch_prompts, device, max_new_tokens,
            csv_writer, logit_csv_writer, gpu_name, model_specific,
            batch_start_idx, output_dir, start_time, 
            in_out_value_checkpointing, save_kv_cache, kv_cache_config
        )
    
    if track_activations:
        print(f"  Activation tracking: ENABLED")
        return process_batch_with_activations(model, tokenizer, batch_prompts, device, max_new_tokens, csv_writer, logit_csv_writer, gpu_name, model_specific, batch_start_idx, output_dir, start_time, in_out_value_checkpointing, save_kv_cache, kv_cache_config)
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


def process_batch_with_activations(model, tokenizer, batch_prompts, device, max_new_tokens, csv_writer, logit_csv_writer, gpu_name, model_specific, batch_start_idx, output_dir=None, start_time=None, in_out_value_checkpointing=False, save_kv_cache=False, kv_cache_config=None):
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
    
    # Get batch size
    batch_size = len(batch_prompts)
    
    # Initialize LayerOperationTracker for detailed operation I/O capture
    layer_tracker = None
    # print('Test', output_dir and start_time and in_out_value_checkpointing)
    if output_dir and start_time:
        if in_out_value_checkpointing:
            layer_tracker = LayerOperationTracker(
                output_dir=output_dir,
                start_time=start_time,
                gpu_name=gpu_name,
                model_specific=model_specific,
                batch_size=batch_size,
                layer_indices=[0, -1]  # First and last layers
            )
            # Register hooks for detailed operation tracking
            layer_tracker.register_hooks(model, tokenizer)
    
    # Initialize KVCacheManager for KV cache saving
    kv_manager = None
    if save_kv_cache and kv_cache_config:
        kv_manager = KVCacheManager(
            base_dir=kv_cache_config.get('base_dir', f"{output_dir}/kv_caches"),
            gpu_name=gpu_name,
            model_specific=model_specific,
            batch_size=batch_size,
            save_mode=kv_cache_config.get('save_mode', 'delta')
        )
        print(f"  KV Cache saving: ENABLED (mode: {kv_cache_config.get('save_mode', 'delta')})")
    
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
    
    # KV Cache for generation (will be updated each step)
    past_kv = None
    
    # Extract input token IDs (for KV Cache metadata)
    input_token_ids_list = []
    for batch_idx in range(batch_size):
        mask = inputs['attention_mask'][batch_idx]
        non_pad_indices = mask.nonzero(as_tuple=True)[0]
        if len(non_pad_indices) > 0:
            start_idx = non_pad_indices[0].item()
            end_idx = non_pad_indices[-1].item() + 1
            tokens = inputs['input_ids'][batch_idx][start_idx:end_idx].tolist()
        else:
            tokens = []
        input_token_ids_list.append(tokens)
    
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
        
        # Prepare inputs for this step
        if step == 0:
            # Prefill: use full input with original attention_mask
            step_input_ids = generated_ids
            current_attention_mask = torch.cat([
                inputs['attention_mask'],
                torch.ones((batch_size, generated_ids.size(1) - inputs['attention_mask'].size(1)), 
                          dtype=inputs['attention_mask'].dtype, device=device)
            ], dim=1)
        else:
            # Decoding: use only last token with past_key_values
            step_input_ids = generated_ids[:, -1:]
            # Extend attention_mask to include all generated tokens
            current_attention_mask = torch.cat([
                inputs['attention_mask'],
                torch.ones((batch_size, generated_ids.size(1) - inputs['attention_mask'].size(1)), 
                          dtype=inputs['attention_mask'].dtype, device=device)
            ], dim=1)
        
        with torch.no_grad():
            outputs = model(
                input_ids=step_input_ids,
                attention_mask=current_attention_mask,  # Always pass attention_mask
                past_key_values=past_kv if step > 0 else None,
                use_cache=True,  # Enable KV caching
                return_dict=True
            )
        
        # Update past_key_values for next iteration
        past_kv = outputs.past_key_values
        
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
        
        # Save detailed layer operation data for each sample
        for batch_idx in range(batch_size):
            if finished[batch_idx]:
                continue
            
            input_index = batch_start_idx + batch_idx
            prompt = batch_prompts[batch_idx]
            token_id = next_token_ids[batch_idx].item()
            token_text = tokenizer.decode(next_token_ids[batch_idx], skip_special_tokens=False)
            
            # Save KV Cache if enabled
            if kv_manager is not None:
                if step == 0:
                    # Prefill step: save full KV cache
                    # Extract individual attention mask for this sample (remove padding)
                    sample_mask = inputs['attention_mask'][batch_idx]
                    non_pad_indices = sample_mask.nonzero(as_tuple=True)[0]
                    if len(non_pad_indices) > 0:
                        # Create mask without left padding
                        actual_length = len(input_token_ids_list[batch_idx])
                        sample_attention_mask = torch.ones((1, actual_length), dtype=sample_mask.dtype, device='cpu')
                    else:
                        sample_attention_mask = torch.ones((1, 1), dtype=sample_mask.dtype, device='cpu')
                    
                    kv_manager.save_prefill(
                        input_index=input_index,
                        batch_idx=batch_idx,
                        kv_cache=past_kv,
                        input_text=prompt,
                        input_tokens=input_token_ids_list[batch_idx],
                        attention_mask=sample_attention_mask
                    )
                else:
                    # Decoding step: save delta
                    kv_manager.save_decoding_step(
                        input_index=input_index,
                        batch_idx=batch_idx,
                        step=step - 1,  # 0-based decoding step
                        kv_cache=past_kv,
                        token_id=token_id,
                        token_text=token_text
                    )
            
            # Save layer operation data if tracker is enabled
            # print("Test2", layer_tracker)
            if layer_tracker is not None:
                layer_tracker.save_step_data(
                    input_index=input_index,
                    decoding_step=step,
                    input_text=prompt,
                    output_token_id=token_id,
                    output_text=token_text,
                    device=gpu_name,
                    batch_idx=batch_idx  # 배치 내 인덱스 전달
                )
        
        # Clear layer tracker buffer after all batch samples are processed
        if layer_tracker is not None:
            layer_tracker.clear_step_buffer()
        
        # Write activations to CSV (배치 내 각 샘플별로)
        if csv_writer is not None:
            #logit save
            for batch_idx in range(batch_size):
                # 이미 완료된 샘플은 건너뛰기
                if finished[batch_idx]:
                    continue
                
                input_index = batch_start_idx + batch_idx
                prompt = batch_prompts[batch_idx]
                token_id = next_token_ids[batch_idx].item()
                token_text = tokenizer.decode(next_token_ids[batch_idx], skip_special_tokens=False)
                token_text = token_text.replace('\n', '\\n').replace(',', '[COMMA]')
                
                row = [gpu_name, model_specific, device.type, batch_size, input_index, prompt[:100], step, token_id, token_text]
                logit_values = next_token_logits[batch_idx].cpu().numpy()
                row.extend([f"{val:.8f}" for val in logit_values])
                logit_csv_writer.writerow(row)
                
                # 메모리에서 즉시 삭제
                del logit_values
                
            #activation save
            for batch_idx in range(batch_size):
                # 이미 완료된 샘플은 건너뛰기
                if finished[batch_idx]:
                    continue
                
                input_index = batch_start_idx + batch_idx
                prompt = batch_prompts[batch_idx]
                token_id = next_token_ids[batch_idx].item()
                token_text = tokenizer.decode(next_token_ids[batch_idx], skip_special_tokens=False)
                token_text = token_text.replace('\n', '\\n').replace(',', '[COMMA]')
                
                # 각 layer의 activation 저장 (model_specific은 이미 safe한 이름으로 전달됨)
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
    
    # Finalize KV Cache for each input
    if kv_manager is not None:
        for batch_idx in range(batch_size):
            input_index = batch_start_idx + batch_idx
            kv_manager.finalize_input(
                input_index=input_index,
                generated_text=generated_texts[batch_idx],
                generated_token_ids=generated_token_ids[batch_idx]
            )
    
    # 메모리 정리
    del generated_ids, inputs, next_token_ids, finished
    if past_kv is not None:
        del past_kv
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"  ✓ Generated {len(generated_texts)} responses with activation tracking")
    return generated_texts, generated_token_ids, input_token_ids


def process_batch_with_kv_cache_loading(model, tokenizer, batch_prompts, device, max_new_tokens, csv_writer, logit_csv_writer, gpu_name, model_specific, batch_start_idx, output_dir=None, start_time=None, in_out_value_checkpointing=False, save_kv_cache=False, kv_cache_config=None):
    """
    KV Cache를 로드하여 step-by-step으로 생성을 재현하는 함수.
    
    다른 디바이스에서 저장된 KV Cache를 로드하여, 각 decoding step마다
    해당 시점까지의 KV Cache를 사용하여 다음 토큰을 생성하고 원본과 비교.
    
    Args:
        model: 언어 모델
        tokenizer: 토크나이저
        batch_prompts: 문자열 리스트 (사용되지 않음, input_index로 KV Cache 로드)
        device: 현재 실행 디바이스
        max_new_tokens: 생성할 최대 토큰 수
        csv_writer: CSV writer object (선택)
        logit_csv_writer: CSV writer object (선택)
        gpu_name: 현재 GPU 이름
        model_specific: 현재 모델명
        batch_start_idx: 배치 시작 인덱스
        output_dir: Output directory
        start_time: Run timestamp
        in_out_value_checkpointing: Layer operation tracking
        save_kv_cache: 현재 디바이스의 KV Cache도 저장할지 여부
        kv_cache_config: KV cache 설정
    
    Returns:
        생성된 텍스트 리스트, 토큰 ID 리스트, 입력 토큰 ID 리스트
    """
    
    print(f"\n  [KV Cache Load Mode] Reproducing generation from saved KV caches...")
    
    # Load configuration
    load_from_device = kv_cache_config.get('load_from_device', 'NVIDIA_H200')
    load_from_batch_size = kv_cache_config.get('load_from_batch_size', 16)
    load_base_dir = kv_cache_config.get('load_base_dir', 'ref_cache')
    
    # Use current model_specific automatically (no need to specify in config)
    load_from_model = model_specific
    
    print(f"    Loading KV caches from:")
    print(f"      Base dir: {load_base_dir}")
    print(f"      Device: {load_from_device}")
    print(f"      Model: {load_from_model} (auto-detected from current model)")
    print(f"      Batch size: {load_from_batch_size}")
    print(f"      Full path: {load_base_dir}/{load_from_device}/{load_from_model}/")
    
    # Initialize KV Cache Manager for loading
    # KVCacheManager will construct path as: load_base_dir/device/model/
    load_kv_manager = KVCacheManager(
        base_dir=load_base_dir,
        gpu_name=load_from_device,
        model_specific=load_from_model,
        batch_size=load_from_batch_size,
        save_mode='delta'
    )
    
    # Initialize KV Cache Manager for saving (if enabled)
    save_kv_manager = None
    if save_kv_cache:
        save_kv_manager = KVCacheManager(
            base_dir=kv_cache_config.get('base_dir', f"{output_dir}/kv_caches"),
            gpu_name=gpu_name,
            model_specific=model_specific,
            batch_size=len(batch_prompts),
            save_mode=kv_cache_config.get('save_mode', 'delta')
        )
        print(f"    Current device KV cache saving: ENABLED")
    
    # Get transformer layers for activation tracking
    try:
        transformer_layers = model.model.layers
        num_layers = len(transformer_layers)
    except AttributeError:
        transformer_layers = None
        num_layers = 0
    
    batch_size = len(batch_prompts)
    generated_texts = []
    generated_token_ids_list = []
    input_token_ids_list = []
    
    # Process each input independently
    for batch_idx in range(batch_size):
        input_index = batch_start_idx + batch_idx
        
        print(f"\n    Processing input {input_index}...")
        
        # Check if KV cache exists
        if not load_kv_manager.cache_exists(input_index):
            print(f"      ⚠️  KV cache not found for input {input_index}, skipping")
            generated_texts.append("")
            generated_token_ids_list.append([])
            input_token_ids_list.append([])
            continue
        
        # Get total steps from saved cache
        try:
            total_steps = load_kv_manager.get_total_steps(input_index)
            steps_to_generate = min(max_new_tokens, total_steps)
            print(f"      Total saved steps: {total_steps}, Will reproduce: {steps_to_generate}")
        except Exception as e:
            print(f"      ⚠️  Error getting total steps: {e}, skipping")
            generated_texts.append("")
            generated_token_ids_list.append([])
            input_token_ids_list.append([])
            continue
        
        # Load prefill KV cache
        try:
            prefill_data = load_kv_manager.load_kv_cache_for_step(input_index, -1, device)
            input_tokens = prefill_data['input_tokens']
            input_text = tokenizer.decode(input_tokens)
            input_token_ids_list.append(input_tokens)
            
            print(f"      Input text: {input_text[:100]}...")
            print(f"      Input tokens: {len(input_tokens)}")
            print(f"      Reproducing: ", end='', flush=True)
        except Exception as e:
            print(f"      ⚠️  Error loading prefill: {e}, skipping")
            generated_texts.append("")
            generated_token_ids_list.append([])
            input_token_ids_list.append([])
            continue
        
        generated_tokens = []
        comparison_results = []
        init_time = time.time()
        
        # Step-by-step generation
        for step in range(steps_to_generate):
            try:
                # Load KV cache up to previous step
                if step == 0:
                    # First decoding step: use prefill only
                    kv_data = prefill_data
                else:
                    # Load cumulative KV cache up to step-1
                    kv_data = load_kv_manager.load_kv_cache_for_step(input_index, step - 1, device)
                
                past_kv = kv_data['past_key_values']
                generated_so_far = kv_data['generated_tokens_so_far']
                
                # Prepare input for this step
                if step == 0:
                    # First step: use full input
                    input_ids = torch.tensor([input_tokens], device=device)
                    attention_mask = torch.ones((1, len(input_tokens)), device=device)
                else:
                    # Subsequent steps: use last generated token
                    last_token = generated_so_far[-1]
                    input_ids = torch.tensor([[last_token]], device=device)
                    # Attention mask should cover past_kv length + 1
                    seq_len = kv_data['seq_length'] + 1
                    attention_mask = torch.ones((1, seq_len), device=device)
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_kv if step > 0 else None,
                        use_cache=True,
                        return_dict=True
                    )
                
                # Get next token
                next_token_logits = outputs.logits[:, -1, :]
                predicted_token_id = torch.argmax(next_token_logits, dim=-1).item()
                predicted_text = tokenizer.decode([predicted_token_id], skip_special_tokens=False)
                
                # Load original token from cache for comparison
                cache_file_data = torch.load(load_kv_manager.get_filepath(input_index), map_location='cpu')
                original_token_id = cache_file_data['decoding_deltas'][step]['token_id']
                original_text = cache_file_data['decoding_deltas'][step]['token_text']
                
                # Compare
                match = (predicted_token_id == original_token_id)
                comparison_results.append({
                    'step': step,
                    'predicted': predicted_token_id,
                    'original': original_token_id,
                    'match': match
                })
                
                # Print token
                display_text = predicted_text.replace('\n', '\\n')
                if match:
                    print(display_text, end='', flush=True)
                else:
                    print(f"[{display_text}≠{original_text}]", end='', flush=True)
                
                generated_tokens.append(predicted_token_id)
                
                # Save current device's KV cache if enabled
                if save_kv_manager is not None:
                    current_past_kv = outputs.past_key_values
                    if step == 0:
                        # Save prefill
                        save_kv_manager.save_prefill(
                            input_index=input_index,
                            batch_idx=0,
                            kv_cache=current_past_kv,
                            input_text=input_text,
                            input_tokens=input_tokens,
                            attention_mask=attention_mask
                        )
                    else:
                        # Save decoding delta
                        save_kv_manager.save_decoding_step(
                            input_index=input_index,
                            batch_idx=0,
                            step=step - 1,
                            kv_cache=current_past_kv,
                            token_id=predicted_token_id,
                            token_text=predicted_text
                        )
                
                # Memory cleanup
                del outputs, next_token_logits, past_kv
                if step % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                
            except Exception as e:
                print(f"\n      ⚠️  Error at step {step}: {e}")
                break
        
        end_time = time.time()
        
        # Summary
        matches = sum(1 for r in comparison_results if r['match'])
        total = len(comparison_results)
        match_rate = 100 * matches / total if total > 0 else 0
        
        print(f"\n      Steps: {total}, Match: {matches}/{total} ({match_rate:.1f}%), TPS: {total/(end_time-init_time):.2f}")
        
        # Generate text from tokens
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_texts.append(generated_text)
        generated_token_ids_list.append(generated_tokens)
        
        # Finalize if saving
        if save_kv_manager is not None:
            save_kv_manager.finalize_input(
                input_index=input_index,
                generated_text=generated_text,
                generated_token_ids=generated_tokens
            )
    
    print(f"\n  ✓ Reproduced {len(generated_texts)} responses from KV caches")
    return generated_texts, generated_token_ids_list, input_token_ids_list