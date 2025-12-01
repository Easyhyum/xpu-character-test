"""
Batch text generation utilities with activation tracking for the XPU character test project.
"""

import torch
import gc
import time
import os
from transformers import DynamicCache
from .text_generator import create_activation_hook
from .layer_operation_tracker import LayerOperationTracker
from .kv_cache_manager import KVCacheManager
import traceback

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
    
    # Always use the unified batch processing function with activation tracking
    # This function handles all cases: KV cache load=true/false, save=true/false
    return process_batch_with_activations(
        model, tokenizer, batch_prompts, device, max_new_tokens,
        csv_writer, logit_csv_writer, gpu_name, model_specific,
        batch_start_idx, output_dir, start_time,
        in_out_value_checkpointing, save_kv_cache, kv_cache_config
    )


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
    
    # Check KV cache configuration
    load_kv_cache = kv_cache_config.get('load', False) if kv_cache_config else False
    
    # Validate: Cannot load and save at the same time
    if load_kv_cache and save_kv_cache:
        raise ValueError("Cannot enable both KV cache loading and saving simultaneously. Please set either 'load' or 'save' to false.")
    
    # Set pad_token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Warning: pad_token was None, set to eos_token ({tokenizer.eos_token})")
    
    # 모델 정보 가져오기
    transformer_layers = model.model.layers
    model_hidden_size = model.config.hidden_size
    num_layers = len(transformer_layers)
    
    if load_kv_cache:
        print(f"\n  Processing batch with KV cache LOADING and activation tracking...")
    else:
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
    elif load_kv_cache and kv_cache_config:
        # Initialize manager for loading with load_base_dir
        load_base_dir = kv_cache_config.get('load_base_dir', 'ref_cache')
        load_from_device = kv_cache_config.get('load_from_device', gpu_name)
        load_from_batch_size = kv_cache_config.get('load_from_batch_size', batch_size)
        
        kv_manager = KVCacheManager(
            base_dir=load_base_dir,
            gpu_name=load_from_device,
            model_specific=model_specific,
            batch_size=load_from_batch_size,
            save_mode='disabled'  # Not saving, only loading
        )
        print(f"  KV Cache loading: ENABLED")
        print(f"    Loading from: {load_base_dir}/{load_from_device}/{model_specific}/batch{load_from_batch_size}_input*_step*.pt")
    
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
    
    original_loaded_kv_caches = {}  
    original_saved_attention_masks = {} 
    
    # Load KV cache if enabled
    if load_kv_cache and kv_manager is not None:
        print(f"  Loading KV caches for batch (step 0 - prefill)...")
        loaded_kv_list = []
        loaded_input_lengths = []
        
        for batch_idx in range(batch_size):
            input_index = batch_start_idx + batch_idx
            try:
                # Load step 0 (prefill) for each sample
                step0_data = kv_manager.load_step(input_index, step=0, device='cpu')
                loaded_kv_list.append(step0_data['kv_cache'])
                loaded_input_lengths.append(step0_data['metadata']['seq_length'])
                
                # Store in cache for reuse
                if input_index not in original_loaded_kv_caches:
                    original_loaded_kv_caches[input_index] = {}
                original_loaded_kv_caches[input_index][0] = step0_data['kv_cache']
                
                # Store original attention mask from step 0
                if 'attention_mask' in step0_data:
                    original_saved_attention_masks[input_index] = step0_data['attention_mask']
                
                # Verify input matches
                expected_text = step0_data['input_data']['text']
                if expected_text != batch_prompts[batch_idx]:
                    print(f"    WARNING: Input text mismatch for input{input_index}")
                    print(f"      Expected: {expected_text[:50]}...")
                    print(f"      Got:      {batch_prompts[batch_idx][:50]}...")
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Cannot load KV cache for input{input_index}: {e}")
        
        # Combine loaded KV caches into batch
        past_kv, kv_seq_lens = kv_manager.combine_batch_kv_caches(loaded_kv_list, device=device)
        print(f"  ✓ Loaded and batched {batch_size} KV caches (seq_lengths: {loaded_input_lengths})")
        print(f"  ✓ Batched KV seq_lens after padding: {kv_seq_lens}")
        
        # Update generated_ids to include input tokens
        # Reconstruct input_ids from loaded data
        loaded_input_ids = []
        for batch_idx in range(batch_size):
            input_index = batch_start_idx + batch_idx
            step0_data = kv_manager.load_step(input_index, step=0, device='cpu')
            token_ids = step0_data['input_data']['token_ids']
            loaded_input_ids.append(torch.tensor(token_ids, device=device))
        
        # Pad loaded input_ids to same length
        max_len = max(len(ids) for ids in loaded_input_ids)
        padded_input_ids = []
        for ids in loaded_input_ids:
            if len(ids) < max_len:
                # Left padding
                padding = torch.full((max_len - len(ids),), tokenizer.pad_token_id, device=device, dtype=ids.dtype)
                padded_ids = torch.cat([padding, ids], dim=0)
            else:
                padded_ids = ids
            padded_input_ids.append(padded_ids)
        
        generated_ids = torch.stack(padded_input_ids, dim=0)
        
        # Update attention mask to match loaded input
        new_attention_mask = torch.zeros_like(generated_ids)
        for batch_idx, orig_len in enumerate(loaded_input_lengths):
            new_attention_mask[batch_idx, -orig_len:] = 1
        inputs['attention_mask'] = new_attention_mask
        
        print(f"  Generated IDs shape after loading: {generated_ids.shape}")
        print(f"  Starting generation from loaded KV cache...")
    
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
        # Load KV cache for this step if loading mode is enabled
        if load_kv_cache and kv_manager is not None:
            # For step N, we need to load and accumulate all KV from step 0 to N
            accumulated_kv_list = []
            
            for batch_idx in range(batch_size):
                if finished[batch_idx]:
                    # For finished samples, use last accumulated KV
                    if accumulated_kv_list:
                        accumulated_kv_list.append(accumulated_kv_list[-1])
                    continue
                
                input_index = batch_start_idx + batch_idx
                
                # Start with step 0 (prefill) - use cached version
                if 0 not in original_loaded_kv_caches[input_index]:
                    # Load if not cached
                    step0_data = kv_manager.load_step(input_index, step=0, device='cpu')
                    original_loaded_kv_caches[input_index][0] = step0_data['kv_cache']
                
                accumulated_kv = list(original_loaded_kv_caches[input_index][0])  # Convert tuple to list for modification

                # Accumulate deltas from step 1 to current step (only if step > 0)
                for s in range(1, step + 1):
                    # Load delta if not cached
                    if s not in original_loaded_kv_caches[input_index]:
                        try:
                            step_data = kv_manager.load_step(input_index, step=s, device='cpu')
                            original_loaded_kv_caches[input_index][s] = step_data['kv_cache']
                        except FileNotFoundError:
                            print(f"\n    Warning: Step {s} file not found for input{input_index}, stopping accumulation")
                            break
                    
                    delta_kv = original_loaded_kv_caches[input_index][s]
                    
                    
                    # Concatenate delta to accumulated KV for each layer
                    for layer_idx in range(len(accumulated_kv)):
                        # Before concat
                        
                        # Concatenate keys and values along sequence dimension (dim=2)
                        accumulated_keys = torch.cat([accumulated_kv[layer_idx][0], delta_kv[layer_idx][0]], dim=2)
                        accumulated_values = torch.cat([accumulated_kv[layer_idx][1], delta_kv[layer_idx][1]], dim=2)
                        accumulated_kv[layer_idx] = (accumulated_keys, accumulated_values)
                        
                accumulated_kv_list.append(tuple(accumulated_kv))
            
            # Combine accumulated KV caches into batch
            past_kv, current_kv_seq_lens = kv_manager.combine_batch_kv_caches(accumulated_kv_list, device=device)
        
        step_activations = {}
        hooks = []
        
        # Register hooks for each layer
        for layer_idx, layer in enumerate(transformer_layers):
            hook = create_activation_hook(f"layer_{layer_idx}", step_activations)
            hooks.append(layer.register_forward_hook(hook))
        
        # Prepare inputs for this step
        if step == 0:
            if load_kv_cache:
                # When loading: past_kv already contains prefill, start with decoding
                # Skip prefill step, generate first new token
                step_input_ids = generated_ids[:, -1:]  # Last token from loaded input
                current_attention_mask = inputs['attention_mask']
            else:
                # Normal prefill: use full input with original attention_mask
                step_input_ids = generated_ids
                current_attention_mask = torch.cat([
                    inputs['attention_mask'],
                    torch.ones((batch_size, generated_ids.size(1) - inputs['attention_mask'].size(1)), 
                              dtype=inputs['attention_mask'].dtype, device=device)
                ], dim=1)
        else:
            # Decoding: use only last token with past_key_values
            step_input_ids = generated_ids[:, -1:]
            
            if load_kv_cache:
                # When loading: attention mask MUST match KV cache padding exactly
                # KV cache has been left-padded, so attention mask must have 0s in the same positions
                
                if past_kv is not None:
                    kv_total_len = past_kv[0][0].shape[2]  # Max length after padding
                    batch_attention_masks = []
                    
                    for b_idx in range(batch_size):
                        input_idx = batch_start_idx + b_idx
                        
                        # Get original sequence length for this sample (before padding)
                        original_kv_len = current_kv_seq_lens[b_idx]
                        
                        # Calculate left padding amount
                        left_pad_len = kv_total_len - original_kv_len
                        
                        # Create mask: [0...0, 1...1] where 1s match the actual KV positions
                        mask = torch.cat([
                            torch.zeros((1, left_pad_len), dtype=torch.long, device=device),
                            torch.ones((1, original_kv_len), dtype=torch.long, device=device)
                        ], dim=1)
                        
                        batch_attention_masks.append(mask)
                    
                    current_attention_mask = torch.cat(batch_attention_masks, dim=0)  # [batch_size, kv_total_len]
                    
                    if step <= 3:
                        print(f"\n    [ATTENTION DEBUG LOAD] Step {step}:")
                        print(f"      KV cache total_len={kv_total_len}, batch KV lens={current_kv_seq_lens}")
                        print(f"      Mask shape={current_attention_mask.shape}")
                        for b_idx in range(min(batch_size, 3)):
                            mask_sum = current_attention_mask[b_idx].sum().item()
                            left_pad = (kv_total_len - current_kv_seq_lens[b_idx])
                            print(f"      batch_idx={b_idx}: sum={mask_sum}, left_pad={left_pad}, expected_ones={current_kv_seq_lens[b_idx]}")
                else:
                    # Fallback if no past_kv
                    current_attention_mask = inputs['attention_mask']
            else:
                # Normal mode: Extend attention_mask to include all generated tokens
                current_attention_mask = torch.cat([
                    inputs['attention_mask'],
                    torch.ones((batch_size, generated_ids.size(1) - inputs['attention_mask'].size(1)), 
                              dtype=inputs['attention_mask'].dtype, device=device)
                ], dim=1)
                
                if step <= 3:
                    print(f"    [ATTENTION DEBUG SAVE] Step {step}: Attention mask shape={current_attention_mask.shape}, batch_idx=0 sum={current_attention_mask[0].sum().item()}")
        
        # Debug: Print KV cache corresponding tokens for batch_idx=0
        if load_kv_cache and save_kv_cache and step <= 5:
            print(f"\n    [KV CACHE DEBUG] Step {step}, batch_idx=0:")
            
            # Print attention mask info
            print(f"      current_attention_mask shape={current_attention_mask.shape}")
            print(f"      current_attention_mask[0]={current_attention_mask[0]}")
            print(f"      Attention mask sum={current_attention_mask[0].sum().item()}")
            
            # Find where attention mask becomes 1
            mask_ones = (current_attention_mask[0] == 1).nonzero(as_tuple=True)[0]
            if len(mask_ones) > 0:
                first_one_pos = mask_ones[0].item()
                last_one_pos = mask_ones[-1].item()
                print(f"      Attention mask: 1s from position {first_one_pos} to {last_one_pos} (length={len(mask_ones)})")
            else:
                print(f"      Attention mask: No 1s found!")
            
            if past_kv is not None and len(past_kv) > 0:
                kv_seq_len = past_kv[0][0].shape[2]  # [batch_size, num_heads, seq_len, head_dim]
                print(f"      KV cache seq_len={kv_seq_len}, Layer 0 keys shape={past_kv[0][0].shape}")
                
                # Get the tokens corresponding to the KV cache for batch_idx=0
                # The KV cache stores keys/values for each position in the sequence
                # We need to figure out which tokens these correspond to
                
                # In generated_ids, we have [padding, original_input, generated_tokens_so_far]
                # The KV cache should correspond to the non-padding portion
                batch_0_generated = generated_ids[0]  # [seq_len]
                batch_0_mask = inputs['attention_mask'][0] if not load_kv_cache else current_attention_mask[0]
                
                # Find non-padding positions
                non_pad_positions = (batch_0_mask == 1).nonzero(as_tuple=True)[0]
                if len(non_pad_positions) > 0:
                    # Get the actual tokens (skip padding)
                    actual_tokens = batch_0_generated[non_pad_positions[:kv_seq_len]]
                    token_texts = [tokenizer.decode([t.item()], skip_special_tokens=False) for t in actual_tokens]
                    print(f"      KV cache corresponds to tokens (first 15): {actual_tokens[:15].tolist()}")
                    print(f"      Token texts: {token_texts[:15]}")
                else:
                    print(f"      No non-padding positions found in attention mask")
            else:
                print(f"      past_kv is None")
        
        with torch.no_grad():
            outputs = model(
                input_ids=step_input_ids,
                attention_mask=current_attention_mask,  # Always pass attention_mask
                past_key_values=past_kv if (step > 0 or load_kv_cache) else None,
                use_cache=True,  # Enable KV caching
                return_dict=True
            )
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Get next tokens (greedy decoding for batch)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Update past_key_values for next iteration (only when not loading)
        if not load_kv_cache:
            past_kv = outputs.past_key_values
        
        # Save KV Cache with new logic:
        # Step 0: Save prefill cache [:N-1] as step 0, then save step 1 cache [N] 
        # Step Y > 0: Save step Y+1 cache
        # Last step: Don't save
        if not load_kv_cache and kv_manager is not None and save_kv_cache:
            # Check if this is the last step (will break next iteration)
            is_last_step = (step == max_new_tokens - 1) or finished.all()
            
            if not is_last_step:
                for batch_idx in range(batch_size):
                    if finished[batch_idx]:
                        continue
                    
                    input_index = batch_start_idx + batch_idx
                    prompt = batch_prompts[batch_idx]
                    token_id = next_token_ids[batch_idx].item()
                    token_text = tokenizer.decode(next_token_ids[batch_idx], skip_special_tokens=False)
                    
                    if step == 0:
                        # Step 0: Save TWO files
                        # 1) step0.pt: Prefill cache [:N-1] (excluding last input token)
                        sample_mask = inputs['attention_mask'][batch_idx]
                        non_pad_indices = sample_mask.nonzero(as_tuple=True)[0]
                        if len(non_pad_indices) > 0:
                            # Save prefill excluding last token
                            actual_length = len(input_token_ids_list[batch_idx]) - 1  # Exclude last token
                            if actual_length > 0:
                                sample_attention_mask_prefill = torch.ones((1, actual_length), dtype=sample_mask.dtype, device='cpu')
                                kv_manager.save_step(
                                    input_index=input_index,
                                    batch_idx=batch_idx,
                                    step=0,
                                    kv_cache=outputs.past_key_values,
                                    input_text=prompt,
                                    input_tokens=input_token_ids_list[batch_idx][:-1],  # Exclude last token
                                    attention_mask=sample_attention_mask_prefill,
                                    token_id=None,
                                    token_text=None,
                                    save_length=actual_length  # Save only [:N-1]
                                )
                        
                        # 2) step1.pt: First generated token (position N)
                        sample_attention_mask_step1 = torch.ones((1, 1), dtype=torch.long, device='cpu')
                        kv_manager.save_step(
                            input_index=input_index,
                            batch_idx=batch_idx,
                            step=1,
                            kv_cache=outputs.past_key_values,  # Full cache including position N
                            input_text=prompt,
                            input_tokens=input_token_ids_list[batch_idx],
                            attention_mask=sample_attention_mask_step1,
                            token_id=token_id,  # First generated token
                            token_text=token_text
                        )
                    else:
                        # Step Y > 0: Save step Y+1 cache (delta for next step)
                        sample_attention_mask = torch.ones((1, 1), dtype=torch.long, device='cpu')
                        kv_manager.save_step(
                            input_index=input_index,
                            batch_idx=batch_idx,
                            step=step + 1,  # Save for NEXT step
                            kv_cache=outputs.past_key_values,
                            input_text=prompt,
                            input_tokens=input_token_ids_list[batch_idx],
                            attention_mask=sample_attention_mask,
                            token_id=token_id,
                            token_text=token_text
                        )
        
        # Validate tokens if loading KV cache
        if load_kv_cache and kv_manager is not None:
            # Now the logic is: Step N file contains KV BEFORE generating token N
            # So when we load step N KV and generate, we get token N (which is also saved in step N file)
            
            for batch_idx in range(batch_size):
                if finished[batch_idx]:
                    continue
                
                input_index = batch_start_idx + batch_idx
                try:
                    # Load the expected token from current step file
                    step_data = kv_manager.load_step(input_index, step=step, device='cpu')
                    expected_token_id = step_data.get('generated_token', {}).get('token_id')
                    expected_token_text = step_data.get('generated_token', {}).get('token_text', '')
                    
                    # Compare with actually generated token
                    actual_token_id = next_token_ids[batch_idx].item()
                    actual_token_text = tokenizer.decode(next_token_ids[batch_idx], skip_special_tokens=False)
                    
                    if batch_idx == 0:
                        if expected_token_id is not None and expected_token_id != actual_token_id:
                            print(f"\n    ⚠️  Token mismatch at step {step}, input {input_index}:")
                            print(f"        Expected (step{step}): {expected_token_id} ({expected_token_text})")
                            print(f"        Got:      {actual_token_id} ({actual_token_text})")
                        else:
                            print(f"\n    ✓ Token match at step {step}, input {input_index}: {actual_token_id} ({actual_token_text})")
                except FileNotFoundError:
                    # Step file doesn't exist (maybe generation was shorter)
                    pass
        
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
            
            # KV Cache saving is now done earlier, before past_kv update
            # This section only handles layer operation tracking
            
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
    
    # Finalize KV Cache for each input - no longer needed with step-by-step saving
    # if kv_manager is not None:
    #     for batch_idx in range(batch_size):
    #         input_index = batch_start_idx + batch_idx
    #         kv_manager.finalize_input(
    #             input_index=input_index,
    #             generated_text=generated_texts[batch_idx],
    #             generated_token_ids=generated_token_ids[batch_idx]
    #         )
    
    
    # 메모리 정리
    del generated_ids, inputs, next_token_ids, finished
    if past_kv is not None:
        del past_kv
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print(f"  ✓ Generated {len(generated_texts)} responses with activation tracking")
    return generated_texts, generated_token_ids, input_token_ids


