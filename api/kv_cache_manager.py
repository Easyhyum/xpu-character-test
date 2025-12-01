"""
KV Cache Manager for saving and loading Key-Value caches in transformer models.

This module implements a delta-based approach to save KV caches efficiently:
- Prefill stage: Save full KV cache
- Decoding steps: Save only the incremental delta (last token's KV)
- Loading: Reconstruct full KV cache by concatenating deltas

File naming: batch{batch_size}_input{input_index}.pt
"""

import os
import torch
import time


class KVCacheManager:
    """
    Manages KV Cache storage and retrieval with delta compression.
    
    Storage structure:
        outputs/{timestamp}/kv_caches/{gpu_name}/{model_name}/batch{N}_input{M}.pt
    
    File contents:
        {
            'metadata': {...},
            'input_data': {text, token_ids, ...},
            'prefill_kv': {past_key_values, seq_length, attention_mask},
            'decoding_deltas': [{step, token_id, kv_delta, ...}, ...],
            'generation_result': {generated_text, generated_token_ids, ...}
        }
    """
    
    def __init__(self, base_dir, gpu_name, model_specific, batch_size, save_mode='delta'):
        """
        Initialize KV Cache Manager.
        
        Args:
            base_dir: Base directory for KV cache storage
            gpu_name: GPU device name (e.g., 'NVIDIA_H200', 'CPU')
            model_specific: Model identifier (safe for filename)
            batch_size: Current batch size
            save_mode: Storage mode - 'delta' (incremental), 'prefill_only', or 'final_only'
        """
        self.base_dir = base_dir
        self.gpu_name = gpu_name
        self.model_specific = model_specific
        self.batch_size = batch_size
        self.save_mode = save_mode
        
        # Create save directory: base_dir/gpu_name/model_name/
        self.save_dir = os.path.join(base_dir, gpu_name, model_specific)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Input-specific buffers (accumulate data before final save)
        self.input_buffers = {}
        
        print(f"  KVCacheManager initialized:")
        print(f"    Save directory: {self.save_dir}")
        print(f"    Save mode: {self.save_mode}")
    
    def get_filepath(self, input_index, step):
        """
        Generate file path for a specific input and step.
        
        Format: batch{batch_size}_input{input_index}_step{step}.pt
        
        Args:
            input_index: Global input index
            step: Generation step (0 for prefill, >0 for decoding)
            
        Returns:
            Full file path
        """
        filename = f"batch{self.batch_size}_input{input_index}_step{step}.pt"
        return os.path.join(self.save_dir, filename)
    
    def save_step(self, input_index, batch_idx, step, kv_cache, input_text, input_tokens, attention_mask, token_id=None, token_text=None, save_length=None):
        """
        Save KV cache for a specific step.
        
        Args:
            input_index: Global input index in dataset
            batch_idx: Index within the current batch
            step: Generation step (0 for prefill, >0 for decoding)
            kv_cache: past_key_values from model output
            input_text: Input prompt text
            input_tokens: Input token IDs (list)
            attention_mask: Attention mask tensor
            token_id: Generated token ID (for step > 0)
            token_text: Generated token text (for step > 0)
            save_length: If provided, save only this many tokens from the end (for prefill [:N-1])
        """
        if self.save_mode == 'disabled':
            return
        
        # Extract single sample from batch
        if step == 0:
            # Prefill: extract KV cache (possibly excluding last token)
            if attention_mask.dim() == 2 and attention_mask.size(0) == 1:
                single_mask = attention_mask.cpu() if attention_mask.device != torch.device('cpu') else attention_mask
            else:
                single_mask = attention_mask[batch_idx:batch_idx+1].cpu()
            
            actual_seq_len = save_length if save_length is not None else single_mask.size(1)
            single_kv = self._extract_single_from_batch(kv_cache, batch_idx, actual_seq_len)
            
            if batch_idx == 0:
                print(f"\n    [SAVE DEBUG] Step 0: Extracting prefill KV (length={actual_seq_len}), Layer 0 keys shape = {single_kv[0][0].shape}")
        else:
            # Decoding: extract only last token's KV (delta)
            if batch_idx == 0:
                print(f"\n    [SAVE DEBUG] Step {step}: Before delta extraction, full KV Layer 0 keys shape = {kv_cache[0][0].shape if len(kv_cache) > 0 else 'N/A'}")
            
            single_kv = self._extract_delta_from_batch(kv_cache, batch_idx)
            single_mask = None
            
            if batch_idx == 0:
                print(f"    [SAVE DEBUG] Step {step}: After delta extraction, Layer 0 keys shape = {single_kv[0][0].shape}")
        
        # Get metadata
        num_layers = len(single_kv)
        if num_layers > 0:
            sample_tensor = single_kv[0][0]
            dtype_str = str(sample_tensor.dtype)
            shape = sample_tensor.shape  # [1, num_heads, seq_len, head_dim]
            num_heads = shape[1]
            seq_length = shape[2]
            head_dim = shape[3]
        else:
            dtype_str = 'unknown'
            num_heads = 0
            seq_length = 0
            head_dim = 0
        
        # Create data structure
        data = {
            'metadata': {
                'input_index': input_index,
                'batch_size': self.batch_size,
                'batch_idx': batch_idx,
                'step': step,
                'model_name': self.model_specific,
                'device': self.gpu_name,
                'dtype': dtype_str,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'head_dim': head_dim,
                'seq_length': seq_length,
                'timestamp': time.strftime("%Y%m%d-%H%M%S")
            },
            'kv_cache': single_kv,
            'input_data': {
                'text': input_text,
                'token_ids': input_tokens,
                'token_length': len(input_tokens)
            }
        }
        
        if step == 0:
            # Prefill step
            data['attention_mask'] = single_mask
            data['generated_token'] = {
                'token_id': token_id,
                'token_text': token_text
            }
        else:
            # Decoding step
            data['generated_token'] = {
                'token_id': token_id,
                'token_text': token_text
            }
        
        # Save to file
        filepath = self.get_filepath(input_index, step)
        torch.save(data, filepath)
        
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        if step == 0:
            print(f"    Saved step {step} (prefill): input{input_index}_step{step}.pt ({file_size_mb:.2f} MB, {seq_length} tokens)")
        elif step == 1:
            print(f"    Saved step {step}: input{input_index}_step{step}.pt ({file_size_mb:.2f} MB, token_id={token_id})")
    
    def load_step(self, input_index, step, device='cpu'):
        """
        Load KV cache for a single step
        
        Args:
            input_index: Global input index
            step: Generation step (0 = prefill, >0 = decoding steps)
            device: Device to load the tensors to
            
        Returns:
            Dictionary containing:
                - metadata: Step metadata
                - kv_cache: Tuple of (keys, values) for each layer
                - input_data: Input text and tokens
                - attention_mask: Attention mask (step 0 only)
                - generated_token: Token info (step > 0 only)
        """
        filepath = self.get_filepath(input_index, step)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"KV cache file not found: {filepath}")
        
        data = torch.load(filepath, map_location=device)
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        if step == 0:
            seq_len = data['metadata']['seq_length']
            # print(f"    Loaded step {step} (prefill): input{input_index}_step{step}.pt ({file_size_mb:.2f} MB, {seq_len} tokens)")
        else:
            token_id = data.get('generated_token', {}).get('token_id', 'N/A')
            # print(f"    Loaded step {step}: input{input_index}_step{step}.pt ({file_size_mb:.2f} MB, token_id={token_id})")
        
        return data
    
    def combine_batch_kv_caches(self, kv_caches_list, device='cpu'):
        """
        Combine multiple single KV caches into a batched KV cache (DynamicCache format)
        
        Args:
            kv_caches_list: List of KV cache tuples [(layer0_keys, layer0_values), (layer1_keys, layer1_values), ...]
                            Each element represents one sample in the batch
            device: Device to move the combined cache to
            
        Returns:
            Tuple of (DynamicCache object, list of original sequence lengths before padding)
        """
        if not kv_caches_list:
            return None, []
        
        num_layers = len(kv_caches_list[0])
        batch_size = len(kv_caches_list)
        
        # Find max sequence length across all samples and track original lengths
        max_seq_len = 0
        original_seq_lens = []
        for kv_cache in kv_caches_list:
            if kv_cache and len(kv_cache) > 0:
                seq_len = kv_cache[0][0].shape[2]  # [1, num_heads, seq_len, head_dim]
                original_seq_lens.append(seq_len)
                max_seq_len = max(max_seq_len, seq_len)
            else:
                original_seq_lens.append(0)
        
        # Create a DynamicCache and populate it
        from transformers import DynamicCache
        batched_cache = DynamicCache()
        
        for layer_idx in range(num_layers):
            # Collect keys and values for this layer from all samples
            layer_keys = []
            layer_values = []
            
            for batch_idx in range(batch_size):
                keys = kv_caches_list[batch_idx][layer_idx][0]  # [1, num_heads, seq_len, head_dim]
                values = kv_caches_list[batch_idx][layer_idx][1]
                
                current_seq_len = keys.shape[2]
                
                # Apply left padding if needed
                if current_seq_len < max_seq_len:
                    pad_len = max_seq_len - current_seq_len
                    # Pad on the left side (dim=2): (left, right, top, bottom, ...)
                    keys = torch.nn.functional.pad(keys, (0, 0, pad_len, 0), value=0.0)
                    values = torch.nn.functional.pad(values, (0, 0, pad_len, 0), value=0.0)
                
                layer_keys.append(keys)
                layer_values.append(values)
            
            # Stack along batch dimension (dim=0)
            batched_keys = torch.cat(layer_keys, dim=0).to(device)
            batched_values = torch.cat(layer_values, dim=0).to(device)
            
            # Update the DynamicCache
            batched_cache.update(batched_keys, batched_values, layer_idx)
        
        return batched_cache, original_seq_lens
    
    # Keep old methods for compatibility but mark as deprecated
    def save_prefill(self, input_index, batch_idx, kv_cache, input_text, input_tokens, attention_mask, first_token_id=None, first_token_text=None):
        """
        Save Prefill stage KV Cache.
        
        Args:
            input_index: Global input index in dataset
            batch_idx: Index within the current batch
            kv_cache: Full past_key_values from model output (tuple of layer tuples)
            input_text: Input prompt text
            input_tokens: Input token IDs (list)
            attention_mask: Attention mask tensor (should be [1, actual_length] without padding)
        """
        if self.save_mode == 'disabled':
            return
        
        # Attention mask should already be for single sample (passed from batch_generator)
        if attention_mask.dim() == 2 and attention_mask.size(0) == 1:
            # Already single sample mask
            single_mask = attention_mask.cpu() if attention_mask.device != torch.device('cpu') else attention_mask
        else:
            # Fallback: extract from batch (shouldn't happen with new code)
            single_mask = attention_mask[batch_idx:batch_idx+1].cpu()
        
        # Get actual sequence length from attention mask
        actual_seq_len = single_mask.size(1)
        
        # Extract single sample from batch with only actual tokens (no padding)
        single_kv = self._extract_single_from_batch(kv_cache, batch_idx, actual_seq_len)
        
        # Get metadata from KV cache structure
        num_layers = len(single_kv)
        if num_layers > 0:
            sample_tensor = single_kv[0][0]  # First layer, keys
            dtype_str = str(sample_tensor.dtype)
            shape = sample_tensor.shape  # [1, num_heads, seq_len, head_dim]
            num_heads = shape[1]
            seq_length = shape[2]
            head_dim = shape[3]
        else:
            dtype_str = 'unknown'
            num_heads = 0
            seq_length = 0
            head_dim = 0
        
        # Create data structure
        data = {
            'metadata': {
                'input_index': input_index,
                'batch_size': self.batch_size,
                'batch_idx': batch_idx,
                'model_name': self.model_specific,
                'device': self.gpu_name,
                'dtype': dtype_str,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'head_dim': head_dim,
                'save_mode': self.save_mode,
                'timestamp': time.strftime("%Y%m%d-%H%M%S")
            },
            'input_data': {
                'text': input_text,
                'token_ids': input_tokens,
                'token_length': len(input_tokens)
            },
            'prefill_kv': {
                'past_key_values': single_kv,
                'seq_length': seq_length,
                'attention_mask': single_mask
            }
        }
        
        # Handle different save modes
        if self.save_mode == 'prefill_only':
            # Save immediately (no decoding will be saved)
            filepath = self.get_filepath(input_index)
            torch.save(data, filepath)
            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"    Saved prefill KV cache: batch{self.batch_size}_input{input_index}.pt ({file_size_mb:.1f} MB)")
        else:
            # Store in buffer (will save after all decoding steps)
            self.input_buffers[input_index] = data
            self.input_buffers[input_index]['decoding_deltas'] = []
    
    def save_decoding_step(self, input_index, batch_idx, step, kv_cache, token_id, token_text):
        """
        Save Decoding step KV Cache delta (only the last token's KV).
        
        Args:
            input_index: Global input index
            batch_idx: Index within the current batch
            step: Decoding step number (0-based)
            kv_cache: Full past_key_values from model output
            token_id: Generated token ID
            token_text: Generated token text
        """
        if self.save_mode == 'prefill_only' or self.save_mode == 'disabled':
            return
        
        # Extract only the LAST token's KV (delta) for this specific input
        single_kv_delta = self._extract_delta_from_batch(kv_cache, batch_idx)
        
        # Get sequence length
        cumulative_seq_length = self._get_current_seq_length(input_index, step)
        
        step_data = {
            'step': step,
            'token_id': token_id,
            'token_text': token_text,
            'kv_delta': single_kv_delta,
            'cumulative_seq_length': cumulative_seq_length
        }
        
        if self.save_mode == 'delta':
            # Accumulate all deltas
            if input_index in self.input_buffers:
                self.input_buffers[input_index]['decoding_deltas'].append(step_data)
        elif self.save_mode == 'final_only':
            # Keep only the last delta (overwrite)
            if input_index not in self.input_buffers:
                self.input_buffers[input_index] = {'decoding_deltas': []}
            self.input_buffers[input_index]['final_kv'] = step_data
    
    def finalize_input(self, input_index, generated_text, generated_token_ids):
        """
        Finalize and save KV cache for a completed input.
        
        Args:
            input_index: Global input index
            generated_text: Full generated text
            generated_token_ids: List of generated token IDs
        """
        if self.save_mode == 'prefill_only' or self.save_mode == 'disabled':
            return
        
        if input_index not in self.input_buffers:
            print(f"    Warning: No buffer found for input {input_index}, skipping save")
            return
        
        data = self.input_buffers[input_index]
        
        # Add generation result
        data['generation_result'] = {
            'generated_text': generated_text,
            'generated_token_ids': generated_token_ids,
            'total_steps': len(generated_token_ids)
        }
        
        # Save to file
        filepath = self.get_filepath(input_index)
        torch.save(data, filepath)
        
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        num_deltas = len(data.get('decoding_deltas', []))
        print(f"    Saved KV cache: batch{self.batch_size}_input{input_index}.pt ({file_size_mb:.1f} MB, {num_deltas} deltas)")
        
        # Clear buffer to save memory
        del self.input_buffers[input_index]
    
    def load_kv_cache_for_step(self, input_index, target_step, device='cuda'):
        """
        Load KV cache up to a specific decoding step.
        
        For target_step=-1: Returns only prefill KV cache
        For target_step=N: Returns prefill + deltas from step 0 to N (reconstructed)
        
        Args:
            input_index: Global input index
            target_step: Target decoding step (-1 for prefill only, 0+ for decoding)
            device: Target device to move tensors to
            
        Returns:
            dict with keys:
                - 'past_key_values': Reconstructed KV cache (tuple of layer tuples)
                - 'seq_length': Total sequence length
                - 'input_tokens': Input token IDs
                - 'generated_tokens_so_far': List of generated token IDs up to target_step
                - 'attention_mask': Attention mask
        """
        filepath = self.get_filepath(input_index)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"KV cache file not found: {filepath}")
        
        # Load data from disk
        data = torch.load(filepath, map_location='cpu')
        
        if target_step == -1:
            # Return only prefill KV cache
            prefill_data = data['prefill_kv']
            kv_cache = self._move_kv_to_device(prefill_data['past_key_values'], device)
            attention_mask = prefill_data['attention_mask'].to(device)
            
            return {
                'past_key_values': kv_cache,
                'seq_length': prefill_data['seq_length'],
                'input_tokens': data['input_data']['token_ids'],
                'generated_tokens_so_far': [],
                'attention_mask': attention_mask
            }
        
        # Reconstruct KV cache from prefill + deltas
        deltas = data.get('decoding_deltas', [])
        
        if target_step >= len(deltas):
            raise ValueError(f"target_step {target_step} exceeds available steps (max: {len(deltas)-1})")
        
        prefill_kv = data['prefill_kv']['past_key_values']
        num_layers = len(prefill_kv)
        
        # Reconstruct by concatenating deltas
        reconstructed_kv = []
        
        for layer_idx in range(num_layers):
            # Start with prefill keys and values
            prefill_keys = prefill_kv[layer_idx][0]    # [1, num_heads, prefill_len, head_dim]
            prefill_values = prefill_kv[layer_idx][1]
            
            # Collect all deltas up to target_step
            key_parts = [prefill_keys]
            value_parts = [prefill_values]
            
            for step_idx in range(target_step + 1):
                delta = deltas[step_idx]['kv_delta'][layer_idx]
                key_parts.append(delta[0])    # [1, num_heads, 1, head_dim]
                value_parts.append(delta[1])
            
            # Concatenate along sequence dimension (dim=2)
            combined_keys = torch.cat(key_parts, dim=2)
            combined_values = torch.cat(value_parts, dim=2)
            
            reconstructed_kv.append((combined_keys, combined_values))
        
        reconstructed_kv = tuple(reconstructed_kv)
        
        # Move to target device
        reconstructed_kv = self._move_kv_to_device(reconstructed_kv, device)
        
        # Collect generated tokens
        generated_tokens = [deltas[i]['token_id'] for i in range(target_step + 1)]
        
        # Reconstruct attention mask
        prefill_len = data['prefill_kv']['seq_length']
        total_len = prefill_len + target_step + 1
        attention_mask = torch.ones((1, total_len), dtype=torch.long, device=device)
        
        return {
            'past_key_values': reconstructed_kv,
            'seq_length': total_len,
            'input_tokens': data['input_data']['token_ids'],
            'generated_tokens_so_far': generated_tokens,
            'attention_mask': attention_mask
        }
    
    def get_total_steps(self, input_index):
        """
        Get the total number of decoding steps saved for an input.
        
        Args:
            input_index: Global input index
            
        Returns:
            Number of decoding steps
        """
        filepath = self.get_filepath(input_index)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"KV cache file not found: {filepath}")
        
        data = torch.load(filepath, map_location='cpu')
        return len(data.get('decoding_deltas', []))
    
    def cache_exists(self, input_index):
        """
        Check if KV cache file exists for an input.
        
        Args:
            input_index: Global input index
            
        Returns:
            True if file exists, False otherwise
        """
        filepath = self.get_filepath(input_index)
        if os.path.exists(filepath):
            return True
        else:
            print((f"KV cache file not found: {filepath}"))
            return False
    
    def load_batch_kv_cache_for_step(self, input_indices, target_step, device='cuda'):
        """
        Load KV cache for multiple inputs (batch) at a specific decoding step.
        
        Args:
            input_indices: List of global input indices
            target_step: Target decoding step (-1 for prefill only, 0+ for decoding)
            device: Target device to move tensors to
            
        Returns:
            dict with keys:
                - 'past_key_values': Batch KV cache (tuple of layer tuples)
                - 'seq_lengths': List of sequence lengths for each sample
                - 'input_tokens_list': List of input token ID lists
                - 'generated_tokens_list': List of generated tokens so far
                - 'attention_masks': Batch attention masks [batch_size, max_seq_len]
                - 'valid_mask': Boolean mask for successfully loaded samples
        """
        batch_size = len(input_indices)
        batch_kv_list = []
        input_tokens_list = []
        generated_tokens_list = []
        attention_masks_list = []
        seq_lengths = []
        valid_mask = []
        
        # Load each input's KV cache
        for input_index in input_indices:
            try:
                kv_data = self.load_kv_cache_for_step(input_index, target_step, 'cpu')
                batch_kv_list.append(kv_data['past_key_values'])
                input_tokens_list.append(kv_data['input_tokens'])
                generated_tokens_list.append(kv_data['generated_tokens_so_far'])
                attention_masks_list.append(kv_data['attention_mask'].cpu())
                seq_lengths.append(kv_data['seq_length'])
                valid_mask.append(True)
            except Exception as e:
                print(f"      ⚠️  Failed to load input {input_index}: {e}")
                # Add dummy data for failed loads
                batch_kv_list.append(None)
                input_tokens_list.append([])
                generated_tokens_list.append([])
                attention_masks_list.append(None)
                seq_lengths.append(0)
                valid_mask.append(False)
        
        # Find max sequence length for padding
        max_seq_len = max(seq_lengths) if seq_lengths else 0
        
        # Combine into batch (only valid samples)
        if not any(valid_mask):
            raise ValueError("No valid KV caches loaded")
        
        # Stack KV caches into batch format
        num_layers = len(batch_kv_list[0]) if batch_kv_list[0] is not None else 0
        batch_kv = []
        
        for layer_idx in range(num_layers):
            batch_keys = []
            batch_values = []
            
            for sample_idx, kv in enumerate(batch_kv_list):
                if kv is None:
                    # Add dummy for invalid sample (will be masked)
                    if len(batch_keys) > 0:
                        dummy_shape = list(batch_keys[0].shape)
                        dummy_shape[0] = 1
                        batch_keys.append(torch.zeros(dummy_shape))
                        batch_values.append(torch.zeros(dummy_shape))
                    continue
                
                layer_kv = kv[layer_idx]
                keys = layer_kv[0]  # [1, num_heads, seq_len, head_dim]
                values = layer_kv[1]
                
                # Pad to max_seq_len if needed (left padding)
                current_seq_len = keys.size(2)
                if current_seq_len < max_seq_len:
                    pad_len = max_seq_len - current_seq_len
                    keys = torch.nn.functional.pad(keys, (0, 0, pad_len, 0), value=0)
                    values = torch.nn.functional.pad(values, (0, 0, pad_len, 0), value=0)
                
                batch_keys.append(keys)
                batch_values.append(values)
            
            # Stack into batch: [batch_size, num_heads, max_seq_len, head_dim]
            batch_keys_tensor = torch.cat(batch_keys, dim=0).to(device)
            batch_values_tensor = torch.cat(batch_values, dim=0).to(device)
            batch_kv.append((batch_keys_tensor, batch_values_tensor))
        
        # Create batch attention masks with left padding
        batch_attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        for idx, (seq_len, valid) in enumerate(zip(seq_lengths, valid_mask)):
            if valid and seq_len > 0:
                # Left padding: set last seq_len positions to 1
                batch_attention_mask[idx, -seq_len:] = 1
        
        batch_attention_mask = batch_attention_mask.to(device)
        
        return {
            'past_key_values': tuple(batch_kv),
            'seq_lengths': seq_lengths,
            'input_tokens_list': input_tokens_list,
            'generated_tokens_list': generated_tokens_list,
            'attention_masks': batch_attention_mask,
            'valid_mask': valid_mask
        }
    
    def _extract_single_from_batch(self, batch_kv, batch_idx, actual_seq_len=None):
        """
        Extract single sample's KV cache from batch.
        
        Args:
            batch_kv: Tuple of (keys, values) for each layer
                     Shape: (batch_size, num_heads, seq_len, head_dim)
            batch_idx: Index within batch
            actual_seq_len: If provided, extract only this many tokens from the end (remove left padding)
            
        Returns:
            Tuple of (keys, values) for single sample (on CPU)
        """
        single_kv = []
        for layer_kv in batch_kv:
            if actual_seq_len is not None:
                # Remove left padding by taking only the last actual_seq_len tokens
                keys = layer_kv[0][batch_idx:batch_idx+1, :, -actual_seq_len:, :].cpu()
                values = layer_kv[1][batch_idx:batch_idx+1, :, -actual_seq_len:, :].cpu()
            else:
                # Take full sequence
                keys = layer_kv[0][batch_idx:batch_idx+1].cpu()
                values = layer_kv[1][batch_idx:batch_idx+1].cpu()
            single_kv.append((keys, values))
        
        return tuple(single_kv)
    
    def _extract_delta_from_batch(self, batch_kv, batch_idx):
        """
        Extract only the LAST token's KV (delta) for a single sample.
        
        Args:
            batch_kv: Tuple of (keys, values) for each layer
                     Shape: (batch_size, num_heads, seq_len, head_dim)
            batch_idx: Index within batch
            
        Returns:
            Tuple of (keys, values) for last token only (on CPU)
        """
        delta_kv = []
        for layer_kv in batch_kv:
            # Extract last token only: [..., -1:, :]
            keys = layer_kv[0][batch_idx:batch_idx+1, :, -1:, :].cpu()
            values = layer_kv[1][batch_idx:batch_idx+1, :, -1:, :].cpu()
            delta_kv.append((keys, values))
        
        return tuple(delta_kv)
    
    def _get_current_seq_length(self, input_index, step):
        """
        Calculate cumulative sequence length at a given step.
        
        Args:
            input_index: Global input index
            step: Current decoding step
            
        Returns:
            Total sequence length (prefill + steps)
        """
        if input_index in self.input_buffers:
            prefill_len = self.input_buffers[input_index]['prefill_kv']['seq_length']
            return prefill_len + step + 1
        return 0
    
    def _move_kv_to_device(self, kv_cache, device):
        """
        Move KV cache to specified device.
        
        Args:
            kv_cache: Tuple of (keys, values) for each layer
            device: Target device
            
        Returns:
            KV cache on target device
        """
        return tuple(
            (keys.to(device), values.to(device))
            for keys, values in kv_cache
        )
    
    def get_cache_info(self, input_index):
        """
        Get metadata about a saved KV cache without loading full tensors.
        
        Args:
            input_index: Global input index
            
        Returns:
            Dictionary with metadata
        """
        filepath = self.get_filepath(input_index)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"KV cache file not found: {filepath}")
        
        # Load only metadata (tensors stay on disk)
        data = torch.load(filepath, map_location='cpu')
        
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        
        info = {
            'file_path': filepath,
            'file_size_mb': file_size_mb,
            'metadata': data['metadata'],
            'input_text': data['input_data']['text'][:100] + '...',
            'prefill_length': data['prefill_kv']['seq_length'],
            'num_decoding_steps': len(data.get('decoding_deltas', [])),
            'generated_text': data.get('generation_result', {}).get('generated_text', 'N/A')
        }
        
        return info
