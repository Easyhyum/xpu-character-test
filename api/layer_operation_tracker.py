"""
Layer operation tracker for detailed input/output capture at each operation step.

Tracks operations like:
- MLP: input -> gate_proj -> up_proj -> activation -> down_proj -> output
- Attention: input -> q_proj/k_proj/v_proj -> attention -> o_proj -> output
- LayerNorm: input -> norm -> output
"""

import torch
import csv
import os
import time


class LayerOperationTracker:
    """Captures input/output tensors at each operation within first/last transformer layers."""
    
    def __init__(self, output_dir, start_time, gpu_name, model_specific, batch_size, layer_indices=[0, -1], print_order=True):
        """
        Args:
            output_dir: Base output directory
            start_time: Run timestamp
            gpu_name: GPU device name
            model_specific: Model name (safe for filename)
            batch_size: Current batch size
            layer_indices: Which layers to track (default: first and last)
            print_order: Whether to print and log operation order (default: True)
        """
        self.output_dir = output_dir
        self.start_time = start_time
        self.gpu_name = gpu_name
        self.model_specific = model_specific
        self.batch_size = batch_size
        self.layer_indices = layer_indices
        self.print_order = print_order
        
        # CSV file handles and writers for each operation
        self.csv_handles = {}
        self.csv_writers = {}
        
        # Hook handles
        self.hooks = []
        
        # Data buffer for current step
        self.current_step_data = {}
        
        # Operation call order tracking
        self.operation_call_order = []
        self.current_step_order = []
        
        # Model architecture info
        self.activation_function_type = None
        self.model_architecture = None
        
        # Log file for operation order
        self.log_file_handle = None
        if self.print_order:
            self._open_log_file()
    
    def _open_log_file(self):
        """Open a log file for recording operation order."""
        run_dir = f"{self.output_dir}/{self.start_time}/{self.model_specific}"
        os.makedirs(run_dir, exist_ok=True)
        log_filename = f"{self.gpu_name}_{self.model_specific}_operation_order.log"
        log_path = os.path.join(run_dir, log_filename)
        
        self.log_file_handle = open(log_path, 'w', encoding='utf-8')
        self.log_file_handle.write(f"Operation Order Log\n")
        self.log_file_handle.write(f"GPU: {self.gpu_name}\n")
        self.log_file_handle.write(f"Model: {self.model_specific}\n")
        self.log_file_handle.write(f"Batch Size: {self.batch_size}\n")
        self.log_file_handle.write(f"Tracked Layers: {self.layer_indices}\n")
        self.log_file_handle.write(f"Timestamp: {self.start_time}\n")
        self.log_file_handle.write("="*80 + "\n\n")
        
        # Write expected architecture order
        self._write_architecture_info()
        
        self.log_file_handle.flush()
        print(f"  Opened operation order log: {log_path}")
    
    def _write_architecture_info(self):
        """Write expected operation order based on known architecture patterns."""
        # Add activation function info if detected
        act_info = ""
        if self.activation_function_type:
            act_info = f" (Detected: {self.activation_function_type})"
        
        self.log_file_handle.write("Expected Architecture Order (LLaMA/Qwen style):\n")
        self.log_file_handle.write("-" * 80 + "\n")
        self.log_file_handle.write("Each Transformer Layer:\n")
        self.log_file_handle.write("  1. input_layernorm(x) -> norm_out\n")
        self.log_file_handle.write("  2. Attention Block (parallel input = norm_out):\n")
        self.log_file_handle.write("     - attn_q_proj(norm_out) -> Q\n")
        self.log_file_handle.write("     - attn_k_proj(norm_out) -> K  (parallel with Q)\n")
        self.log_file_handle.write("     - attn_v_proj(norm_out) -> V  (parallel with Q, K)\n")
        self.log_file_handle.write("     - attention_compute(Q, K, V) -> attn_out (not hooked)\n")
        self.log_file_handle.write("     - attn_o_proj(attn_out) -> O\n")
        self.log_file_handle.write("  3. residual_add: x = x + O (not hooked)\n")
        self.log_file_handle.write("  4. post_attention_layernorm(x) -> norm_out2\n")
        self.log_file_handle.write("  5. MLP Block (parallel input = norm_out2):\n")
        self.log_file_handle.write("     - mlp_gate_proj(norm_out2) -> gate\n")
        self.log_file_handle.write("     - mlp_up_proj(norm_out2) -> up  (parallel with gate)\n")
        self.log_file_handle.write(f"     - mlp_activation(gate) -> activated{act_info}\n")
        self.log_file_handle.write("     - element_wise_multiply(activated, up) -> combined (not hooked)\n")
        self.log_file_handle.write("     - mlp_down_proj(combined) -> mlp_out\n")
        self.log_file_handle.write("  6. residual_add: x = x + mlp_out (not hooked)\n")
        self.log_file_handle.write("\n")
        self.log_file_handle.write("Note: Operations with same input can execute in parallel.\n")
        self.log_file_handle.write("      Hook timestamps may not reflect true dependency order.\n")
        self.log_file_handle.write("="*80 + "\n\n")
        
    def _get_csv_path(self, layer_idx, operation_name):
        """Generate CSV file path for a specific layer and operation."""
        run_dir = f"{self.output_dir}/{self.start_time}/{self.model_specific}"
        os.makedirs(run_dir, exist_ok=True)
        filename = f"{self.gpu_name}_{self.model_specific}_layer{layer_idx}_{operation_name}.csv"
        return os.path.join(run_dir, filename)
    
    def _open_csv(self, layer_idx, operation_name, num_features):
        """Open a CSV file for a specific operation and write header."""
        key = (layer_idx, operation_name)
        if key in self.csv_handles:
            return  # Already open
        
        csv_path = self._get_csv_path(layer_idx, operation_name)
        
        csv_flag = os.path.exists(csv_path)

        handle = open(csv_path, 'a', newline='', encoding='utf-8')
        writer = csv.writer(handle)
        
        # Header: device, input_index, decoding_index, input_text, output_token_id, output_text, layer_name, feature_0, feature_1, ...
        if not csv_flag:
            header = ["device", "input_index", "decoding_index", "input_text", "layer_name", "output_token_id", "output_text"]
            # header.extend([f"feature_{i}" for i in range(num_features)])
            writer.writerow(header)
        
        self.csv_handles[key] = handle
        self.csv_writers[key] = writer
        
        # print(f"  Opened layer operation CSV: {csv_path}")
    
    def _make_io_hook(self, layer_idx, operation_name, capture_input=True, capture_output=True):
        """Create a hook that captures input and/or output of a module."""
        def hook(module, input, output):
            # Record operation call order
            timestamp = time.time()
            call_info = {
                'timestamp': timestamp,
                'layer_idx': layer_idx,
                'operation': operation_name,
                'phase': 'call'
            }
            self.current_step_order.append(call_info)
            
            # CRITICAL: This hook must NOT modify the forward pass
            # We only observe and copy data, never modify originals
            try:
                key_base = f"layer{layer_idx}_{operation_name}"
                
                if capture_input and input is not None:
                    # input is a tuple, take first element
                    inp = input[0] if isinstance(input, tuple) else input
                    if isinstance(inp, torch.Tensor):
                        # Store reference to be copied later (outside forward pass)
                        # Using .detach() creates a new tensor that shares storage but has no gradient
                        # This is safe and doesn't affect the forward pass
                        self.current_step_data[f"{key_base}_input"] = inp.detach().clone()
                        
                        # Record input shape
                        call_info['input_shape'] = tuple(inp.shape)
                        call_info['input_dtype'] = str(inp.dtype)
                
                if capture_output and output is not None:
                    # output might be tuple (e.g., attention returns (output, attn_weights))
                    out = output[0] if isinstance(output, tuple) else output
                    if isinstance(out, torch.Tensor):
                        # Store reference to be copied later
                        self.current_step_data[f"{key_base}_output"] = out.detach().clone()
                        
                        # Record output shape
                        call_info['output_shape'] = tuple(out.shape)
                        call_info['output_dtype'] = str(out.dtype)
            except Exception as e:
                # Silently ignore errors to avoid breaking forward pass
                call_info['error'] = str(e)
            
            # CRITICAL: Must return None - this tells PyTorch we're not modifying the output
            # If we return anything else, it replaces the actual output!
            return None
        
        return hook
    
    def register_hooks(self, model, tokenizer):
        """Register hooks on first and last transformer layers to capture operation I/O.
        
        For LLaMA/Qwen architectures, typical structure:
        - input_layernorm
        - self_attn (with q_proj, k_proj, v_proj, o_proj)
        - post_attention_layernorm
        - mlp (with gate_proj, up_proj, act_fn, down_proj)
        """
        try:
            layers = model.model.layers
        except AttributeError:
            print("Warning: Could not find model.model.layers - skipping layer operation tracking")
            return
        
        num_layers = len(layers)
        
        # Detect activation function type from first layer
        if num_layers > 0 and hasattr(layers[0], 'mlp') and hasattr(layers[0].mlp, 'act_fn'):
            act_fn = layers[0].mlp.act_fn
            act_type = type(act_fn).__name__
            self.activation_function_type = act_type
            print(f"  Detected activation function: {act_type}")
        
        for idx in self.layer_indices:
            if idx < 0:
                idx = num_layers + idx  # Convert negative index
            
            if idx < 0 or idx >= num_layers:
                continue
            
            layer = layers[idx]
            non_flag = True
            print(f"  Registering operation hooks for layer {idx}...")
            
            # Input LayerNorm
            if hasattr(layer, 'input_layernorm'):
                non_flag = False
                h = layer.input_layernorm.register_forward_hook(
                    self._make_io_hook(idx, 'input_layernorm', capture_input=True, capture_output=True)
                )
                self.hooks.append(h)
                
            # Self-attention submodules
            if hasattr(layer, 'self_attn'):
                non_flag = False
                attn = layer.self_attn
                
                # Q/K/V projections
                if hasattr(attn, 'q_norm'):
                    h = attn.q_norm.register_forward_hook(
                        self._make_io_hook(idx, 'attn_q_norm', capture_input=True, capture_output=True)
                    )
                    self.hooks.append(h)
                
                if hasattr(attn, 'k_norm'):
                    h = attn.k_norm.register_forward_hook(
                        self._make_io_hook(idx, 'attn_k_norm', capture_input=True, capture_output=True)
                    )
                    self.hooks.append(h)

                if hasattr(attn, 'q_proj'):
                    h = attn.q_proj.register_forward_hook(
                        self._make_io_hook(idx, 'attn_q_proj', capture_input=True, capture_output=True)
                    )
                    self.hooks.append(h)
                
                if hasattr(attn, 'k_proj'):
                    h = attn.k_proj.register_forward_hook(
                        self._make_io_hook(idx, 'attn_k_proj', capture_input=True, capture_output=True)
                    )
                    self.hooks.append(h)
                
                if hasattr(attn, 'v_proj'):
                    h = attn.v_proj.register_forward_hook(
                        self._make_io_hook(idx, 'attn_v_proj', capture_input=True, capture_output=True)
                    )
                    self.hooks.append(h)
                
                # Output projection
                if hasattr(attn, 'o_proj'):
                    h = attn.o_proj.register_forward_hook(
                        self._make_io_hook(idx, 'attn_o_proj', capture_input=True, capture_output=True)
                    )
                    self.hooks.append(h)
                else:
                    print(f"    Warning: No known attention projections found in layer {idx} self_attn")
            
            # Post-attention LayerNorm
            if hasattr(layer, 'post_attention_layernorm'):
                non_flag = False
                h = layer.post_attention_layernorm.register_forward_hook(
                    self._make_io_hook(idx, 'post_attention_layernorm', capture_input=True, capture_output=True)
                )
                self.hooks.append(h)
            
            # MLP submodules
            if hasattr(layer, 'mlp'):
                non_flag = False
                mlp = layer.mlp
                
                # Gate projection
                if hasattr(mlp, 'gate_proj'):
                    h = mlp.gate_proj.register_forward_hook(
                        self._make_io_hook(idx, 'mlp_gate_proj', capture_input=True, capture_output=True)
                    )
                    self.hooks.append(h)
                
                # Up projection
                if hasattr(mlp, 'up_proj'):
                    h = mlp.up_proj.register_forward_hook(
                        self._make_io_hook(idx, 'mlp_up_proj', capture_input=True, capture_output=True)
                    )
                    self.hooks.append(h)
                
                # Activation function
                if hasattr(mlp, 'act_fn'):
                    h = mlp.act_fn.register_forward_hook(
                        self._make_io_hook(idx, 'mlp_activation', capture_input=True, capture_output=True)
                    )
                    self.hooks.append(h)
                
                # Down projection
                if hasattr(mlp, 'down_proj'):
                    h = mlp.down_proj.register_forward_hook(
                        self._make_io_hook(idx, 'mlp_down_proj', capture_input=True, capture_output=True)
                    )
                    self.hooks.append(h)
                else:
                    print(f"    Warning: No known MLP submodules found in layer {idx} mlp")
            if non_flag:
                print(f"    Warning: No known submodules found in layer {idx} for operation tracking")
        print(f"  Registered {len(self.hooks)} operation hooks")
    
    def print_operation_order(self, decoding_step):
        """Print and log the operation call order for the current step."""
        if not self.print_order or not self.current_step_order:
            return
        
        # Sort by timestamp
        sorted_order = sorted(self.current_step_order, key=lambda x: x['timestamp'])
        
        # Group operations by layer and categorize them
        layer_ops = {}
        for op in sorted_order:
            layer_idx = op['layer_idx']
            if layer_idx not in layer_ops:
                layer_ops[layer_idx] = []
            layer_ops[layer_idx].append(op)
        
        # Prepare output lines
        lines = []
        lines.append(f"\n=== Operation Order for Decoding Step {decoding_step} ===")
        lines.append("(Actual hook call order - see architecture diagram for true dependencies)")
        lines.append("")
        
        seq_num = 1
        for layer_idx in sorted(layer_ops.keys()):
            ops = layer_ops[layer_idx]
            
            # Categorize operations
            input_norm = None
            attn_projs = []
            attn_o = None
            post_norm = None
            mlp_gate = None
            mlp_up = None
            mlp_act = None
            mlp_down = None
            
            for op in ops:
                name = op['operation']
                if 'input_layernorm' in name:
                    input_norm = op
                elif 'attn_q_proj' in name or 'attn_k_proj' in name or 'attn_v_proj' in name or 'k_norm 'in name or 'q_norm' in name:
                    attn_projs.append(op)
                elif 'attn_o_proj' in name:
                    attn_o = op
                elif 'post_attention_layernorm' in name:
                    post_norm = op
                elif 'mlp_gate_proj' in name:
                    mlp_gate = op
                elif 'mlp_up_proj' in name:
                    mlp_up = op
                elif 'mlp_activation' in name:
                    mlp_act = op
                elif 'mlp_down_proj' in name:
                    mlp_down = op
            
            # Write in logical order with annotations
            lines.append(f"Layer {layer_idx}:")
            
            if input_norm:
                lines.append(self._format_op_line(seq_num, input_norm, ""))
                seq_num += 1
            
            if attn_projs:
                lines.append("  [Attention Block - Q/K/V parallel]")
                for proj in sorted(attn_projs, key=lambda x: x['operation']):
                    lines.append(self._format_op_line(seq_num, proj, "  (parallel)"))
                    seq_num += 1
            
            if attn_o:
                lines.append(self._format_op_line(seq_num, attn_o, "  (after attention)"))
                seq_num += 1
            
            if post_norm:
                lines.append(self._format_op_line(seq_num, post_norm, ""))
                seq_num += 1
            
            if mlp_gate or mlp_up:
                lines.append("  [MLP Block]")
                if mlp_gate:
                    lines.append(self._format_op_line(seq_num, mlp_gate, "  (parallel with up)"))
                    seq_num += 1
                if mlp_up:
                    lines.append(self._format_op_line(seq_num, mlp_up, "  (parallel with gate)"))
                    seq_num += 1
                if mlp_act:
                    act_annotation = "  (on gate output)"
                    if self.activation_function_type:
                        act_annotation += f" [{self.activation_function_type}]"
                    lines.append(self._format_op_line(seq_num, mlp_act, act_annotation))
                    seq_num += 1
                lines.append("    → element_wise_multiply(activated_gate, up) [not hooked]")
                if mlp_down:
                    lines.append(self._format_op_line(seq_num, mlp_down, "  (on multiplied)"))
                    seq_num += 1
            
            lines.append("")
        
        lines.append(f"=== Total Operations Hooked: {len(sorted_order)} ===")
        lines.append("")
        
        # Print to console (only for first step to avoid spam)
        if decoding_step == 0:
            for line in lines:
                print(line)
        
        # Write to log file
        if self.log_file_handle:
            for line in lines:
                self.log_file_handle.write(line + "\n")
            self.log_file_handle.flush()
    
    def _format_op_line(self, seq_num, op, annotation):
        """Format a single operation line."""
        operation = op['operation']
        info_str = f"  {seq_num:3d}. {operation:30s}"
        
        if 'input_shape' in op:
            info_str += f" | IN: {str(op['input_shape']):20s}"
        if 'output_shape' in op:
            info_str += f" | OUT: {str(op['output_shape']):20s}"
        if annotation:
            info_str += f" {annotation}"
        if 'error' in op:
            info_str += f" | ERROR: {op['error']}"
        
        return info_str
    
    def save_step_data(self, input_index, decoding_step, input_text, output_token_id, output_text, device, batch_idx=0):
        """Save captured data for current decoding step to CSV files.
        
        Args:
            input_index: Index of the input sample in the overall dataset
            decoding_step: Current decoding step
            input_text: Input text prompt
            output_token_id: Generated token ID
            output_text: Generated token text
            device: Device name
            batch_idx: Index within the batch (default: 0)
        """
        # Print and log operation order for this step (only for first batch sample to avoid spam)
        if input_index == 0 and self.print_order:
            self.print_operation_order(decoding_step)
        
        for key, tensor in self.current_step_data.items():
            # Parse key: "layer{idx}_{operation_name}_{input|output}"
            parts = key.split('_', 2)  # layer0, operation, type
            if len(parts) < 3:
                continue
            
            layer_part = parts[0]  # "layer0"
            op_and_type = parts[1:]  # ["mlp", "gate", "proj", "input"] or similar
            
            # Extract layer index
            try:
                layer_idx = int(layer_part.replace('layer', ''))
            except ValueError:
                continue
            
            # Reconstruct operation name and type
            # print(parts)
            if 'input' in op_and_type[-1]:
                io_type = 'input'
                operation_name = '_'.join(op_and_type[:-1])
            elif 'output' in op_and_type[-1]:
                io_type = 'output'
                operation_name = '_'.join(op_and_type[:-1])
            else:
                operation_name = '_'.join(op_and_type)
                io_type = 'unknown'
            
            full_operation_name = f"{'-'.join(parts)}"
            
            # ===== GPU에서 slice 수행 (디바이스별 연산 차이 보존) =====
            # text_generator.py와 동일한 방식으로 GPU에서 먼저 처리
            
            # Extract only the specific batch sample
            if tensor.dim() >= 2:
                # GPU에서 slice - 디바이스별 부동소수점 연산 차이가 여기서 발생
                # tensor shape: (batch_size, seq_len, hidden_dim)
                sample_tensor = tensor[batch_idx:batch_idx+1, -1, :]  # shape: (1, hidden_dim)
                flat_gpu = sample_tensor.flatten()
            else:
                # 1D tensor case (unlikely)
                flat_gpu = tensor.flatten()
            
            # GPU 연산 완료 후 CPU로 이동
            flat = flat_gpu.cpu().numpy()
            # ===== 수정 끝 =====
            
            num_features = len(flat)
            
            # Open CSV if not already open
            self._open_csv(layer_idx, full_operation_name, num_features)
            
            # Write row
            writer_key = (layer_idx, full_operation_name)
            if writer_key in self.csv_writers:
                row = [
                    device,
                    input_index,
                    decoding_step,
                    input_text[:100],  # Truncate to 100 chars
                    full_operation_name,
                    output_token_id,
                    output_text.replace('\n', '\\n').replace(',', '[COMMA]'),
                ]
                row.extend([f"{val:.8f}" for val in flat])
                self.csv_writers[writer_key].writerow(row)
                # Flush immediately to ensure data is written to disk
                self.csv_handles[writer_key].flush()
            else:
                print(f"WARNING: writer_key {writer_key} not found in csv_writers!")
                print(f"Available keys: {list(self.csv_writers.keys())}")
                print(f"Attempting to open CSV for layer {layer_idx}, operation {full_operation_name}")
            
            # Clean up
            del flat_gpu, flat
    
    def clear_step_buffer(self):
        """Clear the current step data buffer after all batch samples are processed."""
        self.current_step_data.clear()
        # Also save operation order to history
        if self.current_step_order:
            self.operation_call_order.append(list(self.current_step_order))
            self.current_step_order.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()
    
    def close(self):
        """Close all CSV files and remove hooks."""
        self.remove_hooks()
        
        # Close log file
        if self.log_file_handle:
            try:
                self.log_file_handle.write("\n" + "="*80 + "\n")
                self.log_file_handle.write(f"Log completed\n")
                self.log_file_handle.write(f"Total steps recorded: {len(self.operation_call_order)}\n")
                self.log_file_handle.close()
                print(f"  Closed operation order log file")
            except Exception:
                pass
        
        for handle in self.csv_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        
        self.csv_handles.clear()
        self.csv_writers.clear()
        
        print(f"  Closed all layer operation CSV files")
