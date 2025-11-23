"""
Layer operation tracker for detailed input/output capture at each operation step - TPU version.

Tracks operations like:
- MLP: input -> gate_proj -> up_proj -> activation -> down_proj -> output
- Attention: input -> q_proj/k_proj/v_proj -> attention -> o_proj -> output
- LayerNorm: input -> norm -> output
"""
import os
os.environ['PJRT_DEVICE'] = 'TPU'

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import csv
import time


class LayerOperationTrackerTPU:
    """Captures input/output tensors at each operation within first/last transformer layers on TPU."""
    
    def __init__(self, output_dir, start_time, tpu_name, model_specific, batch_size, layer_indices=[0, -1], print_order=True):
        """
        Args:
            output_dir: Base output directory
            start_time: Run timestamp
            tpu_name: TPU device name
            model_specific: Model name (safe for filename)
            batch_size: Current batch size
            layer_indices: Which layers to track (default: first and last)
            print_order: Whether to print and log operation order (default: True)
        """
        self.output_dir = output_dir
        self.start_time = start_time
        self.tpu_name = tpu_name
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
        log_filename = f"{self.tpu_name}_{self.model_specific}_operation_order.log"
        log_path = os.path.join(run_dir, log_filename)
        
        self.log_file_handle = open(log_path, 'w', encoding='utf-8')
        self.log_file_handle.write(f"Operation Order Log (TPU)\n")
        self.log_file_handle.write(f"TPU: {self.tpu_name}\n")
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
            act_info = f" (type: {self.activation_function_type})"
        
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
        filename = f"{self.tpu_name}_{self.model_specific}_layer{layer_idx}_{operation_name}.csv"
        return os.path.join(run_dir, filename)
    
    def _open_csv(self, layer_idx, operation_name, num_features):
        """Open a CSV file for a specific operation and write header."""
        key = (layer_idx, operation_name)
        if key in self.csv_handles:
            return
        
        csv_path = self._get_csv_path(layer_idx, operation_name)
        
        csv_flag = os.path.exists(csv_path)

        handle = open(csv_path, 'a', newline='', encoding='utf-8')
        writer = csv.writer(handle)
        
        # Header: device, input_index, decoding_index, input_text, output_token_id, output_text, layer_name, feature_0, feature_1, ...
        if not csv_flag:
            header = ["device", "input_index", "decoding_index", "input_text", "output_token_id", "output_text", "layer_name"]
            header.extend([f"feature_{i}" for i in range(num_features)])
            writer.writerow(header)
        
        self.csv_handles[key] = handle
        self.csv_writers[key] = writer
    
    def _make_io_hook(self, layer_idx, operation_name, capture_input=True, capture_output=True):
        """Create a hook that captures input and/or output of a module."""
        def hook(module, input, output):
            timestamp = time.time()
            
            # Capture input if requested
            if capture_input and input is not None:
                if isinstance(input, tuple):
                    input_tensor = input[0]
                else:
                    input_tensor = input
                
                if isinstance(input_tensor, torch.Tensor):
                    key = (layer_idx, f"{operation_name}_input")
                    self.current_step_data[key] = input_tensor.detach().clone()
                    
                    # Record order
                    self.current_step_order.append({
                        'timestamp': timestamp,
                        'layer_idx': layer_idx,
                        'operation': f"{operation_name}_input",
                        'input_shape': tuple(input_tensor.shape),
                        'dtype': str(input_tensor.dtype)
                    })
            
            # Capture output if requested
            if capture_output and output is not None:
                if isinstance(output, tuple):
                    output_tensor = output[0]
                else:
                    output_tensor = output
                
                if isinstance(output_tensor, torch.Tensor):
                    key = (layer_idx, f"{operation_name}_output")
                    self.current_step_data[key] = output_tensor.detach().clone()
                    
                    # Record order
                    self.current_step_order.append({
                        'timestamp': timestamp,
                        'layer_idx': layer_idx,
                        'operation': f"{operation_name}_output",
                        'output_shape': tuple(output_tensor.shape),
                        'dtype': str(output_tensor.dtype)
                    })
        
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
            print("  Warning: Could not find model.model.layers, trying model.layers")
            layers = model.layers
        
        num_layers = len(layers)
        
        # Detect activation function type from first layer
        if num_layers > 0 and hasattr(layers[0], 'mlp') and hasattr(layers[0].mlp, 'act_fn'):
            act_fn = layers[0].mlp.act_fn
            self.activation_function_type = type(act_fn).__name__
            print(f"  Detected activation function: {self.activation_function_type}")
        
        for idx in self.layer_indices:
            if idx < 0:
                idx = num_layers + idx
            
            if idx < 0 or idx >= num_layers:
                print(f"  Warning: layer index {idx} out of range [0, {num_layers-1}], skipping")
                continue
            
            layer = layers[idx]
            print(f"  Registering hooks for layer {idx}...")
            
            # Hook LayerNorms
            if hasattr(layer, 'input_layernorm'):
                hook = self._make_io_hook(idx, 'input_layernorm', capture_input=True, capture_output=True)
                self.hooks.append(layer.input_layernorm.register_forward_hook(hook))
            
            if hasattr(layer, 'post_attention_layernorm'):
                hook = self._make_io_hook(idx, 'post_attention_layernorm', capture_input=True, capture_output=True)
                self.hooks.append(layer.post_attention_layernorm.register_forward_hook(hook))
            
            # Hook Attention components
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                
                if hasattr(attn, 'q_proj'):
                    hook = self._make_io_hook(idx, 'attn_q_proj', capture_input=True, capture_output=True)
                    self.hooks.append(attn.q_proj.register_forward_hook(hook))
                
                if hasattr(attn, 'k_proj'):
                    hook = self._make_io_hook(idx, 'attn_k_proj', capture_input=True, capture_output=True)
                    self.hooks.append(attn.k_proj.register_forward_hook(hook))
                
                if hasattr(attn, 'v_proj'):
                    hook = self._make_io_hook(idx, 'attn_v_proj', capture_input=True, capture_output=True)
                    self.hooks.append(attn.v_proj.register_forward_hook(hook))
                
                if hasattr(attn, 'o_proj'):
                    hook = self._make_io_hook(idx, 'attn_o_proj', capture_input=True, capture_output=True)
                    self.hooks.append(attn.o_proj.register_forward_hook(hook))
            
            # Hook MLP components
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                
                if hasattr(mlp, 'gate_proj'):
                    hook = self._make_io_hook(idx, 'mlp_gate_proj', capture_input=True, capture_output=True)
                    self.hooks.append(mlp.gate_proj.register_forward_hook(hook))
                
                if hasattr(mlp, 'up_proj'):
                    hook = self._make_io_hook(idx, 'mlp_up_proj', capture_input=True, capture_output=True)
                    self.hooks.append(mlp.up_proj.register_forward_hook(hook))
                
                if hasattr(mlp, 'down_proj'):
                    hook = self._make_io_hook(idx, 'mlp_down_proj', capture_input=True, capture_output=True)
                    self.hooks.append(mlp.down_proj.register_forward_hook(hook))
                
                if hasattr(mlp, 'act_fn'):
                    hook = self._make_io_hook(idx, 'mlp_activation', capture_input=True, capture_output=True)
                    self.hooks.append(mlp.act_fn.register_forward_hook(hook))
        
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
        lines.append(f"\n=== Operation Order for Decoding Step {decoding_step} (TPU) ===")
        lines.append("(Actual hook call order - see architecture diagram for true dependencies)")
        lines.append("")
        
        seq_num = 1
        for layer_idx in sorted(layer_ops.keys()):
            lines.append(f"Layer {layer_idx}:")
            for op in layer_ops[layer_idx]:
                operation = op['operation']
                info_str = f"  {seq_num:3d}. {operation:30s}"
                
                if 'input_shape' in op:
                    info_str += f" input_shape={op['input_shape']}"
                if 'output_shape' in op:
                    info_str += f" output_shape={op['output_shape']}"
                if 'dtype' in op:
                    info_str += f" dtype={op['dtype']}"
                
                lines.append(info_str)
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
    
    def save_step_data(self, input_index, decoding_step, input_text, output_token_id, output_text, device, batch_idx=0):
        """Save captured data for current step to CSV files."""
        # Print operation order if enabled
        self.print_operation_order(decoding_step)
        
        # Save each captured operation's input/output
        for key, tensor in self.current_step_data.items():
            layer_idx, operation_name = key
            
            # Extract the relevant slice for this batch index
            # tensor shape is typically (batch_size, seq_len, hidden_size)
            if tensor.dim() >= 2 and tensor.size(0) > batch_idx:
                # Take batch_idx slice and the last token (seq_len=-1)
                if tensor.dim() == 3:
                    data_slice = tensor[batch_idx, -1, :].cpu().numpy()
                elif tensor.dim() == 2:
                    data_slice = tensor[batch_idx, :].cpu().numpy()
                else:
                    data_slice = tensor[batch_idx].cpu().numpy()
                
                data_flat = data_slice.flatten()
                num_features = len(data_flat)
                
                # Open CSV if not already open
                self._open_csv(layer_idx, operation_name, num_features)
                
                # Write row
                writer = self.csv_writers[(layer_idx, operation_name)]
                row = [
                    str(device),
                    input_index,
                    decoding_step,
                    input_text.replace('\n', '\\n').replace(',', '[COMMA]'),
                    output_token_id,
                    output_text.replace('\n', '\\n').replace(',', '[COMMA]'),
                    f"layer_{layer_idx}"
                ]
                row.extend([f"{val:.8f}" for val in data_flat])
                writer.writerow(row)
    
    def clear_step_buffer(self):
        """Clear the current step data buffer."""
        self.current_step_data.clear()
        self.current_step_order.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        print("  Removed all operation hooks")
    
    def close(self):
        """Close all CSV files and log file."""
        for handle in self.csv_handles.values():
            handle.close()
        self.csv_handles.clear()
        self.csv_writers.clear()
        
        if self.log_file_handle:
            self.log_file_handle.close()
            self.log_file_handle = None
        
        self.remove_hooks()
        print("  Closed all layer operation tracking files")
