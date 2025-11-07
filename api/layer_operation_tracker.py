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


class LayerOperationTracker:
    """Captures input/output tensors at each operation within first/last transformer layers."""
    
    def __init__(self, output_dir, start_time, gpu_name, model_specific, batch_size, layer_indices=[0, -1]):
        """
        Args:
            output_dir: Base output directory
            start_time: Run timestamp
            gpu_name: GPU device name
            model_specific: Model name (safe for filename)
            batch_size: Current batch size
            layer_indices: Which layers to track (default: first and last)
        """
        self.output_dir = output_dir
        self.start_time = start_time
        self.gpu_name = gpu_name
        self.model_specific = model_specific
        self.batch_size = batch_size
        self.layer_indices = layer_indices
        
        # CSV file handles and writers for each operation
        self.csv_handles = {}
        self.csv_writers = {}
        
        # Hook handles
        self.hooks = []
        
        # Data buffer for current step
        self.current_step_data = {}
        
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
        handle = open(csv_path, 'w', newline='', encoding='utf-8')
        writer = csv.writer(handle)
        
        # Header: device, input_index, decoding_index, input_text, output_token_id, output_text, feature_0, feature_1, ...
        header = ["device", "input_index", "decoding_index", "input_text", "output_token_id", "output_text"]
        # header.extend([f"feature_{i}" for i in range(num_features)])
        writer.writerow(header)
        
        self.csv_handles[key] = handle
        self.csv_writers[key] = writer
        
        # print(f"  Opened layer operation CSV: {csv_path}")
    
    def _make_io_hook(self, layer_idx, operation_name, capture_input=True, capture_output=True):
        """Create a hook that captures input and/or output of a module."""
        def hook(module, input, output):
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
                        self.current_step_data[f"{key_base}_input"] = inp.detach()
                
                if capture_output and output is not None:
                    # output might be tuple (e.g., attention returns (output, attn_weights))
                    out = output[0] if isinstance(output, tuple) else output
                    if isinstance(out, torch.Tensor):
                        # Store reference to be copied later
                        self.current_step_data[f"{key_base}_output"] = out.detach()
            except Exception as e:
                # Silently ignore errors to avoid breaking forward pass
                pass
            
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
                
                # Activation function (captures input = before activation, output = after activation)
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
    
    def save_step_data(self, input_index, decoding_step, input_text, output_token_id, output_text, device):
        """Save captured data for current decoding step to CSV files."""
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
            if 'input' in op_and_type[-1]:
                io_type = 'input'
                operation_name = '_'.join(op_and_type[:-1])
            elif 'output' in op_and_type[-1]:
                io_type = 'output'
                operation_name = '_'.join(op_and_type[:-1])
            else:
                operation_name = '_'.join(op_and_type)
                io_type = 'unknown'
            
            full_operation_name = f"{operation_name}_{io_type}"
            
            # NOW copy to CPU and flatten (this happens AFTER forward pass is complete)
            # Clone to ensure we don't modify the original tensor
            tensor_cpu = tensor.clone().cpu()
            
            # Flatten tensor (take last token position: [:, -1, :])
            if tensor_cpu.dim() >= 2:
                # Shape: (batch_size, seq_len, hidden_dim) -> take last token
                flat = tensor_cpu[:, -1, :].flatten().numpy()
            else:
                flat = tensor_cpu.flatten().numpy()
            
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
                    output_token_id,
                    output_text.replace('\n', '\\n').replace(',', '[COMMA]')
                ]
                row.extend([f"{val:.8f}" for val in flat])
                self.csv_writers[writer_key].writerow(row)
            
            # Clean up
            del tensor_cpu, flat
        
        # Clear buffer for next step
        self.current_step_data.clear()
    
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
        
        for handle in self.csv_handles.values():
            try:
                handle.close()
            except Exception:
                pass
        
        self.csv_handles.clear()
        self.csv_writers.clear()
        
        print(f"  Closed all layer operation CSV files")
