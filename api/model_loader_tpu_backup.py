"""
Model loading utilities for TPU devices in the XPU character test project.
"""

import os
os.environ['PJRT_DEVICE'] = 'TPU'

print("  [API] Loading model_loader_tpu module...")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import torch_xla
import torch_xla.core.xla_model as xm

print("  [API] ✓ model_loader_tpu loaded successfully")


def create_dequant_forward(original_forward, weight_scale):
    """Helper to create a forward method that dequantizes INT8 weights on the fly"""
    def dequant_forward(self, input):
        # Dequantize on-the-fly
        x = input
        target_dtype = x.dtype
        
        # Dequantize weight
        # weight is (out, in), scale is (out)
        # Reshape scale to (out, 1) for broadcasting
        if weight_scale.dim() == 1:
            scale = weight_scale.view(-1, 1).to(target_dtype)
        else:
            scale = weight_scale.to(target_dtype)
            
        weight_float = self.weight.to(target_dtype) * scale
        
        # Perform linear operation
        return torch.nn.functional.linear(x, weight_float, self.bias)
    return dequant_forward


def model_load_function_tpu(model_name):
    """Load model optimized for TPU execution"""
    print(f"\n{'='*80}")
    print(f"Loading model for TPU: {model_name}")
    print(f"{'='*80}")
    
    # Get TPU device
    tpu_device = xm.xla_device()
    print(f"Target TPU device: {tpu_device}")
    
    try:
        if model_name == "RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8":
            # INT8 quantized model (W8A8) - TPU에서 INT8 연산 지원
            print("Loading INT8 (W8A8) quantized model for TPU...")
            # Patch _load_parameter_into_model to handle int8 tensors
            from transformers import modeling_utils
            original_load_param = modeling_utils._load_parameter_into_model
            
            def patched_load_param(model, param_name, tensor):
                # For int8 tensors or any non-float tensors, set requires_grad=False explicitly
                if tensor.dtype in [torch.int8, torch.uint8] or not tensor.is_floating_point():
                    module_name, param_type = param_name.rsplit(".", 1)
                    module = model
                    for name in module_name.split("."):
                        module = getattr(module, name)
                    # Directly set the parameter/buffer without gradient checking
                    param = torch.nn.Parameter(tensor, requires_grad=False)
                    if param_type == "weight":
                        module.weight = param
                    elif param_type == "bias":
                        module.bias = param
                    else:
                        # For scale, zero_point, etc., set as attribute
                        setattr(module, param_type, param)
                else:
                    original_load_param(model, param_name, tensor)
            
            modeling_utils._load_parameter_into_model = patched_load_param
            
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    attn_implementation="eager",
                    device_map=None,
                    low_cpu_mem_usage=True,
                )
                
                # INT8 quantized layers - keep them as INT8 for TPU
                quantized_layer_count = 0
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight_scale') and module.weight.dtype == torch.int8:
                        # Replace forward method for W8A8 layers too
                        module.forward = create_dequant_forward(module.forward, module.weight_scale).__get__(module, type(module))
                        quantized_layer_count += 1
                
                print(f"Found {quantized_layer_count} INT8 quantized layers (will run with on-the-fly dequantization)")
                        
            finally:
                modeling_utils._load_parameter_into_model = original_load_param
        else:
            # Default loading for other models - preserve original dtype
            print("Loading model with default settings (preserving original dtype)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=None,
                trust_remote_code=True,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
            print("Model loaded with original dtype")
        
        # Explicitly disable gradients for all parameters after loading
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        
        # Move model to TPU device
        print(f"\nMoving model to TPU device: {tpu_device}")
        print("This may take several minutes for large models...")
        print("Please be patient, the model is being transferred to TPU memory...")
        
        try:
            # For TPU, we need to move the model in a specific way
            # Move model to TPU (this is the slow part due to PCIe transfer)
            print("Step 1/2: Moving model to TPU (transferring ~8-16GB data, may take 2-5 minutes)...")
            import time
            start_transfer = time.time()
            model = model.to(tpu_device)
            transfer_time = time.time() - start_transfer
            print(f"  Transfer completed in {transfer_time:.1f} seconds")
            
            # Mark XLA step to ensure transfer is complete and compile
            print("Step 2/2: XLA compilation and initialization...")
            xm.mark_step()
            print("  XLA ready")
            
            # Verify model is on TPU
            first_param_device = next(model.parameters()).device
            print(f"✓ Model successfully loaded on device: {first_param_device}")
        except Exception as e:
            print(f"Error moving model to TPU: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Print parameter dtypes
        activation_dtypes = []
        weight_dtypes = []
        bias_dtypes = []
        for name, param in model.named_parameters():
            if "weight" in name:
                weight_dtypes.append(param.dtype)
            elif "bias" in name:
                bias_dtypes.append(param.dtype)
            elif "activation" in name:
                activation_dtypes.append(param.dtype)
        print(f"Weight dtypes: {set(weight_dtypes)}")
        print(f"Bias dtypes: {set(bias_dtypes)}")
        print(f"Activation dtypes: {set(activation_dtypes)}")
        
        return model
        
    except Exception as e:
        print(f"Failed to load model {model_name} on TPU: {e}")
        import traceback
        traceback.print_exc()
        return None
