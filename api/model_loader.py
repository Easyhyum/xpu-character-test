"""
Model loading utilities for the XPU character test project.
"""

# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch


def model_load_function(model_name):
    print(f"Loading model: {model_name}")
    try:
        if model_name == "unsloth/Meta-Llama-3.1-8B-bnb-4bit":
            # 4-bit quantized model - already quantized, just load it
            # This model is pre-quantized, so we don't need to pass quantization_config
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
        elif model_name == "RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8":
            # INT8 quantized model (W8A8) - uses compressed-tensors
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
                    device_map=None,  # CPU에 먼저 로드
                    low_cpu_mem_usage=True,
                )
                
                # After loading, replace forward method for INT8 quantized layers
                quantized_layer_count = 0
                
                def create_dequant_forward(original_forward, weight_scale):
                    def dequant_forward(self, input):
                        # Dequantize on-the-fly
                        x = input
                        target_dtype = x.dtype
                        
                        # Dequantize weight
                        weight_float = self.weight.to(target_dtype) * weight_scale.to(target_dtype)
                        
                        # Perform linear operation
                        return torch.nn.functional.linear(x, weight_float, self.bias)
                    return dequant_forward
                
                # Replace forward method for all quantized Linear layers
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight_scale') and module.weight.dtype == torch.int8:
                        # Save weight_scale
                        weight_scale = module.weight_scale
                        # Replace the forward method
                        module.forward = create_dequant_forward(module.forward, weight_scale).__get__(module, type(module))
                        quantized_layer_count += 1
                
                print(f"Replaced forward method for {quantized_layer_count} quantized Linear layers")
                        
            finally:
                modeling_utils._load_parameter_into_model = original_load_param
        elif model_name == "RedHatAI/Meta-Llama-3.1-8B-FP8":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                attn_implementation="eager",  # For consistency and compatibility
                low_cpu_mem_usage=True,
            )
        else:
            # Default loading for other models
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                attn_implementation="eager",
                low_cpu_mem_usage=True,
            )
        
        # Explicitly disable gradients for all parameters after loading
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        
        # Determine special device
        special_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            special_device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            special_device = torch.device("mps")
        else:
            special_device = None
        
        # For models using device_map="auto", check if all parameters are on the same device
        # If not, explicitly move them (important for compressed_tensors models)
        if special_device is not None:
            devices_found = set()
            for param in model.parameters():
                devices_found.add(param.device)
            
            if len(devices_found) > 1 or any(d.type == 'cpu' for d in devices_found):
                print(f"  Warning: Model parameters on multiple devices: {devices_found}")
                print(f"  Moving all parameters to {special_device}...")
                try:
                    # Move model to target device
                    model = model.to(special_device)
                    
                    # For compressed_tensors models, also move buffers and attributes
                    for name, module in model.named_modules():
                        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                            if module.weight.device != special_device:
                                module.weight = module.weight.to(special_device)
                        if hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor):
                            if module.bias is not None and module.bias.device != special_device:
                                module.bias = module.bias.to(special_device)
                        # Move any scale/zero_point attributes for quantized layers
                        for attr_name in ['weight_scale', 'weight_zero_point', 'input_scale', 'input_zero_point']:
                            if hasattr(module, attr_name):
                                attr = getattr(module, attr_name)
                                if isinstance(attr, torch.Tensor) and attr.device != special_device:
                                    setattr(module, attr_name, attr.to(special_device))
                    
                    print(f"  ✓ All parameters moved to {special_device}")
                except Exception as move_error:
                    print(f"  Could not move to GPU, keeping on current device: {move_error}")
        if torch.backends.mps.is_available():
            # For MPS, ensure all parameters are on MPS device
            for param in model.parameters():
                if param.device.type != 'mps':
                    print(f"  Moving parameter from {param.device} to mps")
                    model = model.to(torch.device('mps'))
                    break
        # print(f"Model loaded successfully on {next(model.parameters()).device}")
        # print(f"parameter dtypes weight and bias and activation")
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
        print(f"Failed to load model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None