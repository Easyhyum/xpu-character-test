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
        # elif model_name == "Qwen/Qwen3-VL-8B-Instruct":
        #     model = Qwen3VLForConditionalGeneration.from_pretrained(
        #         "Qwen/Qwen3-VL-8B-Instruct",
        #         device_map="auto" if torch.cuda.is_available() else None,
        #         trust_remote_code=True,
        #         attn_implementation="eager",
        #         low_cpu_mem_usage=True,
        #     )
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
        
        # Now move to appropriate device if needed
        if torch.cuda.is_available():
            try:
                model = model.to(special_device)
                # print(f"Model moved to {special_device}")
            except Exception as move_error:
                print(f"Could not move to GPU, keeping on CPU: {move_error}")
        
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
        exit()
        return None