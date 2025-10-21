"""
Text generation utilities with activation tracking for the XPU character test project.
"""

import torch
import gc
import time

def create_activation_hook(layer_name, activations_dict):
    """
    Create a hook function to capture layer activations.
    
    Args:
        layer_name (str): Name identifier for the layer
        activations_dict (dict): Dictionary to store activations
        
    Returns:
        function: Hook function for the layer
    """
    def hook(module, input, output):
        activation = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
        activations_dict[layer_name] = activation[:, -1:, :].clone()
    return hook


def generate_with_activations(model, tokenizer, inputs, device, max_new_tokens, temperature=0.0):
    """
    Generate text while capturing layer activations at each step.
    
    Args:
        model: The language model to use for generation
        tokenizer: Tokenizer for the model
        inputs: Tokenized input data
        device: Device to run generation on
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Temperature for sampling (0.0 for greedy)
        
    Returns:
        tuple: (generated_ids, all_step_activations)
    """
    # check if model is already on the correct device
    model = model.to(device)
    model.eval()  # evaluation 모드 확실히 설정
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generated_ids = inputs['input_ids'].clone()
    all_step_activations = []
    
    # Get transformer layers and eos token
    transformer_layers = model.model.layers
    eos_token_id = tokenizer.eos_token_id
    init_time = time.time()
    for step in range(max_new_tokens):
        step_activations = {}
        hooks = []
        
        # Register hooks
        for i, layer in enumerate(transformer_layers):
            hook = create_activation_hook(f"layer_{i}", step_activations)
            hooks.append(layer.register_forward_hook(hook))
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=generated_ids, attention_mask=torch.ones_like(generated_ids))
            
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        # Get next token with temperature control
        next_token_logits = outputs.logits[:, -1, :]
        
        if temperature == 0.0:
            # Temperature = 0: Greedy decoding (완전히 결정론적)
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            # Temperature > 0: Sampling (확률적 생성)
            # Apply temperature scaling
            scaled_logits = next_token_logits / temperature
            probabilities = torch.softmax(scaled_logits, dim=-1)
            next_token_id = torch.multinomial(probabilities, num_samples=1)
        
        # Decode and print the new token immediately
        new_token_text = tokenizer.decode(next_token_id[0], skip_special_tokens=False)
        # Replace newlines with visible \n representation
        display_text = new_token_text.replace('\n', ' \\n ')
        print(display_text, end='', flush=True)  # Print without newline, flush immediately
        
        # Store activations with step info
        step_activations['step'] = step
        step_activations['token_id'] = next_token_id.item()
        step_activations['token_text'] = new_token_text
        all_step_activations.append(step_activations)
        
        # Add next token to sequence
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        
        # Check for EOS
        if next_token_id.item() == eos_token_id:
            break
    end_time = time.time()
    print(f"\ntotal steps: {step+1}, tps: {(step+1)/(end_time - init_time):.2f}, total seconds: {end_time - init_time:.2f}")
    print()
    
    # 생성 완료 후 중간 변수들 메모리 정리
    del next_token_logits, outputs
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    return generated_ids, all_step_activations