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


def generate_with_activations(input_index, gpu_name, model_specific, model, messages, tokenizer, inputs, device, max_new_tokens, temperature=0.0, csv_writer=None, model_hidden_size=None, num_layers=None):
    """
    Generate text while capturing layer activations at each step.
    
    Args:
        model: The language model to use for generation
        tokenizer: Tokenizer for the model
        inputs: Tokenized input data
        device: Device to run generation on
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Temperature for sampling (0.0 for greedy)
        csv_writer: CSV writer object to write activations directly
        model_hidden_size: Size of the hidden layer
        num_layers: Number of transformer layers
        
    Returns:
        tuple: (generated_ids, step_count)
    """
    # check if model is already on the correct device
    model = model.to(device)
    model.eval()  # evaluation 모드 확실히 설정
    inputs = {k: v.to(device) for k, v in inputs.items()}
    generated_ids = inputs['input_ids'].clone()
    
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
        
        # Write activations directly to CSV if writer provided
        if csv_writer is not None:
            token_id = next_token_id.item()
            token_text = new_token_text.replace('\n', '\\n').replace(',', '[COMMA]')
            
            for i in range(num_layers):
                layer_name = f"layer_{i}"
                if layer_name in step_activations:
                    activation = step_activations[layer_name].cpu().numpy()
                    activation_flat = activation.flatten()

                    row = [gpu_name, model_specific, device.type, input_index, messages, i, step, token_id, token_text]
                    row.extend([f"{val:.8f}" for val in activation_flat])
                    csv_writer.writerow(row)
                    
                    # 즉시 메모리에서 삭제
                    del activation, activation_flat
            
            # step_activations 삭제
            del step_activations
            step_activations = None
        
        # Add next token to sequence
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        
        # 중간 변수 정리
        del next_token_logits, outputs
        if 'scaled_logits' in locals():
            del scaled_logits
        if 'probabilities' in locals():
            del probabilities
        
        # 메모리 정리 (5스텝마다)
        if step % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # print_gpu_memory(label=f"After step {step}")
        
        # Check for EOS
        if next_token_id.item() == eos_token_id:
            break
    end_time = time.time()
    print(f"\ntotal steps: {step+1}, tps: {(step+1)/(end_time - init_time):.2f}, total seconds: {end_time - init_time:.2f}")
    print()
    
    # 생성 완료 후 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return generated_ids, step+1