from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, set_seed
import torch
import numpy as np
import csv
import random
import os, time
import contextlib
import json
from api import model_load_function, generate_with_activations, create_activation_hook

def set_deterministic_mode(seed=42):
    print(f"Setting deterministic mode with seed: {seed}")
    
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    set_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    print("✓ Deterministic mode enabled")
set_deterministic_mode(42)


cpu_device = torch.device("cpu")
special_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    special_device = torch.device("cuda")
elif torch.backends.mps.is_available():
    special_device = torch.device("mps")
else:
    special_device = None

with open("hyperparameter.json", "r") as f:
    hyperparams = json.load(f)
model_list = hyperparams['models']
inputs = hyperparams['inputs']
print(model_list)
print(inputs)
print(hyperparams)
max_new_tokens = hyperparams['max_new_tokens']
cpu_enable = hyperparams.get('cpu', 'Enable').lower() != 'disable'
output_dir = "/workspace/outputs"
os.makedirs(output_dir, exist_ok=True)
import argparse

# Allow passing start time from environment or CLI so run.sh and Python share the same timestamp
parser = argparse.ArgumentParser()
parser.add_argument('--start-time', dest='start_time', default=None, help='Start time string (YYYYmmdd-HHMMSS) passed from the wrapper')
args, unknown = parser.parse_known_args()

# Determine start_time: CLI arg > ENV START_TIME > current time
start_time = None
if args.start_time:
    start_time = args.start_time
elif os.environ.get('START_TIME'):
    start_time = os.environ.get('START_TIME')
else:
    start_time = time.strftime("%Y%m%d-%H%M%S")

print(f"Run start_time: {start_time}")

def save_activations_to_csv(all_activations, csv_file_path, model_hidden_size, transformer_layers):
    """
    Save activations data to CSV file.
    
    Args:
        all_activations: List of step activation data
        csv_file_path: Path to save the CSV file
        model_hidden_size: Size of the hidden layer
        transformer_layers: List of transformer layers
    """
    with open(csv_file_path, "w", newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        header = ["layer", "decoding_step", "token_id", "token_text"]
        for col in range(model_hidden_size):
            header.append(f"hidden_col_{col}")
        writer.writerow(header)
        
        # Write data for each step and layer
        for step_data in all_activations:
            step = step_data['step']
            token_id = step_data['token_id']
            token_text = step_data['token_text'].replace('\n', '\\n').replace(',', '[COMMA]')
            
            for i in range(len(transformer_layers)):
                layer_name = f"layer_{i}"
                if layer_name in step_data:
                    activation = step_data[layer_name].cpu().numpy()  # Shape: [1, 1, hidden_size]
                    activation_flat = activation.flatten()  # Flatten to [hidden_size]
                    
                    row = [i, step, token_id, token_text]
                    row.extend([f"{val:.8f}" for val in activation_flat])
                    writer.writerow(row)

def save_activation_differences_to_csv(all_cpu_activations, all_special_activations, diff_csv_file, model_hidden_size, transformer_layers):
    """
    Save activation differences between CPU and special device to CSV file.
    
    Args:
        all_cpu_activations: CPU activations data
        all_special_activations: Special device activations data
        diff_csv_file: Path to save the differences CSV file
        model_hidden_size: Size of the hidden layer
        transformer_layers: List of transformer layers
        
    Returns:
        float: Total cumulative difference
    """
    with open(diff_csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        
        header = ["layer", "decoding_step", "mean_abs_diff", "max_abs_diff"]
        for col in range(model_hidden_size):
            header.append(f"col_{col}_diff")
        writer.writerow(header)
        
        total_diff_sum = 0
        for step_idx in range(len(all_cpu_activations)):
            cpu_step = all_cpu_activations[step_idx]
            special_step = all_special_activations[step_idx]
            
            for i in range(len(transformer_layers)):
                layer_name = f"layer_{i}"
                if layer_name in cpu_step and layer_name in special_step:
                    cpu_act = cpu_step[layer_name].cpu().numpy()
                    special_act = special_step[layer_name].cpu().numpy()
                    
                    abs_diff = np.abs(cpu_act - special_act)
                    mean_diff = np.mean(abs_diff)
                    max_diff = np.max(abs_diff)
                    total_diff_sum += mean_diff
                    
                    row = [i, step_idx, f"{mean_diff:.8f}", f"{max_diff:.8f}"]
                    row.extend([f"{val:.8f}" for val in abs_diff.flatten()])
                    writer.writerow(row)
                    
    return total_diff_sum

def main():
    """Main execution function."""    
    for model_name in model_list:
        model_specific = model_name.split("/")[-1].lower()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use the custom model loading function
        model = model_load_function(model_name)
        if model is None:
            print(f"Skipping model {model_name} due to loading failure")
            continue
        print(f"\nModel: {model_specific}")
        for messages in inputs:
            tokenized_chat = tokenizer(messages,return_tensors="pt")
            print(f"Input: {messages}")
            
            device_result = False
            cpu_result = False

            tokenized_inputs = tokenized_chat
            all_cpu_activations = []
            all_special_activations = []

            transformer_layers = model.model.layers
            model_hidden_size = model.config.hidden_size

            eos_token_id = tokenizer.eos_token_id

            # Generate with Special Device
            if special_device:
                try:
                    print(f"{special_device.type:10s} Generating: ", end='', flush=True)
                    special_generated, all_special_activations = generate_with_activations(
                        model, tokenizer, tokenized_inputs, special_device, max_new_tokens, temperature=0.0
                    )
                    device_result = True
                    # Delete Model from device
                    model = model.to(cpu_device)
                except Exception as e:
                    print(f"Error during generation on {special_device.type}: {e}")
                    all_special_activations = []

            if device_result and special_device and all_special_activations:
                # print("\n--- Saving Special Device Activations to CSV ---")
                special_csv_file = f"{output_dir}/{model_specific}_{special_device.type}_{model_specific}_activations_per_step_{start_time}.csv"
                save_activations_to_csv(all_special_activations, special_csv_file, model_hidden_size, transformer_layers)
                # print(f"Special device activations saved to: {special_csv_file}")
            if cpu_enable:
                try:
                    print(f"{'cpu':10s} Generating: ", end='', flush=True)
                    cpu_generated, all_cpu_activations = generate_with_activations(
                        model, tokenizer, tokenized_inputs, cpu_device, max_new_tokens, temperature=0.0
                    )
                    cpu_result = True
                except Exception as e:
                    print(f"Error during generation on CPU: {e}")
                    all_cpu_activations = []

            cpu_csv_file = f"{output_dir}/{model_specific}_cpu_{model_specific}_activations_per_step_{start_time}.csv"
            if cpu_result and all_cpu_activations:
                save_activations_to_csv(all_cpu_activations, cpu_csv_file, model_hidden_size, transformer_layers)
                # print(f"CPU activations saved to: {cpu_csv_file}")

                # Compare activations if both are available
                if device_result and special_device and all_special_activations and len(all_cpu_activations) == len(all_special_activations):
                    # print("\n--- Comparing CPU vs Special Device Activations ---")
                    
                    # Save differences to CSV
                    diff_csv_file = f"{output_dir}/{model_specific}_cpu_vs_{special_device.type}_{model_specific}_activation_differences.csv"
                    total_diff_sum = save_activation_differences_to_csv(
                        all_cpu_activations, all_special_activations, diff_csv_file, model_hidden_size, transformer_layers
                    )



if __name__ == "__main__":
    main()

