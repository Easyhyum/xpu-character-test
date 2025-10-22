from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, set_seed
import torch
import numpy as np
import csv
import random
import os, time
import contextlib
import json
import gc
import subprocess
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

print(f"Available special device: {special_device}")

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

def get_gpu_name():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return 'CPU'

def save_input_output_csv(model_name, gpu_name, input_text, output_text, csv_file_path):

    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(["Model", "GPU", "Input", "Output"])
        
        # Write data
        writer.writerow([model_name, gpu_name, input_text, output_text])

def main():
    gpu_name = get_gpu_name()
    print(f"GPU Name: {gpu_name}")
    
    # Create input-output CSV file path
    io_csv_file = f"{output_dir}/{start_time}/{gpu_name}_input_output_summary_{start_time}.csv"
    for model_name in model_list:
        model_specific = model_name.split("/")[-1].lower()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use the custom model loading function
        model = model_load_function(model_name)
        if model is None:
            print(f"Skipping model {model_name} due to loading failure")
            continue
        
        print(f"\nModel: {model_specific}")
        input_index = 0
        for messages in inputs:
            # 입력 처리 전 메모리 상태
            
            tokenized_chat = tokenizer(messages,return_tensors="pt")
            print(f"Input: {messages}")
            

            tokenized_inputs = tokenized_chat

            transformer_layers = model.model.layers
            model_hidden_size = model.config.hidden_size

            eos_token_id = tokenizer.eos_token_id

            # Generate with Special Device
            special_generated = None
            if special_device:
                # CSV 파일 먼저 열고 헤더 작성
                special_csv_file = f"{output_dir}/{start_time}/{gpu_name}_{input_index}_{model_specific}_{special_device.type}_{model_specific}_activations_per_step_{start_time}.csv"
                
                try:
                    with open(special_csv_file, "w", newline='') as f:
                        writer = csv.writer(f)
                        
                        # Write header
                        header = ["device","model", "type", "index", "input", "layer", "decoding_step", "token_id", "token_text"]
                        for col in range(model_hidden_size):
                            header.append(f"hidden_col_{col}")
                        writer.writerow(header)
                        
                        print(f"{special_device.type:10s} Generating: ", end='', flush=True)
                        special_generated, step_count = generate_with_activations(
                            input_index, gpu_name, model_specific, model, messages, tokenizer, tokenized_inputs, special_device, max_new_tokens, 
                            temperature=0.0, csv_writer=writer, 
                            model_hidden_size=model_hidden_size, num_layers=len(transformer_layers)
                        )
                    
                    # Save input-output to CSV
                    if special_generated is not None:
                        output_text = tokenizer.decode(special_generated[0], skip_special_tokens=True)
                        save_input_output_csv(model_name, gpu_name, messages, output_text, io_csv_file)
                    
                    # Delete Model from device and clear GPU memory
                    model = model.to(cpu_device)
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                except Exception as e:
                    print(f"Error during generation on {special_device.type}: {e}")

            if cpu_enable:
                # CPU용 CSV 파일 먼저 열고 헤더 작성
                cpu_csv_file = f"{output_dir}/{start_time}/{gpu_name}_{input_index}_{model_specific}_cpu_{model_specific}_activations_per_step_{start_time}.csv"
                
                try:
                    with open(cpu_csv_file, "w", newline='') as f:
                        writer = csv.writer(f)
                        
                        # Write header
                        header = ["layer", "decoding_step", "token_id", "token_text"]
                        for col in range(model_hidden_size):
                            header.append(f"hidden_col_{col}")
                        writer.writerow(header)
                        
                        print(f"{'cpu':10s} Generating: ", end='', flush=True)
                        cpu_generated, step_count = generate_with_activations(
                            model, tokenizer, tokenized_inputs, cpu_device, max_new_tokens, 
                            temperature=0.0, csv_writer=writer,
                            model_hidden_size=model_hidden_size, num_layers=len(transformer_layers)
                        )
                    cpu_result = True
                    # CPU 생성 후에도 메모리 정리
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                except Exception as e:
                    print(f"Error during generation on CPU: {e}")
                    cpu_result = False

            # 각 입력 처리 후 메모리 정리
            if 'cpu_generated' in locals():
                del cpu_generated
            if 'special_generated' in locals():
                del special_generated
            if 'tokenized_inputs' in locals():
                del tokenized_inputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
            input_index += 1

        model = model.cpu()
        
        del model, tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
    
        gc.collect()



if __name__ == "__main__":
    main()

