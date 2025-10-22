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
from datasets import load_dataset
import traceback

# Load dataset - using HuggingFaceH4/ultrachat_200k which is widely used and well-maintained

print("Loading dataset...")
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft[:1024]")
ds_processed = ds_processed.filter(lambda x: len(x['prompt_text']) <= 250)
print(f"✓ Dataset loaded successfully: {len(ds)} examples")
print(f"  Features: {list(ds.features.keys())}")

# Step 1: 데이터셋 전처리 함수
def format_messages_for_inference(example):
    """
    데이터셋의 messages를 모델 입력 형태로 변환
    example['messages']는 [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}] 형태
    """
    # messages에서 user와 assistant의 대화를 추출
    messages = example['messages']
    
    # user의 첫 번째 메시지만 사용 (prompt로)
    user_messages = [msg for msg in messages if msg['role'] == 'user']
    if user_messages:
        return {'prompt_text': user_messages[0]['content']}
    else:
        return {'prompt_text': example.get('prompt', '')}

print("\n[Step 1] Processing dataset for inference...")
# 데이터셋에 전처리 적용
ds_processed = ds.map(format_messages_for_inference)

# Step 1.5: 길이가 250 이하인 데이터만 필터링
print("\n[Step 1.5] Filtering dataset by length (≤ 250 characters)...")
original_size = len(ds_processed)
ds_processed = ds_processed.filter(lambda x: len(x['prompt_text']) <= 250)
print(f"✓ Filtered: {original_size} → {len(ds_processed)} examples (removed {original_size - len(ds_processed)})")

if len(ds_processed) > 0:
    print(f"  First example prompt preview:")
    print(f"  {ds_processed[0]['prompt_text'][:100]}...")
else:
    print("  Warning: No examples remain after filtering!")

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

parser = argparse.ArgumentParser()
parser.add_argument('--start-time', dest='start_time', default=None, help='Start time string (YYYYmmdd-HHMMSS) passed from the wrapper')
args, unknown = parser.parse_known_args()
print(args)

args_bool = True
start_time = None
if args.start_time:
    start_time = args.start_time
elif os.environ.get('START_TIME'):
    start_time = os.environ.get('START_TIME')
    args_bool = False
else:
    start_time = time.strftime("%Y%m%d-%H%M%S")
    args_bool = False

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

def process_batch_inference(model, tokenizer, batch_prompts, device, max_new_tokens=50):
    """
    Step 2: Batch로 여러 prompt를 한번에 처리하는 함수
    
    Args:
        model: 언어 모델
        tokenizer: 토크나이저
        batch_prompts: 문자열 리스트 (여러 개의 prompt)
        device: 실행할 디바이스
        max_new_tokens: 생성할 최대 토큰 수
    
    Returns:
        생성된 텍스트 리스트
    """
    print(f"\n[Step 2] Processing batch of {len(batch_prompts)} prompts...")
    
    # Tokenize all prompts at once (batch processing)
    inputs = tokenizer(
        batch_prompts, 
        return_tensors="pt", 
        padding=True,  # 길이를 맞추기 위해 padding
        truncation=True,  # 너무 긴 경우 자르기
        max_length=512  # 최대 입력 길이
    ).to(device)
    
    print(f"  Input shape: {inputs['input_ids'].shape}")
    
    # Generate outputs for the batch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding (deterministic)
            pad_token_id=tokenizer.eos_token_id
        )
    
    # outputs는 [입력 + 생성된 텍스트]를 포함하므로, 입력 길이만큼 잘라내기
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[:, input_length:]  # 입력 부분 제거, 생성된 부분만 추출
    
    print(f"  Input length: {input_length}, Output length: {outputs.shape[1]}, Generated length: {generated_tokens.shape[1]}")
    
    # Decode only the generated part
    generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    
    print(f"  ✓ Generated {len(generated_texts)} responses")
    
    return generated_texts

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
        
        # Step 3: 데이터셋을 batch로 처리
        print(f"\n[Step 3] Using dataset instead of fixed inputs...")
        print(f"  Dataset size: {len(ds_processed)}")
        
        # Batch 크기 설정
        batch_size = 4  # 한번에 4개씩 처리
        data_len = len(ds_processed)
        data_len = 4
        # 데이터셋을 batch로 나누어 처리
        for batch_start in range(0, data_len, batch_size):
            batch_end = min(batch_start + batch_size, data_len)
            
            print(f"\n--- Processing batch {batch_start//batch_size + 1}: examples {batch_start}-{batch_end-1} ---")
            
            # Batch에서 prompt 텍스트 추출
            batch_prompts = []
            for idx in range(batch_start, batch_end):
                prompt = ds_processed[idx]['prompt_text'][:500]  # 너무 길면 잘라내기
                batch_prompts.append(prompt)
            
            # Batch inference 실행
            generated_texts = process_batch_inference(
                model, tokenizer, batch_prompts, special_device, max_new_tokens=max_new_tokens
            )
            
            # 결과 출력 및 저장
            for i, generated in enumerate(generated_texts):
                example_idx = batch_start + i
                prompt = batch_prompts[i]
                print(f"\n  Example {example_idx}:")
                print(f"    Prompt    {len(prompt):10d}: {prompt}")
                print(f"    Generated {len(generated):10d}: {generated}")

                # CSV에 저장 (원하는 경우)
                if args_bool:
                    save_input_output_csv(
                        model_name, gpu_name, 
                        prompt, generated, 
                        io_csv_file
                    )
            
            # 메모리 정리
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        print("\n[Step 3] ✓ All batches processed!")
        
        # 기존 코드 (비활성화) - 필요하면 주석 해제
        """
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
            if special_device and False:  # Disabled - using batch inference instead
                # CSV 파일 먼저 열고 헤더 작성
                special_csv_file = f"{output_dir}/{start_time}/{gpu_name}_activations_per_step_{start_time}.csv"
                
                try:
                    if args_bool:
                        with open(special_csv_file, "w", newline='') as f:
                            writer = csv.writer(f)
                            # Write header
                            header = ["device","model", "type", "index", "input", "layer", "decoding_step", "token_id", "token_text"]
                            for col in range(model_hidden_size):
                                header.append(f"hidden_col_{col}")
                            writer.writerow(header)
                    else:
                        writer = None

                    print(f"{special_device.type:10s} Generating: ", end='', flush=True)
                    special_generated, step_count = generate_with_activations(
                        input_index, gpu_name, model_specific, model, messages, tokenizer, tokenized_inputs, special_device, max_new_tokens, 
                        temperature=0.0, csv_writer=writer, 
                        model_hidden_size=model_hidden_size, num_layers=len(transformer_layers)
                    )
                    
                    # Save input-output to CSV
                    if special_generated is not None and args_bool:
                        output_text = tokenizer.decode(special_generated[0], skip_special_tokens=True)
                        save_input_output_csv(model_name, gpu_name, messages, output_text, io_csv_file)
                    
                    # Delete Model from device and clear GPU memory
                    model = model.to(cpu_device)
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                except Exception as e:
                    print(f"Error during generation on {special_device.type}: {e}")
                    traceback.print_exc()
                    assert False, "Generation failed"

            if cpu_enable and False:  # Disabled - using batch inference instead
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
        """  # End of commented out old code
        
        model = model.cpu()
        
        del model, tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
    
        gc.collect()



if __name__ == "__main__":
    main()

