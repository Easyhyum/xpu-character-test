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
from api import model_load_function, generate_with_activations, create_activation_hook, process_batch_inference
from datasets import load_dataset, load_from_disk, disable_caching
import traceback

# datasets 캐시를 완전히 비활성화하여 cache*.arrow 파일 생성 방지
disable_caching()

# Load dataset configuration from hyperparameter.json
print("Loading configuration...")
with open("hyperparameter.json", "r") as f:
    config = json.load(f)

dataset_config = config.get('dataset', {})
dataset_name = dataset_config.get('name', 'HuggingFaceH4/ultrachat_200k')
dataset_split = dataset_config.get('split', 'test_sft')
dataset_num_samples = dataset_config.get('num_samples', 1024)

print(f"Dataset configuration:")
print(f"  Name: {dataset_name}")
print(f"  Split: {dataset_split}")
print(f"  Num samples: {dataset_num_samples}")

# 로컬 캐시 경로 생성 (전처리된 데이터셋 경로)
dataset_name_clean = dataset_name.replace('/', '_')
local_dataset_path = f"./datas/{dataset_name_clean}_{dataset_split}_{dataset_num_samples}_processed"

print("\nLoading dataset...")
# 전처리된 데이터셋이 있는지 확인
if os.path.exists(local_dataset_path):
    print(f"  Loading preprocessed dataset from cache: {local_dataset_path}")
    ds_processed = load_from_disk(local_dataset_path)
    print(f"✓ Preprocessed dataset loaded successfully: {len(ds_processed)} examples")
else:
    print(f"  Downloading from HuggingFace: {dataset_name}")
    # split과 num_samples를 결합하여 다운로드
    full_split = f"{dataset_split}[:{dataset_num_samples}]"
    ds = load_dataset(dataset_name, split=full_split)
    
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
    ds_processed = ds.map(format_messages_for_inference)

    # Step 1.5: 길이가 250 이하인 데이터만 필터링
    print("\n[Step 1.5] Filtering dataset by length (≤ 250 characters)...")
    original_size = len(ds_processed)
    ds_processed = ds_processed.filter(lambda x: len(x['prompt_text']) <= 250)
    print(f"✓ Filtered: {original_size} → {len(ds_processed)} examples (removed {original_size - len(ds_processed)})")
    
    # 전처리된 데이터셋 저장
    print(f"\n  Saving preprocessed dataset: {local_dataset_path}")
    os.makedirs(os.path.dirname(local_dataset_path), exist_ok=True)
    ds_processed.save_to_disk(local_dataset_path)
    print(f"✓ Preprocessed dataset saved")

# hyperparameter.json에서 기타 설정 가져오기 (이미 로드됨)
inputs_from_config = config.get('inputs', [])
model_list = config['models']
max_new_tokens = config['max_new_tokens']
cpu_enable = config.get('cpu', 'Enable').lower() != 'disable'
batch_size = config.get('batch_size', 4)
decoding_number = config.get('decoding_number', 'None')
print(f"\nModel configuration:")
print(f"  Models: {model_list}")
print(f"  Max new tokens: {max_new_tokens}")
print(f"  CPU enable: {cpu_enable}")

if inputs_from_config:
    print(f"\n[Step 1.6] Adding {len(inputs_from_config)} inputs from hyperparameter.json to the front...")
    from datasets import Dataset, concatenate_datasets
    
    # inputs를 데이터셋으로 변환
    inputs_dataset = Dataset.from_dict({"prompt_text": inputs_from_config})
    
    # 맨 앞에 추가 (inputs_dataset + ds_processed)
    ds_processed = concatenate_datasets([inputs_dataset, ds_processed])
    print(f"✓ Total dataset size: {len(ds_processed)} examples")

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

def save_input_output_csv(model_name, gpu_name, device, input_text, input_token_ids, output_text, output_token_ids, csv_file_path):

    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(["device", "model", "type", "batch_size", "input_text", "input_tokens", "output_text", "output_tokens"])
        
        # input_text, input_token_ids, output_text, output_token_ids are already joined strings with |||
        # Just pass them directly without further processing
        
        # Write data
        writer.writerow([gpu_name, model_name, device.type, batch_size, input_text, input_token_ids, output_text, output_token_ids])

def main():
    gpu_name = get_gpu_name()
    print(f"GPU Name: {gpu_name}")
    
    # Create input-output CSV file path
    io_csv_file = f"{output_dir}/{start_time}/{gpu_name}_input_output_summary_{start_time}.csv"
    
    # 실행별로 하나의 activation CSV 파일 생성
    activation_csv_file = f"{output_dir}/{start_time}/{gpu_name}_activations_per_step_{start_time}.csv"
    csv_writer = None
    activation_csv_handle = None
    
    if args_bool:  # activation 추적 활성화
        try:
            activation_csv_handle = open(activation_csv_file, "w", newline='', encoding='utf-8')
            csv_writer = csv.writer(activation_csv_handle)
            header = ["device", "model", "type", "batch_size", "index", "input", "layer", "decoding_step", "token_id", "token_text"]
            csv_writer.writerow(header)
        except Exception as e:
            print(f"\nActivation tracking: FAILED ({e}) -> Proceeding without activation tracking")
    else:
        print(f"\nActivation tracking: DISABLED (args_bool=False)")
    for model_name in model_list:
        model_specific = model_name.split("/")[-1].lower()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use the custom model loading function
        model = model_load_function(model_name)
        if model is None:
            print(f"Skipping model {model_name} due to loading failure")
            continue
        
        print(f"\nModel: {model_specific}")
        
        # 모델 정보 가져오기 (activation 추적용)
        transformer_layers = model.model.layers
        model_hidden_size = model.config.hidden_size
        num_layers = len(transformer_layers)
        
        print(f"  Model info - Hidden size: {model_hidden_size}, Layers: {num_layers}")
        
        # Step 3: 데이터셋을 batch로 처리
        print(f"\n[Step 3] Using dataset instead of fixed inputs...")
        print(f"  Dataset size: {len(ds_processed)}")
        
        # Batch 크기 설정
        if decoding_number == 'None':
            data_len = len(ds_processed)
        else:
            data_len = decoding_number
        # 데이터셋을 batch로 나누어 처리
        for batch_start in range(0, data_len, batch_size):
            batch_end = min(batch_start + batch_size, data_len)
            
            print(f"\n--- Processing batch {batch_start//batch_size + 1}: examples {batch_start}-{batch_end-1} ---")
            
            # Batch에서 prompt 텍스트 추출
            batch_prompts = []
            for idx in range(batch_start, batch_end):
                prompt = ds_processed[idx]['prompt_text'][:500]  # 너무 길면 잘라내기
                batch_prompts.append(prompt)
            
            # Batch inference 실행 (activation 추적 포함)
            generated_texts, generated_token_ids, input_token_ids = process_batch_inference(
                model, tokenizer, batch_prompts, special_device, 
                max_new_tokens=max_new_tokens,
                track_activations=(csv_writer is not None),
                csv_writer=csv_writer,
                gpu_name=gpu_name,
                model_specific=model_specific,
                batch_start_idx=batch_start,
            )
            
            # 결과 출력 및 저장
            for i, (generated, gen_token_ids, inp_token_ids) in enumerate(zip(generated_texts, generated_token_ids, input_token_ids)):
                example_idx = batch_start + i
                prompt = batch_prompts[i]
                print(f"\n  Example {example_idx}:")
                print(f"    Prompt    {len(prompt):10d}: {prompt}")
                print(f"    Generated {len(generated):10d}: {generated}")

            # CSV에 저장 (배치 단위로 저장)
            if args_bool:
                for i, (gen_token_ids, inp_token_ids) in enumerate(zip(generated_token_ids, input_token_ids)):
                    prompt = batch_prompts[i]
                    generated = generated_texts[i]
                    
                    # Decode each token individually and join with |||
                    input_text_tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in inp_token_ids]
                    output_text_tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in gen_token_ids]
                    
                    combined_input_text = '|||'.join(input_text_tokens)
                    combined_output_text = '|||'.join(output_text_tokens)
                    
                    # Token IDs also joined with |||
                    combined_input_tokens = '|||'.join(map(str, inp_token_ids))
                    combined_output_tokens = '|||'.join(map(str, gen_token_ids))
                    
                    save_input_output_csv(
                        model_name, gpu_name, special_device,
                        combined_input_text, combined_input_tokens, 
                        combined_output_text, combined_output_tokens,
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
    
    # 모든 모델 처리 완료 후 CSV 파일 닫기
    if csv_writer is not None:
        try:
            activation_csv_handle.close()
            print(f"\n✓ All activation data saved to: {activation_csv_file}")
        except Exception as e:
            print(f"Warning: Failed to close activation CSV file: {e}")


if __name__ == "__main__":
    main()

