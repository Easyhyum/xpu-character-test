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
##
# output_dir = "/home/work/easyhyum/xpu-character-test/outputs" #original


# Utility: export and load editable JSONL versions of preprocessed dataset
def export_dataset_to_jsonl(ds, jsonl_path):
    """Export a HuggingFace Dataset or iterable of dicts to newline-delimited JSON (JSONL).

    Each line will be a JSON object representing one example. Existing file will be overwritten.
    """
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, 'w', encoding='utf-8') as jf:
        for ex in ds:
            jf.write(json.dumps(ex, ensure_ascii=False) + '\n')
    print(f"✓ Exported preprocessed dataset to JSONL: {jsonl_path}")

def load_preprocessed_from_jsonl(jsonl_path):
    """Load an editable JSONL (one JSON per line) back into a HuggingFace Dataset.

    Returns a Dataset object. This avoids using Arrow files so you can edit the JSONL and reload.
    """
    print(f"Loading preprocessed dataset from editable JSONL: {jsonl_path}")
    ds = load_dataset('json', data_files=jsonl_path, split='train')
    print(f"✓ Editable dataset loaded successfully: {len(ds)} examples")
    return ds

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
# Editable JSONL override: set environment variable EDITABLE_DATASET to path of an edited JSONL file
editable_jsonl = "/home/work/easyhyum/xpu-character-test/HuggingFaceH4_ultrachat_200k_test_sft_64_processed.jsonl"
# editable_jsonl = os.environ.get('EDITABLE_DATASET', None)
if editable_jsonl and os.path.exists(editable_jsonl):
    ds_processed = load_preprocessed_from_jsonl(editable_jsonl)
elif os.path.exists(local_dataset_path):
    print(f"  Loading preprocessed dataset from cache: {local_dataset_path}")
    ds_processed = load_from_disk(local_dataset_path)
    print(f"✓ Preprocessed dataset loaded successfully: {len(ds_processed)} examples")
    # Export editable JSONL for this cached dataset so you can edit and reload later
    try:
        jsonl_path = f"{local_dataset_path}.jsonl"
        export_dataset_to_jsonl(ds_processed, jsonl_path)
        print(f"  Editable JSONL exported to: {jsonl_path}")
    except Exception as e:
        print(f"  Warning: failed to export editable JSONL from cache: {e}")
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
        messages = example.get('messages', [])
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        if user_messages:
            return {'prompt_text': user_messages[0].get('content', '')}
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

    # Also export an editable JSONL so you can edit and reload subsets without Arrow files
    jsonl_path = f"{local_dataset_path}.jsonl"
    try:
        export_dataset_to_jsonl(ds_processed, jsonl_path)
        print(f"  You can edit and reload this file by setting EDITABLE_DATASET={jsonl_path} before running the script.")
    except Exception as e:
        print(f"  Warning: failed to export editable JSONL: {e}")
    # Helper: generated dataset JSONL path builder and saver
    # Use get_generated_jsonl_path(start_time) to get a per-run file path
    # and save_generated_example(jsonl_path, record) to append one record (newline-delimited JSON).
    def get_generated_jsonl_path(start_time):
        """Return a path for saving generated examples for this run and ensure directory exists."""
        out_dir = f"{output_dir}/{start_time}"
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"generated_dataset_{start_time}.jsonl")

    def save_generated_example(jsonl_path, record):
        """Append a single JSON record to jsonl_path (no loading of other CSVs required).

        record should be a JSON-serializable dict containing fields like:
            {"device": ..., "model": ..., "batch_size": ..., "data_index": ..., "input_text": ..., "output_text": ..., ...}
        """
        try:
            with open(jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Warning: failed to append generated record to {jsonl_path}: {e}")
# exit()
# hyperparameter.json에서 기타 설정 가져오기 (이미 로드됨)
inputs_from_config = config.get('inputs', [])
model_list = config['models']
max_new_tokens = config['max_new_tokens']
cpu_enable = config.get('cpu', 'Enable').lower() != 'disable'
batch_size_list = config.get('batch_size', [1, 4, 8, 16])
if not isinstance(batch_size_list, list):
    batch_size_list = [batch_size_list]  # Convert single value to list for backward compatibility
request_number = config.get('request_number', 'None')
print(f"\nModel configuration:")
print(f"  Models: {model_list}")
print(f"  Max new tokens: {max_new_tokens}")
print(f"  Batch sizes: {batch_size_list}")
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
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    
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


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start-time', dest='start_time', default=None, help='Start time string (YYYYmmdd-HHMMSS) passed from the wrapper')
parser.add_argument('--path', dest='path', default=None, help='PWD path passed from the wrapper')
args, unknown = parser.parse_known_args()
output_dir = args.path + "/outputs"
os.makedirs(output_dir, exist_ok=True)
token_checkpoint = config.get('token_checkpoint', True)
activation_checkpointing = config.get('activation_checkpointing', False)
in_out_value_checkpointing = config.get('in_out_value_check', False)
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

# KV Cache 설정 로드
kv_cache_config = config.get('kv_cache', {})
save_kv_cache = kv_cache_config.get('save', False)
load_kv_cache = kv_cache_config.get('load', False)
kv_cache_config['base_dir'] = kv_cache_config.get('base_dir', f"{output_dir}/{start_time}/kv_caches")

if save_kv_cache:
    print(f"\nKV Cache Save configuration:")
    print(f"  Save: {save_kv_cache}")
    print(f"  Mode: {kv_cache_config.get('save_mode', 'delta')}")
    print(f"  Base dir: {kv_cache_config['base_dir']}")
    os.makedirs(kv_cache_config['base_dir'], exist_ok=True)

if load_kv_cache:
    print(f"\nKV Cache Load configuration:")
    print(f"  Load: {load_kv_cache}")
    print(f"  Load base dir: {kv_cache_config.get('load_base_dir', 'N/A')}")
    print(f"  Load from device: {kv_cache_config.get('load_from_device', 'N/A')}")
    print(f"  Load from batch size: {kv_cache_config.get('load_from_batch_size', 'N/A')}")
    print(f"  Note: Model will be auto-detected from current execution")

print(f"Run start_time: {start_time}")

def get_gpu_name():
    if torch.cuda.is_available():
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except:
            return 'ERROR'
    elif torch.backends.mps.is_available():
        return 'MPS'
    else:
        return 'CPU_Only'

def save_input_output_csv(model_name, gpu_name, device, input_text, input_token_ids, output_text, output_token_ids, csv_file_path, batch_size=4, data_index=0):

    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Write header only if file doesn't exist
        if not file_exists:
            writer.writerow(["device", "model", "type", "batch_size", "data_index", "input_text", "input_tokens", "output_text", "output_tokens"])
        
        # input_text, input_token_ids, output_text, output_token_ids are already joined strings with |||
        # Just pass them directly without further processing
        
        # Write data
        writer.writerow([gpu_name, model_name, device.type, batch_size, data_index, input_text, input_token_ids, output_text, output_token_ids])

def main():
    gpu_name = get_gpu_name()
    print(f"GPU Name: {gpu_name}")
    
    # Create input-output CSV file path
    io_csv_file = f"{output_dir}/{start_time}/{gpu_name}_input_output_summary_{start_time}.csv"
    
    for model_name in model_list:
        # Use full model path to avoid conflicts (e.g., unsloth/Qwen3-8B-FP8 vs Qwen/Qwen3-8B-FP8)
        model_specific = model_name.replace("/", "_").replace("-", "_").lower()
        
        try:
            print(f"\nLoading tokenizer for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"\n⚠️  Failed to load tokenizer for {model_name}: {e}")
            print(f"{'='*60}")
            continue
        
        # Use the custom model loading function (retry up to 3 times)
        model = None
        for attempt in range(3):
            try:
                model = model_load_function(model_name)
                if model is not None:
                    break
            except Exception as e:
                print(f"\n⚠️  Model loading attempt {attempt+1} failed for {model_name}: {e}")
            # Memory cleanup between retries
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
            gc.collect()
            time.sleep(2)
        if model is None:
            print(f"\n⚠️  Skipping model {model_name} due to model loading failure after 3 attempts")
            print(f"{'='*60}")
            continue

        print(f"\nModel: {model_specific}")

        try:
            layers = model.model.layers
        except AttributeError:
            print("Warning: Could not find model.model.layers - skipping layer operation tracking")
            return
        
        #print all layers in model
        print("Model layers:", layers)
        # continue
        # 모델 정보 가져오기 (activation 추적용)
        transformer_layers = model.model.layers
        model_hidden_size = model.config.hidden_size
        num_layers = len(transformer_layers)
        
        print(f"  Model info - Hidden size: {model_hidden_size}, Layers: {num_layers}")
        
        # Step 3: 데이터셋을 batch로 처리
        print(f"\n[Step 3] Using dataset instead of fixed inputs...")
        print(f"  Dataset size: {len(ds_processed)}")
        
        # Iterate over different batch sizes
        for batch_size in batch_size_list:
            print(f"\n{'='*60}")
            print(f"Processing with batch_size = {batch_size}")
            print(f"{'='*60}")
            
            # Model별, Batch별 activation CSV 파일 생성
            activation_csv_file = f"{output_dir}/{start_time}/{gpu_name}_{model_specific}_batch{batch_size}_activations_{start_time}.csv"
            csv_writer = None
            activation_csv_handle = None
            logit_csv_file = f"{output_dir}/{start_time}/{gpu_name}_{model_specific}_batch{batch_size}_logit_{start_time}.csv"
            logit_csv_writer = None
            logit_csv_handle = None
            if args_bool and activation_checkpointing:  # activation 추적 활성화
                try:
                    activation_csv_handle = open(activation_csv_file, "w", newline='', encoding='utf-8')
                    csv_writer = csv.writer(activation_csv_handle)
                    header = ["device", "model", "type", "batch_size", "index", "input", "layer", "decoding_step", "token_id", "token_text"]
                    csv_writer.writerow(header)
                    print(f"  Activation tracking: ENABLED -> {activation_csv_file}")
                except Exception as e:
                    print(f"  Activation tracking: FAILED ({e}) -> Proceeding without activation tracking")
                try:
                    logit_csv_handle = open(logit_csv_file, "w", newline='', encoding='utf-8')
                    logit_csv_writer = csv.writer(logit_csv_handle)
                    header = ["device", "model", "type", "batch_size", "index", "input", "decoding_step", "token_id", "token_text"]
                    logit_csv_writer.writerow(header)
                    print(f"  Logit tracking: ENABLED -> {logit_csv_file}")
                except Exception as e:
                    print(f"  Logit tracking: FAILED ({e}) -> Proceeding without logit tracking")
            else:
                print(f"  Activation tracking: DISABLED")
            
            # Batch 크기 설정
            if request_number == 'None':
                data_len = len(ds_processed)
            else:
                data_len = request_number
            print(data_len, batch_size)
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
                    track_activations= activation_checkpointing or in_out_value_checkpointing or token_checkpoint,
                    csv_writer=csv_writer,
                    logit_csv_writer=logit_csv_writer,
                    gpu_name=gpu_name,
                    model_specific=model_specific,
                    batch_start_idx=batch_start,
                    output_dir=output_dir,
                    start_time=start_time,
                    in_out_value_checkpointing=in_out_value_checkpointing,
                    save_kv_cache=save_kv_cache,
                    kv_cache_config=kv_cache_config
                )
                
                # 결과 출력 및 저장
                for i, (generated, gen_token_ids, inp_token_ids) in enumerate(zip(generated_texts, generated_token_ids, input_token_ids)):
                    data_idx = batch_start + i
                    prompt = batch_prompts[i]
                    # Format prompt for display (replace <br> with actual newlines for readability)
                    display_prompt = prompt.replace('<br>', '\n')
                    print(f"\n  Example {data_idx}:")
                    print(f"    Prompt    {len(inp_token_ids):10d}: {display_prompt}")
                    print(f"    Generated {len(gen_token_ids):10d}: {generated}")

                # CSV에 저장 (배치 단위로 저장)
                if args_bool and token_checkpoint:
                    for i, (gen_token_ids, inp_token_ids) in enumerate(zip(generated_token_ids, input_token_ids)):
                        data_idx = batch_start + i
                        prompt = batch_prompts[i]
                        generated = generated_texts[i]
                        
                        # Decode each token individually and join with |||
                        input_text_tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in inp_token_ids]
                        output_text_tokens = [tokenizer.decode([token_id], skip_special_tokens=False) for token_id in gen_token_ids]
                        
                        # Replace commas and newlines to prevent CSV parsing issues
                        input_text_tokens = [token.replace(',', ' <comma> ').replace('\n', ' <br> ') for token in input_text_tokens]
                        output_text_tokens = [token.replace(',', ' <comma> ').replace('\n', ' <br> ') for token in output_text_tokens]
                        
                        combined_input_text = '|||'.join(input_text_tokens)
                        combined_output_text = '|||'.join(output_text_tokens)
                        
                        # Token IDs also joined with |||
                        combined_input_tokens = '|||'.join(map(str, inp_token_ids))
                        combined_output_tokens = '|||'.join(map(str, gen_token_ids))
                        
                        save_input_output_csv(
                            model_name, gpu_name, special_device,
                            combined_input_text, combined_input_tokens, 
                            combined_output_text, combined_output_tokens,
                            io_csv_file, batch_size=batch_size, data_index=data_idx
                        )
                
                # 메모리 정리
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
            
            # 각 batch size별 activation CSV 파일 닫기
            if csv_writer is not None and activation_csv_handle is not None:
                try:
                    activation_csv_handle.close()
                    print(f"  ✓ Activation data saved to: {activation_csv_file}")
                except Exception as e:
                    print(f"  Warning: Failed to close activation CSV file: {e}")
            
            print(f"\n✓ Batch size {batch_size} completed!")
        
        print("\n[Step 3] ✓ All batch sizes processed!")
        
        # model = model.cpu()
        
        del model, tokenizer
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
    
        gc.collect()


if __name__ == "__main__":
    main()

