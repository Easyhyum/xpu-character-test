import os
os.environ['PJRT_DEVICE'] = 'TPU'

print("="*80)
print("TPU Test Script Starting...")
print("="*80)
print(f"PJRT_DEVICE set to: {os.environ.get('PJRT_DEVICE')}")

print("Importing transformers...")
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, set_seed

print("Importing torch and torch_xla...")
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

print("Importing other libraries...")
import json
import csv
import time
import argparse
import random
import numpy as np

print("Importing API modules...")
from api import model_load_function_tpu, generate_with_activations_tpu, create_activation_hook, process_batch_inference_tpu

print("Importing datasets...")
from datasets import load_dataset, load_from_disk, disable_caching

print("✓ All imports successful!")
print("="*80)

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
editable_jsonl = "/workspace/HuggingFaceH4_ultrachat_200k_test_sft_64_processed.jsonl"
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
            example['prompt_text'] = user_messages[0].get('content', '')
        else:
            example['prompt_text'] = ''
        return example

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
        run_dir = f"/workspace/outputs/{start_time}"
        os.makedirs(run_dir, exist_ok=True)
        return os.path.join(run_dir, f"generated_examples_{start_time}.jsonl")

    def save_generated_example(jsonl_path, record):
        with open(jsonl_path, 'a', encoding='utf-8') as jf:
            jf.write(json.dumps(record, ensure_ascii=False) + '\n')
        
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
    xm.set_rng_state(seed)  # TPU random state
    
    set_seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print("✓ Deterministic mode enabled for TPU")
set_deterministic_mode(42)


cpu_device = torch.device("cpu")
# TPU device setup
tpu_device = xm.xla_device()
print(f"TPU Device: {tpu_device}")

output_dir = "/workspace/outputs"
os.makedirs(output_dir, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--start-time', dest='start_time', default=None, help='Start time string (YYYYmmdd-HHMMSS) passed from the wrapper')
args, unknown = parser.parse_known_args()
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

print(f"Run start_time: {start_time}")

def get_tpu_name():
    """Get TPU device name"""
    try:
        return f"TPU_{xm.xla_device()}"
    except:
        return "TPU_Unknown"

def save_input_output_csv(model_name, tpu_name, device, input_text, input_token_ids, output_text, output_token_ids, csv_file_path, batch_size=4, data_index=0):
    file_exists = os.path.isfile(csv_file_path)
    with open(csv_file_path, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["model_name", "tpu_name", "device", "batch_size", "data_index", "input_text", "input_token_ids", "output_text", "output_token_ids"])
        writer.writerow([model_name, tpu_name, device, batch_size, data_index, input_text, str(input_token_ids), output_text, str(output_token_ids)])

def main():
    tpu_name = get_tpu_name()
    print(f"TPU Name: {tpu_name}")
    
    # Create input-output CSV file path
    io_csv_file = f"{output_dir}/{start_time}/{tpu_name}_input_output_summary_{start_time}.csv"
    
    for model_name in model_list:
        print(f"\n{'='*80}")
        print(f"Processing model: {model_name}")
        print(f"{'='*80}")
        
        # Load model for TPU
        model = model_load_function_tpu(model_name)
        if model is None:
            print(f"Failed to load model {model_name}, skipping...")
            continue
        
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model_specific = model_name.split('/')[-1].replace('.', '_')
        
        # Process batches
        for batch_size in batch_size_list:
            print(f"\n{'='*80}")
            print(f"Processing batch size: {batch_size}")
            print(f"{'='*80}")
            
            num_batches = (len(ds_processed) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(ds_processed))
                batch_prompts = [ds_processed[i]['prompt_text'] for i in range(start_idx, end_idx)]
                
                print(f"\nProcessing batch {batch_idx + 1}/{num_batches} (samples {start_idx}-{end_idx-1})")
                
                # Process batch with TPU
                results = process_batch_inference_tpu(
                    model=model,
                    tokenizer=tokenizer,
                    batch_prompts=batch_prompts,
                    device=tpu_device,
                    max_new_tokens=max_new_tokens,
                    track_activations=activation_checkpointing,
                    csv_writer=None,
                    logit_csv_writer=None,
                    tpu_name=tpu_name,
                    model_specific=model_specific,
                    batch_start_idx=start_idx,
                    output_dir=output_dir,
                    start_time=start_time,
                    in_out_value_checkpointing=in_out_value_checkpointing
                )
                
                if results:
                    generated_texts, generated_token_ids, input_token_ids = results
                    
                    # Save results to CSV
                    for i, (input_text, gen_text, gen_tokens, inp_tokens) in enumerate(zip(batch_prompts, generated_texts, generated_token_ids, input_token_ids)):
                        save_input_output_csv(
                            model_name=model_name,
                            tpu_name=tpu_name,
                            device=str(tpu_device),
                            input_text=input_text,
                            input_token_ids=inp_tokens,
                            output_text=gen_text,
                            output_token_ids=gen_tokens,
                            csv_file_path=io_csv_file,
                            batch_size=batch_size,
                            data_index=start_idx + i
                        )
        
        print(f"\n✓ Completed processing for model: {model_name}")
        
        # Clean up model
        del model
        del tokenizer
        import gc
        gc.collect()

if __name__ == "__main__":
    main()
