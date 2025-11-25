"""
Example script demonstrating how to use the KV Cache functionality.

This script shows:
1. How to enable KV Cache saving during generation
2. How to load and reproduce generation step-by-step on different devices
3. How to verify consistency between original and reproduced generation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from api import KVCacheManager
import os


def example_1_save_kv_cache():
    """Example 1: Save KV Cache during generation (already integrated in test.py)"""
    
    print("="*80)
    print("Example 1: Enable KV Cache saving in hyperparameter.json")
    print("="*80)
    
    config = """
    {
      "kv_cache": {
        "save": true,              # Enable KV Cache saving
        "save_mode": "delta",       # Use delta compression (recommended)
        "base_dir": "outputs/{timestamp}/kv_caches"
      }
    }
    """
    
    print("\n1. Add this configuration to hyperparameter.json:")
    print(config)
    
    print("\n2. Run test.py normally:")
    print("   python test.py")
    
    print("\n3. KV Cache files will be saved at:")
    print("   outputs/{timestamp}/kv_caches/{gpu_name}/{model_name}/")
    print("   - batch16_input0.pt")
    print("   - batch16_input1.pt")
    print("   - ...")


def example_2_load_and_reproduce():
    """Example 2: Load KV Cache and reproduce generation step-by-step"""
    
    print("\n" + "="*80)
    print("Example 2: Load KV Cache and reproduce generation")
    print("="*80)
    
    # Configuration (adjust these paths)
    kv_cache_dir = "ref_cache"                    # Simplified: just reference cache folder
    gpu_name = "NVIDIA_H200"
    model_name = "redhatai_meta_llama_3_1_8b_fp8"  # Auto-detected from current model
    batch_size = 16
    input_index = 0
    
    print(f"\nConfiguration:")
    print(f"  KV Cache directory: {kv_cache_dir}")
    print(f"  GPU name: {gpu_name}")
    print(f"  Model: {model_name} (auto-detected)")
    print(f"  Batch size: {batch_size}")
    print(f"  Input index: {input_index}")
    print(f"\n  Full path: {kv_cache_dir}/{gpu_name}/{model_name}/")
    
    print(f"\n📝 Note: You don't need to specify model name in config!")
    print(f"   It's automatically detected from the model you're running.")
    
    # Initialize KV Cache Manager
    kv_manager = KVCacheManager(
        base_dir=kv_cache_dir,
        gpu_name=gpu_name,
        model_specific=model_name,  # This comes from current execution
        batch_size=batch_size,
        save_mode='delta'
    )
    
    # Check if cache exists
    if not kv_manager.cache_exists(input_index):
        print(f"\n❌ KV Cache not found for input {input_index}")
        print(f"   Expected path: {kv_manager.get_filepath(input_index)}")
        return
    
    print(f"\n✓ KV Cache found!")
    
    # Get cache info
    info = kv_manager.get_cache_info(input_index)
    print(f"\nCache Information:")
    print(f"  File path: {info['file_path']}")
    print(f"  File size: {info['file_size_mb']:.1f} MB")
    print(f"  Input text: {info['input_text']}")
    print(f"  Prefill length: {info['prefill_length']} tokens")
    print(f"  Decoding steps: {info['num_decoding_steps']}")
    print(f"  Generated text: {info['generated_text'][:100]}...")
    
    # Load model on different device (e.g., CPU for testing)
    print(f"\nLoading model on CPU for reproduction...")
    # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    # device = torch.device('cpu')
    # model = model.to(device)
    
    print("\n⚠️  Note: Actual model loading commented out to avoid memory issues in example")
    print("    Uncomment the lines above to run with real model")
    
    # Reproduce specific steps
    print(f"\nTo reproduce generation:")
    print(f"  # Load KV cache for step 0 (after prefill)")
    print(f"  kv_data = kv_manager.load_kv_cache_for_step(input_index={input_index}, target_step=0, device='cpu')")
    print(f"  ")
    print(f"  # Use the KV cache for next token generation")
    print(f"  past_kv = kv_data['past_key_values']")
    print(f"  seq_length = kv_data['seq_length']")
    print(f"  ")
    print(f"  # Continue generation with past_kv...")


def example_3_step_by_step_comparison():
    """Example 3: Compare generation step-by-step between devices"""
    
    print("\n" + "="*80)
    print("Example 3: Step-by-step comparison between devices")
    print("="*80)
    
    code = """
def compare_generation_across_devices(kv_manager, input_index, model_cpu, tokenizer, max_steps=10):
    '''
    Compare generation results step-by-step between original GPU and CPU.
    
    Args:
        kv_manager: KVCacheManager instance
        input_index: Input index to compare
        model_cpu: Model loaded on CPU
        tokenizer: Tokenizer
        max_steps: Number of steps to compare
    '''
    
    print(f"Comparing generation for input {input_index}...")
    
    device = torch.device('cpu')
    results = []
    
    for step in range(max_steps):
        # Load KV cache up to this step
        kv_data = kv_manager.load_kv_cache_for_step(input_index, step - 1, device)
        
        past_kv = kv_data['past_key_values']
        generated_so_far = kv_data['generated_tokens_so_far']
        
        # Prepare input
        if step == 0:
            input_ids = torch.tensor([kv_data['input_tokens']], device=device)
        else:
            last_token = generated_so_far[-1]
            input_ids = torch.tensor([[last_token]], device=device)
        
        # Generate next token
        with torch.no_grad():
            outputs = model_cpu(
                input_ids=input_ids,
                past_key_values=past_kv if step > 0 else None,
                use_cache=True,
                return_dict=True
            )
        
        next_token_logits = outputs.logits[:, -1, :]
        predicted_token_id = torch.argmax(next_token_logits, dim=-1).item()
        predicted_text = tokenizer.decode([predicted_token_id])
        
        # Load original token from cache
        cache_data = torch.load(kv_manager.get_filepath(input_index), map_location='cpu')
        original_token_id = cache_data['decoding_deltas'][step]['token_id']
        original_text = cache_data['decoding_deltas'][step]['token_text']
        
        # Compare
        match = (predicted_token_id == original_token_id)
        
        result = {
            'step': step,
            'predicted': f"{predicted_token_id} ({predicted_text})",
            'original': f"{original_token_id} ({original_text})",
            'match': '✓' if match else '✗'
        }
        
        results.append(result)
        
        print(f"  Step {step}: {result['match']} Pred: {result['predicted']}, Orig: {result['original']}")
    
    # Summary
    matches = sum(1 for r in results if r['match'] == '✓')
    print(f"\\nSummary: {matches}/{len(results)} steps matched ({100*matches/len(results):.1f}%)")
    
    return results
"""
    
    print("\nCode template:")
    print(code)


def example_4_file_structure():
    """Example 4: Understanding the file structure"""
    
    print("\n" + "="*80)
    print("Example 4: KV Cache file structure")
    print("="*80)
    
    structure = """
outputs/
└── 20251125-143000/
    └── kv_caches/
        ├── NVIDIA_H200/
        │   ├── meta_llama_llama_3_1_8b/
        │   │   ├── batch16_input0.pt     # Input 0, batch size 16
        │   │   ├── batch16_input1.pt     # Input 1, batch size 16
        │   │   ├── batch16_input2.pt
        │   │   └── ...
        │   └── meta_llama_llama_3_1_8b_instruct/
        │       └── ...
        └── CPU/
            └── ...

Each .pt file contains:
{
    'metadata': {
        'input_index': 0,
        'batch_size': 16,
        'model_name': 'meta_llama_llama_3_1_8b',
        'device': 'NVIDIA_H200',
        'num_layers': 32,
        'num_heads': 32,
        ...
    },
    'input_data': {
        'text': 'Hello, how are you?',
        'token_ids': [123, 456, 789, ...],
        'token_length': 10
    },
    'prefill_kv': {
        'past_key_values': tuple_of_tuples,  # Full KV cache after prefill
        'seq_length': 10,
        'attention_mask': tensor
    },
    'decoding_deltas': [
        {
            'step': 0,
            'token_id': 234,
            'token_text': 'I',
            'kv_delta': tuple_of_tuples,  # Only last token's KV (delta)
            'cumulative_seq_length': 11
        },
        {
            'step': 1,
            'token_id': 567,
            'token_text': ' am',
            'kv_delta': tuple_of_tuples,
            'cumulative_seq_length': 12
        },
        # ... more steps
    ],
    'generation_result': {
        'generated_text': 'I am doing great!',
        'generated_token_ids': [234, 567, 890, ...],
        'total_steps': 5
    }
}
"""
    
    print("\nFile structure:")
    print(structure)


def example_5_configuration_options():
    """Example 5: Configuration options"""
    
    print("\n" + "="*80)
    print("Example 5: Configuration options in hyperparameter.json")
    print("="*80)
    
    options = """
{
  "kv_cache": {
    "save": true,                    # Enable/disable KV Cache saving
    "save_mode": "delta",            # Options: "delta", "prefill_only", "final_only"
    "base_dir": "outputs/{timestamp}/kv_caches"  # Base directory for KV caches
  }
}

Save modes:
  - "delta" (recommended):
      * Saves prefill + incremental deltas for each decoding step
      * File size: ~114 MB per input (64 steps)
      * Allows step-by-step reproduction
      * Memory efficient
  
  - "prefill_only":
      * Saves only the prefill KV cache
      * File size: ~50 MB per input
      * Useful for prefill-only analysis
      * Cannot reproduce decoding
  
  - "final_only":
      * Saves only the final KV cache state
      * File size: ~72 MB per input (after 64 steps)
      * Useful for final state analysis
      * Cannot reproduce intermediate steps
"""
    
    print(options)


def main():
    """Run all examples"""
    
    print("\n" + "="*100)
    print(" "*30 + "KV Cache Usage Examples")
    print("="*100)
    
    example_1_save_kv_cache()
    example_2_load_and_reproduce()
    example_3_step_by_step_comparison()
    example_4_file_structure()
    example_5_configuration_options()
    
    print("\n" + "="*100)
    print("Examples completed!")
    print("="*100)
    
    print("\nNext steps:")
    print("  1. Enable KV Cache in hyperparameter.json")
    print("  2. Run test.py to generate and save KV caches")
    print("  3. Use the code templates above to load and reproduce generation")
    print("  4. Compare results across different devices")


if __name__ == "__main__":
    main()
