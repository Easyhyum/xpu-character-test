#!/usr/bin/env python3
"""
Debug script to check attention mask and generation behavior
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer
model_name = "redhat-et/Meta-Llama-3.1-8B-fp8"
print(f"Loading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Test prompts with different lengths
prompts = [
    "Hello, how are you?",  # Short
    "Compose a speech about the need for more affordable dental care.",  # Medium
    "What is AI?",  # Very short
]

print("\n" + "="*80)
print("Testing left padding with batch processing")
print("="*80)

# Set left padding (for batch generation)
tokenizer.padding_side = "left"

# Tokenize with padding
inputs = tokenizer(
    prompts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
)

print(f"\nInput IDs shape: {inputs['input_ids'].shape}")
print(f"Attention mask shape: {inputs['attention_mask'].shape}")

# Print each sample's details
for idx in range(len(prompts)):
    mask = inputs['attention_mask'][idx]
    non_pad_indices = mask.nonzero(as_tuple=True)[0]
    
    if len(non_pad_indices) > 0:
        start_idx = non_pad_indices[0].item()
        end_idx = non_pad_indices[-1].item() + 1
        actual_tokens = inputs['input_ids'][idx][start_idx:end_idx]
        padding_count = start_idx
    else:
        actual_tokens = []
        padding_count = len(mask)
    
    print(f"\nSample {idx}: {prompts[idx]}")
    print(f"  Total length: {inputs['input_ids'].shape[1]}")
    print(f"  Padding tokens (left): {padding_count}")
    print(f"  Actual tokens: {len(actual_tokens)}")
    print(f"  Attention mask: {mask.tolist()}")
    print(f"  Input IDs: {inputs['input_ids'][idx].tolist()}")
    print(f"  Actual token IDs (no padding): {actual_tokens.tolist()}")
    print(f"  Decoded (full): {tokenizer.decode(inputs['input_ids'][idx], skip_special_tokens=False)}")
    print(f"  Decoded (actual): {tokenizer.decode(actual_tokens, skip_special_tokens=False)}")

print("\n" + "="*80)
print("Key Issue:")
print("="*80)
print("When using left padding in batch:")
print("1. Each sample has different amount of left padding (0s in attention_mask)")
print("2. Model's KV Cache will include positions for ALL tokens (including padding)")
print("3. KV Cache shape: [batch_size, num_heads, FULL_SEQ_LEN, head_dim]")
print("4. We need to extract only non-padding positions when saving individual files")
print("\nCorrect extraction:")
print("  - Use attention_mask to find actual token positions")
print("  - Extract KV Cache slice: [:, :, -actual_length:, :]  (last N positions)")
print("  - This removes left padding from saved KV Cache")
