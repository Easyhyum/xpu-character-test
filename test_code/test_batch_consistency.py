"""
Batch 크기에 따른 출력 일관성 테스트
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from api import model_load_function

# Deterministic 설정
set_seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# 모델과 토크나이저 로드
model_name = "RedHatAI/Meta-Llama-3.1-8B-FP8"
print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = model_load_function(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 테스트 입력
test_prompt = "Hello, how are you?"
max_new_tokens = 20

print(f"\n{'='*60}")
print(f"Test Prompt: '{test_prompt}'")
print(f"Max new tokens: {max_new_tokens}")
print(f"{'='*60}")

# Test 1: Batch size = 1
print(f"\n[Test 1] Batch size = 1")
inputs_1 = tokenizer([test_prompt], return_tensors="pt", padding=True).to(device)
print(f"  Input IDs: {inputs_1['input_ids']}")
print(f"  Attention mask: {inputs_1['attention_mask']}")

with torch.no_grad():
    outputs_1 = model.generate(
        input_ids=inputs_1['input_ids'],
        attention_mask=inputs_1['attention_mask'],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    )

generated_1 = tokenizer.decode(outputs_1[0, inputs_1['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"  Output: {generated_1}")

# Test 2: Batch size = 4 (같은 프롬프트 4개)
print(f"\n[Test 2] Batch size = 4 (same prompt)")
inputs_4 = tokenizer([test_prompt] * 4, return_tensors="pt", padding=True).to(device)
print(f"  Input IDs shape: {inputs_4['input_ids'].shape}")
print(f"  Attention mask shape: {inputs_4['attention_mask'].shape}")
print(f"  First input IDs: {inputs_4['input_ids'][0]}")
print(f"  First attention mask: {inputs_4['attention_mask'][0]}")

with torch.no_grad():
    outputs_4 = model.generate(
        input_ids=inputs_4['input_ids'],
        attention_mask=inputs_4['attention_mask'],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    )

generated_4_first = tokenizer.decode(outputs_4[0, inputs_4['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"  Output (first): {generated_4_first}")

# 결과 비교
print(f"\n{'='*60}")
print(f"Consistency Check:")
print(f"  Batch=1: {generated_1}")
print(f"  Batch=4: {generated_4_first}")
print(f"  Match: {generated_1 == generated_4_first}")
print(f"{'='*60}")
