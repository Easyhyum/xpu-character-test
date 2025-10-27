"""
CSV 파일에서 device, model, type, batch_size, input_tokens가 동일한 행들의 output_tokens 비교
"""
import pandas as pd
import os
from pathlib import Path

# 현재 파일 경로
current_dir = Path(__file__).parent

# CSV 파일 찾기
csv_files = list(current_dir.glob("*.csv"))

if not csv_files:
    print("No CSV files found in current directory")
    exit()

print(f"Found {len(csv_files)} CSV files:")
for i, csv_file in enumerate(csv_files):
    print(f"  {i}: {csv_file.name}")

# CSV 파일 읽기
all_data = []
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        df['source_file'] = csv_file.name
        all_data.append(df)
        print(f"\nLoaded: {csv_file.name}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")

if not all_data:
    print("\nNo data loaded")
    exit()

# 모든 데이터 합치기
df_all = pd.concat(all_data, ignore_index=True)
print(f"\n{'='*80}")
print(f"Total rows loaded: {len(df_all)}")
print(f"{'='*80}")

# device, model, type, batch_size, input_tokens로 그룹화
required_cols = ['device', 'model', 'type', 'batch_size', 'input_tokens']
if not all(col in df_all.columns for col in required_cols):
    print(f"Error: Required columns not found")
    print(f"Available columns: {df_all.columns.tolist()}")
    exit()

print(f"\nGrouping by: {required_cols}")
grouped = df_all.groupby(required_cols)

# 비교 결과 저장
inconsistent_groups = []
total_groups = 0
identical_groups = 0

for group_key, group_df in grouped:
    total_groups += 1
    
    # output_tokens가 2개 이상인 경우만 비교
    if len(group_df) < 2:
        continue
    
    # 모든 output_tokens가 동일한지 확인
    output_tokens_list = group_df['output_tokens'].tolist()
    first_output = output_tokens_list[0]
    
    # NaN 처리
    if pd.isna(first_output):
        all_same = all(pd.isna(x) for x in output_tokens_list)
    else:
        all_same = all(str(x) == str(first_output) for x in output_tokens_list)
    
    if all_same:
        identical_groups += 1
    else:
        # 불일치 발견
        inconsistent_groups.append({
            'group_key': group_key,
            'group_df': group_df,
            'output_tokens_list': output_tokens_list
        })

print(f"\n{'='*80}")
print(f"Comparison Results:")
print(f"  Total groups: {total_groups}")
print(f"  Identical groups: {identical_groups}")
print(f"  Inconsistent groups: {len(inconsistent_groups)}")
print(f"{'='*80}")

if inconsistent_groups:
    print(f"\n{'='*80}")
    print(f"INCONSISTENCIES FOUND!")
    print(f"{'='*80}")
    
    for i, item in enumerate(inconsistent_groups[:10]):  # 처음 10개만 표시
        print(f"\n[Group {i+1}]")
        device, model, type_val, batch_size, input_tokens = item['group_key']
        print(f"  Device: {device}")
        print(f"  Model: {model}")
        print(f"  Type: {type_val}")
        print(f"  Batch size: {batch_size}")
        print(f"  Input tokens (first 100 chars): {str(input_tokens)[:100]}...")
        print(f"  Number of rows: {len(item['group_df'])}")
        print(f"\n  Output tokens comparison:")
        
        for idx, (row_idx, row) in enumerate(item['group_df'].iterrows()):
            output_str = str(row['output_tokens'])
            print(f"    Row {idx+1} (from {row['source_file']}):")
            print(f"      First 150 chars: {output_str[:150]}...")
            print(f"      Length: {len(output_str)}")
        
        # 차이점 찾기
        outputs = item['output_tokens_list']
        if len(outputs) >= 2:
            out1 = str(outputs[0])
            out2 = str(outputs[1])
            
            # 첫 번째 차이점 찾기
            diff_pos = -1
            for pos in range(min(len(out1), len(out2))):
                if out1[pos] != out2[pos]:
                    diff_pos = pos
                    break
            
            if diff_pos >= 0:
                print(f"\n    First difference at position {diff_pos}:")
                start = max(0, diff_pos - 50)
                end = min(len(out1), len(out2), diff_pos + 50)
                print(f"      Output 1: ...{out1[start:end]}...")
                print(f"      Output 2: ...{out2[start:end]}...")
            elif len(out1) != len(out2):
                print(f"\n    Same content but different lengths:")
                print(f"      Output 1 length: {len(out1)}")
                print(f"      Output 2 length: {len(out2)}")
    
    if len(inconsistent_groups) > 10:
        print(f"\n... and {len(inconsistent_groups) - 10} more inconsistent groups")
    
    # 불일치 결과를 CSV로 저장
    inconsistent_rows = []
    for item in inconsistent_groups:
        for _, row in item['group_df'].iterrows():
            inconsistent_rows.append(row)
    
    if inconsistent_rows:
        inconsistent_df = pd.DataFrame(inconsistent_rows)
        output_file = current_dir / "inconsistent_outputs.csv"
        inconsistent_df.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print(f"Inconsistent rows saved to: {output_file}")
        print(f"{'='*80}")
else:
    print(f"\n✓ All output_tokens are consistent within same groups!")
    print(f"  (device, model, type, batch_size, input_tokens가 같은 경우 output_tokens도 모두 동일)")
