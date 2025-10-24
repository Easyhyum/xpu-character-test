import pandas as pd
import csv
import os
from datetime import datetime
path = "./result"
create_path = "./result/result"
extend_before = 5  # Number of tokens before diff index
extend_after = 5   # Number of tokens after diff index
def get_output_folders():
    """Get all non-result folders from outputs directory."""
    output_dir = path
    if not os.path.exists(output_dir):
        print(f"Error: {output_dir} directory not found")
        return []
    
    folders = []
    for item in os.listdir(output_dir):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path) and not item.startswith("result"):
            folders.append(item)
    
    return sorted(folders)

def select_folders(folders):
    """Display folders and let user select multiple folders."""
    if not folders:
        print("No folders found.")
        return []
    
    print("\nAvailable folders:")
    for idx, folder in enumerate(folders):
        print(f"{idx}: {folder}")
    
    print("\nEnter folder indices separated by spaces (e.g., '0 1 2'):")
    user_input = input().strip()
    
    if not user_input:
        return []
    
    try:
        indices = [int(x) for x in user_input.split()]
        selected = [folders[i] for i in indices if 0 <= i < len(folders)]
        return selected
    except (ValueError, IndexError) as e:
        print(f"Invalid input: {e}")
        return []

def read_input_output_files(folder_paths):
    """Read all input_output_summary CSV files from selected folders."""
    all_data = []
    
    for folder in folder_paths:
        folder_path = os.path.join(path, folder)
        
        # Find all files containing "input_output_summary"
        for filename in os.listdir(folder_path):
            if "input_output_summary" in filename and filename.endswith(".csv"):
                file_path = os.path.join(folder_path, filename)
                try:
                    df = pd.read_csv(file_path)
                    df['source_folder'] = folder
                    df['source_file'] = filename
                    
                    # Split output_text and output_tokens by ||| and calculate token lengths
                    df['output_token_length'] = df['output_tokens'].apply(
                        lambda x: len(str(x).split('|||')) if pd.notna(x) else 0
                    )
                    
                    all_data.append(df)
                    print(f"Loaded: {file_path}")
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    if not all_data:
        return None
    
    return pd.concat(all_data, ignore_index=True)

def find_batch_differences(df):
    """Find where output strings differ for same device/model/input but different batch_size."""
    results = []
    
    # Group by device, model, and Input
    grouped = df.groupby(['device', 'model', 'Input'])
    
    for (device, model, input_text), group in grouped:
        # Check if there are multiple batch_sizes
        if len(group['batch_size'].unique()) < 2:
            continue
        
        # Sort by batch_size
        group = group.sort_values('batch_size')
        
        batch_sizes = group['batch_size'].tolist()
        output_list = group['Output'].tolist()
        
        # Compare outputs
        for i in range(len(output_list)):
            for j in range(i + 1, len(output_list)):
                output_i = str(output_list[i]) if pd.notna(output_list[i]) else ""
                output_j = str(output_list[j]) if pd.notna(output_list[j]) else ""
                
                # Find first difference index
                diff_index = -1
                min_len = min(len(output_i), len(output_j))
                
                for idx in range(min_len):
                    if output_i[idx] != output_j[idx]:
                        diff_index = idx
                        break
                
                # If no difference found but lengths differ
                if diff_index == -1 and len(output_i) != len(output_j):
                    diff_index = min_len
                
                # Get substring up to diff_index + extend_string_num
                end_index = diff_index + extend_string_num if diff_index >= 0 else 0
                output_substr_1 = output_i[:end_index] if diff_index >= 0 else ""
                output_substr_2 = output_j[:end_index] if diff_index >= 0 else ""
                
                results.append({
                    'device': device,
                    'model': model,
                    'Input': input_text,
                    'batch_size_1': batch_sizes[i],
                    'batch_size_2': batch_sizes[j],
                    'first_difference_index': diff_index,
                    'output_length_1': len(output_i),
                    'output_length_2': len(output_j),
                    'output_substr_batch_1': output_substr_1,
                    'output_substr_batch_2': output_substr_2
                })
    
    return pd.DataFrame(results)

def find_device_differences(df):
    """Find where output tokens differ for same model/input_text/batch_size but different device."""
    results = []
    
    # Group by model, input_text (not Input column), and batch_size
    grouped = df.groupby(['model', 'input_text', 'batch_size'])
    
    for (model, input_text, batch_size), group in grouped:
        # Check if there are multiple devices
        if len(group['device'].unique()) < 2:
            continue
        
        # Sort by device
        group = group.sort_values('device')
        
        devices = group['device'].tolist()
        output_tokens_list = group['output_tokens'].tolist()
        output_text_list = group['output_text'].tolist()
        output_token_length_list = group['output_token_length'].tolist()
        
        # Compare outputs
        for i in range(len(output_tokens_list)):
            for j in range(i + 1, len(output_tokens_list)):
                # Split tokens by |||
                tokens_i = str(output_tokens_list[i]).split('|||') if pd.notna(output_tokens_list[i]) else []
                tokens_j = str(output_tokens_list[j]).split('|||') if pd.notna(output_tokens_list[j]) else []
                
                text_i = str(output_text_list[i]).split('|||') if pd.notna(output_text_list[i]) else []
                text_j = str(output_text_list[j]).split('|||') if pd.notna(output_text_list[j]) else []
                
                # Find first difference index in tokens
                diff_index = -1
                min_len = min(len(tokens_i), len(tokens_j))
                
                # Check if outputs are empty (only padding tokens)
                # Empty output means all tokens are padding (128004) or end tokens (128001)
                # Filter out special tokens and check if anything meaningful remains
                special_tokens = ['128001', '128004', '128000']  # EOS, PAD, BOS tokens
                meaningful_tokens_i = [t for t in tokens_i if t.strip() and t.strip() not in special_tokens]
                meaningful_tokens_j = [t for t in tokens_j if t.strip() and t.strip() not in special_tokens]
                
                # Check empty status first, before checking for differences
                if len(meaningful_tokens_i) == 0 and len(meaningful_tokens_j) == 0:
                    # Both outputs have no meaningful tokens - mark as -2 (both empty)
                    diff_index = -2
                elif len(meaningful_tokens_i) == 0:
                    # Only device_1 output is empty - mark as -3
                    diff_index = -3
                elif len(meaningful_tokens_j) == 0:
                    # Only device_2 output is empty - mark as -4
                    diff_index = -4
                else:
                    # Both have meaningful tokens, check for differences
                    for idx in range(min_len):
                        if tokens_i[idx] != tokens_j[idx]:
                            diff_index = idx
                            break
                    
                    # If no difference found but lengths differ
                    if diff_index == -1 and len(tokens_i) != len(tokens_j):
                        diff_index = min_len
                
                # Get substring from (diff_index - 5) to (diff_index + 5)
                # Remove ||| separators when joining
                # For -1, -2, -3, -4 cases, show full output text
                if diff_index == -1 or diff_index == -2 or diff_index == -3 or diff_index == -4:
                    # Show full output for identical or empty outputs
                    output_substr_1 = ''.join(text_i)
                    output_substr_2 = ''.join(text_j)
                elif diff_index >= 0:
                    start_index = max(0, diff_index - extend_before)
                    end_index = min(len(text_i), diff_index + extend_after + 1)
                    output_substr_1 = ''.join(text_i[start_index:end_index])
                    
                    end_index_j = min(len(text_j), diff_index + extend_after + 1)
                    output_substr_2 = ''.join(text_j[start_index:end_index_j])
                else:
                    output_substr_1 = ""
                    output_substr_2 = ""
                
                results.append({
                    'model': model,
                    'input_text': input_text.replace('|||', '').replace('\n', '<br>'),
                    'batch_size': batch_size,
                    'device_1': devices[i],
                    'device_2': devices[j],
                    'first_difference_index': diff_index,
                    'output_token_length_1': output_token_length_list[i],
                    'output_token_length_2': output_token_length_list[j],
                    'output_substr_device_1': output_substr_1.replace('\n', '<br>'),
                    'output_substr_device_2': output_substr_2.replace('\n', '<br>')
                })
    
    return pd.DataFrame(results)

def select_analysis_type():
    """Let user select the type of analysis to perform."""
    print("\nSelect analysis type:")
    print("0: Both (run batch and device comparison sequentially)")
    print("1: Batch size comparison (same device/model/input, different batch_size)")
    print("2: Device comparison (same model/input/batch_size, different device)")
    
    user_input = input("Enter index (0, 1, or 2): ").strip()
    
    if user_input == "0":
        return ["batch", "device"]
    elif user_input == "1":
        return ["batch"]
    elif user_input == "2":
        return ["device"]
    else:
        print("Invalid selection. Defaulting to device comparison only.")
        return ["device"]

def create_diff_index_summary(diff_df, result_dir, timestamp, analysis_type):
    """Create a summary of first_difference_index counts."""
    if diff_df.empty:
        return
    
    # Count occurrences of each first_difference_index value
    summary = diff_df['first_difference_index'].value_counts().sort_index()
    
    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'first_difference_index': summary.index,
        'count': summary.values
    })
    
    # Save overall summary to CSV
    summary_file = os.path.join(result_dir, f"diff_index_summary_{analysis_type}_{timestamp}.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Difference index summary saved to: {summary_file}")
    
    # Create summary by model
    model_summary_list = []
    for model in diff_df['model'].unique():
        model_df = diff_df[diff_df['model'] == model]
        model_counts = model_df['first_difference_index'].value_counts().sort_index()
        
        for idx, count in model_counts.items():
            model_summary_list.append({
                'model': model,
                'first_difference_index': idx,
                'count': count
            })
    
    if model_summary_list:
        model_summary_df = pd.DataFrame(model_summary_list)
        model_summary_file = os.path.join(result_dir, f"diff_index_summary_by_model_{analysis_type}_{timestamp}.csv")
        model_summary_df.to_csv(model_summary_file, index=False)
        print(f"Difference index summary by model saved to: {model_summary_file}")
    
    # Create summary by batch_size (if batch_size column exists)
    if 'batch_size' in diff_df.columns:
        batch_summary_list = []
        for batch_size in sorted(diff_df['batch_size'].unique()):
            batch_df = diff_df[diff_df['batch_size'] == batch_size]
            batch_counts = batch_df['first_difference_index'].value_counts().sort_index()
            
            for idx, count in batch_counts.items():
                batch_summary_list.append({
                    'batch_size': batch_size,
                    'first_difference_index': idx,
                    'count': count
                })
        
        if batch_summary_list:
            batch_summary_df = pd.DataFrame(batch_summary_list)
            batch_summary_file = os.path.join(result_dir, f"diff_index_summary_by_batch_{analysis_type}_{timestamp}.csv")
            batch_summary_df.to_csv(batch_summary_file, index=False)
            print(f"Difference index summary by batch size saved to: {batch_summary_file}")
    
    # Create summary by model and batch_size combination
    if 'batch_size' in diff_df.columns:
        model_batch_summary_list = []
        for model in diff_df['model'].unique():
            for batch_size in sorted(diff_df['batch_size'].unique()):
                model_batch_df = diff_df[(diff_df['model'] == model) & (diff_df['batch_size'] == batch_size)]
                if not model_batch_df.empty:
                    model_batch_counts = model_batch_df['first_difference_index'].value_counts().sort_index()
                    
                    for idx, count in model_batch_counts.items():
                        model_batch_summary_list.append({
                            'model': model,
                            'batch_size': batch_size,
                            'first_difference_index': idx,
                            'count': count
                        })
        
        if model_batch_summary_list:
            model_batch_summary_df = pd.DataFrame(model_batch_summary_list)
            model_batch_summary_file = os.path.join(result_dir, f"diff_index_summary_by_model_batch_{analysis_type}_{timestamp}.csv")
            model_batch_summary_df.to_csv(model_batch_summary_file, index=False)
            print(f"Difference index summary by model and batch size saved to: {model_batch_summary_file}")
    
    return summary_df

def main():
    """Main execution function."""
    print("=== Token Difference Analyzer ===\n")
    
    # Get available folders
    folders = get_output_folders()
    if not folders:
        return
    
    # Let user select folders
    selected_folders = select_folders(folders)
    if not selected_folders:
        print("No folders selected.")
        return
    
    print(f"\nSelected folders: {selected_folders}")
    
    # Let user select analysis type
    analysis_types = select_analysis_type()
    print(f"\nAnalysis type(s): {', '.join(analysis_types)} comparison")
    
    # Read all input_output_summary files
    print("\nReading CSV files...")
    df = read_input_output_files(selected_folders)
    
    if df is None or df.empty:
        print("No data found.")
        return
    
    print(f"\nTotal rows loaded: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Process each analysis type
    for analysis_type in analysis_types:
        print(f"\n{'='*60}")
        print(f"Running {analysis_type.upper()} comparison analysis")
        print('='*60)
        
        # Find differences based on analysis type
        print(f"\nAnalyzing token differences ({analysis_type} comparison)...")
        if analysis_type == "batch":
            diff_df = find_batch_differences(df)
        else:
            diff_df = find_device_differences(df)
        
        if diff_df.empty:
            print(f"No differences found between different {analysis_type}s.")
            continue
        
        print(f"\nFound {len(diff_df)} comparisons with differences.")
        
        # Create result folder name with selected folders and analysis type
        folder_suffix = "_".join(selected_folders)
        result_dir_name = f"result_{analysis_type}_{folder_suffix}"
        result_dir = os.path.join(path, result_dir_name)
        os.makedirs(result_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_file = os.path.join(result_dir, f"token_differences_{analysis_type}_{timestamp}.csv")
        
        diff_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Create and save diff index summary
        create_diff_index_summary(diff_df, result_dir, timestamp, analysis_type)
    
    print(f"\n{'='*60}")
    print("All analyses completed!")
    print('='*60)

if __name__ == "__main__":
    main()
