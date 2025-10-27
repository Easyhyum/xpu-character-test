import pandas as pd
import csv
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
    from itertools import combinations
    
    results = []
    
    # Group by model, input_text, and batch_size
    grouped = df.groupby(['model', 'input_text', 'batch_size'])
    
    for (model, input_text, batch_size), group in grouped:
        # Get all unique devices in this group
        unique_devices = sorted(group['device'].unique())
        
        # Generate all possible pairs of devices (nC2)
        if len(unique_devices) < 2:
            continue
        
        device_pairs = list(combinations(unique_devices, 2))
        
        # Compare each pair of devices
        for device_i, device_j in device_pairs:
            # Get data for each device
            device_i_data = group[group['device'] == device_i]
            device_j_data = group[group['device'] == device_j]
            
            if device_i_data.empty or device_j_data.empty:
                continue
            
            # Get the first row from each device group
            output_tokens_i = device_i_data.iloc[0]['output_tokens']
            output_tokens_j = device_j_data.iloc[0]['output_tokens']
            output_text_i = device_i_data.iloc[0]['output_text']
            output_text_j = device_j_data.iloc[0]['output_text']
            output_token_length_i = device_i_data.iloc[0]['output_token_length']
            output_token_length_j = device_j_data.iloc[0]['output_token_length']
            
            # Split tokens by |||
            tokens_i = str(output_tokens_i).split('|||') if pd.notna(output_tokens_i) else []
            tokens_j = str(output_tokens_j).split('|||') if pd.notna(output_tokens_j) else []
            
            text_i = str(output_text_i).split('|||') if pd.notna(output_text_i) else []
            text_j = str(output_text_j).split('|||') if pd.notna(output_text_j) else []
            
            # Find first difference index in tokens
            diff_index = -1
            min_len = min(len(tokens_i), len(tokens_j))
            
            # Check if outputs are empty (only padding tokens)
            special_tokens = ['128001', '128004', '128000']  # EOS, PAD, BOS tokens
            meaningful_tokens_i = [t for t in tokens_i if t.strip() and t.strip() not in special_tokens]
            meaningful_tokens_j = [t for t in tokens_j if t.strip() and t.strip() not in special_tokens]
            
            # Check empty status first, before checking for differences
            if len(meaningful_tokens_i) == 0 and len(meaningful_tokens_j) == 0:
                diff_index = -2  # Both empty
            elif len(meaningful_tokens_i) == 0:
                diff_index = -3  # Only device_1 empty
            elif len(meaningful_tokens_j) == 0:
                diff_index = -4  # Only device_2 empty
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
            if diff_index == -1 or diff_index == -2 or diff_index == -3 or diff_index == -4:
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
                'device_1': device_i,
                'device_2': device_j,
                'device_pair': f"{device_i}_vs_{device_j}",
                'first_difference_index': diff_index,
                'output_token_length_1': output_token_length_i,
                'output_token_length_2': output_token_length_j,
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
    """Create a summary of first_difference_index counts, separated by device_pair if available."""
    if diff_df.empty:
        return {}
    
    summary_files = {}
    
    # Check if device_pair column exists (for device comparison)
    if 'device_pair' in diff_df.columns:
        device_pairs = diff_df['device_pair'].unique()
        
        for device_pair in device_pairs:
            pair_df = diff_df[diff_df['device_pair'] == device_pair]
            
            # Create summary by model and batch_size for this device pair
            if 'batch_size' in pair_df.columns:
                model_batch_summary_list = []
                for model in pair_df['model'].unique():
                    for batch_size in sorted(pair_df['batch_size'].unique()):
                        model_batch_df = pair_df[(pair_df['model'] == model) & (pair_df['batch_size'] == batch_size)]
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
                    summary_file = os.path.join(result_dir, f"diff_index_summary_{device_pair}_{timestamp}.csv")
                    model_batch_summary_df.to_csv(summary_file, index=False)
                    summary_files[device_pair] = summary_file
                    print(f"Summary for {device_pair} saved to: {summary_file}")
    
    else:
        # Original logic for batch comparison (no device_pair)
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
                summary_file = os.path.join(result_dir, f"diff_index_summary_by_model_batch_{analysis_type}_{timestamp}.csv")
                model_batch_summary_df.to_csv(summary_file, index=False)
                summary_files['all'] = summary_file
                print(f"Difference index summary by model and batch size saved to: {summary_file}")
    
    return summary_files


def plot_difference_analysis(summary_file, result_dir, analysis_type, timestamp, device_pair=None):
    """
    Create plots from diff_index_summary CSV file.
    Creates 4 plots:
    1. Line plot for each model individually
    2. Combined line plot with all models
    3. Stacked bar chart for each model individually
    4. Combined stacked bar chart
    
    If device_pair is provided, plots will include device_pair in title and filename.
    """
    if not os.path.exists(summary_file):
        print(f"Summary file not found: {summary_file}")
        return
    
    df = pd.read_csv(summary_file)
    
    # Check if required columns exist
    required_cols = ['model', 'batch_size', 'first_difference_index', 'count']
    if not all(col in df.columns for col in required_cols):
        print(f"Required columns not found in {summary_file}")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Filter out first_difference_index <= -2
    df_filtered = df[df['first_difference_index'] > -2].copy()
    
    if df_filtered.empty:
        print(f"No data remaining after filtering (all first_difference_index <= -2)")
        return
    
    # Get unique models and batch sizes
    models = df_filtered['model'].unique()
    batch_sizes = sorted(df_filtered['batch_size'].unique())
    
    # Create color map for batch sizes
    colors = plt.cm.Set1(range(len(batch_sizes)))
    batch_color_map = {batch: colors[i] for i, batch in enumerate(batch_sizes)}
    
    # Create plots directory
    plots_dir = os.path.join(result_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Setup title suffix based on device_pair
    title_suffix = f" - {device_pair.replace('_', ' ')}" if device_pair else ""
    file_suffix = f"_{device_pair}" if device_pair else ""
    
    # Plot 1: Line plot for each model individually
    for model in models:
        model_clean = model.replace('/', '_').replace('\\', '_')
        model_df = df_filtered[df_filtered['model'] == model]
        batch_totals = model_df.groupby('batch_size')['count'].sum()
        total_comparisons = model_df['count'].sum()
        
        plt.figure(figsize=(12, 6))
        
        for batch_size in batch_sizes:
            batch_df = model_df[model_df['batch_size'] == batch_size]
            
            if not batch_df.empty:
                batch_df_sorted = batch_df.sort_values('first_difference_index').copy()
                total_count = batch_totals[batch_size]
                batch_df_sorted['percentage'] = (batch_df_sorted['count'] / total_count) * 100
                
                plt.plot(
                    batch_df_sorted['first_difference_index'],
                    batch_df_sorted['percentage'],
                    marker='o',
                    label=f'Batch {batch_size}',
                    color=batch_color_map[batch_size],
                    linewidth=2,
                    markersize=6
                )
        
        plt.xlabel('First Difference Index', fontsize=12)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.title(f'Token Difference Analysis - {model}{title_suffix}\n({analysis_type.upper()} comparison, Total: {total_comparisons} comparisons)', fontsize=14)
        plt.legend(title='Batch Size', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add vertical line at x=-1 (identical)
        plt.axvline(x=-1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Identical')
        
        # Ensure -1 is shown on x-axis
        ax = plt.gca()
        x_ticks = ax.get_xticks()
        if -1 not in x_ticks:
            x_ticks = sorted(list(x_ticks) + [-1])
            ax.set_xticks(x_ticks)
        
        plt.tight_layout()
        
        plot_file = os.path.join(plots_dir, f"diff_analysis_{model_clean}{file_suffix}_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved (individual model): {plot_file}")
    
    # Plot 2: Combined line plot with all models
    num_models = len(models)
    fig, axes = plt.subplots(num_models, 1, figsize=(14, 6 * num_models))
    if num_models == 1:
        axes = [axes]
    
    for idx, model in enumerate(models):
        model_df = df_filtered[df_filtered['model'] == model]
        batch_totals = model_df.groupby('batch_size')['count'].sum()
        total_comparisons = model_df['count'].sum()
        
        for batch_size in batch_sizes:
            batch_df = model_df[model_df['batch_size'] == batch_size]
            
            if not batch_df.empty:
                batch_df_sorted = batch_df.sort_values('first_difference_index').copy()
                total_count = batch_totals[batch_size]
                batch_df_sorted['percentage'] = (batch_df_sorted['count'] / total_count) * 100
                
                axes[idx].plot(
                    batch_df_sorted['first_difference_index'],
                    batch_df_sorted['percentage'],
                    marker='o',
                    label=f'Batch {batch_size}',
                    color=batch_color_map[batch_size],
                    linewidth=2,
                    markersize=6
                )
        
        axes[idx].set_xlabel('First Difference Index', fontsize=11)
        axes[idx].set_ylabel('Percentage (%)', fontsize=11)
        axes[idx].set_title(f'{model} (Total: {total_comparisons} comparisons)', fontsize=12)
        axes[idx].legend(title='Batch Size', fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        
        # Add vertical line at x=-1 (identical)
        axes[idx].axvline(x=-1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Ensure -1 is shown on x-axis
        x_ticks = axes[idx].get_xticks()
        if -1 not in x_ticks:
            x_ticks = sorted(list(x_ticks) + [-1])
            axes[idx].set_xticks(x_ticks)
    
    overall_total = df_filtered['count'].sum()
    plt.suptitle(f'Token Difference Analysis - All Models{title_suffix}\n({analysis_type.upper()} comparison, Overall Total: {overall_total} comparisons)', fontsize=14, y=0.998)
    plt.tight_layout()
    
    plot_file = os.path.join(plots_dir, f"diff_analysis_all_models{file_suffix}_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved (combined): {plot_file}")
    
    # Plot 3: Stacked bar chart for each model individually - Identical vs Differences
    for model in models:
        model_clean = model.replace('/', '_').replace('\\', '_')
        model_df = df_filtered[df_filtered['model'] == model]
        
        identical_data = []
        difference_data = []
        batch_labels = []
        
        for batch_size in batch_sizes:
            batch_df = model_df[model_df['batch_size'] == batch_size]
            
            if not batch_df.empty:
                total_count = batch_df['count'].sum()
                
                # Count for -1 (identical)
                identical_count = batch_df[batch_df['first_difference_index'] == -1]['count'].sum()
                identical_pct = (identical_count / total_count) * 100
                
                # Count for others (differences)
                difference_count = batch_df[batch_df['first_difference_index'] > -1]['count'].sum()
                difference_pct = (difference_count / total_count) * 100
                
                identical_data.append(identical_pct)
                difference_data.append(difference_pct)
                batch_labels.append(f'Batch {batch_size}')
        
        if identical_data:  # Only create plot if there's data
            plt.figure(figsize=(10, 6))
            x_pos = range(len(batch_labels))
            
            # Create stacked bars
            bars1 = plt.bar(x_pos, identical_data, label='Identical (-1)', color='#2ecc71', alpha=0.8)
            bars2 = plt.bar(x_pos, difference_data, bottom=identical_data, label='Differences (>-1)', color='#e74c3c', alpha=0.8)
            
            plt.xticks(x_pos, batch_labels)
            plt.ylabel('Percentage (%)', fontsize=12)
            plt.title(f'Identical vs Differences - {model}{title_suffix}\n({analysis_type.upper()} comparison)', fontsize=14)
            plt.legend(fontsize=10)
            plt.grid(True, alpha=0.3, axis='y')
            plt.ylim(0, 100)
            
            # Add percentage labels on bars
            for i, (ident, diff) in enumerate(zip(identical_data, difference_data)):
                if ident > 3:  # Only show if > 3%
                    plt.text(i, ident/2, f'{ident:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold')
                if diff > 3:  # Only show if > 3%
                    plt.text(i, ident + diff/2, f'{diff:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            
            plot_file = os.path.join(plots_dir, f"identical_vs_diff_{model_clean}{file_suffix}_{timestamp}.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Plot saved (individual model, identical vs diff): {plot_file}")
    
    # Plot 4: Stacked bar chart - Identical (-1) vs Differences (>-1) for all models
    fig, axes = plt.subplots(num_models, 1, figsize=(10, 5 * num_models))
    if num_models == 1:
        axes = [axes]
    
    for idx, model in enumerate(models):
        model_df = df_filtered[df_filtered['model'] == model]
        
        identical_data = []
        difference_data = []
        batch_labels = []
        
        for batch_size in batch_sizes:
            batch_df = model_df[model_df['batch_size'] == batch_size]
            
            if not batch_df.empty:
                total_count = batch_df['count'].sum()
                
                # Count for -1 (identical)
                identical_count = batch_df[batch_df['first_difference_index'] == -1]['count'].sum()
                identical_pct = (identical_count / total_count) * 100
                
                # Count for others (differences)
                difference_count = batch_df[batch_df['first_difference_index'] > -1]['count'].sum()
                difference_pct = (difference_count / total_count) * 100
                
                identical_data.append(identical_pct)
                difference_data.append(difference_pct)
                batch_labels.append(f'Batch {batch_size}')
        
        x_pos = range(len(batch_labels))
        
        # Create stacked bars
        bars1 = axes[idx].bar(x_pos, identical_data, label='Identical (-1)', color='#2ecc71', alpha=0.8)
        bars2 = axes[idx].bar(x_pos, difference_data, bottom=identical_data, label='Differences (>-1)', color='#e74c3c', alpha=0.8)
        
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(batch_labels)
        axes[idx].set_ylabel('Percentage (%)', fontsize=11)
        axes[idx].set_title(f'{model}', fontsize=12)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].set_ylim(0, 100)
        
        # Add percentage labels on bars
        for i, (ident, diff) in enumerate(zip(identical_data, difference_data)):
            if ident > 3:  # Only show if > 3%
                axes[idx].text(i, ident/2, f'{ident:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold')
            if diff > 3:  # Only show if > 3%
                axes[idx].text(i, ident + diff/2, f'{diff:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold')
    
    plt.suptitle(f'Identical vs Differences Comparison - All Models{title_suffix}\n({analysis_type.upper()} comparison)', fontsize=14, y=0.998)
    plt.tight_layout()
    
    plot_file = os.path.join(plots_dir, f"identical_vs_diff_all_models{file_suffix}_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {plot_file}")
    
    print(f"\nAll plots saved to: {plots_dir}")


def plot_batch16_device_histogram(diff_df, result_dir, timestamp):
    """
    Create histogram comparing device pairs for batch_size=16 only.
    Shows first_difference_index distribution for each device pair.
    """
    if 'device_pair' not in diff_df.columns or 'batch_size' not in diff_df.columns:
        print("device_pair or batch_size column not found, skipping batch16 histogram")
        return
    
    # Filter for batch_size=16 and first_difference_index > -2
    batch16_df = diff_df[(diff_df['batch_size'] == 16) & (diff_df['first_difference_index'] > -2)].copy()
    
    if batch16_df.empty:
        print("No data for batch_size=16 after filtering")
        return
    
    # Get unique device pairs and models
    device_pairs = sorted(batch16_df['device_pair'].unique())
    models = batch16_df['model'].unique()
    
    if not device_pairs:
        print("No device pairs found for batch16 histogram")
        return
    
    # Create plots directory
    plots_dir = os.path.join(result_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create color map for device pairs
    colors = plt.cm.tab10(range(len(device_pairs)))
    device_pair_color_map = {pair: colors[i] for i, pair in enumerate(device_pairs)}
    
    # Plot for each model
    for model in models:
        model_clean = model.replace('/', '_').replace('\\', '_')
        model_df = batch16_df[batch16_df['model'] == model]
        
        if model_df.empty:
            continue
        
        plt.figure(figsize=(14, 7))
        
        # Get all unique first_difference_index values for x-axis
        all_indices = sorted(model_df['first_difference_index'].unique())
        
        # For each device pair, count occurrences by first_difference_index
        for device_pair in device_pairs:
            pair_df = model_df[model_df['device_pair'] == device_pair]
            
            if not pair_df.empty:
                # Count occurrences of each first_difference_index
                counts_by_index = pair_df['first_difference_index'].value_counts().to_dict()
                
                # Create list of counts for all indices (0 if not present)
                counts = [counts_by_index.get(idx, 0) for idx in all_indices]
                
                # Calculate total for percentage
                total_count = len(pair_df)
                percentages = [(c / total_count * 100) if total_count > 0 else 0 for c in counts]
                
                # Plot line
                plt.plot(
                    all_indices,
                    percentages,
                    marker='o',
                    label=device_pair.replace('_', ' '),
                    color=device_pair_color_map[device_pair],
                    linewidth=2,
                    markersize=6,
                    alpha=0.8
                )
        
        plt.xlabel('First Difference Index', fontsize=12)
        plt.ylabel('Percentage (%)', fontsize=12)
        plt.title(f'Device Comparison (Batch Size 16) - {model}\nToken Difference Distribution by Device Pair', fontsize=14)
        plt.legend(title='Device Pairs', fontsize=10, loc='best')
        plt.grid(True, alpha=0.3)
        
        # Add vertical line at x=-1 (identical)
        plt.axvline(x=-1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Ensure -1 is shown on x-axis
        ax = plt.gca()
        x_ticks = ax.get_xticks()
        if -1 not in x_ticks:
            x_ticks = sorted(list(x_ticks) + [-1])
            ax.set_xticks(x_ticks)
        
        plt.tight_layout()
        
        plot_file = os.path.join(plots_dir, f"batch16_device_comparison_{model_clean}_{timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Batch16 device comparison plot saved: {plot_file}")
    
    # Create combined plot for all models
    num_models = len(models)
    fig, axes = plt.subplots(num_models, 1, figsize=(14, 7 * num_models))
    if num_models == 1:
        axes = [axes]
    
    for idx, model in enumerate(models):
        model_df = batch16_df[batch16_df['model'] == model]
        
        if model_df.empty:
            continue
        
        all_indices = sorted(model_df['first_difference_index'].unique())
        
        for device_pair in device_pairs:
            pair_df = model_df[model_df['device_pair'] == device_pair]
            
            if not pair_df.empty:
                # Count occurrences of each first_difference_index
                counts_by_index = pair_df['first_difference_index'].value_counts().to_dict()
                counts = [counts_by_index.get(idx, 0) for idx in all_indices]
                total_count = len(pair_df)
                percentages = [(c / total_count * 100) if total_count > 0 else 0 for c in counts]
                
                axes[idx].plot(
                    all_indices,
                    percentages,
                    marker='o',
                    label=device_pair.replace('_', ' '),
                    color=device_pair_color_map[device_pair],
                    linewidth=2,
                    markersize=6,
                    alpha=0.8
                )
        
        axes[idx].set_xlabel('First Difference Index', fontsize=11)
        axes[idx].set_ylabel('Percentage (%)', fontsize=11)
        axes[idx].set_title(f'{model}', fontsize=12)
        axes[idx].legend(title='Device Pairs', fontsize=9, loc='best')
        axes[idx].grid(True, alpha=0.3)
        
        # Add vertical line at x=-1 (identical)
        axes[idx].axvline(x=-1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Ensure -1 is shown on x-axis
        x_ticks = axes[idx].get_xticks()
        if -1 not in x_ticks:
            x_ticks = sorted(list(x_ticks) + [-1])
            axes[idx].set_xticks(x_ticks)
    
    plt.suptitle(f'Device Comparison (Batch Size 16) - All Models\nToken Difference Distribution by Device Pair', fontsize=14, y=0.998)
    plt.tight_layout()
    
    plot_file = os.path.join(plots_dir, f"batch16_device_comparison_all_models_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Batch16 device comparison plot saved (all models): {plot_file}")
    
    # Additional plot: Identical vs Differences for batch_size=16
    # Create stacked bar chart showing identical (-1) vs differences (>-1) by device pair
    num_models = len(models)
    fig, axes = plt.subplots(num_models, 1, figsize=(12, 6 * num_models))
    if num_models == 1:
        axes = [axes]
    
    for idx, model in enumerate(models):
        model_df = batch16_df[batch16_df['model'] == model]
        
        if model_df.empty:
            continue
        
        identical_data = []
        difference_data = []
        pair_labels = []
        
        for device_pair in device_pairs:
            pair_df = model_df[model_df['device_pair'] == device_pair]
            
            if not pair_df.empty:
                total_count = len(pair_df)
                
                # Count for -1 (identical)
                identical_count = len(pair_df[pair_df['first_difference_index'] == -1])
                identical_pct = (identical_count / total_count) * 100
                
                # Count for others (differences)
                difference_count = len(pair_df[pair_df['first_difference_index'] > -1])
                difference_pct = (difference_count / total_count) * 100
                
                identical_data.append(identical_pct)
                difference_data.append(difference_pct)
                pair_labels.append(device_pair.replace('_', ' '))
        
        x_pos = range(len(pair_labels))
        
        # Create stacked bars
        bars1 = axes[idx].bar(x_pos, identical_data, label='Identical (-1)', color='#2ecc71', alpha=0.8)
        bars2 = axes[idx].bar(x_pos, difference_data, bottom=identical_data, label='Differences (>-1)', color='#e74c3c', alpha=0.8)
        
        axes[idx].set_xticks(x_pos)
        axes[idx].set_xticklabels(pair_labels, rotation=15, ha='right')
        axes[idx].set_ylabel('Percentage (%)', fontsize=11)
        axes[idx].set_title(f'{model}', fontsize=12)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3, axis='y')
        axes[idx].set_ylim(0, 100)
        
        # Add percentage labels on bars
        for i, (ident, diff) in enumerate(zip(identical_data, difference_data)):
            if ident > 3:  # Only show if > 3%
                axes[idx].text(i, ident/2, f'{ident:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold')
            if diff > 3:  # Only show if > 3%
                axes[idx].text(i, ident + diff/2, f'{diff:.1f}%', ha='center', va='center', fontsize=9, fontweight='bold')
    
    plt.suptitle(f'Identical vs Differences (Batch Size 16) - All Models\nComparison by Device Pair', fontsize=14, y=0.998)
    plt.tight_layout()
    
    plot_file = os.path.join(plots_dir, f"batch16_identical_vs_diff_all_models_{timestamp}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Batch16 identical vs diff plot saved (all models): {plot_file}")


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
        
        # Create result folder name with selected folders, analysis type, and timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_suffix = "_".join(selected_folders)
        result_dir_name = f"result_{analysis_type}_{folder_suffix}_{timestamp}"
        result_dir = os.path.join(path, result_dir_name)
        os.makedirs(result_dir, exist_ok=True)
        
        output_file = os.path.join(result_dir, f"token_differences_{analysis_type}_{timestamp}.csv")
        
        diff_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
        # Create and save diff index summary
        summary_files = create_diff_index_summary(diff_df, result_dir, timestamp, analysis_type)
        
        # Create plots from summaries
        if summary_files:
            print(f"\nGenerating plots...")
            
            # For device comparison with device_pairs
            if analysis_type == "device" and 'device_pair' in diff_df.columns:
                # Generate plots for each device pair
                for device_pair, summary_file in summary_files.items():
                    if os.path.exists(summary_file):
                        print(f"\nGenerating plots for {device_pair}...")
                        plot_difference_analysis(summary_file, result_dir, analysis_type, timestamp, device_pair)
                
                # Generate batch16 device comparison histogram
                print(f"\nGenerating batch_size=16 device comparison histogram...")
                plot_batch16_device_histogram(diff_df, result_dir, timestamp)
            
            # For batch comparison (no device_pairs)
            elif 'all' in summary_files:
                summary_file = summary_files['all']
                if os.path.exists(summary_file):
                    plot_difference_analysis(summary_file, result_dir, analysis_type, timestamp)
    
    print(f"\n{'='*60}")
    print("All analyses completed!")
    print('='*60)

if __name__ == "__main__":
    main()
