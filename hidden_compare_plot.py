import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Configuration
results_dir = "result/final_activation/results_20251118_051252"
output_dir = os.path.join(results_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

# Metrics to plot
metrics_main = ['mae', 'mse', 'rmse', 'cosine_similarity','magnitude_percent']
metrics_magnitude = []

# Get all CSV files
csv_files = glob.glob(os.path.join(results_dir, "*_hidden_value_trends.csv"))
print(f"Found {len(csv_files)} CSV files")

# Read all CSV files
all_data = []
for csv_file in csv_files:
    print(f"Reading {os.path.basename(csv_file)}...")
    df = pd.read_csv(csv_file)
    all_data.append(df)

# Combine all data
combined_df = pd.concat(all_data, ignore_index=True)
print(f"Total rows: {len(combined_df)}")
print(f"Columns: {combined_df.columns.tolist()}")

# Convert identical to string for better grouping
combined_df['identical'] = combined_df['identical'].astype(str)

# Filter data to keep only transition points (identical changes from -1 to 0)
# Group by (device, model, type, batch_size, index) and find transition points
filtered_rows = []

grouped = combined_df.groupby(['device', 'model', 'type', 'batch_size', 'index'])

for group_key, group_df in grouped:
    # Sort by decoding_step
    group_df = group_df.sort_values('decoding_step')
    
    # Find the first occurrence where identical == '0'
    first_diff = group_df[group_df['identical'] == '0']
    
    if len(first_diff) == 0:
        # No transition point, skip this group entirely
        continue
    
    # Get the decoding_step where transition happens (first identical == '0')
    transition_step = first_diff.iloc[0]['decoding_step']
    
    # Skip if transition happens at decoding_step 0
    if transition_step == 0:
        continue
    
    # Keep rows where:
    # 1. decoding_step == transition_step and identical == '0'
    # 2. decoding_step == transition_step - 1 and identical == '-1'
    transition_rows = group_df[
        ((group_df['decoding_step'] == transition_step) & (group_df['identical'] == '0')) |
        ((group_df['decoding_step'] == transition_step - 1) & (group_df['identical'] == '-1'))
    ]
    
    filtered_rows.extend(transition_rows.index.tolist())

# Apply filter
combined_df = combined_df.loc[filtered_rows].reset_index(drop=True)
print(f"Filtered rows (transition points only): {len(combined_df)}")

# Get unique models and identical values
unique_models = combined_df['model'].unique()
unique_identical = combined_df['identical'].unique()

print(f"\nUnique models: {unique_models}")
print(f"Unique identical values: {unique_identical}")

# Get unique devices
unique_devices = combined_df['device'].unique()
print(f"Unique devices: {unique_devices}")

# Group by device and model, then aggregate by layer
for device in unique_devices:
    device_dir = os.path.join(output_dir, device.replace('/', '_').replace(' ', '_'))
    os.makedirs(device_dir, exist_ok=True)
    
    # Filter data for this device
    device_df = combined_df[combined_df['device'] == device]
    device_models = device_df['model'].unique()
    
    for model in device_models:
        model_dir = os.path.join(device_dir, model.replace('/', '_').replace(' ', '_'))
        os.makedirs(model_dir, exist_ok=True)
        
        # Prepare data for each metric
        metrics_data = {}
        for metric in metrics_main + metrics_magnitude:
            metrics_data[metric] = {}
        
        # Collect data for identical values -1 and 0
        for identical_val in ['-1', '0']:
            if identical_val not in unique_identical:
                continue
                
            
            # Filter data
            filtered_df = device_df[
                (device_df['model'] == model) & 
                (device_df['identical'] == identical_val)
            ]
            
            if len(filtered_df) == 0:
                continue
            
            print(f"\nCollecting {device} - {model} - identical={identical_val} ({len(filtered_df)} rows)")
            
            # Group by layer and compute mean for each metric
            all_metrics = metrics_main + metrics_magnitude
            layer_stats = filtered_df.groupby('layer')[all_metrics].mean().reset_index()
            
            # Store data for each metric
            label = "Identical" if identical_val == '-1' else "First-diff"
            for metric in all_metrics:
                metrics_data[metric][label] = {
                    'layer': layer_stats['layer'].values,
                    'values': layer_stats[metric].values
                }
        
        # Create plots for each metric
        for metric in metrics_main:
            if len(metrics_data[metric]) == 0:
                continue
                
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Get all unique layers from both datasets
            all_layers = set()
            for label_data in metrics_data[metric].values():
                all_layers.update(label_data['layer'])
            all_layers = sorted(list(all_layers))
            positions = np.arange(len(all_layers))
            
            width = 0.35
            colors = {'Identical': '#2ca02c', 'First-diff': '#d62728'}
            
            # Count unique samples (by index) for each label
            sample_counts = {}
            for identical_val in ['-1', '0']:
                label = "Identical" if identical_val == '-1' else "First-diff"
                count = device_df[
                    (device_df['model'] == model) & 
                    (device_df['identical'] == identical_val)
                ]['index'].nunique()
                sample_counts[label] = count
            
            # Special handling for cosine_similarity: plot percentage change
            if metric == 'cosine_similarity' and len(metrics_data[metric]) == 2:
                # Get Identical and First-diff data
                identical_data = metrics_data[metric].get('Identical', None)
                first_diff_data = metrics_data[metric].get('First-diff', None)
                
                if identical_data is not None and first_diff_data is not None:
                    # Create full arrays for both
                    identical_values = np.full(len(all_layers), np.nan)
                    first_diff_values = np.full(len(all_layers), np.nan)
                    
                    for i, layer in enumerate(all_layers):
                        if layer in identical_data['layer']:
                            layer_idx = np.where(identical_data['layer'] == layer)[0][0]
                            identical_values[i] = identical_data['values'][layer_idx]
                        if layer in first_diff_data['layer']:
                            layer_idx = np.where(first_diff_data['layer'] == layer)[0][0]
                            first_diff_values[i] = first_diff_data['values'][layer_idx]
                    
                    # Calculate percentage change: abs((identical - first_diff) / identical * 100)
                    pct_change = np.where(identical_values != 0, 
                                         np.abs((identical_values - first_diff_values) / identical_values * 100), 
                                         np.nan)
                    
                    # Plot percentage change as single bar chart
                    ax.bar(positions, pct_change, width*2, 
                           label=f'% Change (n={sample_counts.get("First-diff", 0)})', 
                           color='#ff7f0e', alpha=0.8)
                    
                    # Customize plot for percentage change
                    ax.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Percentage Change (%)', fontsize=12, fontweight='bold')
                    ax.set_title(f'Cosine Similarity % Change by Layer - {device} - {model}', 
                                 fontsize=14, fontweight='bold', pad=20)
                else:
                    # Fallback to regular plotting
                    for idx, (label, data) in enumerate(metrics_data[metric].items()):
                        values = np.full(len(all_layers), np.nan)
                        for i, layer in enumerate(all_layers):
                            if layer in data['layer']:
                                layer_idx = np.where(data['layer'] == layer)[0][0]
                                values[i] = data['values'][layer_idx]
                        
                        offset = width * (idx - 0.5)
                        label_with_count = f"{label} (n={sample_counts.get(label, 0)})"
                        ax.bar(positions + offset, values, width, label=label_with_count, 
                               color=colors.get(label, '#1f77b4'), alpha=0.8)
                    
                    ax.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
                    ax.set_ylabel(f'{metric.upper()}', fontsize=12, fontweight='bold')
                    ax.set_title(f'{metric.upper()} by Layer - {device} - {model}', 
                                 fontsize=14, fontweight='bold', pad=20)
            else:
                # Regular plotting for other metrics
                for idx, (label, data) in enumerate(metrics_data[metric].items()):
                    # Create full array with NaN for missing layers
                    values = np.full(len(all_layers), np.nan)
                    for i, layer in enumerate(all_layers):
                        if layer in data['layer']:
                            layer_idx = np.where(data['layer'] == layer)[0][0]
                            values[i] = data['values'][layer_idx]
                    
                    offset = width * (idx - 0.5)
                    label_with_count = f"{label} (n={sample_counts.get(label, 0)})"
                    ax.bar(positions + offset, values, width, label=label_with_count, 
                           color=colors.get(label, '#1f77b4'), alpha=0.8)
                
                # Customize plot
                ax.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
                ax.set_ylabel(f'{metric.upper()}', fontsize=12, fontweight='bold')
                ax.set_title(f'{metric.upper()} by Layer - {device} - {model}', 
                             fontsize=14, fontweight='bold', pad=20)
            ax.set_xticks(positions)
            ax.set_xticklabels(all_layers)
            
            ax.legend(loc='upper left', frameon=True, shadow=True)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            plt.tight_layout()
            
            # Save figure
            safe_model_name = model.replace('/', '_').replace(' ', '_')
            output_filename = f"{safe_model_name}_{metric}.png"
            output_path = os.path.join(model_dir, output_filename)
            
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  Saved: {output_filename}")
            plt.close()
    
    # Create plot for magnitude_percent
    metric = 'magnitude_percent'
    if len(metrics_data[metric]) > 0:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get all unique layers
        all_layers = set()
        for label_data in metrics_data[metric].values():
            all_layers.update(label_data['layer'])
        all_layers = sorted(list(all_layers))
        positions = np.arange(len(all_layers))
        
        width = 0.35
        colors = {'Identical': '#2ca02c', 'First-diff': '#d62728'}
        
        # Count unique samples (by index) for each label
        sample_counts = {}
        for identical_val in ['-1', '0']:
            label = "Identical" if identical_val == '-1' else "First-diff"
            count = device_df[
                (device_df['model'] == model) & 
                (device_df['identical'] == identical_val)
            ]['index'].nunique()
            sample_counts[label] = count
        
        # Plot bars for each identical value
        for idx, (label, data) in enumerate(metrics_data[metric].items()):
            # Create full array with NaN for missing layers
            values = np.full(len(all_layers), np.nan)
            for i, layer in enumerate(all_layers):
                if layer in data['layer']:
                    layer_idx = np.where(data['layer'] == layer)[0][0]
                    values[i] = data['values'][layer_idx]
            
            offset = width * (idx - 0.5)
            label_with_count = f"{label} (n={sample_counts.get(label, 0)})"
            ax.bar(positions + offset, values, width, label=label_with_count, 
                   color=colors.get(label, '#9467bd'), alpha=0.8)
        
        # Customize plot
        ax.set_xlabel('Layer Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Magnitude Percent (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Magnitude Percent by Layer - {device} - {model}', 
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(positions)
        ax.set_xticklabels(all_layers)
        ax.legend(loc='upper left', frameon=True, shadow=True)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save figure
        safe_model_name = model.replace('/', '_').replace(' ', '_')
        output_filename = f"{safe_model_name}_{metric}.png"
        output_path = os.path.join(model_dir, output_filename)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_filename}")
        plt.close()

print(f"\n✓ All plots saved to: {output_dir}")
print(f"✓ Total plots generated: {len(unique_models) * (len(metrics_main) + len(metrics_magnitude))}")
