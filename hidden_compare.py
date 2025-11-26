import os, time
import re
import glob
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import torch.nn.functional as F
import torch

baseline = 'H200'
main_dict = {}
log_dir = "result/final_activation"
start_time = time.strftime("%Y%m%d_%H%M%S")

def parse_layer_dirs():
    dir_list = os.listdir(log_dir)
    # print(dir_list)
    for csv in dir_list:
        if 'input_output_summary' in csv:
            continue
        baseline_flag = False
        gpu_name = csv.split('_')[0]
        if baseline in gpu_name:
            baseline_flag = True
        model_name = csv.split(gpu_name+'_')[1]
        model_name = model_name.split('_batch')[0]

        if model_name not in main_dict.keys():
            main_dict[model_name] = {'baseline': None, 'compare': []}
        
        if baseline_flag:
            main_dict[model_name]['baseline'] = os.path.join(log_dir, csv)
        else:
            main_dict[model_name]['compare'].append({
                'device': gpu_name,
                'file_path': os.path.join(log_dir, csv)
            })

if __name__ == "__main__":
    
    parse_layer_dirs()
    
    csv_chunk_read = 512
    result_path = f'{log_dir}/results_{start_time}'
    print(result_path)
    # print(main_dict)
    
    os.makedirs(result_path, exist_ok=True)
    complete_index = 0
    column_info = ['device', 'model', 'type', 'batch_size', 'index', 'input', 'layer', 'decoding_step', 'token_id', 'token_text']
    column_len = len(column_info)

    compare_column_info = ['device', 'model', 'type', 'batch_size', 'index', 'input', 'layer', 'decoding_step', 'identical', 'token_id1', 'token_id2', 'token_text1', 'token_text2', 'mae', 'mse', 'rmse', 'cosine_similarity', 'magnitude_percent']   
    file_index = 0
    first_flag = True
    for model_name, model_data in main_dict.items():
        if 'redhat' not in model_name:
            continue
        print(model_name)
        if model_data['baseline'] is None or len(model_data['compare']) == 0:
            print(f"Skipping model {model_name} due to missing baseline or comparison data.")
            continue
        
        ## baseline file and compare file open with chunk read because file is too large

        print(f"open baseline: {model_data['baseline']}")
        baseline_csv = open(model_data['baseline'], 'r')
        baseline_reader = csv.reader(baseline_csv)
        baseline_lines = list(baseline_reader)



        for compare_file in model_data['compare']:
            compare_csv_files = []
            compare_readers = []
            compare_lines_list = []
            compare_csv_writers = []
            compare_csv_writer_objs = []

            identical_flag = {}
            print(f"open compare: {compare_file['file_path']}")
            compare_csv = open(compare_file['file_path'], 'r')
            compare_reader = csv.reader(compare_csv)
            compare_lines = list(compare_reader)
            compare_csv_files.append(compare_csv)
            compare_readers.append(compare_reader)
            compare_lines_list.append(compare_lines)

            compare_csv_name = f'{compare_file["device"]}_{model_name}_hidden_value_trends.csv'
            compare_csv_writer = open(os.path.join(result_path, compare_csv_name), 'w', newline='')
            compare_csv_writers.append(compare_csv_writer)
            compare_csv_writer_obj = csv.writer(compare_csv_writer)
            compare_csv_writer_obj.writerow(compare_column_info)
            compare_csv_writer_objs.append(compare_csv_writer_obj)

            compare_number = len(compare_lines_list)

            if compare_file['device'] not in identical_flag.keys():
                identical_flag[compare_file['device']] = {}

            for baseline_line, *compare_lines in zip(baseline_lines[1:], *[cl[1:] for cl in compare_lines_list]):
                
                base_info = dict(zip(column_info, baseline_line[:column_len]))
                base_values = list(map(float, baseline_line[column_len:]))
                # print(base_info)
                for compare_line_set, i in zip(compare_lines, range(compare_number)):
                    compare_info = dict(zip(column_info, compare_line_set[:column_len]))

                    if compare_info['index'] not in identical_flag[compare_info['device']].keys():
                        identical_flag[compare_info['device']][compare_info['index']] = { 'index': -1, 'diff_flag': False}

                    if int(base_info['layer']) == 0:
                        if base_info['token_id'] != compare_info['token_id']:
                            identical_flag[compare_info['device']][compare_info['index']]['diff_flag'] = True
                        
                        if identical_flag[compare_info['device']][compare_info['index']]['diff_flag']:
                            identical_flag[compare_info['device']][compare_info['index']]['index'] += 1

                        if identical_flag[compare_info['device']][compare_info['index']]['index'] > 2:
                            continue
                    compare_values = list(map(float, compare_line_set[column_len:]))

                    if (base_info['model'] != compare_info['model'] or
                        base_info['batch_size'] != compare_info['batch_size'] or
                        base_info['index'] != compare_info['index'] or
                        base_info['layer'] != compare_info['layer'] or
                        base_info['decoding_step'] != compare_info['decoding_step']):
                        print("Mismatched data between baseline and comparison.")
                        continue
                    
                    # compute metrics
                    base_tensor = torch.tensor(base_values)
                    compare_tensor = torch.tensor(compare_values)
                    mae = F.l1_loss(base_tensor, compare_tensor).item()
                    mse = F.mse_loss(base_tensor, compare_tensor).item()
                    rmse = torch.sqrt(F.mse_loss(base_tensor, compare_tensor)).item()
                    cosine_similarity = F.cosine_similarity(base_tensor.unsqueeze(0), compare_tensor.unsqueeze(0)).item()
                    magnitude_percent = (torch.norm(compare_tensor) / torch.norm(base_tensor)).item() * 100 if torch.norm(base_tensor) != 0 else float('inf')

                    compare_csv_writer_objs[i].writerow([
                        compare_info['device'],
                        compare_info['model'],
                        compare_info['type'],
                        compare_info['batch_size'],
                        compare_info['index'],
                        compare_info['input'],
                        compare_info['layer'],
                        compare_info['decoding_step'],
                        identical_flag[compare_info['device']][compare_info['index']]['index'],
                        base_info['token_id'],
                        compare_info['token_id'],
                        base_info['token_text'],
                        compare_info['token_text'],
                        f"{mae:.8f}",
                        f"{mse:.8f}",
                        f"{rmse:.8f}",
                        f"{cosine_similarity:.8f}",
                        f"{magnitude_percent:.8f}"
                    ])
            for compare_csv_file, compare_csv_writer in zip(compare_csv_files, compare_csv_writers):
                compare_csv_file.close()
                compare_csv_writer.close() 
        baseline_csv.close()
    print(f"Results saved in {result_path}")

            

