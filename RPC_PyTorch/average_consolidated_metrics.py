#!/usr/bin/env python3
"""Average all runs for each split in consolidated metrics."""

import json
import sys
from collections import defaultdict

def average_runs_per_split(filename):
    """Average all runs for each split in consolidated metrics file."""
    
    # Read the file
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Create new results with averaged values
    averaged_results = {}
    
    # Process each split
    for split_id, split_results in data.get('results', {}).items():
        # Collect all metrics across runs for this split
        metrics_sum = defaultdict(float)
        metrics_count = defaultdict(int)
        
        # Track the model name and split index (should be same across runs)
        model_name = None
        split_index = None
        static_network_delay_ms = None
        
        # Sum up all metrics across runs
        for run_id, run_data in split_results.items():
            model_name = run_data.get('model_name', model_name)
            split_index = run_data.get('split_index', split_index)
            static_network_delay_ms = run_data.get('static_network_delay_ms', static_network_delay_ms)
            
            # Add system throughput
            if 'system_inference_throughput_imgs_per_s' in run_data:
                metrics_sum['system_inference_throughput_imgs_per_s'] += run_data['system_inference_throughput_imgs_per_s']
                metrics_count['system_inference_throughput_imgs_per_s'] += 1
            
            # Add average metrics per batch
            avg_metrics = run_data.get('average_metrics_per_batch', {})
            for metric_name, metric_value in avg_metrics.items():
                if metric_value is not None:
                    key = f'average_metrics_per_batch.{metric_name}'
                    metrics_sum[key] += metric_value
                    metrics_count[key] += 1
        
        # Calculate averages
        averaged_split = {
            'model_name': model_name,
            'split_index': split_index,
            'static_network_delay_ms': static_network_delay_ms,
            'system_inference_throughput_imgs_per_s': None,
            'average_metrics_per_batch': {}
        }
        
        # Set averaged system throughput
        if metrics_count.get('system_inference_throughput_imgs_per_s', 0) > 0:
            averaged_split['system_inference_throughput_imgs_per_s'] = round(
                metrics_sum['system_inference_throughput_imgs_per_s'] / 
                metrics_count['system_inference_throughput_imgs_per_s'], 2
            )
        
        # Set averaged batch metrics
        for key in metrics_sum:
            if key.startswith('average_metrics_per_batch.'):
                metric_name = key.replace('average_metrics_per_batch.', '')
                if metrics_count[key] > 0:
                    avg_value = metrics_sum[key] / metrics_count[key]
                    # Round based on metric type
                    if metric_name == 'intermediate_data_size_bytes':
                        averaged_split['average_metrics_per_batch'][metric_name] = int(avg_value)
                    elif metric_name == 'network_throughput_mbps':
                        averaged_split['average_metrics_per_batch'][metric_name] = round(avg_value, 2)
                    else:
                        # For time-based metrics, use more precision
                        averaged_split['average_metrics_per_batch'][metric_name] = round(avg_value, 6)
                else:
                    averaged_split['average_metrics_per_batch'][metric_name] = None
        
        averaged_results[split_id] = averaged_split
    
    # Sort results by split number (convert to int for proper numerical sorting)
    sorted_results = {}
    for split_id in sorted(averaged_results.keys(), key=lambda x: int(x)):
        sorted_results[split_id] = averaged_results[split_id]
    
    # Create new data structure with averaged results
    averaged_data = {
        'session_id': data['session_id'],
        'timestamp': data['timestamp'],
        'configuration': data['configuration'],
        'device_mapping': data['device_mapping'],
        'results': sorted_results
    }
    
    # Save to new file
    output_filename = filename.replace('.json', '_averaged.json')
    with open(output_filename, 'w') as f:
        json.dump(averaged_data, f, indent=2)
    
    print(f"Created averaged metrics file: {output_filename}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Original: {len(data.get('results', {}))} splits with multiple runs each")
    print(f"Averaged: {len(averaged_results)} splits with averaged values")
    
    # Show sample comparison for one split
    if averaged_results:
        sample_split = list(averaged_results.keys())[0]
        original_runs = len(data.get('results', {}).get(sample_split, {}))
        print(f"\nExample - Split {sample_split}:")
        print(f"  Original: {original_runs} runs")
        print(f"  Averaged: 1 entry with averaged values")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "/home/xfu601/Projects/distributed_inference/metrics/session_20250723_134418/consolidated_metrics.json"
    
    average_runs_per_split(filename)