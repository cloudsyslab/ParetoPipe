import json
import matplotlib.pyplot as plt
import numpy as np

# Read the JSON data
with open('metrics/session_20250724_202850/consolidated_metrics_averaged.json', 'r') as f:
    data = json.load(f)

# Extract metrics for each split configuration
points = []
for split_id, metrics in data['results'].items():
    points.append({
        'split': int(split_id),
        'throughput': metrics['system_inference_throughput_imgs_per_s'],
        'data_mb': metrics['average_metrics_per_batch']['intermediate_data_size_bytes'] / 1024 / 1024,
        'latency': metrics['average_metrics_per_batch']['end_to_end_latency_s'],
        'network_time_ms': metrics['average_metrics_per_batch']['network_time_s'] * 1000
    })

# Sort by split index
points.sort(key=lambda x: x['split'])

# Extract arrays
splits = [p['split'] for p in points]
throughputs = [p['throughput'] for p in points]
data_sizes = [p['data_mb'] for p in points]
latencies = [p['latency'] for p in points]

# Create a clean plot with style similar to reference
fig, ax = plt.subplots(figsize=(10, 8))

# Find Pareto frontier using the same logic as the notebook
# A point is not Pareto optimal if there exists another point with lower latency AND higher throughput
pareto_indices = []
for i in range(len(points)):
    is_pareto = True
    for j in range(len(points)):
        if i != j:
            # Check if j dominates i (j has lower latency AND higher throughput)
            if latencies[j] < latencies[i] and throughputs[j] > throughputs[i]:
                is_pareto = False
                break
    if is_pareto:
        pareto_indices.append(i)

# Sort Pareto points by latency for plotting
pareto_points = [points[i] for i in pareto_indices]
pareto_points.sort(key=lambda x: x['latency'])

# Plot all points first (non-Pareto in blue)
non_pareto_indices = [i for i in range(len(points)) if i not in pareto_indices]
for i in non_pareto_indices:
    ax.scatter(points[i]['latency'], points[i]['throughput'], 
              c='blue', s=80, zorder=2, label='All Points' if i == non_pareto_indices[0] else "")

# Plot Pareto points in red
for i, idx in enumerate(pareto_indices):
    ax.scatter(points[idx]['latency'], points[idx]['throughput'], 
              c='red', s=80, zorder=3, label='Pareto Frontier' if i == 0 else "")

# Draw Pareto frontier line
if len(pareto_points) > 1:
    pareto_x = [p['latency'] for p in pareto_points]
    pareto_y = [p['throughput'] for p in pareto_points]
    ax.plot(pareto_x, pareto_y, 'r-', linewidth=2, zorder=1)

# Add labels for all points as P0, P1, etc.
for p in points:
    ax.annotate(f'P{p["split"]}', (p['latency'], p['throughput']), 
                xytext=(3, 3), textcoords='offset points', fontsize=9)

# Formatting
ax.set_xlabel('End-to-End Latency (s)', fontsize=12)
ax.set_ylabel('Inference Throughput (imgs/s)', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(loc='best', fontsize=11)

# Set reasonable axis limits with padding
ax.set_xlim(min(latencies) * 0.98, max(latencies) * 1.02)
ax.set_ylim(min(throughputs) * 0.95, max(throughputs) * 1.05)

plt.tight_layout()
plt.savefig('clean_pareto_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print analysis
print("\nPareto Optimal Configuration:")
print("Split | Throughput | Latency   | Data Size")
print("-" * 50)
for p in pareto_points:
    print(f"P{p['split']:4d} | {p['throughput']:10.2f} | {p['latency']:9.3f} | {p['data_mb']:9.2f} MB")

# Summary
if len(pareto_points) == 1:
    print(f"\nSplit P{pareto_points[0]['split']} dominates all other configurations!")
    print(f"It achieves both the highest throughput ({pareto_points[0]['throughput']:.2f} imgs/s) ")
    print(f"and the lowest latency ({pareto_points[0]['latency']:.3f}s)")
else:
    print(f"\n{len(pareto_points)} Pareto optimal configurations found")
    print("These represent the best trade-offs between throughput and latency")