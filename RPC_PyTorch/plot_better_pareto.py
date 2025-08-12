import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

# Read the JSON data
with open('metrics/session_20250724_202850/consolidated_metrics_averaged.json', 'r') as f:
    data = json.load(f)

# Extract metrics
points = []
for split_id, metrics in data['results'].items():
    points.append({
        'split': int(split_id),
        'throughput': metrics['system_inference_throughput_imgs_per_s'],
        'data_mb': metrics['average_metrics_per_batch']['intermediate_data_size_bytes'] / 1024 / 1024,
        'latency': metrics['average_metrics_per_batch']['end_to_end_latency_s']
    })

points.sort(key=lambda x: x['split'])

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left plot: Throughput vs Data Size (want high throughput, low data)
splits = [p['split'] for p in points]
throughputs = np.array([p['throughput'] for p in points])
data_sizes = np.array([p['data_mb'] for p in points])

scatter1 = ax1.scatter(data_sizes, throughputs, c=splits, cmap='viridis', s=150, alpha=0.8, edgecolors='black')

# Find Pareto optimal points (simple approach)
pareto_indices = []
for i in range(len(points)):
    is_pareto = True
    for j in range(len(points)):
        if i != j:
            # Check if j dominates i (j is better in both dimensions)
            if throughputs[j] >= throughputs[i] and data_sizes[j] <= data_sizes[i]:
                if throughputs[j] > throughputs[i] or data_sizes[j] < data_sizes[i]:
                    is_pareto = False
                    break
    if is_pareto:
        pareto_indices.append(i)

# Highlight Pareto points
pareto_points = [(data_sizes[i], throughputs[i], splits[i]) for i in pareto_indices]
pareto_points.sort(key=lambda x: x[0])  # Sort by data size

# Draw stepped Pareto frontier
if len(pareto_points) > 1:
    # Create stepped line
    x_steps = []
    y_steps = []
    for i in range(len(pareto_points)-1):
        x_steps.extend([pareto_points[i][0], pareto_points[i+1][0]])
        y_steps.extend([pareto_points[i][1], pareto_points[i][1]])
    # Add last segment
    x_steps.append(pareto_points[-1][0])
    y_steps.append(pareto_points[-1][1])
    
    ax1.plot(x_steps, y_steps, 'r-', linewidth=2, alpha=0.7, label='Pareto Frontier')
    
    # Highlight Pareto points
    for x, y, split in pareto_points:
        ax1.scatter(x, y, s=300, facecolors='none', edgecolors='red', linewidth=3)
        ax1.annotate(f'{split}', (x, y), xytext=(5, 5), textcoords='offset points', 
                    fontsize=11, fontweight='bold', color='red')

# Add all split labels
for p in points:
    if p['split'] not in [pt[2] for pt in pareto_points]:
        ax1.annotate(f'{p["split"]}', (p['data_mb'], p['throughput']), 
                    xytext=(3, 3), textcoords='offset points', fontsize=9, alpha=0.7)

ax1.set_xlabel('Intermediate Data Size (MB)', fontsize=12)
ax1.set_ylabel('System Inference Throughput (imgs/s)', fontsize=12)
ax1.set_title('Throughput vs Data Size Trade-off', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Right plot: Throughput vs Latency (want high throughput, low latency)
latencies = np.array([p['latency'] for p in points])
scatter2 = ax2.scatter(latencies, throughputs, c=splits, cmap='viridis', s=150, alpha=0.8, edgecolors='black')

# Find Pareto optimal for throughput vs latency
pareto_indices_2 = []
for i in range(len(points)):
    is_pareto = True
    for j in range(len(points)):
        if i != j:
            if throughputs[j] >= throughputs[i] and latencies[j] <= latencies[i]:
                if throughputs[j] > throughputs[i] or latencies[j] < latencies[i]:
                    is_pareto = False
                    break
    if is_pareto:
        pareto_indices_2.append(i)

# Highlight Pareto points for second plot
for i in pareto_indices_2:
    ax2.scatter(latencies[i], throughputs[i], s=300, facecolors='none', edgecolors='red', linewidth=3)
    ax2.annotate(f'{splits[i]}', (latencies[i], throughputs[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=11, fontweight='bold', color='red')

# Add all split labels
for i, p in enumerate(points):
    if i not in pareto_indices_2:
        ax2.annotate(f'{p["split"]}', (p['latency'], p['throughput']), 
                    xytext=(3, 3), textcoords='offset points', fontsize=9, alpha=0.7)

ax2.set_xlabel('End-to-End Latency (s)', fontsize=12)
ax2.set_ylabel('System Inference Throughput (imgs/s)', fontsize=12)
ax2.set_title('Throughput vs Latency Trade-off', fontsize=14)
ax2.grid(True, alpha=0.3)

# Add colorbars
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('Split Index', fontsize=11)
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Split Index', fontsize=11)

plt.suptitle('MobileNetV2 Distributed Inference: Pareto Analysis', fontsize=16)
plt.tight_layout()
plt.savefig('better_pareto_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print analysis
print("Pareto Optimal Points (Throughput vs Data Size):")
print("Split | Throughput | Data Size | Efficiency")
print("-" * 50)
for x, y, split in pareto_points:
    efficiency = y / x if x > 0 else 0
    print(f"{split:5d} | {y:10.2f} | {x:9.2f} | {efficiency:10.2f}")

print("\nKey Insights:")
print(f"- Split {splits[np.argmin(data_sizes)]} has smallest data transfer ({min(data_sizes):.2f} MB)")
print(f"- Split {splits[np.argmin(latencies)]} has lowest latency ({min(latencies):.3f} s)")

# Identify sweet spots
mid_range_splits = [p for p in points if 0.3 <= p['data_mb'] <= 1.0 and p['throughput'] >= 5.0]
if mid_range_splits:
    best_mid = max(mid_range_splits, key=lambda x: x['throughput'])
    print(f"- Best balanced option: Split {best_mid['split']} ({best_mid['throughput']:.2f} imgs/s, {best_mid['data_mb']:.2f} MB)")