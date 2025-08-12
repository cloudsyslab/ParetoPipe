import json
import matplotlib.pyplot as plt
import numpy as np

# Read the JSON data
with open('metrics/session_20250724_202850/consolidated_metrics_averaged.json', 'r') as f:
    data = json.load(f)

# Extract metrics for each split configuration
split_indices = []
throughput_imgs = []
network_time = []
latency = []
intermediate_data = []

for split_id, metrics in data['results'].items():
    split_indices.append(int(split_id))
    throughput_imgs.append(metrics['system_inference_throughput_imgs_per_s'])
    network_time.append(metrics['average_metrics_per_batch']['network_time_s'])
    latency.append(metrics['average_metrics_per_batch']['end_to_end_latency_s'])
    intermediate_data.append(metrics['average_metrics_per_batch']['intermediate_data_size_bytes'])

# Sort by split index
sorted_indices = np.argsort(split_indices)
split_indices = [split_indices[i] for i in sorted_indices]
throughput_imgs = [throughput_imgs[i] for i in sorted_indices]
network_time = [network_time[i] for i in sorted_indices]
latency = [latency[i] for i in sorted_indices]
intermediate_data = [intermediate_data[i] for i in sorted_indices]

# Convert to numpy arrays for easier manipulation
intermediate_data_mb = np.array(intermediate_data) / 1024 / 1024

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Main Pareto frontier: System Throughput vs Intermediate Data Size
# For Pareto optimality: maximize throughput, minimize data size
ax1.scatter(intermediate_data_mb, throughput_imgs, c=split_indices, cmap='viridis', s=100)
ax1.set_xlabel('Intermediate Data Size (MB)')
ax1.set_ylabel('System Inference Throughput (imgs/s)')
ax1.set_title('Pareto Analysis: System Throughput vs Data Transfer Size')
ax1.grid(True, alpha=0.3)

# Add split index labels
for i, txt in enumerate(split_indices):
    ax1.annotate(txt, (intermediate_data_mb[i], throughput_imgs[i]), fontsize=8)

# Calculate Pareto frontier (maximize throughput, minimize data size)
pareto_mask = []
for i in range(len(throughput_imgs)):
    is_dominated = False
    for j in range(len(throughput_imgs)):
        if i != j:
            # Point j dominates point i if j has higher throughput AND lower data size
            if throughput_imgs[j] >= throughput_imgs[i] and intermediate_data_mb[j] <= intermediate_data_mb[i]:
                if throughput_imgs[j] > throughput_imgs[i] or intermediate_data_mb[j] < intermediate_data_mb[i]:
                    is_dominated = True
                    break
    pareto_mask.append(not is_dominated)

# Plot Pareto frontier
pareto_points = [(intermediate_data_mb[i], throughput_imgs[i], split_indices[i]) 
                 for i in range(len(throughput_imgs)) if pareto_mask[i]]
pareto_points.sort(key=lambda x: x[0])  # Sort by data size
if pareto_points:
    pareto_x, pareto_y, pareto_splits = zip(*pareto_points)
    ax1.plot(pareto_x, pareto_y, 'r--', linewidth=2, label='Pareto Frontier')
    ax1.legend()

# 2. System Throughput vs Latency
ax2.scatter(latency, throughput_imgs, c=split_indices, cmap='viridis', s=100)
ax2.set_xlabel('End-to-End Latency (s)')
ax2.set_ylabel('System Inference Throughput (imgs/s)')
ax2.set_title('System Throughput vs Latency Trade-off')
ax2.grid(True, alpha=0.3)
ax2.invert_xaxis()  # Lower latency is better

# 3. Network Time vs Intermediate Data Size
ax3.scatter(intermediate_data_mb, np.array(network_time)*1000, c=split_indices, cmap='viridis', s=100)
ax3.set_xlabel('Intermediate Data Size (MB)')
ax3.set_ylabel('Network Time (ms)')
ax3.set_title('Data Transfer Overhead Analysis')
ax3.grid(True, alpha=0.3)

# 4. Multi-metric visualization
ax4.plot(split_indices, throughput_imgs, 'b-', marker='o', label='System Throughput (imgs/s)', linewidth=2)
ax4_twin = ax4.twinx()
ax4_twin.plot(split_indices, intermediate_data_mb, 'r-', marker='s', label='Data Size (MB)', linewidth=2)
ax4.set_xlabel('Split Index')
ax4.set_ylabel('System Throughput (imgs/s)', color='b')
ax4_twin.set_ylabel('Intermediate Data Size (MB)', color='r')
ax4.set_title('Performance Metrics by Split Index')
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='y', labelcolor='b')
ax4_twin.tick_params(axis='y', labelcolor='r')

# Highlight Pareto optimal points in the multi-metric plot
for i, is_pareto in enumerate(pareto_mask):
    if is_pareto:
        ax4.scatter(split_indices[i], throughput_imgs[i], s=200, c='b', marker='*', edgecolors='black', linewidth=2)
        ax4_twin.scatter(split_indices[i], intermediate_data_mb[i], s=200, c='r', marker='*', edgecolors='black', linewidth=2)

# Add legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

# Add colorbar for split indices
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(split_indices), vmax=max(split_indices)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax1, pad=0.1)
cbar.set_label('Split Index')

plt.suptitle(f'MobileNetV2 Distributed Inference Pareto Analysis\n'
             f'Model: {data["configuration"]["model"]}, '
             f'Batch Size: {data["configuration"]["batch_size"]}, '
             f'World Size: {data["configuration"]["world_size"]}', 
             fontsize=14)
plt.tight_layout()
plt.savefig('pareto_analysis_corrected.png', dpi=300, bbox_inches='tight')
plt.show()

# Print analysis results
print("\nPareto Optimal Configurations:")
print("Split Index | System Throughput (imgs/s) | Data Size (MB) | Network Time (ms) | Latency (s)")
print("-" * 90)
for i in range(len(split_indices)):
    if pareto_mask[i]:
        print(f"{split_indices[i]:11d} | {throughput_imgs[i]:26.2f} | {intermediate_data_mb[i]:14.2f} | "
              f"{network_time[i]*1000:16.2f} | {latency[i]:.6f}")

# Find best configurations for different objectives
best_throughput_idx = np.argmax(throughput_imgs)
best_data_idx = np.argmin(intermediate_data_mb)
best_latency_idx = np.argmin(latency)
best_network_time_idx = np.argmin(network_time)

print(f"\nBest Configurations by Objective:")
print(f"Highest Throughput: Split {split_indices[best_throughput_idx]} with {throughput_imgs[best_throughput_idx]:.2f} imgs/s")
print(f"Smallest Data Size: Split {split_indices[best_data_idx]} with {intermediate_data_mb[best_data_idx]:.2f} MB")
print(f"Lowest Latency: Split {split_indices[best_latency_idx]} with {latency[best_latency_idx]:.4f} s")
print(f"Fastest Network: Split {split_indices[best_network_time_idx]} with {network_time[best_network_time_idx]*1000:.2f} ms")

# Calculate efficiency metrics
print("\nEfficiency Analysis:")
for i in range(len(split_indices)):
    if pareto_mask[i]:
        efficiency = throughput_imgs[i] / intermediate_data_mb[i]  # imgs/s per MB
        print(f"Split {split_indices[i]}: {efficiency:.2f} imgs/s per MB of data transfer")