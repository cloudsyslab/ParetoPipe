import json
import matplotlib.pyplot as plt
import numpy as np

# Read the JSON data
with open('metrics/session_20250724_202850/consolidated_metrics_averaged.json', 'r') as f:
    data = json.load(f)

# Extract metrics for each split configuration
split_indices = []
throughput_imgs = []
network_throughput = []
latency = []
intermediate_data = []

for split_id, metrics in data['results'].items():
    split_indices.append(int(split_id))
    throughput_imgs.append(metrics['system_inference_throughput_imgs_per_s'])
    network_throughput.append(metrics['average_metrics_per_batch']['network_throughput_mbps'])
    latency.append(metrics['average_metrics_per_batch']['end_to_end_latency_s'])
    intermediate_data.append(metrics['average_metrics_per_batch']['intermediate_data_size_bytes'])

# Sort by split index
sorted_indices = np.argsort(split_indices)
split_indices = [split_indices[i] for i in sorted_indices]
throughput_imgs = [throughput_imgs[i] for i in sorted_indices]
network_throughput = [network_throughput[i] for i in sorted_indices]
latency = [latency[i] for i in sorted_indices]
intermediate_data = [intermediate_data[i] for i in sorted_indices]

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Pareto frontier: System Throughput vs Network Throughput
ax1.scatter(network_throughput, throughput_imgs, c=split_indices, cmap='viridis', s=100)
ax1.set_xlabel('Network Throughput (Mbps)')
ax1.set_ylabel('System Inference Throughput (imgs/s)')
ax1.set_title('Pareto Analysis: System vs Network Throughput')
ax1.grid(True, alpha=0.3)

# Add split index labels
for i, txt in enumerate(split_indices):
    ax1.annotate(txt, (network_throughput[i], throughput_imgs[i]), fontsize=8)

# Highlight Pareto optimal points
pareto_mask = []
for i in range(len(throughput_imgs)):
    is_dominated = False
    for j in range(len(throughput_imgs)):
        if i != j and throughput_imgs[j] >= throughput_imgs[i] and network_throughput[j] <= network_throughput[i]:
            if throughput_imgs[j] > throughput_imgs[i] or network_throughput[j] < network_throughput[i]:
                is_dominated = True
                break
    pareto_mask.append(not is_dominated)

pareto_points = [(network_throughput[i], throughput_imgs[i]) for i in range(len(throughput_imgs)) if pareto_mask[i]]
pareto_points.sort(key=lambda x: x[0])
if pareto_points:
    pareto_x, pareto_y = zip(*pareto_points)
    ax1.plot(pareto_x, pareto_y, 'r--', linewidth=2, label='Pareto Frontier')
    ax1.legend()

# 2. System Throughput vs Latency
ax2.scatter(latency, throughput_imgs, c=split_indices, cmap='viridis', s=100)
ax2.set_xlabel('End-to-End Latency (s)')
ax2.set_ylabel('System Inference Throughput (imgs/s)')
ax2.set_title('System Throughput vs Latency Trade-off')
ax2.grid(True, alpha=0.3)

# 3. Intermediate Data Size vs Network Throughput
ax3.scatter(np.array(intermediate_data)/1024/1024, network_throughput, c=split_indices, cmap='viridis', s=100)
ax3.set_xlabel('Intermediate Data Size (MB)')
ax3.set_ylabel('Network Throughput (Mbps)')
ax3.set_title('Data Transfer Analysis')
ax3.grid(True, alpha=0.3)

# 4. Split Index vs Multiple Metrics (normalized)
ax4_twin = ax4.twinx()
ax4.plot(split_indices, throughput_imgs, 'b-', marker='o', label='System Throughput (imgs/s)')
ax4_twin.plot(split_indices, network_throughput, 'r-', marker='s', label='Network Throughput (Mbps)')
ax4.set_xlabel('Split Index')
ax4.set_ylabel('System Throughput (imgs/s)', color='b')
ax4_twin.set_ylabel('Network Throughput (Mbps)', color='r')
ax4.set_title('Performance Metrics by Split Index')
ax4.grid(True, alpha=0.3)
ax4.tick_params(axis='y', labelcolor='b')
ax4_twin.tick_params(axis='y', labelcolor='r')

# Add legends
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1 + lines2, labels1 + labels2, loc='best')

# Add colorbar for split indices
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(split_indices), vmax=max(split_indices)))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax1, pad=0.1)
cbar.set_label('Split Index')

plt.suptitle(f'MobileNetV2 Distributed Inference Performance Analysis\n'
             f'Model: {data["configuration"]["model"]}, '
             f'Batch Size: {data["configuration"]["batch_size"]}, '
             f'World Size: {data["configuration"]["world_size"]}', 
             fontsize=14)
plt.tight_layout()
plt.savefig('pareto_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print optimal configurations
print("\nPareto Optimal Configurations:")
print("Split Index | System Throughput (imgs/s) | Network Throughput (Mbps) | Latency (s)")
print("-" * 80)
for i in range(len(split_indices)):
    if pareto_mask[i]:
        print(f"{split_indices[i]:11d} | {throughput_imgs[i]:26.2f} | {network_throughput[i]:24.2f} | {latency[i]:.6f}")

# Find best configurations
best_throughput_idx = np.argmax(throughput_imgs)
best_network_idx = np.argmin(network_throughput)
best_latency_idx = np.argmin(latency)

print(f"\nBest System Throughput: Split {split_indices[best_throughput_idx]} with {throughput_imgs[best_throughput_idx]:.2f} imgs/s")
print(f"Best Network Efficiency: Split {split_indices[best_network_idx]} with {network_throughput[best_network_idx]:.2f} Mbps")
print(f"Best Latency: Split {split_indices[best_latency_idx]} with {latency[best_latency_idx]:.6f} s")