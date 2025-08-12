import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Read the consolidated metrics file
with open('metrics/session_20250724_202850/consolidated_metrics_averaged.json', 'r') as f:
    data = json.load(f)

latencies = []
throughputs = []
labels = []

# Extract data from the consolidated file (mimicking the professor's structure)
for split_id, metrics in data['results'].items():
    throughput = metrics.get('system_inference_throughput_imgs_per_s')
    avg_metrics = metrics.get('average_metrics_per_batch', {})
    latency = avg_metrics.get('end_to_end_latency_s')
    if latency is not None and throughput is not None:
        latencies.append(latency)
        throughputs.append(throughput)
        label = f"P{split_id}"
        labels.append(label)

# Convert to numpy arrays for easier processing
latencies = np.array(latencies)
throughputs = np.array(throughputs)
labels = np.array(labels)

# Pareto frontier: minimize latency, maximize throughput
points = np.array(list(zip(latencies, throughputs)))
pareto_mask = np.ones(points.shape[0], dtype=bool)
for i, (lat, thr) in enumerate(points):
    # A point is not Pareto optimal if there exists another point with lower latency and higher throughput
    pareto_mask[i] = not np.any((latencies < lat) & (throughputs > thr))

pareto_points = points[pareto_mask]
pareto_labels = labels[pareto_mask]

# Sort Pareto points by latency
sort_idx = np.argsort(pareto_points[:,0])
pareto_points = pareto_points[sort_idx]
pareto_labels = pareto_labels[sort_idx]

print('Labels of points on Pareto frontier:')
print(list(pareto_labels))

# Plotting
plt.figure(figsize=(8,6))
plt.scatter(latencies, throughputs, color='blue', label='All Points')
plt.plot(pareto_points[:,0], pareto_points[:,1], color='red', marker='o', linestyle='-', label='Pareto Frontier')

# Annotate each point with its label
for i, label in enumerate(labels):
    plt.annotate(label, (latencies[i], throughputs[i]), textcoords="offset points", xytext=(5,-5), ha='left', fontsize=9)

plt.xlabel('End-to-End Latency (s)', fontsize=14)
plt.ylabel('Inference Throughput (imgs/s)', fontsize=14)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('clean_pareto_analysis2.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed info
print("\nDetailed Pareto Analysis:")
print("-" * 50)
for i in range(len(pareto_labels)):
    print(f"{pareto_labels[i]}: {pareto_points[i,1]:.2f} imgs/s at {pareto_points[i,0]:.3f}s latency")