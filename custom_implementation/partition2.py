# Partition 2
import torch
import socket
import pickle
import time
import os
import json
import numpy as np
import argparse
import torch.nn as nn
from torchvision import models

# --- Configuration ---
PORT = 5555
BATCH_SIZE = 8

# --- Helper Classes & Functions ---
class ModelPart(nn.Module):
    def __init__(self, part):
        super().__init__()
        self.part = part
    def forward(self, x):
        return self.part(x)

def recvall(sock, length):
    """Helper function to receive a specific number of bytes from a socket."""
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise EOFError("Socket closed early while trying to receive data.")
        data += more
    return data

def get_part2_model(model_name, split_index, weights_path):
    """
    Loads the correct full model and constructs the second part of the model
    based on the split index.
    """
    # Define a generic weights path, but allow model-specific overrides if needed
    CIFAR10_WEIGHTS_PATH = os.path.join("models", f"{model_name.lower()}_cifar10.pth")
    if not os.path.exists(CIFAR10_WEIGHTS_PATH):
        print(f"[{model_name}] No weights found at {CIFAR10_WEIGHTS_PATH}. Using random initialization.")
        weights_path = None
    else:
        weights_path = CIFAR10_WEIGHTS_PATH
        print(f"[{model_name}] Found weights at {weights_path}.")
        
    part2_model = None

    # --- MobileNetV2 ---
    if model_name == "MobileNetV2":
        full_model = models.mobilenet_v2(weights=None)
        full_model.classifier[1] = nn.Linear(1280, 10)
        if weights_path: full_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        
        feature_blocks = list(full_model.features.children())
        part2_model = ModelPart(nn.Sequential(
            *feature_blocks[split_index:],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            full_model.classifier
        )).eval()

    # --- AlexNet ---
    elif model_name == "AlexNet":
        full_model = models.alexnet(weights=None)
        full_model.classifier[6] = nn.Linear(full_model.classifier[6].in_features, 10)
        if weights_path: full_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        
        feature_blocks = list(full_model.features.children())
        part2_model = ModelPart(nn.Sequential(
            *feature_blocks[split_index:],
            full_model.avgpool,
            nn.Flatten(start_dim=1),
            full_model.classifier
        )).eval()
        
    # --- ResNet18 ---
    elif model_name == "ResNet18":
        full_model = models.resnet18(weights=None)
        full_model.fc = nn.Linear(full_model.fc.in_features, 10)
        if weights_path: full_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        
        all_blocks = [*full_model.layer1, *full_model.layer2, *full_model.layer3, *full_model.layer4]
        part2_layers = [
            *all_blocks[split_index:],
            full_model.avgpool,
            nn.Flatten(),
            full_model.fc
        ]
        part2_model = ModelPart(nn.Sequential(*part2_layers)).eval()

    # --- ResNet50 ---
    elif model_name == "ResNet50":
        full_model = models.resnet50(weights=None)
        full_model.fc = nn.Linear(full_model.fc.in_features, 10)
        if weights_path: full_model.load_state_dict(torch.load(weights_path, map_location="cpu"))

        blocks = [nn.Sequential(full_model.conv1, full_model.bn1, full_model.relu, full_model.maxpool)]
        for layer in [full_model.layer1, full_model.layer2, full_model.layer3, full_model.layer4]:
            blocks.extend(layer.children())
        
        part2_model = ModelPart(nn.Sequential(
            *blocks[split_index:],
            full_model.avgpool,
            nn.Flatten(start_dim=1),
            full_model.fc
        )).eval()
        
    # --- VGG16 ---
    elif model_name == "VGG16":
        full_model = models.vgg16(weights=None)
        full_model.classifier[6] = nn.Linear(full_model.classifier[6].in_features, 10)
        if weights_path: full_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        
        layers = list(full_model.features.children())
        part2_model = ModelPart(nn.Sequential(
            *layers[split_index:],
            full_model.avgpool,
            nn.Flatten(),
            *full_model.classifier
        )).eval()
        
    # --- InceptionV3 ---
    elif model_name == "InceptionV3":
        full_model = models.inception_v3(weights=None, aux_logits=False)
        full_model.fc = nn.Linear(full_model.fc.in_features, 10)
        if weights_path: full_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        
        layers = list(full_model.children())[:-1] # Exclude fc
        part2_model = ModelPart(nn.Sequential(
            *layers[split_index:],
            nn.AdaptiveAvgPool2d(output_size=(1, 1)), # Ensure correct output size
            nn.Flatten(),
            full_model.fc
        )).eval()
        
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return part2_model

# --- Main Client Logic ---
def run_client(model_name, server_host, split_index, output_dir, network_delay_ms):
    print(f"[Client] Preparing for '{model_name}' split at index {split_index} from host {server_host}")
    
    try:
        part2_model = get_part2_model(model_name, split_index, None)
    except ValueError as e:
        print(f"[Client] Error: {e}")
        return

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((server_host, PORT))
        print(f"[Client] Connected to {server_host}:{PORT}")
    except Exception as e:
        print(f"[Client] Connection error: {e}")
        return

    all_batch_metrics = []
    batch_completion_times = []
    try:
        while True:
            size_bytes = recvall(s, 8)
            data_size = int.from_bytes(size_bytes, 'big')
            if data_size == 0:
                print("[Client] Received end-of-data signal.")
                break

            # Network reception time
            t_net_start = time.time()
            data_payload_bytes = recvall(s, data_size)
            network_time_s = time.time() - t_net_start

            payload = pickle.loads(data_payload_bytes)
            tensor = payload['tensor']
            batch_start_time = payload['batch_start_time']
            part1_inference_time = payload['part1_inference_time_s']

            # Part 2 inference
            t_infer2_start = time.time()
            with torch.no_grad():
                _ = part2_model(tensor)
            part2_inference_time = time.time() - t_infer2_start

            batch_end_time = time.time()
            batch_completion_times.append(batch_end_time)

            # Calculate metrics for this batch
            end_to_end_latency = batch_end_time - batch_start_time
            network_throughput_mbps = (data_size / (1024*1024)) / network_time_s if network_time_s > 0 else 0

            all_batch_metrics.append({
                "part1_inference_time_s": part1_inference_time,
                "part2_inference_time_s": part2_inference_time,
                "network_time_s": network_time_s,
                "end_to_end_latency_s": end_to_end_latency,
                "intermediate_data_size_bytes": data_size,
                "network_throughput_mbps": network_throughput_mbps
            })
            print(f"[Client] Processed batch. End-to-end latency: {end_to_end_latency:.4f}s")

    except Exception as e:
        print(f"[Client] Communication Error: {e}")
    finally:
        s.close()
        print("[Client] Connection closed.")

    if all_batch_metrics:
        system_throughput_imgs_s = 0
        if len(batch_completion_times) > 1:
            avg_inter_batch_time = np.mean(np.diff(batch_completion_times))
            system_throughput_imgs_s = BATCH_SIZE / avg_inter_batch_time if avg_inter_batch_time > 0 else 0

        avg_metrics = {key: np.mean([m[key] for m in all_batch_metrics]) for key in all_batch_metrics[0]}

        results = {
            "model_name": model_name,
            "split_index": split_index,
            "static_network_delay_ms": network_delay_ms,
            "system_inference_throughput_imgs_per_s": system_throughput_imgs_s,
            "average_metrics_per_batch": avg_metrics,
        }

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{model_name}_split_{split_index}_metrics.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"[Client] Metrics saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Client-side (Part 2) for split inference.")
    parser.add_argument("--model-name", type=str, required=True,
                        choices=["MobileNetV2", "AlexNet", "ResNet18", "ResNet50", "VGG16", "InceptionV3"],
                        help="The name of the model to use.")
    parser.add_argument("--host", type=str, required=True, help="The IP address of the server.")
    parser.add_argument("--split-index", type=int, required=True, help="The layer/block index where the split occurs.")
    parser.add_argument("--output-dir", type=str, default="split_results", help="Directory to save the JSON metric files.")
    parser.add_argument("--network-delay-ms", type=float, default=0, help="Simulated network delay (for logging).")
    args = parser.parse_args()

    run_client(args.model_name, args.host, args.split_index, args.output_dir, args.network_delay_ms)