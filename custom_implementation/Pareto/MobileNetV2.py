#part 1
import torch
import socket
import pickle
import time
import os
import argparse
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torch.nn as nn

# --- Configuration ---
HOST = '0.0.0.0'  # Listen on all available network interfaces
PORT = 5555
MODEL_NAME = "MobileNetV2"
CIFAR10_WEIGHTS_PATH = "./models/mobilenetv2_cifar10.pth"
BATCH_SIZE = 8
NUM_BATCHES_TO_TEST = 8

# --- Helper Classes & Functions ---

class ModelPart(nn.Module):
    def __init__(self, part):
        super().__init__()
        self.part = part
    def forward(self, x):
        return self.part(x)

def load_full_mobilenetv2():
    """Loads the base MobileNetV2 model and modifies it for CIFAR-10."""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(1280, 10)
    if os.path.exists(CIFAR10_WEIGHTS_PATH):
        try:
            # THIS IS THE CORRECTED LINE: Added weights_only=True
            state_dict = torch.load(CIFAR10_WEIGHTS_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            print(f"[{MODEL_NAME}] Loaded CIFAR-10 trained weights.")
        except Exception as e:
            print(f"[{MODEL_NAME}] Failed to load weights: {e}. Using random initialization.")
    else:
        print(f"[{MODEL_NAME}] No weights found. Using random initialization.")
    return model

def run_server(split_index):
    print(f"[Server] Running for split index: {split_index}")

    model = load_full_mobilenetv2()
    feature_layers = list(model.features.children())
    if not (1 <= split_index < len(feature_layers)):
        print(f"Invalid split index: {split_index}")
        return

    part1_model = ModelPart(nn.Sequential(*feature_layers[:split_index])).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(1)
    print(f"[Server] Listening on {HOST}:{PORT}")
    conn, addr = s.accept()
    print(f"[Server] Connected to {addr}")

    try:
        for i, (images, _) in enumerate(loader):
            if i >= NUM_BATCHES_TO_TEST:
                break
            batch_start = time.time()
            t1 = time.time()
            with torch.no_grad():
                output_tensor = part1_model(images)
            part1_time = time.time() - t1

            payload = {
                'tensor': output_tensor.cpu(),
                'batch_start_time': batch_start,
                'part1_inference_time_s': part1_time,
            }
            data_bytes = pickle.dumps(payload)
            conn.sendall(len(data_bytes).to_bytes(8, 'big'))
            conn.sendall(data_bytes)
            print(f"[Server] Sent batch {i+1}/{NUM_BATCHES_TO_TEST}")

        conn.sendall((0).to_bytes(8, 'big'))
        print("[Server] Done sending data.")

    except Exception as e:
        print(f"[Server] Error: {e}")
    finally:
        conn.close()
        s.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-index", type=int, required=True)
    args = parser.parse_args()
    run_server(args.split_index)


#part 2


import torch
import socket
import pickle
import time
import os
import json
import psutil
import numpy as np
import argparse
import torch.nn as nn
from torchvision import models

# --- Configuration ---
PORT = 5555
MODEL_NAME = "MobileNetV2"
CIFAR10_WEIGHTS_PATH = "./models/mobilenetv2_cifar10.pth"
BATCH_SIZE = 8
# REMOVED: Hardcoded network delay constant. This will now be a command-line argument.

# --- Helper Classes & Functions ---

class ModelPart(nn.Module):
    def __init__(self, part):
        super().__init__()
        self.part = part
    def forward(self, x):
        return self.part(x)

def load_full_mobilenetv2():
    """Loads the base MobileNetV2 model and modifies it for CIFAR-10."""
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(1280, 10)
    if os.path.exists(CIFAR10_WEIGHTS_PATH):
        try:
            # THIS IS THE CORRECTED LINE: Added weights_only=True for security
            state_dict = torch.load(CIFAR10_WEIGHTS_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            print(f"[{MODEL_NAME}] Loaded CIFAR-10 trained weights.")
        except Exception as e:
            print(f"[{MODEL_NAME}] Failed to load weights: {e}. Using random initialization.")
    else:
        print(f"[{MODEL_NAME}] No weights found. Using random initialization.")
    return model

def recvall(sock, length):
    """Helper function to receive a specific number of bytes from a socket."""
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise EOFError("Socket closed early.")
        data += more
    return data

# --- Main Client Logic ---

def run_client(server_host, split_index, output_dir, network_delay_ms):
    """
    Runs the client logic. Calculates and saves all performance metrics.
    """
    print(f"[Client] Preparing to run for split index: {split_index} with {network_delay_ms}ms simulated delay.")

    # 1. Load and create Part 2 of the model
    full_model = load_full_mobilenetv2()
    feature_blocks = list(full_model.features.children())
    
    if not (1 <= split_index < len(feature_blocks)):
        print(f"Error: Split index must be between 1 and {len(feature_blocks) - 1}")
        return
        
    part2_model = ModelPart(nn.Sequential(
        *feature_blocks[split_index:],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        full_model.classifier
    )).eval()
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

            t_net_start = time.time()
            data_payload_bytes = recvall(s, data_size)
            network_time_s = time.time() - t_net_start

            payload = pickle.loads(data_payload_bytes)
            tensor = payload['tensor']
            batch_start_time = payload['batch_start_time']
            part1_inference_time = payload['part1_inference_time_s']

            t_infer2_start = time.time()
            with torch.no_grad():
                model_output = part2_model(tensor)
            part2_inference_time = time.time() - t_infer2_start

            batch_end_time = time.time()
            batch_completion_times.append(batch_end_time)

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
        print(f"[Client] Error: {e}")
    finally:
        s.close()

    if all_batch_metrics:
        system_throughput_imgs_s = 0
        if len(batch_completion_times) > 1:
            avg_inter_batch_time = np.mean(np.diff(batch_completion_times))
            system_throughput_imgs_s = BATCH_SIZE / avg_inter_batch_time if avg_inter_batch_time > 0 else 0

        avg_metrics = {key: np.mean([m[key] for m in all_batch_metrics]) for key in all_batch_metrics[0]}

        results = {
            "model_name": MODEL_NAME,
            "split_index": split_index,
            "static_network_delay_ms": network_delay_ms,
            "system_inference_throughput_imgs_per_s": system_throughput_imgs_s,
            "average_metrics_per_batch": avg_metrics,
        }

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{MODEL_NAME}_split_{split_index}_metrics.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"[Client] Metrics saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Client-side (Part 2) for split inference.")
    parser.add_argument("--host", type=str, required=True, help="The IP address of the server.")
    parser.add_argument("--split-index", type=int, required=True, help="The layer index where the split occurs.")
    parser.add_argument("--output-dir", type=str, default="split_results", help="Directory to save the JSON metric files.")
    parser.add_argument("--network-delay-ms", type=float, default=0, help="The simulated round-trip network delay in milliseconds for logging purposes.")
    args = parser.parse_args()

    run_client(args.host, args.split_index, args.output_dir, args.network_delay_ms)