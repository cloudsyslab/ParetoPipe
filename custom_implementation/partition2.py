#partition2
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
    
#----MobileNetV2---
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
#---END MobileNet---

#---AlexNet---
def load_full_alexnet():
    """Loads the base AlexNet model and modifies it for CIFAR-10."""
    model = models.alexnet(weights=None)
    # The last layer of AlexNet's classifier is at index 6
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, 10) # Modify for 10 classes
    
    if os.path.exists(ALEXNET_WEIGHTS_PATH):
        try:
            state_dict = torch.load(ALEXNET_WEIGHTS_PATH, map_location="cpu", weights_only=True)
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
    Runs the client logic for AlexNet. Calculates and saves all performance metrics.
    """
    print(f"[Client] Preparing to run {MODEL_NAME} for split index: {split_index} with {network_delay_ms}ms simulated delay.")

    # 1. Load and create Part 2 of the model
    full_model = load_full_alexnet() # MODIFIED
    feature_blocks = list(full_model.features.children())
    
    if not (1 <= split_index < len(feature_blocks)):
        print(f"Error: Split index must be between 1 and {len(feature_blocks) - 1}")
        return
        
    # MODIFIED: Reconstruct Part 2 according to AlexNet's architecture
    part2_model = ModelPart(nn.Sequential(
        *feature_blocks[split_index:],
        full_model.avgpool,
        nn.Flatten(start_dim=1),
        full_model.classifier
    )).eval()
#----END ----

#---ResNet18-----
def load_full_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    if os.path.exists(CIFAR10_WEIGHTS_PATH):
        try:
            model.load_state_dict(torch.load(CIFAR10_WEIGHTS_PATH, map_location="cpu"))
            print(f"[{MODEL_NAME}] Loaded CIFAR-10 trained weights.")
        except Exception as e:
            print(f"[{MODEL_NAME}] Failed to load weights: {e}. Using random init.")
    else:
        print(f"[{MODEL_NAME}] No weights found. Using random init.")
    return model

def recvall(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise EOFError("Socket closed early.")
        data += more
    return data

def run_client(server_host, split_index, output_dir, network_delay_ms):
    """
    Runs the client logic for AlexNet. Calculates and saves all performance metrics.
    """
    print(f"[Client] Preparing to run {MODEL_NAME} for split index: {split_index} with {network_delay_ms}ms simulated delay.")

    full_model = load_full_resnet18()
    initial_layers = [
        full_model.conv1, full_model.bn1, full_model.relu, full_model.maxpool
    ]
    all_blocks = [
        *full_model.layer1, # Unpack the 2 blocks from layer1
        *full_model.layer2, # Unpack the 2 blocks from layer2
        *full_model.layer3, # Unpack the 2 blocks from layer3
        *full_model.layer4  # Unpack the 2 blocks from layer4
    ]

    # This is the validation for an index from 1 to 8
    if not (1 <= split_index <= len(all_blocks)):
        print(f"Invalid block split index: {split_index}")
        return

    # --- On the SERVER ---

    # Part 1 includes the initial layers PLUS the first N blocks
    part1_layers = initial_layers + all_blocks[:split_index]
    part1_model = ModelPart(nn.Sequential(*part1_layers)).eval()


    # --- On the CLIENT ---
    # Part 2 includes the remaining blocks PLUS the final layers
    part2_layers = [
        *all_blocks[split_index:],
        full_model.avgpool,
        nn.Flatten(),
        full_model.fc
    ]
    part2_model = ModelPart(nn.Sequential(*part2_layers)).eval()
#----END ResNet18--

#-----Resnet50----
def load_full_resnet50():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    if os.path.exists(RESNET50_WEIGHTS_PATH):
        try:
            state_dict = torch.load(RESNET50_WEIGHTS_PATH, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            print(f"[{MODEL_NAME}] Loaded CIFAR-10 trained weights.")
        except TypeError:
            state_dict = torch.load(RESNET50_WEIGHTS_PATH, map_location="cpu")
            model.load_state_dict(state_dict)
            print(f"[{MODEL_NAME}] Loaded weights (fallback).")
        except Exception as e:
            print(f"[{MODEL_NAME}] Failed to load weights: {e}")
    else:
        print(f"[{MODEL_NAME}] No weights found. Using random initialization.")
    return model

def get_resnet50_blocks(model):
    blocks = []
    # Stem
    blocks.append(nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool))
    # Bottleneck blocks
    for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
        for block in layer:
            blocks.append(block)
    return blocks

def recvall(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise EOFError("Socket closed early.")
        data += more
    return data

def run_client(server_host, split_index, output_dir, network_delay_ms):
    print(f"[Client] Starting {MODEL_NAME} split at index {split_index} with simulated {network_delay_ms}ms delay.")

    full_model = load_full_resnet50()
    blocks = get_resnet50_blocks(full_model)

    # ... inside run_client ...
    if not (1 <= split_index <= 18):
        raise ValueError("split_index for blocks must be between 1 and 18")

    # Part 2 starts after the Nth block
    part2_model = ModelPart(nn.Sequential(
        *blocks[split_index + 1:], # The slice is shifted by 1
        full_model.avgpool,
        nn.Flatten(start_dim=1),
        full_model.fc
    )).eval()
#---End ResNet50--

#----VGG16----
def load_full_vgg16():
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
    if os.path.exists(CIFAR10_WEIGHTS_PATH):
        try:
            model.load_state_dict(torch.load(CIFAR10_WEIGHTS_PATH, map_location="cpu"))
            print(f"[{MODEL_NAME}] Loaded CIFAR-10 trained weights.")
        except Exception as e:
            print(f"[{MODEL_NAME}] Failed to load weights: {e}. Using random init.")
    else:
        print(f"[{MODEL_NAME}] No weights found. Using random init.")
    return model

def recvall(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise EOFError("Socket closed early.")
        data += more
    return data

def run_client(server_host, split_index, output_dir, network_delay_ms):
    """
    Runs the client logic for AlexNet. Calculates and saves all performance metrics.
    """
    print(f"[Client] Preparing to run {MODEL_NAME} for split index: {split_index} with {network_delay_ms}ms simulated delay.")

    model = load_full_vgg16()
    layers = list(model.features.children())
    if not (1 <= split_index < len(layers)):
        print(f"Invalid split index: {split_index}")
        return

    part2_model = ModelPart(nn.Sequential(
        *layers[split_index:],
        model.avgpool,
        nn.Flatten(),
        *model.classifier
    )).eval()

#---END VGG16---

#----InceptionV3----
def load_full_inceptionv3():
    model = models.inception_v3(weights=None, aux_logits=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    if os.path.exists(CIFAR10_WEIGHTS_PATH):
        try:
            model.load_state_dict(torch.load(CIFAR10_WEIGHTS_PATH, map_location="cpu"))
            print(f"[{MODEL_NAME}] Loaded CIFAR-10 trained weights.")
        except Exception as e:
            print(f"[{MODEL_NAME}] Failed to load weights: {e}. Using random init.")
    else:
        print(f"[{MODEL_NAME}] No weights found. Using random init.")
    return model

def recvall(sock, length):
    data = b''
    while len(data) < length:
        more = sock.recv(length - len(data))
        if not more:
            raise EOFError("Socket closed early.")
        data += more
    return data

def run_client(server_host, split_index, output_dir, network_delay_ms):
    """
    Runs the client logic for AlexNet. Calculates and saves all performance metrics.
    """
    print(f"[Client] Preparing to run {MODEL_NAME} for split index: {split_index} with {network_delay_ms}ms simulated delay.")

    full_model = load_full_inceptionv3()
    layers = list(full_model.children())[:-1]  # exclude fc
    if not (1 <= split_index < len(layers)):
        print(f"Invalid split index: {split_index}")
        return

    part2_model = ModelPart(nn.Sequential(
        *layers[split_index:],
        nn.Flatten(),
        full_model.fc
    )).eval()
#---END---

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
