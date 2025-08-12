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

