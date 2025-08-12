# Partition 1
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
BATCH_SIZE = 8
NUM_BATCHES_TO_TEST = 8

# --- Helper Class ---
class ModelPart(nn.Module):
    def __init__(self, part):
        super().__init__()
        self.part = part
    def forward(self, x):
        return self.part(x)

# --- Model Loading & Splitting Logic ---
def get_part1_model_and_transform(model_name, split_index, weights_path):
    """
    Loads the correct model, splits it at the given index,
    and returns the first part of the model and the appropriate data transform.
    """
    # Define a generic weights path, but allow model-specific overrides if needed
    CIFAR10_WEIGHTS_PATH = os.path.join("models", f"{model_name.lower()}_cifar10.pth")
    if not os.path.exists(CIFAR10_WEIGHTS_PATH):
        print(f"[{model_name}] No weights found at {CIFAR10_WEIGHTS_PATH}. Using random initialization.")
        weights_path = None # Ensure we don't try to load non-existent weights
    else:
        weights_path = CIFAR10_WEIGHTS_PATH
        print(f"[{model_name}] Found weights at {weights_path}.")

    part1_model = None
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # --- MobileNetV2 ---
    if model_name == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(1280, 10)
        if weights_path: model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        
        feature_layers = list(model.features.children())
        if not (1 <= split_index < len(feature_layers)):
            raise ValueError(f"Invalid split index for {model_name}: {split_index}")
        part1_model = ModelPart(nn.Sequential(*feature_layers[:split_index])).eval()

    # --- AlexNet ---
    elif model_name == "AlexNet":
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
        if weights_path: model.load_state_dict(torch.load(weights_path, map_location="cpu"))

        feature_layers = list(model.features.children())
        if not (1 <= split_index < len(feature_layers)):
            raise ValueError(f"Invalid split index for {model_name}: {split_index}")
        part1_model = ModelPart(nn.Sequential(*feature_layers[:split_index])).eval()

    # --- ResNet18 ---
    elif model_name == "ResNet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        if weights_path: model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        
        initial_layers = [model.conv1, model.bn1, model.relu, model.maxpool]
        all_blocks = [*model.layer1, *model.layer2, *model.layer3, *model.layer4]
        if not (1 <= split_index <= len(all_blocks)):
            raise ValueError(f"Invalid block split index for {model_name}: {split_index}")
        
        part1_layers = initial_layers + all_blocks[:split_index]
        part1_model = ModelPart(nn.Sequential(*part1_layers)).eval()

    # --- ResNet50 ---
    elif model_name == "ResNet50":
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 10)
        if weights_path: model.load_state_dict(torch.load(weights_path, map_location="cpu"))

        blocks = [nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)]
        for layer in [model.layer1, model.layer2, model.layer3, model.layer4]:
            blocks.extend(layer.children())
        
        if not (1 <= split_index < len(blocks)):
             raise ValueError(f"Invalid block split index for {model_name}: {split_index}")
        
        part1_model = ModelPart(nn.Sequential(*blocks[:split_index])).eval()
        
    # --- VGG16 ---
    elif model_name == "VGG16":
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)
        if weights_path: model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        
        layers = list(model.features.children())
        if not (1 <= split_index < len(layers)):
            raise ValueError(f"Invalid split index for {model_name}: {split_index}")
        part1_model = ModelPart(nn.Sequential(*layers[:split_index])).eval()

    # --- InceptionV3 ---
    elif model_name == "InceptionV3":
        model = models.inception_v3(weights=None, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, 10)
        if weights_path: model.load_state_dict(torch.load(weights_path, map_location="cpu"))

        layers = list(model.children())[:-1]  # Exclude final fc layer
        if not (1 <= split_index < len(layers)):
            raise ValueError(f"Invalid split index for {model_name}: {split_index}")
        part1_model = ModelPart(nn.Sequential(*layers[:split_index])).eval()
        
        # InceptionV3 requires a different input size
        transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return part1_model, transform

# --- Main Server Logic ---
def run_server(model_name, split_index):
    print(f"[Server] Running for model '{model_name}' at split index: {split_index}")
    
    try:
        part1_model, transform = get_part1_model_and_transform(model_name, split_index, None)
    except ValueError as e:
        print(f"[Server] Error: {e}")
        return

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
            
            with torch.no_grad():
                output_tensor = part1_model(images)
            part1_time = time.time() - batch_start

            payload = {
                'tensor': output_tensor.cpu(),
                'batch_start_time': batch_start,
                'part1_inference_time_s': part1_time,
            }
            data_bytes = pickle.dumps(payload)
            conn.sendall(len(data_bytes).to_bytes(8, 'big'))
            conn.sendall(data_bytes)
            print(f"[Server] Sent batch {i+1}/{NUM_BATCHES_TO_TEST} ({len(data_bytes) / 1024:.2f} KB)")

        conn.sendall((0).to_bytes(8, 'big')) # End of data signal
        print("[Server] Done sending data.")

    except Exception as e:
        print(f"[Server] Communication Error: {e}")
    finally:
        conn.close()
        s.close()
        print("[Server] Connection closed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Server-side (Part 1) for split inference.")
    parser.add_argument("--model-name", type=str, required=True, 
                        choices=["MobileNetV2", "AlexNet", "ResNet18", "ResNet50", "VGG16", "InceptionV3"],
                        help="The name of the model to use.")
    parser.add_argument("--split-index", type=int, required=True, help="The layer/block index to split the model at.")
    args = parser.parse_args()
    
    run_server(args.model_name, args.split_index)