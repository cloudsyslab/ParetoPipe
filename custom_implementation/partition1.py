#Partition 1
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
#----MobileNetV2-----
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
#-----END------

#-----AlexNet------
def load_full_alexnet():
    model = models.alexnet(weights=None)
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

def run_server(split_index):
    print(f"[Server] Running for split index: {split_index}")

    model = load_full_alexnet()
    feature_layers = list(model.features.children())
    if not (1 <= split_index < len(feature_layers)):
        print(f"Invalid split index: {split_index}")
        return

    part1_model = ModelPart(nn.Sequential(*feature_layers[:split_index])).eval()
#-----END------

#-----ResNet18-------
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

def run_server(split_index):
    print(f"[Server] Running for split index: {split_index}")
    model = load_full_resnet18()

    # --- CORRECTED LOGIC ---
    # 1. Get all the individual components
    initial_layers = [
        model.conv1, model.bn1, model.relu, model.maxpool
    ]
    all_blocks = [
        *model.layer1, # Unpack the 2 blocks from layer1
        *model.layer2, # Unpack the 2 blocks from layer2
        *model.layer3, # Unpack the 2 blocks from layer3
        *model.layer4  # Unpack the 2 blocks from layer4
    ]

    # 2. Validate the split index against the number of blocks
    if not (1 <= split_index <= len(all_blocks)):
        print(f"Invalid block split index: {split_index}")
        return

    # 3. Build Part 1 correctly
    part1_layers = initial_layers + all_blocks[:split_index]
    part1_model = ModelPart(nn.Sequential(*part1_layers)).eval()
#----END-----

#-----ResNet50-----
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
            print(f"[{MODEL_NAME}] Failed to load weights: {e}. Using random initialization.")
    else:
        print(f"[{MODEL_NAME}] No weights found. Using random initialization.")
    return model

def get_resnet50_blocks(model):
    """
    Block 0: stem (conv1+bn1+relu+maxpool)
    Blocks 1..16: Bottleneck blocks from layer1..layer4
    """
    blocks = []
    names = []

    # Stem
    stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    blocks.append(stem)
    names.append("stem(conv1+bn1+relu+maxpool)")

    # Residual layers (Bottleneck blocks)
    for li, layer in enumerate([model.layer1, model.layer2, model.layer3, model.layer4], start=1):
        for bi, block in enumerate(layer, start=1):
            blocks.append(block)
            names.append(f"layer{li}.block{bi}")

    return blocks, names

def build_part1(blocks, split_index):
    # Validates against the 16 residual blocks
    if not (1 <= split_index <= 18):
        raise ValueError("split_index for blocks must be between 1 and 18")
    
    # Part 1 is the stem + the first N blocks
    # split_index + 1 is used because blocks[0] is the stem
    part1_layers = blocks[:split_index + 1] 
    return ModelPart(nn.Sequential(*part1_layers)).eval()

def run_server(split_index, list_splits=False):
    full_model = load_full_resnet50()
    blocks, names = get_resnet50_blocks(full_model)

    if list_splits:
        print("\n=== Available split points (you split AFTER these) ===")
        for i, n in enumerate(names):
            print(f"{i}: {n}")
        print(f"\nValid --split-index values: 1 .. {len(blocks)-1}")
        return

    print(f"[Server] Preparing to run for split index: {split_index}")
    part1_model = build_part1(blocks, split_index)
#----END----

#---VGG16----
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

def run_server(split_index):
    print(f"[Server] Running for split index: {split_index}")

    model = load_full_vgg16()
    layers = list(model.features.children())
    if not (1 <= split_index < len(layers)):
        print(f"Invalid split index: {split_index}")
        return

    part1_model = ModelPart(nn.Sequential(*layers[:split_index])).eval()
#---END---

    #--- transform is same for MobileNetV2,AlexNet,ResNet18,ResNet50,VGG16 ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

#----InceptionV3---
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

def run_server(split_index):
    print(f"[Server] Running for split index: {split_index}")

    model = load_full_inceptionv3()
    layers = list(model.children())[:-1]  # exclude fc
    if not (1 <= split_index < len(layers)):
        print(f"Invalid split index: {split_index}")
        return

    part1_model = ModelPart(nn.Sequential(*layers[:split_index])).eval()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
#---END----
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

