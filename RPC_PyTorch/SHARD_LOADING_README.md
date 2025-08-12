# Shard-Only Loading for Distributed Inference

This document explains the new shard-only loading feature that allows each Pi device to load ONLY its assigned shard weights instead of loading the entire model.

## Overview

Previously, each worker would:
1. Load the ENTIRE model from disk
2. Extract only the layers it needs
3. Discard the rest

This was inefficient and memory-intensive.

With shard-only loading, each worker:
1. Loads ONLY its pre-split shard weights from disk
2. No unnecessary memory usage
3. Faster startup times

## How to Use

### Step 1: Pre-split Your Model

Before running distributed inference, you need to split your model and save the individual shards:

```bash
# Example: Split MobileNetV2 at block 8 (creates 2 shards)
python prepare_shards.py --model mobilenetv2 --num-splits 1 --split-block 8

# Example: Split ResNet18 into 3 shards
python prepare_shards.py --model resnet18 --num-splits 2
```

This creates:
- Individual shard weight files in `./model_shards/`
- A metadata file describing the shards

### Step 2: Run Distributed Inference

The distributed runner now defaults to using local shard loading:

```bash
# Master node (automatically uses local loading)
python distributed_runner.py --rank 0 --world-size 3 --model mobilenetv2

# Worker nodes will load only their assigned shards
python distributed_runner.py --rank 1 --world-size 3 --model mobilenetv2
python distributed_runner.py --rank 2 --world-size 3 --model mobilenetv2
```

## Key Changes

1. **Removed Caching**: Since these are pass-through devices, caching was removed to simplify the implementation.

2. **Local Loading by Default**: The `--use-local-loading` flag now defaults to `True`.

3. **Pre-split Weights**: Models must be pre-split using `prepare_shards.py` before distributed inference.

4. **Prefetching Retained**: Data prefetching is still available with `--enable-prefetch` for improved throughput.

## Directory Structure

```
distributed_inference/
├── models/              # Original full model weights
├── model_shards/        # Pre-split shard weights
│   ├── mobilenetv2_shard_0_of_2.pth
│   ├── mobilenetv2_shard_1_of_2.pth
│   └── mobilenetv2_shards_metadata.json
└── utils/
    └── model_splitter.py  # Utility to split models
```

## Benefits

1. **Memory Efficiency**: Each device only loads what it needs
2. **Faster Startup**: No need to load and parse the entire model
3. **Scalability**: Better suited for resource-constrained devices

## Troubleshooting

If shard loading fails, the system will fall back to loading the full model. Check:
1. That you've run `prepare_shards.py` for your model
2. The `model_shards/` directory exists and contains the shard files
3. The metadata file matches your model configuration