#!/usr/bin/env python3
"""
Generate shards compatible with the current distributed runner.
Uses intelligent splitting but saves as simple Sequential modules.
"""

import os
import sys
import torch
import torch.nn as nn
import json
import logging
import argparse

sys.path.append('.')
from core import ModelLoader
from profiling import LayerProfiler, split_model_intelligently

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_resnet_shards(model, model_type: str, num_splits: int):
    """Create ResNet shards preserving residual block structure."""
    if num_splits != 1:  # Only support 2 shards for now
        raise ValueError("ResNet models currently only support 2 shards (num_splits=1)")
    
    # Import ShardWithTransitions
    from profiling.intelligent_splitter import ShardWithTransitions
    
    # Create a dummy model structure and layer names for ShardWithTransitions
    model_structure = {'type': model_type}
    
    if model_type == 'resnet18':
        # Split at layer2/layer3 boundary for ResNet18
        shard0_modules = [
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2
        ]
        
        shard1_modules = [
            model.layer3,
            model.layer4,
            model.avgpool,
            nn.Flatten(1),
            model.fc
        ]
        
        # Create layer indices and names for compatibility
        shard0_indices = list(range(len(shard0_modules)))
        shard1_indices = list(range(len(shard0_modules), len(shard0_modules) + len(shard1_modules)))
        
        shard0_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2']
        shard1_names = ['layer3', 'layer4', 'avgpool', 'flatten', 'fc']
        
        all_layer_names = shard0_names + shard1_names
        
        shard0 = ShardWithTransitions(shard0_modules, 0, 2, model_structure, shard0_indices, all_layer_names)
        shard1 = ShardWithTransitions(shard1_modules, 1, 2, model_structure, shard1_indices, all_layer_names)
        
        return [shard0, shard1]
    
    elif model_type == 'resnet50':
        # Split at layer2/layer3 boundary for ResNet50
        shard0_modules = [
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2
        ]
        
        shard1_modules = [
            model.layer3,
            model.layer4,
            model.avgpool,
            nn.Flatten(1),
            model.fc
        ]
        
        # Create layer indices and names for compatibility
        shard0_indices = list(range(len(shard0_modules)))
        shard1_indices = list(range(len(shard0_modules), len(shard0_modules) + len(shard1_modules)))
        
        shard0_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2']
        shard1_names = ['layer3', 'layer4', 'avgpool', 'flatten', 'fc']
        
        all_layer_names = shard0_names + shard1_names
        
        shard0 = ShardWithTransitions(shard0_modules, 0, 2, model_structure, shard0_indices, all_layer_names)
        shard1 = ShardWithTransitions(shard1_modules, 1, 2, model_structure, shard1_indices, all_layer_names)
        
        return [shard0, shard1]
    
    else:
        raise ValueError(f"Unsupported ResNet variant: {model_type}")


def generate_compatible_shards(model_type: str, num_splits: int, output_dir: str, 
                              models_dir: str = "./models", num_classes: int = 10):
    """Generate shards compatible with distributed runner."""
    logger.info(f"Generating {num_splits + 1} compatible shards for {model_type}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model_loader = ModelLoader(models_dir)
    model = model_loader.load_model(model_type, num_classes)
    
    # Profile model
    logger.info("Profiling model for intelligent splitting...")
    sample_input = model_loader.get_sample_input(model_type, batch_size=1)
    profiler = LayerProfiler(device="cpu", warmup_iterations=2, profile_iterations=5)
    profile = profiler.profile_model(model, sample_input, model_type)
    
    # Handle ResNet models specially to preserve residual connections
    if 'resnet' in model_type.lower():
        logger.info(f"Using ResNet-specific splitting for {model_type}")
        shards = create_resnet_shards(model, model_type, num_splits)
        split_config = type('obj', (object,), {
            'load_balance_score': 0.0,
            'estimated_communication_overhead_ms': 0.0
        })
    else:
        # Split model intelligently for non-ResNet models
        logger.info(f"Splitting model into {num_splits + 1} shards...")
        shards, split_config = split_model_intelligently(
            model, profile, num_splits,
            network_config={
                'communication_latency_ms': 200.0,
                'network_bandwidth_mbps': 3.5
            }
        )
    
    # Create metadata
    metadata = {
        'model_type': model_type,
        'num_splits': num_splits,
        'num_shards': len(shards),
        'num_classes': num_classes,
        'split_type': 'intelligent',
        'split_config': {
            'load_balance_score': split_config.load_balance_score,
            'communication_overhead_ms': split_config.estimated_communication_overhead_ms
        },
        'shards': []
    }
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params:,}")
    
    # Save each shard
    for i, shard in enumerate(shards):
        shard_filename = f"{model_type}_shard_{i}_of_{len(shards)}.pth"
        shard_path = os.path.join(output_dir, shard_filename)
        
        # Calculate parameters
        num_params = sum(p.numel() for p in shard.parameters())
        
        # Get layer names
        layers = []
        for idx, (name, module) in enumerate(shard.named_modules()):
            if len(list(module.children())) == 0:  # Leaf modules
                layers.append(f"modules.{idx}: {type(module).__name__}")
        
        # Save shard - make sure it's saved as a proper module
        torch.save({
            'model': shard,  # This should be a nn.Sequential
            'state_dict': shard.state_dict(),
            'shard_id': i,
            'total_shards': len(shards),
            'model_type': model_type,
            'num_classes': num_classes
        }, shard_path)
        
        metadata['shards'].append({
            'shard_id': i,
            'filename': shard_filename,
            'path': shard_path,
            'num_parameters': num_params,
            'layers': layers[:10]  # First 10 layers
        })
        
        param_pct = (num_params / total_params) * 100
        logger.info(f"Shard {i}: {num_params:,} params ({param_pct:.1f}%)")
        logger.info(f"  Saved to: {shard_filename}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"{model_type}_shards_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_path}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Generate compatible model shards')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['alexnet', 'vgg16', 'resnet18', 'resnet50', 'inceptionv3', 'mobilenetv3'],
                       help='Model type to generate shards for')
    parser.add_argument('--output-dir', type=str, 
                       default='/home/xfu601/datasets/model_shards/split_0',
                       help='Output directory for shards')
    
    args = parser.parse_args()
    
    # For now, always use 1 split (2 shards) for compatibility
    num_splits = 1
    
    logger.info(f"Generating compatible shards for {args.model}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Backup existing shards if any
    import shutil
    backup_dir = args.output_dir + "_backup"
    if os.path.exists(args.output_dir):
        for f in os.listdir(args.output_dir):
            if args.model in f and f.endswith('.pth'):
                os.makedirs(backup_dir, exist_ok=True)
                shutil.move(os.path.join(args.output_dir, f), 
                           os.path.join(backup_dir, f))
                logger.info(f"Backed up {f} to {backup_dir}")
    
    # Generate shards
    generate_compatible_shards(args.model, num_splits, args.output_dir)
    
    logger.info("Done! Compatible shards generated.")


if __name__ == "__main__":
    main()