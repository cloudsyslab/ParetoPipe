#!/usr/bin/env python3
"""
Generate working model shards using the enhanced architectural splitter.
"""

import os
import sys
import torch
import torch.nn as nn
import json
import logging
import argparse
from typing import Dict, List, Any

sys.path.append('.')
from core import ModelLoader
from utils.model_splitter_v2 import ArchitecturalModelSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_shard_info(shard: nn.Module, shard_id: int, total_shards: int, 
                    model_type: str, output_dir: str) -> Dict[str, Any]:
    """Save a shard and return its metadata."""
    shard_filename = f"{model_type}_shard_{shard_id}_of_{total_shards}.pth"
    shard_path = os.path.join(output_dir, shard_filename)
    
    # Save the shard
    torch.save({
        'model': shard,
        'state_dict': shard.state_dict(),
        'shard_id': shard_id,
        'total_shards': total_shards,
        'model_type': model_type
    }, shard_path)
    
    # Calculate parameters
    num_params = sum(p.numel() for p in shard.parameters())
    
    # Get layer info
    layers = []
    for name, module in shard.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            layers.append(f"{name}: {type(module).__name__}")
    
    return {
        'shard_id': shard_id,
        'filename': shard_filename,
        'path': shard_path,
        'num_parameters': num_params,
        'layers': layers[:10]  # Show first 10 layers
    }


def generate_shards(model_type: str, num_splits: int, output_dir: str, 
                   models_dir: str = "./models", num_classes: int = 10):
    """Generate working shards for a model."""
    logger.info(f"Generating {num_splits + 1} shards for {model_type}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model_loader = ModelLoader(models_dir)
    model = model_loader.load_model(model_type, num_classes)
    
    # Split model using architectural splitter
    splitter = ArchitecturalModelSplitter()
    shards = splitter.split_model(model, num_splits, model_type)
    
    # Save shards and create metadata
    metadata = {
        'model_type': model_type,
        'num_splits': num_splits,
        'num_shards': len(shards),
        'num_classes': num_classes,
        'split_type': 'architectural',
        'shards': []
    }
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params:,}")
    
    for i, shard in enumerate(shards):
        shard_info = save_shard_info(shard, i, len(shards), model_type, output_dir)
        metadata['shards'].append(shard_info)
        
        param_pct = (shard_info['num_parameters'] / total_params) * 100
        logger.info(f"Shard {i}: {shard_info['num_parameters']:,} params ({param_pct:.1f}%)")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"{model_type}_shards_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_path}")
    return metadata


def test_shard_connection(model_type: str, shards_dir: str):
    """Test if shards connect properly."""
    metadata_path = os.path.join(shards_dir, f"{model_type}_shards_metadata.json")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load shards
    shards = []
    for shard_info in metadata['shards']:
        shard_data = torch.load(shard_info['path'])
        shards.append(shard_data['model'])
    
    # Get sample input
    model_loader = ModelLoader("./models")
    sample_input = model_loader.get_sample_input(model_type, batch_size=1)
    
    logger.info(f"Testing {model_type} with input shape {sample_input.shape}")
    
    # Test forward pass through shards
    try:
        x = sample_input
        for i, shard in enumerate(shards):
            logger.info(f"Shard {i} input shape: {x.shape}")
            x = shard(x)
            logger.info(f"Shard {i} output shape: {x.shape}")
        
        logger.info(f"✓ {model_type} shards connected successfully!")
        logger.info(f"Final output shape: {x.shape}")
        return True
        
    except Exception as e:
        logger.error(f"✗ {model_type} shard connection failed: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Generate working model shards')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['alexnet', 'vgg16', 'resnet18', 'resnet50', 'inceptionv3'],
                       help='Model type to generate shards for')
    parser.add_argument('--splits', type=int, default=1,
                       help='Number of splits (creates splits+1 shards)')
    parser.add_argument('--output-dir', type=str, 
                       default='/home/xfu601/datasets/model_shards/working',
                       help='Output directory for shards')
    parser.add_argument('--test', action='store_true',
                       help='Test shard connections after generation')
    
    args = parser.parse_args()
    
    # Generate shards
    metadata = generate_shards(args.model, args.splits, args.output_dir)
    
    # Test if requested
    if args.test:
        logger.info("\nTesting shard connections...")
        success = test_shard_connection(args.model, args.output_dir)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()