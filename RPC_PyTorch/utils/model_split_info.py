#!/usr/bin/env python3
"""
Utility to get split information for different model architectures.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

def get_model_split_info(model_type: str) -> Dict[str, any]:
    """
    Get split information for a model type.
    
    Returns:
        Dict with:
        - max_splits: Maximum number of split points
        - split_type: Type of splitting ('blocks', 'layers', 'modules')
        - description: Human-readable description
    """
    model_type = model_type.lower()
    
    split_info = {
        'mobilenetv2': {
            'max_splits': 18,  # 19 blocks = 18 split points
            'split_type': 'blocks',
            'description': 'InvertedResidual blocks'
        },
        'resnet18': {
            'max_splits': 8,  # 8 BasicBlocks
            'split_type': 'blocks',
            'description': 'BasicBlock modules'
        },
        'resnet50': {
            'max_splits': 16,  # Typically 16 Bottleneck blocks
            'split_type': 'blocks',
            'description': 'Bottleneck blocks'
        },
        'vgg16': {
            'max_splits': 30,  # 31 layers = 30 split points
            'split_type': 'layers',
            'description': 'Feature layers (Conv/ReLU/Pool)'
        },
        'alexnet': {
            'max_splits': 12,  # 13 feature layers = 12 split points
            'split_type': 'layers',
            'description': 'Feature layers'
        },
        'inceptionv3': {
            'max_splits': 19,  # ~20 major blocks
            'split_type': 'blocks',
            'description': 'Inception blocks'
        },
    }
    
    if model_type not in split_info:
        raise ValueError(f"Unknown model type: {model_type}. Supported: {list(split_info.keys())}")
    
    return split_info[model_type]


def get_split_points(model: nn.Module, model_type: str) -> List[int]:
    """
    Get the actual split point indices for a model.
    
    Returns:
        List of indices where the model can be split
    """
    model_type = model_type.lower()
    
    if model_type == 'mobilenetv2':
        # Split between blocks in features
        num_blocks = len(list(model.features.children()))
        return list(range(1, num_blocks))  # Can split after block 1, 2, ..., n-1
        
    elif model_type == 'resnet18':
        # Split after each BasicBlock
        split_points = []
        idx = 0
        # Initial layers before blocks
        for name in ['conv1', 'bn1', 'relu', 'maxpool']:
            if hasattr(model, name):
                idx += 1
                split_points.append(idx)
        
        # Add split points after each block in layers
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                num_blocks = len(list(layer.children()))
                for _ in range(num_blocks):
                    idx += 1
                    split_points.append(idx)
        
        return split_points[:8]  # Return first 8 split points
        
    elif model_type == 'vgg16':
        # Split between any feature layers
        num_layers = len(list(model.features.children()))
        return list(range(1, num_layers))  # Can split after layer 1, 2, ..., n-1
        
    elif model_type == 'alexnet':
        # Split between feature layers
        num_layers = len(list(model.features.children()))
        return list(range(1, num_layers))  # 12 split points
        
    elif model_type == 'inceptionv3':
        # Split between major blocks
        num_blocks = len([name for name, _ in model.named_children() 
                         if not name.startswith('_')])
        return list(range(1, min(num_blocks, 20)))  # Up to 19 split points
        
    elif model_type == 'resnet50':
        # Split after each Bottleneck block
        split_points = []
        idx = 0
        # Initial layers before blocks
        for name in ['conv1', 'bn1', 'relu', 'maxpool']:
            if hasattr(model, name):
                idx += 1
                split_points.append(idx)
        
        # Add split points after each block in layers
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(model, layer_name):
                layer = getattr(model, layer_name)
                num_blocks = len(list(layer.children()))
                for _ in range(num_blocks):
                    idx += 1
                    split_points.append(idx)
        
        return split_points[:16]  # Return first 16 split points
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_default_split_ranges(model_type: str) -> List[int]:
    """
    Get default split indices to test for a model type.
    
    Returns ALL possible split points for every model.
    """
    info = get_model_split_info(model_type)
    max_splits = info['max_splits']
    
    # Return ALL split points from 0 to max_splits
    return list(range(max_splits + 1))  # 0 to max_splits inclusive


if __name__ == "__main__":
    # Test the utility
    print("Model Split Information:")
    print("-" * 60)
    
    for model_type in ['mobilenetv2', 'resnet18', 'resnet50', 'vgg16', 'alexnet', 'inceptionv3']:
        try:
            info = get_model_split_info(model_type)
            default_ranges = get_default_split_ranges(model_type)
            
            print(f"\n{model_type.upper()}:")
            print(f"  Max splits: {info['max_splits']}")
            print(f"  Type: {info['split_type']}")
            print(f"  Description: {info['description']}")
            print(f"  Default test ranges: {default_ranges}")
        except Exception as e:
            print(f"\n{model_type.upper()}: Error - {e}")