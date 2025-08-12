#!/usr/bin/env python3
"""
Utility to pre-split models and save individual shard weights.
This allows each worker to load only its assigned shard instead of the full model.
"""

import os
import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple
import json
import logging
from core import ModelLoader
from profiling import LayerProfiler, split_model_intelligently

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelSplitter:
    """Utility to split models and save individual shards."""
    
    def __init__(self, models_dir: str = "./models", shards_dir: str = "./model_shards"):
        self.models_dir = models_dir
        self.shards_dir = shards_dir
        os.makedirs(self.shards_dir, exist_ok=True)
        
    def split_and_save_model(self, model_type: str, num_splits: int, 
                           split_block: int = None, num_classes: int = 10) -> Dict[str, Any]:
        """
        Split a model and save individual shard weights.
        
        Returns:
            Dict containing shard metadata and file paths
        """
        logger.info(f"Splitting {model_type} into {num_splits + 1} shards")
        
        # Load the original model
        model_loader = ModelLoader(self.models_dir)
        model = model_loader.load_model(model_type, num_classes)
        
        # Create metadata
        metadata = {
            'model_type': model_type,
            'num_splits': num_splits,
            'num_shards': num_splits + 1,
            'num_classes': num_classes,
            'shards': []
        }
        
        # Handle MobileNetV2 block-level splitting
        if model_type.lower() == 'mobilenetv2' and hasattr(model, 'features'):
            shards = self._split_mobilenetv2_blocks(model, num_splits, split_block)
            metadata['split_type'] = 'block_level'
            metadata['split_block'] = split_block
        else:
            # Use intelligent splitting for other models
            logger.info("Using intelligent splitting based on profiling")
            sample_input = model_loader.get_sample_input(model_type, batch_size=1)
            profiler = LayerProfiler(device="cpu", warmup_iterations=2, profile_iterations=5)
            profile = profiler.profile_model(model, sample_input, model_type)
            
            # Force use of simple fallback for now since intelligent splitting is broken
            force_simple_fallback = True
            
            if force_simple_fallback and model_type.lower() != 'mobilenetv2':
                logger.info("Using simple fallback splitting (forced)")
                from profiling.intelligent_splitter import create_simple_model_split
                shards = create_simple_model_split(model, model_type, num_splits)
                split_config = None
            else:
                try:
                    shards, split_config = split_model_intelligently(
                        model, profile, num_splits,
                        network_config={
                            'communication_latency_ms': 200.0,
                            'network_bandwidth_mbps': 3.5
                        }
                    )
                except Exception as e:
                    logger.warning(f"Intelligent splitting failed: {e}")
                    logger.info("Falling back to simple splitting")
                    from profiling.intelligent_splitter import create_simple_model_split
                    shards = create_simple_model_split(model, model_type, num_splits)
                    split_config = None
            if split_config is not None:
                metadata['split_type'] = 'intelligent'
                metadata['split_config'] = {
                    'load_balance_score': split_config.load_balance_score,
                    'communication_overhead_ms': split_config.estimated_communication_overhead_ms
                }
            else:
                metadata['split_type'] = 'simple_fallback'
        
        # Save each shard
        for i, shard in enumerate(shards):
            shard_filename = f"{model_type}_shard_{i}_of_{len(shards)}.pth"
            shard_path = os.path.join(self.shards_dir, shard_filename)
            
            # Save both the model and state dict
            torch.save({
                'model': shard,  # Save the complete model structure
                'state_dict': shard.state_dict(),
                'shard_id': i,
                'total_shards': len(shards),
                'model_type': model_type,
                'num_classes': num_classes
            }, shard_path)
            
            # Calculate shard info
            num_params = sum(p.numel() for p in shard.parameters())
            
            metadata['shards'].append({
                'shard_id': i,
                'filename': shard_filename,
                'path': shard_path,
                'num_parameters': num_params,
                'layers': self._get_layer_names(shard)
            })
            
            logger.info(f"Saved shard {i} to {shard_path} ({num_params:,} parameters)")
        
        # Save metadata
        metadata_path = os.path.join(self.shards_dir, f"{model_type}_shards_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
        return metadata
    
    def _split_mobilenetv2_blocks(self, model: nn.Module, num_splits: int, 
                                 split_block: int = None) -> List[nn.Module]:
        """Split MobileNetV2 at block boundaries."""
        feature_blocks = list(model.features.children())
        total_blocks = len(feature_blocks)
        
        if split_block is None:
            split_block = 8 if num_splits == 1 else total_blocks // (num_splits + 1)
        
        logger.info(f"Splitting MobileNetV2 at block {split_block} of {total_blocks}")
        
        # Create shards
        shards = []
        
        # Shard 1: first part of features
        shard1 = nn.Sequential(*feature_blocks[:split_block])
        shards.append(shard1)
        
        # Shard 2: remaining features + pooling + classifier
        shard2_modules = feature_blocks[split_block:]
        shard2_modules.append(nn.AdaptiveAvgPool2d((1, 1)))
        shard2_modules.append(nn.Flatten())
        shard2_modules.append(model.classifier)
        shard2 = nn.Sequential(*shard2_modules)
        shards.append(shard2)
        
        return shards
    
    def _get_layer_names(self, shard: nn.Module) -> List[str]:
        """Get descriptive names for layers in a shard."""
        layer_names = []
        for name, module in shard.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                layer_names.append(f"{name}: {module.__class__.__name__}")
        return layer_names[:10]  # Limit to first 10 for brevity


def create_shard_module(shard_metadata: Dict[str, Any], shard_id: int) -> nn.Module:
    """
    Create a shard module structure that matches the saved weights.
    This is used by workers to create the module before loading weights.
    """
    # This would need to be implemented based on the specific model architecture
    # For now, we'll rely on the saved module structure
    pass


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-split models for distributed inference")
    parser.add_argument("--model", type=str, default="mobilenetv2", help="Model type to split")
    parser.add_argument("--num-splits", type=int, default=1, help="Number of split points")
    parser.add_argument("--split-block", type=int, default=None, help="Specific block to split at")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of output classes")
    parser.add_argument("--models-dir", type=str, default="./models", help="Directory with model weights")
    parser.add_argument("--shards-dir", type=str, default="./model_shards", help="Directory to save shards")
    
    args = parser.parse_args()
    
    splitter = ModelSplitter(args.models_dir, args.shards_dir)
    metadata = splitter.split_and_save_model(
        args.model,
        args.num_splits,
        args.split_block,
        args.num_classes
    )
    
    print(f"\nSuccessfully split {args.model} into {metadata['num_shards']} shards")
    print(f"Metadata saved to: {args.shards_dir}/{args.model}_shards_metadata.json")