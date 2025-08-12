#!/usr/bin/env python3
"""
Enhanced model splitter that preserves architectural integrity.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Tuple, Optional
import logging
from collections import OrderedDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ArchitecturalModelSplitter:
    """Enhanced model splitter that respects architectural boundaries."""
    
    def __init__(self):
        self.logger = logger
    
    def split_model(self, model: nn.Module, num_splits: int, model_type: str) -> List[nn.Module]:
        """
        Split model while preserving architectural integrity.
        
        Args:
            model: PyTorch model to split
            num_splits: Number of split points (creates num_splits + 1 shards)
            model_type: Type of model (resnet18, resnet50, vgg16, etc.)
            
        Returns:
            List of model shards
        """
        model_type_lower = model_type.lower()
        
        if 'resnet' in model_type_lower:
            return self._split_resnet(model, num_splits, model_type_lower)
        elif 'inception' in model_type_lower:
            return self._split_inception(model, num_splits)
        elif 'vgg' in model_type_lower or 'alexnet' in model_type_lower:
            return self._split_sequential_cnn(model, num_splits, model_type_lower)
        elif 'mobilenet' in model_type_lower:
            return self._split_mobilenet(model, num_splits)
        else:
            self.logger.warning(f"Unknown model type {model_type}, using generic splitter")
            return self._split_generic(model, num_splits)
    
    def _split_resnet(self, model: nn.Module, num_splits: int, model_type: str) -> List[nn.Module]:
        """Split ResNet models at residual block boundaries."""
        # Get the major components
        initial_layers = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool
        )
        
        # Collect all residual blocks
        blocks = []
        blocks.append(('initial', initial_layers))
        
        # Add all residual blocks from each layer
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(model, layer_name)
            for i, block in enumerate(layer):
                blocks.append((f'{layer_name}.{i}', block))
        
        # Add final layers
        final_layers = nn.Sequential(
            model.avgpool,
            nn.Flatten(1),
            model.fc
        )
        blocks.append(('final', final_layers))
        
        # Calculate split points
        total_blocks = len(blocks)
        blocks_per_shard = total_blocks // (num_splits + 1)
        
        # Create shards at block boundaries
        shards = []
        for i in range(num_splits + 1):
            start_idx = i * blocks_per_shard
            if i == num_splits:  # Last shard gets remaining blocks
                end_idx = total_blocks
            else:
                end_idx = (i + 1) * blocks_per_shard
            
            # Create shard
            shard_dict = OrderedDict()
            for idx in range(start_idx, end_idx):
                block_name, block = blocks[idx]
                shard_dict[block_name] = block
            
            # Create a proper sequential model for the shard
            if len(shard_dict) == 1 and 'initial' in list(shard_dict.keys())[0]:
                # First shard - just the initial layers
                shard = list(shard_dict.values())[0]
            elif len(shard_dict) == 1 and 'final' in list(shard_dict.keys())[0]:
                # Last shard - just the final layers
                shard = list(shard_dict.values())[0]
            else:
                # Middle shards - create a module that can handle residual connections
                shard = ResNetShard(shard_dict)
            
            shards.append(shard)
            self.logger.info(f"Shard {i}: {list(shard_dict.keys())}")
        
        return shards
    
    def _split_inception(self, model: nn.Module, num_splits: int) -> List[nn.Module]:
        """Split Inception models at module boundaries."""
        # Collect all inception modules and other layers
        modules = []
        
        # Initial conv layers
        for name in ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3']:
            if hasattr(model, name):
                modules.append((name, getattr(model, name)))
        
        # Pooling layers
        if hasattr(model, 'maxpool1'):
            modules.append(('maxpool1', model.maxpool1))
        
        # More conv layers
        for name in ['Conv2d_3b_1x1', 'Conv2d_4a_3x3']:
            if hasattr(model, name):
                modules.append((name, getattr(model, name)))
        
        if hasattr(model, 'maxpool2'):
            modules.append(('maxpool2', model.maxpool2))
        
        # Mixed modules (inception blocks)
        mixed_modules = ['Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 
                        'Mixed_6c', 'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c']
        
        for name in mixed_modules:
            if hasattr(model, name):
                modules.append((name, getattr(model, name)))
        
        # Final layers
        if hasattr(model, 'avgpool'):
            modules.append(('avgpool', model.avgpool))
        if hasattr(model, 'dropout'):
            modules.append(('dropout', model.dropout))
        if hasattr(model, 'fc'):
            modules.append(('fc', model.fc))
        
        # Calculate split points
        total_modules = len(modules)
        modules_per_shard = total_modules // (num_splits + 1)
        
        # Create shards
        shards = []
        for i in range(num_splits + 1):
            start_idx = i * modules_per_shard
            if i == num_splits:  # Last shard
                end_idx = total_modules
            else:
                end_idx = (i + 1) * modules_per_shard
            
            # Create shard
            shard_dict = OrderedDict()
            for idx in range(start_idx, end_idx):
                module_name, module = modules[idx]
                shard_dict[module_name] = module
            
            # Check if we need flatten before fc
            needs_flatten = False
            for name in shard_dict:
                if name == 'fc' and i > 0:  # fc layer not in first shard
                    needs_flatten = True
            
            if needs_flatten:
                # Insert flatten before fc
                new_dict = OrderedDict()
                for name, module in shard_dict.items():
                    if name == 'fc':
                        new_dict['flatten'] = nn.Flatten(1)
                    new_dict[name] = module
                shard_dict = new_dict
            
            shard = nn.Sequential(shard_dict)
            shards.append(shard)
            self.logger.info(f"Shard {i}: {list(shard_dict.keys())}")
        
        return shards
    
    def _split_sequential_cnn(self, model: nn.Module, num_splits: int, model_type: str) -> List[nn.Module]:
        """Split VGG/AlexNet style models."""
        # Collect all layers
        all_layers = []
        
        # Features
        if hasattr(model, 'features'):
            for i, layer in enumerate(model.features):
                all_layers.append((f'features.{i}', layer))
        
        # Avgpool
        if hasattr(model, 'avgpool'):
            all_layers.append(('avgpool', model.avgpool))
        
        # Classifier
        if hasattr(model, 'classifier'):
            # Add flatten before classifier if transitioning from conv to linear
            if all_layers and isinstance(all_layers[-1][1], (nn.Conv2d, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                all_layers.append(('flatten', nn.Flatten(1)))
            
            for i, layer in enumerate(model.classifier):
                all_layers.append((f'classifier.{i}', layer))
        
        # Calculate split points - prefer splitting after pooling layers
        split_candidates = []
        for i, (name, layer) in enumerate(all_layers):
            if isinstance(layer, (nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
                split_candidates.append(i + 1)  # Split after pooling
        
        # If not enough pooling layers, add more candidates
        if len(split_candidates) < num_splits:
            # Add splits after ReLU layers
            for i, (name, layer) in enumerate(all_layers):
                if isinstance(layer, nn.ReLU) and i + 1 not in split_candidates:
                    split_candidates.append(i + 1)
        
        # Sort candidates and select split points
        split_candidates.sort()
        if len(split_candidates) >= num_splits and num_splits > 0:
            # Select evenly spaced splits from candidates
            step = max(1, len(split_candidates) // (num_splits + 1))
            split_points = []
            for i in range(1, num_splits + 1):
                idx = min(i * step, len(split_candidates) - 1)
                if split_candidates[idx] not in split_points:
                    split_points.append(split_candidates[idx])
        else:
            # Fall back to even splitting
            total_layers = len(all_layers)
            layers_per_shard = max(1, total_layers // (num_splits + 1))
            split_points = [(i + 1) * layers_per_shard for i in range(num_splits)]
        
        # Create shards
        shards = []
        start_idx = 0
        
        for i in range(num_splits + 1):
            if i < num_splits:
                end_idx = split_points[i]
            else:
                end_idx = len(all_layers)
            
            # Create shard
            shard_layers = []
            for idx in range(start_idx, end_idx):
                layer_name, layer = all_layers[idx]
                shard_layers.append(layer)
            
            shard = nn.Sequential(*shard_layers)
            shards.append(shard)
            self.logger.info(f"Shard {i}: layers {start_idx}-{end_idx-1}, "
                           f"types: {[type(l).__name__ for _, l in all_layers[start_idx:end_idx]]}")
            
            start_idx = end_idx
        
        return shards
    
    def _split_mobilenet(self, model: nn.Module, num_splits: int) -> List[nn.Module]:
        """Split MobileNet models."""
        # For MobileNetV2, we can use the existing block-based splitting
        # since it's already working well
        return self._split_generic(model, num_splits)
    
    def _split_generic(self, model: nn.Module, num_splits: int) -> List[nn.Module]:
        """Generic splitter for unknown model types."""
        # Get all modules
        all_modules = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                all_modules.append((name, module))
        
        # Even splitting
        total_modules = len(all_modules)
        modules_per_shard = total_modules // (num_splits + 1)
        
        shards = []
        for i in range(num_splits + 1):
            start_idx = i * modules_per_shard
            if i == num_splits:
                end_idx = total_modules
            else:
                end_idx = (i + 1) * modules_per_shard
            
            shard_dict = OrderedDict()
            for idx in range(start_idx, end_idx):
                module_name, module = all_modules[idx]
                shard_dict[module_name] = module
            
            shard = nn.Sequential(shard_dict)
            shards.append(shard)
        
        return shards


class ResNetShard(nn.Module):
    """Custom module for ResNet shards that handles residual connections."""
    
    def __init__(self, blocks_dict: OrderedDict):
        super().__init__()
        # Convert dict keys to valid module names
        self.block_names = list(blocks_dict.keys())
        for i, (name, module) in enumerate(blocks_dict.items()):
            setattr(self, f'block_{i}', module)
    
    def forward(self, x):
        for i in range(len(self.block_names)):
            x = getattr(self, f'block_{i}')(x)
        return x


if __name__ == "__main__":
    # Test the splitter
    import torchvision.models as models
    
    splitter = ArchitecturalModelSplitter()
    
    # Test with ResNet18
    model = models.resnet18(pretrained=False)
    shards = splitter.split_model(model, num_splits=1, model_type='resnet18')
    print(f"ResNet18 split into {len(shards)} shards")