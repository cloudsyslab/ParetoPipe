#!/usr/bin/env python3
"""
Intelligent model splitting algorithm based on profiling data.
Uses computational cost analysis to find optimal split points for distributed inference.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
import logging
from dataclasses import dataclass
import numpy as np
from .layer_profiler import ModelProfile, LayerProfile


class ShardWithTransitions(nn.Module):
    """Custom shard module that handles tensor reshaping between architectural boundaries."""
    
    def __init__(self, modules: List[nn.Module], shard_idx: int, total_shards: int, 
                model_structure: Dict[str, Any], layer_indices: List[int], layer_names: List[str]):
        super().__init__()
        self.modules = nn.ModuleList(modules)
        self.shard_idx = shard_idx
        self.total_shards = total_shards
        self.model_structure = model_structure
        self.layer_indices = layer_indices
        self.layer_names = layer_names
        
        # Create a sequential version for better RPC compatibility
        self._sequential = nn.Sequential(*modules) if modules else None
        
        # Determine if this shard needs special handling
        self.needs_pooling = self._needs_pooling()
        self.needs_flattening = self._needs_flattening()
        
        # Add pooling layer if needed
        if self.needs_pooling:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Add flatten layer if needed  
        if self.needs_flattening:
            self.flatten = nn.Flatten()
    
    def _needs_pooling(self) -> bool:
        """Check if this shard needs adaptive pooling."""
        if not self.layer_indices:
            return False
        
        # Check if this shard contains the transition from features to classifier
        has_features = any('features' in self.layer_names[i] 
                          for i in self.layer_indices if i < len(self.layer_names))
        has_classifier = any('classifier' in self.layer_names[i] 
                           for i in self.layer_indices if i < len(self.layer_names))
        
        # If both features and classifier are in this shard, we need pooling
        if has_features and has_classifier:
            return True
        
        # Alternative: Check if this shard ends with features and next shard starts with classifier
        last_layer_idx = self.layer_indices[-1]
        last_layer_name = self.layer_names[last_layer_idx] if last_layer_idx < len(self.layer_names) else ""
        
        if 'features' in last_layer_name:
            # Check if next layer (in next shard) is classifier
            next_layer_idx = last_layer_idx + 1
            if next_layer_idx < len(self.layer_names):
                next_layer_name = self.layer_names[next_layer_idx]
                return 'classifier' in next_layer_name
        
        return False
    
    def _needs_flattening(self) -> bool:
        """Check if this shard needs flattening."""
        # Check if this shard contains both features ending and classifier beginning
        if not self.layer_indices:
            return False
        
        has_features = any('features' in self.layer_names[i] 
                          for i in self.layer_indices if i < len(self.layer_names))
        has_classifier = any('classifier' in self.layer_names[i] 
                           for i in self.layer_indices if i < len(self.layer_names))
        
        # Also check if we're transitioning within this shard
        transitions_within_shard = False
        for i in range(len(self.layer_indices) - 1):
            current_idx = self.layer_indices[i]
            next_idx = self.layer_indices[i + 1]
            if (current_idx < len(self.layer_names) and next_idx < len(self.layer_names)):
                current_name = self.layer_names[current_idx]
                next_name = self.layer_names[next_idx]
                if 'features' in current_name and 'classifier' in next_name:
                    transitions_within_shard = True
                    break
        
        return (has_features and has_classifier) or transitions_within_shard
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import logging
        import torch
        logger = logging.getLogger(__name__)
        
        logger.info(f"Shard {self.shard_idx} input shape: {x.shape}")
        logger.info(f"Shard {self.shard_idx} needs_pooling: {self.needs_pooling}, needs_flattening: {self.needs_flattening}")
        logger.info(f"Shard {self.shard_idx} layer_indices: {self.layer_indices}")
        if self.layer_indices:
            layer_names_in_shard = [self.layer_names[i] for i in self.layer_indices if i < len(self.layer_names)]
            logger.info(f"Shard {self.shard_idx} layers: {layer_names_in_shard}")
        
        # Process through all modules in sequence
        # Use nn.Sequential approach for better RPC compatibility
        if hasattr(self, '_sequential') and self._sequential is not None:
            # Handle each module individually to catch tensor shape transitions
            for i, module in enumerate(self._sequential):
                logger.info(f"Shard {self.shard_idx} processing module {i}: {type(module).__name__}")
                
                # Special handling for adaptive pooling before classifier
                if isinstance(module, nn.AdaptiveAvgPool2d) or 'pool' in str(type(module)).lower():
                    logger.info(f"Shard {self.shard_idx} applying pooling layer: {x.shape}")
                    x = module(x)
                    logger.info(f"Shard {self.shard_idx} after pooling: {x.shape}")
                    continue
                
                # Check if we're about to enter a linear layer and need flattening
                if isinstance(module, nn.Linear) and len(x.shape) > 2:
                    logger.warning(f"Shard {self.shard_idx} Linear layer needs flattened input, got {x.shape}")
                    x = torch.flatten(x, 1)
                    logger.info(f"Shard {self.shard_idx} flattened for Linear layer: {x.shape}")
                
                x = module(x)
                logger.info(f"Shard {self.shard_idx} after module {i}: {x.shape}")
        else:
            # Fallback to manual iteration
            for i, module in enumerate(self.modules):
                logger.info(f"Shard {self.shard_idx} processing module {i}: {type(module).__name__}")
                
                # Check if we're about to enter a linear layer and need flattening  
                if isinstance(module, nn.Linear) and len(x.shape) > 2:
                    logger.warning(f"Shard {self.shard_idx} Linear layer needs flattened input, got {x.shape}")
                    x = torch.flatten(x, 1)
                    logger.info(f"Shard {self.shard_idx} flattened for Linear layer: {x.shape}")
                
                x = module(x)
                logger.info(f"Shard {self.shard_idx} after module {i}: {x.shape}")
        
        logger.info(f"Shard {self.shard_idx} after modules: {x.shape}")
        
        # Apply pooling if needed (typically after feature extraction)
        if hasattr(self, 'needs_pooling') and self.needs_pooling and len(x.shape) == 4:  # 4D tensor (batch, channels, height, width)
            logger.info(f"Shard {self.shard_idx} applying adaptive pooling")
            x = self.adaptive_pool(x)
            logger.info(f"Shard {self.shard_idx} after pooling: {x.shape}")
        
        # Apply flattening if needed (before classifier)
        if hasattr(self, 'needs_flattening') and self.needs_flattening and len(x.shape) > 2:  # More than 2D
            logger.info(f"Shard {self.shard_idx} applying flattening")
            x = self.flatten(x)
            logger.info(f"Shard {self.shard_idx} after flattening: {x.shape}")
        
        # Additional safety check for classifier input
        if len(x.shape) > 2 and any('classifier' in name for name in [self.layer_names[i] for i in self.layer_indices if i < len(self.layer_names)]):
            logger.warning(f"Shard {self.shard_idx} has classifier layers but tensor still has shape {x.shape}, forcing flatten")
            x = torch.flatten(x, 1)  # Flatten all dimensions except batch
            logger.info(f"Shard {self.shard_idx} after forced flattening: {x.shape}")
        
        logger.info(f"Shard {self.shard_idx} final output shape: {x.shape}")
        return x


@dataclass
class SplitPoint:
    """Represents a potential split point in a model."""
    layer_index: int
    layer_name: str
    cumulative_time_ms: float
    cumulative_memory_mb: float
    cumulative_flops: int
    cumulative_parameters: int
    split_cost: float  # Cost of splitting at this point (network transfer overhead)


@dataclass
class ModelSplit:
    """Represents a complete model split configuration."""
    split_points: List[SplitPoint]
    shards: List[List[int]]  # List of layer indices for each shard
    load_balance_score: float
    estimated_total_time_ms: float
    estimated_communication_overhead_ms: float
    
    def get_shard_layers(self, shard_idx: int, layer_names: List[str]) -> List[str]:
        """Get layer names for a specific shard."""
        if shard_idx >= len(self.shards):
            return []
        return [layer_names[i] for i in self.shards[shard_idx]]


class IntelligentSplitter:
    """Intelligent model splitter that uses profiling data to find optimal split points."""
    
    def __init__(self, communication_latency_ms: float = 5.0, 
                 network_bandwidth_mbps: float = 100.0,
                 load_balance_weight: float = 0.7,
                 communication_weight: float = 0.3):
        """
        Initialize the intelligent splitter.
        
        Args:
            communication_latency_ms: Base network latency between nodes
            network_bandwidth_mbps: Network bandwidth in Mbps
            load_balance_weight: Weight for load balancing in optimization
            communication_weight: Weight for communication overhead in optimization
        """
        self.communication_latency_ms = communication_latency_ms
        self.network_bandwidth_mbps = network_bandwidth_mbps
        self.load_balance_weight = load_balance_weight
        self.communication_weight = communication_weight
        self.logger = logging.getLogger(__name__)
    
    def _calculate_transfer_cost(self, tensor_size_bytes: int) -> float:
        """Calculate the cost of transferring a tensor between nodes."""
        # Convert bytes to megabits
        tensor_size_mb = tensor_size_bytes / (1024 * 1024)
        tensor_size_mbits = tensor_size_mb * 8
        
        # Transfer time = latency + (data_size / bandwidth)
        transfer_time_ms = self.communication_latency_ms + (tensor_size_mbits / self.network_bandwidth_mbps) * 1000
        return transfer_time_ms
    
    def _estimate_tensor_size(self, layer_profile: LayerProfile) -> int:
        """Estimate the size of the output tensor from a layer."""
        # Calculate tensor size based on output shape (assuming float32)
        tensor_elements = 1
        for dim in layer_profile.output_shape:
            tensor_elements *= dim
        return tensor_elements * 4  # 4 bytes per float32
    
    def _calculate_load_balance_score(self, shard_times: List[float]) -> float:
        """Calculate load balance score (lower is better)."""
        if not shard_times:
            return float('inf')
        
        mean_time = np.mean(shard_times)
        if mean_time == 0:
            return 0.0
        
        # Coefficient of variation (std/mean) - lower is better balanced
        std_time = np.std(shard_times)
        return std_time / mean_time
    
    def _generate_split_points(self, profile: ModelProfile) -> List[SplitPoint]:
        """Generate all possible split points with their costs, respecting architectural boundaries."""
        split_points = []
        cumulative_time = 0.0
        cumulative_memory = 0.0
        cumulative_flops = 0
        cumulative_parameters = 0
        
        for i, layer_profile in enumerate(profile.layer_profiles):
            cumulative_time += layer_profile.execution_time_ms
            cumulative_memory += layer_profile.memory_usage_mb
            cumulative_flops += layer_profile.flops
            cumulative_parameters += layer_profile.parameters
            
            # Check if this is a valid split point
            if self._is_valid_split_point(i, profile.layer_profiles):
                # Calculate split cost (network transfer overhead)
                tensor_size_bytes = self._estimate_tensor_size(layer_profile)
                split_cost = self._calculate_transfer_cost(tensor_size_bytes)
                
                split_point = SplitPoint(
                    layer_index=i,
                    layer_name=layer_profile.layer_name,
                    cumulative_time_ms=cumulative_time,
                    cumulative_memory_mb=cumulative_memory,
                    cumulative_flops=cumulative_flops,
                    cumulative_parameters=cumulative_parameters,
                    split_cost=split_cost
                )
                split_points.append(split_point)
        
        return split_points
    
    def _is_valid_split_point(self, layer_idx: int, layer_profiles: List[LayerProfile]) -> bool:
        """Check if a layer index is a valid split point respecting architectural boundaries."""
        if layer_idx >= len(layer_profiles) - 1:  # Can't split at the last layer
            return False
        
        current_layer = layer_profiles[layer_idx]
        next_layer = layer_profiles[layer_idx + 1] if layer_idx + 1 < len(layer_profiles) else None
        
        # NEVER split functional operations - they don't exist as nn.Modules
        if 'functional.' in current_layer.layer_name:
            return False
        
        # NEVER split before functional operations in critical sequences
        if next_layer and 'functional.' in next_layer.layer_name:
            # Check if this creates a problematic functional sequence split
            if any(func_op in next_layer.layer_name for func_op in ['adaptive_avg_pool2d', 'flatten']):
                return False
        
        # Don't split within feature blocks or between related operations
        if self._layers_should_stay_together(current_layer, next_layer):
            return False
        
        # Prefer splits at architectural boundaries
        return self._is_architectural_boundary(current_layer, next_layer)
    
    def _layers_should_stay_together(self, current: LayerProfile, next: Optional[LayerProfile]) -> bool:
        """Check if two consecutive layers should not be split."""
        if not next:
            return False
        
        current_name = current.layer_name
        next_name = next.layer_name
        
        # Keep batch norm with its preceding layer (Conv-BN coupling)
        if 'bn' in next_name.lower() or 'batchnorm' in next_name.lower():
            return True
        
        # Keep activation functions with their preceding layer (BN-Activation coupling)
        if any(act in next_name.lower() for act in ['relu', 'sigmoid', 'tanh', 'gelu']):
            return True
        
        # Keep dropout with its preceding layer
        if 'dropout' in next_name.lower():
            return True
        
        # Keep functional operations with their preceding layers
        if 'functional.' in next_name:
            return True
        
        # RESNET: Keep residual blocks together
        if self._in_same_resnet_block(current_name, next_name):
            return True
            
        # INCEPTION: Keep inception modules together
        if self._in_same_inception_module(current_name, next_name):
            return True
        
        # Keep consecutive layers in the same architectural block together
        if ('features' in current_name and 'features' in next_name and
            self._same_feature_block(current_name, next_name)):
            return True
        
        return False
    
    def _same_feature_block(self, name1: str, name2: str) -> bool:
        """Check if two layer names belong to the same feature block."""
        import re
        
        # Extract block numbers from layer names
        def extract_block_num(name):
            match = re.search(r'features\.(\d+)', name)
            return int(match.group(1)) if match else -1
        
        block1 = extract_block_num(name1)
        block2 = extract_block_num(name2)
        
        return block1 == block2 and block1 != -1
    
    def _in_same_resnet_block(self, name1: str, name2: str) -> bool:
        """Check if two layers are in the same ResNet residual block."""
        import re
        
        # ResNet block patterns: layer1.0.conv1, layer1.0.bn1, layer1.0.downsample.0, etc.
        pattern = r'layer(\d+)\.(\d+)\.'
        
        match1 = re.search(pattern, name1)
        match2 = re.search(pattern, name2)
        
        if match1 and match2:
            # Same layer group (e.g., layer1) and same block number (e.g., 0)
            return match1.group(1) == match2.group(1) and match1.group(2) == match2.group(2)
        
        return False
    
    def _in_same_inception_module(self, name1: str, name2: str) -> bool:
        """Check if two layers are in the same Inception module."""
        import re
        
        # Inception patterns: Mixed_5b.branch1x1.conv, Mixed_5b.branch3x3.conv, etc.
        pattern = r'(Mixed_\w+)\.'
        
        match1 = re.search(pattern, name1)
        match2 = re.search(pattern, name2)
        
        if match1 and match2:
            # Same Mixed module (e.g., Mixed_5b)
            return match1.group(1) == match2.group(1)
        
        return False
    
    def _is_architectural_boundary(self, current: LayerProfile, next: Optional[LayerProfile]) -> bool:
        """Check if this is a good architectural boundary for splitting."""
        if not next:
            return False
        
        current_name = current.layer_name
        next_name = next.layer_name
        
        # Major architectural transitions
        if 'features' in current_name and 'classifier' in next_name:
            return True  # Features to classifier boundary
        
        # RESNET: Between different residual blocks
        if not self._in_same_resnet_block(current_name, next_name):
            # Check if we're transitioning between ResNet blocks
            import re
            curr_match = re.search(r'layer(\d+)\.(\d+)\.', current_name)
            next_match = re.search(r'layer(\d+)\.(\d+)\.', next_name)
            if curr_match and next_match:
                # Different blocks = good split point
                if curr_match.group(1) != next_match.group(1) or curr_match.group(2) != next_match.group(2):
                    return True
        
        # INCEPTION: Between different inception modules  
        if not self._in_same_inception_module(current_name, next_name):
            # Check if we're transitioning between Inception modules
            import re
            curr_match = re.search(r'(Mixed_\w+)\.', current_name)
            next_match = re.search(r'(Mixed_\w+)\.', next_name)
            if curr_match and next_match:
                # Different modules = good split point
                if curr_match.group(1) != next_match.group(1):
                    return True
        
        # Between different feature blocks
        if ('features' in current_name and 'features' in next_name and
            not self._same_feature_block(current_name, next_name)):
            return True
        
        # After pooling layers (good split points)
        if any(pool in current_name.lower() for pool in ['pool', 'avgpool', 'maxpool']):
            # But not if the next layer is part of a residual block
            if not any(pattern in next_name for pattern in ['downsample', 'shortcut']):
                return True
        
        # VGG/AlexNet style: After ReLU before next Conv (simple sequential models)
        if ('relu' in current_name.lower() and 'conv' in next_name.lower() and 
            'layer' not in current_name and 'layer' not in next_name):  # Not ResNet
            return True
        
        # Don't allow arbitrary splits - must be at a recognized boundary
        return False
    
    def _evaluate_split_configuration(self, split_indices: List[int], 
                                    split_points: List[SplitPoint],
                                    profile: ModelProfile) -> ModelSplit:
        """Evaluate a specific split configuration."""
        if not split_indices:
            # Single shard - no splits
            shards = [list(range(len(profile.layer_profiles)))]
            return ModelSplit(
                split_points=[],
                shards=shards,
                load_balance_score=0.0,
                estimated_total_time_ms=profile.total_time_ms,
                estimated_communication_overhead_ms=0.0
            )
        
        # Create index mapping from layer index to split point
        split_point_map = {sp.layer_index: sp for sp in split_points}
        
        # Add start and end points
        indices = [0] + sorted(split_indices) + [len(profile.layer_profiles)]
        
        # Create shards
        shards = []
        shard_times = []
        selected_split_points = []
        communication_overhead = 0.0
        
        for i in range(len(indices) - 1):
            start_idx = indices[i]
            end_idx = indices[i + 1]
            
            # Create shard
            shard = list(range(start_idx, end_idx))
            shards.append(shard)
            
            # Calculate shard processing time
            shard_time = sum(profile.layer_profiles[j].execution_time_ms for j in shard)
            shard_times.append(shard_time)
            
            # Add communication overhead (except for last shard)
            if i < len(indices) - 2:
                split_layer_idx = end_idx - 1
                if split_layer_idx in split_point_map:
                    split_point = split_point_map[split_layer_idx]
                    selected_split_points.append(split_point)
                    communication_overhead += split_point.split_cost
                else:
                    # Estimate cost for layers without explicit split points
                    layer_profile = profile.layer_profiles[split_layer_idx]
                    tensor_size_bytes = self._estimate_tensor_size(layer_profile)
                    estimated_cost = self._calculate_transfer_cost(tensor_size_bytes)
                    communication_overhead += estimated_cost
        
        # Calculate load balance score
        load_balance_score = self._calculate_load_balance_score(shard_times)
        
        # Estimate total time (max of shard times + communication overhead)
        max_shard_time = max(shard_times) if shard_times else 0.0
        estimated_total_time = max_shard_time + communication_overhead
        
        return ModelSplit(
            split_points=selected_split_points,
            shards=shards,
            load_balance_score=load_balance_score,
            estimated_total_time_ms=estimated_total_time,
            estimated_communication_overhead_ms=communication_overhead
        )
    
    def _optimize_splits_dynamic_programming(self, profile: ModelProfile, 
                                           num_splits: int) -> ModelSplit:
        """Use dynamic programming to find optimal splits."""
        split_points = self._generate_split_points(profile)
        n_layers = len(profile.layer_profiles)
        
        if num_splits >= n_layers:
            # More splits than layers - each layer becomes a shard
            split_indices = list(range(1, n_layers))
            return self._evaluate_split_configuration(split_indices, split_points, profile)
        
        # DP table: dp[i][j] = minimum cost to split first i layers into j shards
        dp = {}
        parent = {}
        
        def get_cost(start: int, end: int) -> Tuple[float, float]:
            """Get processing cost and communication overhead for a shard."""
            if start >= end:
                return 0.0, 0.0
            
            processing_cost = sum(profile.layer_profiles[i].execution_time_ms for i in range(start, end))
            
            # Communication cost for this shard (if not the last one)
            comm_cost = 0.0
            if end < n_layers:
                tensor_size = self._estimate_tensor_size(profile.layer_profiles[end - 1])
                comm_cost = self._calculate_transfer_cost(tensor_size)
            
            return processing_cost, comm_cost
        
        # Initialize base case
        for i in range(n_layers + 1):
            dp[(i, 1)] = get_cost(0, i)
            parent[(i, 1)] = []
        
        # Fill DP table
        for j in range(2, num_splits + 2):  # j shards
            for i in range(j, n_layers + 1):  # i layers
                best_cost = float('inf')
                best_split = -1
                
                for k in range(j - 1, i):  # try splitting at position k
                    if (k, j - 1) not in dp:
                        continue
                    
                    prev_processing, prev_comm = dp[(k, j - 1)]
                    curr_processing, curr_comm = get_cost(k, i)
                    
                    # Total cost considers both load balance and communication
                    max_processing = max(prev_processing, curr_processing)
                    total_comm = prev_comm + curr_comm
                    
                    # Combined cost function
                    cost = (self.load_balance_weight * max_processing + 
                           self.communication_weight * total_comm)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_split = k
                
                if best_split != -1:
                    prev_processing, prev_comm = dp[(best_split, j - 1)]
                    curr_processing, curr_comm = get_cost(best_split, i)
                    dp[(i, j)] = (max(prev_processing, curr_processing), prev_comm + curr_comm)
                    parent[(i, j)] = parent[(best_split, j - 1)] + [best_split]
        
        # Reconstruct solution
        if (n_layers, num_splits + 1) in parent:
            split_indices = parent[(n_layers, num_splits + 1)]
            # Remove the initial 0 if present
            split_indices = [s for s in split_indices if s > 0]
        else:
            # Fallback to even splits
            split_indices = [i * n_layers // (num_splits + 1) for i in range(1, num_splits + 1)]
        
        return self._evaluate_split_configuration(split_indices, split_points, profile)
    
    def _optimize_splits_greedy(self, profile: ModelProfile, num_splits: int) -> ModelSplit:
        """Use greedy approach to find good splits quickly."""
        split_points = self._generate_split_points(profile)
        n_layers = len(profile.layer_profiles)
        
        # Get list of valid split indices
        valid_split_indices = [sp.layer_index for sp in split_points]
        
        if num_splits >= len(valid_split_indices):
            # Use all available split points
            split_indices = valid_split_indices[:num_splits]
            return self._evaluate_split_configuration(split_indices, split_points, profile)
        
        if not valid_split_indices:
            # No valid splits - return single shard
            return self._evaluate_split_configuration([], split_points, profile)
        
        # Start with evenly distributed splits from valid points
        step = len(valid_split_indices) // (num_splits + 1) if num_splits > 0 else 1
        split_indices = [valid_split_indices[i * step] for i in range(1, num_splits + 1) 
                        if i * step < len(valid_split_indices)]
        
        # Ensure we have exactly num_splits
        while len(split_indices) < num_splits and len(split_indices) < len(valid_split_indices):
            # Add more splits from remaining valid points
            remaining = [idx for idx in valid_split_indices if idx not in split_indices]
            if remaining:
                split_indices.append(remaining[0])
        
        split_indices = sorted(split_indices[:num_splits])
        best_split = self._evaluate_split_configuration(split_indices, split_points, profile)
        
        # Iteratively improve splits using only valid split points
        max_iterations = 20
        for iteration in range(max_iterations):
            improved = False
            
            for i in range(len(split_indices)):
                original_idx = split_indices[i]
                best_local_score = float('inf')
                best_local_idx = original_idx
                
                # Try other valid split points near current position
                current_pos = valid_split_indices.index(original_idx) if original_idx in valid_split_indices else 0
                window_size = max(1, len(valid_split_indices) // 10)
                
                for offset in range(-window_size, window_size + 1):
                    new_pos = current_pos + offset
                    
                    # Ensure valid bounds
                    if new_pos < 0 or new_pos >= len(valid_split_indices):
                        continue
                    
                    new_idx = valid_split_indices[new_pos]
                    
                    # Ensure no duplicates
                    if new_idx in split_indices[:i] + split_indices[i+1:]:
                        continue
                    
                    # Test this configuration
                    test_indices = split_indices[:i] + [new_idx] + split_indices[i+1:]
                    test_split = self._evaluate_split_configuration(test_indices, split_points, profile)
                    
                    # Score combines load balance and communication overhead
                    score = (self.load_balance_weight * test_split.load_balance_score + 
                            self.communication_weight * test_split.estimated_communication_overhead_ms / 1000.0)
                    
                    if score < best_local_score:
                        best_local_score = score
                        best_local_idx = new_idx
                
                # Update if improvement found
                if best_local_idx != original_idx:
                    split_indices[i] = best_local_idx
                    improved = True
            
            # Re-evaluate current best
            current_split = self._evaluate_split_configuration(split_indices, split_points, profile)
            
            # Check if overall improvement
            current_score = (self.load_balance_weight * current_split.load_balance_score + 
                           self.communication_weight * current_split.estimated_communication_overhead_ms / 1000.0)
            best_score = (self.load_balance_weight * best_split.load_balance_score + 
                         self.communication_weight * best_split.estimated_communication_overhead_ms / 1000.0)
            
            if current_score < best_score:
                best_split = current_split
            
            if not improved:
                break
        
        return best_split
    
    def find_optimal_splits(self, profile: ModelProfile, num_splits: int, 
                          method: str = "greedy") -> ModelSplit:
        """
        Find optimal split points for a model.
        
        Args:
            profile: Model profiling data
            num_splits: Number of splits to create (results in num_splits+1 shards)
            method: Optimization method ("greedy" or "dp")
        
        Returns:
            Optimal model split configuration
        """
        if num_splits <= 0:
            return self._evaluate_split_configuration([], self._generate_split_points(profile), profile)
        
        self.logger.info(f"Finding optimal splits for {profile.model_name} with {num_splits} splits using {method} method")
        
        if method == "dp" and num_splits <= 8:  # DP only for small number of splits
            result = self._optimize_splits_dynamic_programming(profile, num_splits)
        else:
            result = self._optimize_splits_greedy(profile, num_splits)
        
        self.logger.info(f"Found optimal split with load balance score: {result.load_balance_score:.4f}")
        self.logger.info(f"Estimated total time: {result.estimated_total_time_ms:.2f}ms")
        self.logger.info(f"Communication overhead: {result.estimated_communication_overhead_ms:.2f}ms")
        
        return result
    
    def create_pytorch_shards(self, model: nn.Module, split: ModelSplit, 
                            layer_names: List[str]) -> List[nn.Module]:
        """
        Create PyTorch Sequential modules for each shard based on split configuration.
        
        Args:
            model: Original PyTorch model
            split: Split configuration
            layer_names: List of layer names corresponding to model modules
        
        Returns:
            List of Sequential modules representing shards
        """
        # First, we need to build an ordered list of modules that matches the profiler's order
        # The profiler captures modules in execution order, so we need to preserve this
        
        self.logger.info(f"Creating shards from {len(layer_names)} profiled layers")
        self.logger.info(f"Split configuration has {len(split.shards)} shards")
        
        # Build ordered module list based on model architecture
        ordered_modules = self._extract_modules_in_order(model, layer_names)
        
        if len(ordered_modules) != len(layer_names):
            self.logger.warning(f"Module count mismatch: extracted {len(ordered_modules)}, expected {len(layer_names)}")
        
        # Get the full model structure to understand architectural boundaries
        model_structure = self._analyze_model_structure(model)
        
        shards = []
        for shard_idx, layer_indices in enumerate(split.shards):
            shard_modules = []
            
            self.logger.info(f"Creating shard {shard_idx} with layer indices {layer_indices[:5]}...")
            
            for layer_idx in layer_indices:
                if layer_idx < len(ordered_modules):
                    module = ordered_modules[layer_idx]
                    if module is not None:  # Skip None entries (functional ops)
                        shard_modules.append(module)
                        self.logger.debug(f"Added module at index {layer_idx} to shard {shard_idx}")
                else:
                    self.logger.warning(f"Layer index {layer_idx} out of range (max: {len(ordered_modules)-1})")
            
            self.logger.info(f"Shard {shard_idx} has {len(shard_modules)} modules")
            
            # Add architectural transition layers if needed
            shard_with_transitions = self._add_transition_layers(
                shard_modules, shard_idx, len(split.shards), model_structure, layer_indices, layer_names
            )
            
            if shard_with_transitions:
                shards.append(shard_with_transitions)
            else:
                self.logger.warning(f"Empty shard {shard_idx} created")
        
        self.logger.info(f"Created {len(shards)} shards from split configuration")
        return shards
    
    def _extract_modules_in_order(self, model: nn.Module, layer_names: List[str]) -> List[Optional[nn.Module]]:
        """
        Extract modules in the same order as they were profiled.
        Returns a list where each index corresponds to the profiled layer at that index.
        None entries indicate functional operations that don't have corresponding modules.
        """
        ordered_modules = []
        
        # Different extraction strategy based on model architecture
        model_type = model.__class__.__name__.lower()
        
        if hasattr(model, 'features') and hasattr(model, 'classifier'):
            # VGG/AlexNet style models
            self.logger.info(f"Extracting modules for features/classifier architecture")
            ordered_modules = self._extract_features_classifier_model(model, layer_names)
        elif hasattr(model, 'layer1') and hasattr(model, 'fc'):
            # ResNet style models
            self.logger.info(f"Extracting modules for ResNet-style architecture")
            ordered_modules = self._extract_resnet_model(model, layer_names)
        elif 'inception' in model_type:
            # Inception style models
            self.logger.info(f"Extracting modules for Inception-style architecture")
            ordered_modules = self._extract_inception_model(model, layer_names)
        else:
            # Generic extraction
            self.logger.info(f"Using generic module extraction")
            ordered_modules = self._extract_generic_model(model, layer_names)
        
        return ordered_modules
    
    def _extract_features_classifier_model(self, model: nn.Module, layer_names: List[str]) -> List[Optional[nn.Module]]:
        """Extract modules for models with features/classifier structure (VGG, AlexNet)."""
        ordered_modules = []
        features_modules = list(model.features.children())
        classifier_modules = list(model.classifier.children()) if hasattr(model.classifier, 'children') else [model.classifier]
        
        # Add avgpool if it exists
        has_avgpool = hasattr(model, 'avgpool')
        
        feature_idx = 0
        classifier_idx = 0
        avgpool_added = False
        
        for layer_name in layer_names:
            if 'functional.' in layer_name:
                ordered_modules.append(None)
            elif 'features' in layer_name and feature_idx < len(features_modules):
                ordered_modules.append(features_modules[feature_idx])
                feature_idx += 1
            elif 'avgpool' in layer_name and has_avgpool and not avgpool_added:
                ordered_modules.append(model.avgpool)
                avgpool_added = True
            elif 'classifier' in layer_name and classifier_idx < len(classifier_modules):
                ordered_modules.append(classifier_modules[classifier_idx])
                classifier_idx += 1
            else:
                # Try to find by name
                found = False
                for name, module in model.named_modules():
                    if name == layer_name or name.endswith('.' + layer_name):
                        ordered_modules.append(module)
                        found = True
                        break
                if not found:
                    self.logger.warning(f"Could not find module for layer: {layer_name}")
                    ordered_modules.append(None)
        
        return ordered_modules
    
    def _extract_resnet_model(self, model: nn.Module, layer_names: List[str]) -> List[Optional[nn.Module]]:
        """Extract modules for ResNet-style models."""
        ordered_modules = []
        
        # Build a mapping of all modules
        module_dict = {name: module for name, module in model.named_modules()}
        
        for layer_name in layer_names:
            if 'functional.' in layer_name:
                ordered_modules.append(None)
            else:
                # Try exact match first
                if layer_name in module_dict:
                    module = module_dict[layer_name]
                    # Only add leaf modules
                    if len(list(module.children())) == 0:
                        ordered_modules.append(module)
                    else:
                        # For container modules, we skip them as they're not actual computation
                        ordered_modules.append(None)
                else:
                    # Try to find a matching module
                    found = False
                    for name, module in module_dict.items():
                        if name.endswith('.' + layer_name) or name.endswith(layer_name):
                            if len(list(module.children())) == 0:
                                ordered_modules.append(module)
                                found = True
                                break
                    if not found:
                        self.logger.warning(f"Could not find module for layer: {layer_name}")
                        ordered_modules.append(None)
        
        return ordered_modules
    
    def _extract_inception_model(self, model: nn.Module, layer_names: List[str]) -> List[Optional[nn.Module]]:
        """Extract modules for Inception-style models."""
        ordered_modules = []
        module_dict = {name: module for name, module in model.named_modules()}
        
        for layer_name in layer_names:
            if 'functional.' in layer_name:
                ordered_modules.append(None)
            else:
                # Inception has complex nested structure
                if layer_name in module_dict:
                    module = module_dict[layer_name]
                    if len(list(module.children())) == 0:
                        ordered_modules.append(module)
                    else:
                        ordered_modules.append(None)
                else:
                    # Try partial matching
                    found = False
                    for name, module in module_dict.items():
                        if layer_name in name and len(list(module.children())) == 0:
                            ordered_modules.append(module)
                            found = True
                            break
                    if not found:
                        ordered_modules.append(None)
        
        return ordered_modules
    
    def _extract_generic_model(self, model: nn.Module, layer_names: List[str]) -> List[Optional[nn.Module]]:
        """Generic module extraction for unknown architectures."""
        ordered_modules = []
        all_modules = list(model.modules())
        module_dict = {name: module for name, module in model.named_modules()}
        
        for layer_name in layer_names:
            if 'functional.' in layer_name:
                ordered_modules.append(None)
            elif layer_name in module_dict:
                module = module_dict[layer_name]
                if len(list(module.children())) == 0:
                    ordered_modules.append(module)
                else:
                    ordered_modules.append(None)
            else:
                ordered_modules.append(None)
        
        return ordered_modules

    def _analyze_model_structure(self, model: nn.Module) -> Dict[str, Any]:
        """Analyze model structure to identify architectural boundaries."""
        structure = {
            'has_adaptive_pool': False,
            'has_flatten': False,
            'has_classifier': False,
            'feature_extractor_end': None,
            'classifier_start': None
        }
        
        # Look for common architectural patterns
        for name, module in model.named_modules():
            if isinstance(module, nn.AdaptiveAvgPool2d):
                structure['has_adaptive_pool'] = True
                structure['feature_extractor_end'] = name
            elif isinstance(module, nn.Flatten):
                structure['has_flatten'] = True
            elif 'classifier' in name.lower() and isinstance(module, (nn.Linear, nn.Sequential)):
                structure['has_classifier'] = True
                structure['classifier_start'] = name
        
        return structure
    
    def _add_transition_layers(self, shard_modules: List[nn.Module], shard_idx: int, 
                             total_shards: int, model_structure: Dict[str, Any],
                             layer_indices: List[int], layer_names: List[str]) -> nn.Module:
        """Add necessary transition layers between shards."""
        
        if not shard_modules:
            return nn.Sequential()  # Empty sequential for empty shards
        
        # For simple cases, just wrap in Sequential
        # The more complex ShardWithTransitions is causing issues
        return nn.Sequential(*shard_modules)
    
    def _shard_needs_explicit_transition(self, layer_indices: List[int], layer_names: List[str]) -> bool:
        """Check if this shard contains features→classifier transition and needs explicit layers."""
        if not layer_indices:
            return False
        
        has_features = False
        has_classifier = False
        
        for layer_idx in layer_indices:
            if layer_idx < len(layer_names):
                layer_name = layer_names[layer_idx]
                if 'features' in layer_name:
                    has_features = True
                elif 'classifier' in layer_name:
                    has_classifier = True
        
        # Return True if this shard contains both features and classifier layers
        return has_features and has_classifier


def create_simple_model_split(model: nn.Module, model_type: str, num_splits: int = 1) -> List[nn.Module]:
    """
    Create a simple but working split for models based on their architecture.
    This is a fallback when intelligent splitting fails.
    """
    if num_splits != 1:
        raise ValueError("Currently only supports num_splits=1 (creates 2 shards)")
    
    model_type = model_type.lower()
    
    # ResNet models
    if 'resnet' in model_type:
        # Split between layer2 and layer3
        shard1 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2
        )
        shard2 = nn.Sequential(
            model.layer3,
            model.layer4,
            model.avgpool,
            nn.Flatten(),
            model.fc
        )
        return [shard1, shard2]
    
    # VGG models
    elif 'vgg' in model_type:
        features = list(model.features.children())
        # Find split point after a pooling layer
        split_idx = len(features) // 2
        for i in range(split_idx, len(features)):
            if isinstance(features[i], nn.MaxPool2d):
                split_idx = i + 1
                break
        
        shard1 = nn.Sequential(*features[:split_idx])
        shard2 = nn.Sequential(
            *features[split_idx:],
            model.avgpool,
            nn.Flatten(),
            model.classifier
        )
        return [shard1, shard2]
    
    # AlexNet
    elif 'alexnet' in model_type:
        shard1 = model.features
        shard2 = nn.Sequential(
            model.avgpool,
            nn.Flatten(),
            model.classifier
        )
        return [shard1, shard2]
    
    # Inception models
    elif 'inception' in model_type:
        # This is more complex, collect modules
        modules = []
        mixed_modules = []
        
        for name, module in model.named_children():
            if name.startswith('Mixed'):
                mixed_modules.append(module)
            else:
                modules.append((name, module))
        
        # Split mixed modules in half
        if mixed_modules:
            split_point = len(mixed_modules) // 2
            
            # First shard: initial modules + first half of Mixed
            shard1_modules = []
            for name, mod in modules:
                if name in ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3', 
                           'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'maxpool1', 'maxpool2']:
                    shard1_modules.append(mod)
            shard1_modules.extend(mixed_modules[:split_point])
            
            # Second shard: remaining Mixed + final layers
            shard2_modules = mixed_modules[split_point:]
            for name, mod in modules:
                if name in ['avgpool', 'dropout', 'fc']:
                    if name == 'avgpool':
                        shard2_modules.append(nn.AdaptiveAvgPool2d((1, 1)))
                    elif name == 'fc':
                        shard2_modules.append(nn.Flatten())
                        shard2_modules.append(mod)
                    else:
                        shard2_modules.append(mod)
            
            shard1 = nn.Sequential(*shard1_modules)
            shard2 = nn.Sequential(*shard2_modules)
            return [shard1, shard2]
    
    # MobileNetV2
    elif 'mobilenet' in model_type:
        features = list(model.features.children())
        split_point = len(features) // 2
        
        shard1 = nn.Sequential(*features[:split_point])
        shard2 = nn.Sequential(
            *features[split_point:],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            model.classifier
        )
        return [shard1, shard2]
    
    # Fallback - just return the whole model
    else:
        raise ValueError(f"Unsupported model type for simple splitting: {model_type}")

def split_model_intelligently(model: nn.Module, profile: ModelProfile, 
                            num_splits: int, device_capabilities: Optional[Dict[str, Any]] = None,
                            network_config: Optional[Dict[str, float]] = None) -> Tuple[List[nn.Module], ModelSplit]:
    """
    Convenience function to split a model intelligently based on profiling data.
    
    Args:
        model: PyTorch model to split
        profile: Profiling data for the model
        num_splits: Number of splits to create
        device_capabilities: Optional device capability information
        network_config: Optional network configuration (latency, bandwidth)
    
    Returns:
        Tuple of (shard_modules, split_configuration)
    """
    # Configure splitter based on network conditions
    kwargs = {}
    if network_config:
        kwargs.update(network_config)
    
    splitter = IntelligentSplitter(**kwargs)
    
    # Find optimal split
    split = splitter.find_optimal_splits(profile, num_splits)
    
    # Create layer names list
    layer_names = [lp.layer_name for lp in profile.layer_profiles]
    
    # Create PyTorch shards
    shards = splitter.create_pytorch_shards(model, split, layer_names)
    
    return shards, split


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    print("Intelligent splitter module loaded successfully")