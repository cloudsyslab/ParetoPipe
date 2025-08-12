#!/usr/bin/env python3
"""
Enhanced layer-by-layer profiling system that captures both named modules and functional operations.
This module profiles individual layers AND functional operations (like adaptive_avg_pool2d, flatten)
to determine accurate computational costs, memory usage, and optimal split points for distributed inference.
"""

import time
import torch
import torch.nn as nn
import torch.profiler
import psutil
import logging
import os
from typing import Dict, List, Tuple, Any, Callable, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class LayerProfile:
    """Profile data for a single layer or functional operation."""
    layer_name: str
    layer_type: str
    execution_time_ms: float
    memory_usage_mb: float
    flops: int
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    parameters: int
    cpu_utilization: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'layer_name': self.layer_name,
            'layer_type': self.layer_type,
            'execution_time_ms': self.execution_time_ms,
            'memory_usage_mb': self.memory_usage_mb,
            'flops': self.flops,
            'input_shape': list(self.input_shape),
            'output_shape': list(self.output_shape),
            'parameters': self.parameters,
            'cpu_utilization': self.cpu_utilization
        }


@dataclass
class ModelProfile:
    """Complete profile for a model including functional operations."""
    model_name: str
    layer_profiles: List[LayerProfile]
    total_time_ms: float
    total_memory_mb: float
    total_flops: int
    total_parameters: int
    block_profiles: Optional[Dict[int, Dict[str, Any]]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_name': self.model_name,
            'layer_profiles': [lp.to_dict() for lp in self.layer_profiles],
            'total_time_ms': self.total_time_ms,
            'total_memory_mb': self.total_memory_mb,
            'total_flops': self.total_flops,
            'total_parameters': self.total_parameters,
            'block_profiles': self.block_profiles if self.block_profiles else {}
        }
    
    def save_to_file(self, filepath: str):
        """Save profile to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def save_block_times_to_csv(self, filepath: str):
        """Save block-level execution times to CSV file."""
        import csv
        
        if not self.block_profiles:
            return
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['block_id', 'execution_time_ms', 'memory_usage_mb', 'flops', 'parameters'])
            
            # Write data for each block
            for block_id in sorted(self.block_profiles.keys()):
                bp = self.block_profiles[block_id]
                writer.writerow([
                    block_id,
                    f"{bp['execution_time_ms']:.4f}",
                    f"{bp['memory_usage_mb']:.4f}",
                    bp['flops'],
                    bp['parameters']
                ])


class LayerProfiler:
    """Enhanced profiler that captures both named modules and functional operations."""
    
    def __init__(self, device: str = "cpu", warmup_iterations: int = 10, profile_iterations: int = 25):
        """
        Initialize the enhanced profiler.
        
        Args:
            device: Device to run profiling on ("cpu" or "cuda")
            warmup_iterations: Number of warmup iterations (default: 10, matching GAPP)
            profile_iterations: Number of profiling iterations for averaging (default: 25, matching GAPP)
        """
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.profile_iterations = profile_iterations
        self.logger = logging.getLogger(__name__)
        
        # Check if CUDA is available and warn if using GPU
        if self.device == "cuda" and torch.cuda.is_available():
            self.logger.info(f"Using GPU for profiling: {torch.cuda.get_device_name(0)}")
        elif self.device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        
        # Model-specific functional operation definitions
        self.model_functional_ops = {
            'mobilenetv2': self._get_mobilenetv2_functional_ops,
            'mobilenet_v2': self._get_mobilenetv2_functional_ops,
            'resnet18': self._get_resnet_functional_ops,
            'resnet': self._get_resnet_functional_ops,
        }
    
    def _get_mobilenetv2_functional_ops(self) -> List[Dict[str, Any]]:
        """Define functional operations for MobileNetV2."""
        return [
            {
                'name': 'adaptive_avg_pool2d',
                'layer_name': 'functional.adaptive_avg_pool2d',
                'operation': lambda x: torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)),
                'insert_after': 'features.18.2',  # Last ReLU6 in features
                'flops_calc': lambda inp, out: int(inp.numel())  # Read each input element once
            },
            {
                'name': 'flatten',
                'layer_name': 'functional.flatten',
                'operation': lambda x: torch.flatten(x, 1),
                'insert_after': 'functional.adaptive_avg_pool2d',
                'flops_calc': lambda inp, out: 0  # Reshape operation, no FLOPs
            }
        ]
    
    def _get_resnet_functional_ops(self) -> List[Dict[str, Any]]:
        """Define functional operations for ResNet models."""
        return [
            {
                'name': 'adaptive_avg_pool2d',
                'layer_name': 'functional.adaptive_avg_pool2d',
                'operation': lambda x: torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)),
                'insert_after': 'layer4.1.relu',  # Last ReLU in ResNet18
                'flops_calc': lambda inp, out: int(inp.numel())
            },
            {
                'name': 'flatten',
                'layer_name': 'functional.flatten', 
                'operation': lambda x: torch.flatten(x, 1),
                'insert_after': 'functional.adaptive_avg_pool2d',
                'flops_calc': lambda inp, out: 0
            }
        ]
    
    def _calculate_flops(self, module: nn.Module, input_tensor: torch.Tensor, output_tensor: torch.Tensor) -> int:
        """Calculate FLOPs for a given module."""
        if isinstance(module, nn.Conv2d):
            # Conv2D: FLOPs = batch_size * output_height * output_width * kernel_height * kernel_width * input_channels * output_channels
            batch_size, in_channels, in_height, in_width = input_tensor.shape
            out_channels, out_height, out_width = output_tensor.shape[1], output_tensor.shape[2], output_tensor.shape[3]
            kernel_ops = module.kernel_size[0] * module.kernel_size[1] * in_channels
            output_elements = batch_size * out_height * out_width * out_channels
            flops = kernel_ops * output_elements + (output_elements if module.bias is not None else 0)
            return int(flops)
            
        elif isinstance(module, nn.Linear):
            # Linear: FLOPs = batch_size * input_features * output_features + bias_term
            batch_size = input_tensor.shape[0]
            flops = batch_size * module.in_features * module.out_features + (module.out_features if module.bias is not None else 0)
            return int(flops)
            
        elif isinstance(module, nn.BatchNorm2d):
            # BatchNorm: FLOPs ≈ 2 * num_elements (normalize + scale+shift)
            return int(2 * output_tensor.numel())
            
        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.ELU, nn.LeakyReLU)):
            # Activation functions: FLOPs ≈ num_elements
            return int(output_tensor.numel())
            
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            # Pooling: FLOPs ≈ num_output_elements * kernel_area
            if hasattr(module, 'kernel_size'):
                if isinstance(module.kernel_size, int):
                    kernel_area = module.kernel_size ** 2
                else:
                    kernel_area = module.kernel_size[0] * module.kernel_size[1]
                return int(output_tensor.numel() * kernel_area)
            else:
                # AdaptiveAvgPool
                return int(output_tensor.numel())
        else:
            # Conservative estimate for unknown layers
            return int(output_tensor.numel())
    
    def profile_layer(self, module: nn.Module, input_tensor: torch.Tensor, layer_name: str) -> LayerProfile:
        """Profile a single layer or module."""
        module.eval()
        module = module.to(self.device)
        input_tensor = input_tensor.to(self.device)
        
        # CPU-specific optimizations for Raspberry Pi
        if self.device == "cpu":
            # Set number of threads for better performance on Pi
            torch.set_num_threads(4)  # Raspberry Pi 4 has 4 cores
            # Disable gradient computation for better performance
            torch.set_grad_enabled(False)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = module(input_tensor)
        
        # Clear cache
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Profile execution time
        execution_times = []
        memory_usages = []
        cpu_utilizations = []
        
        for _ in range(self.profile_iterations):
            # Measure memory before
            memory_before = psutil.virtual_memory().used / (1024 * 1024)
            cpu_before = psutil.cpu_percent()
            
            # Ensure GPU operations are complete before timing
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            # Time the execution using high-precision timer
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = module(input_tensor)
            
            # Ensure GPU operations are complete after execution
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            # Measure memory after
            memory_after = psutil.virtual_memory().used / (1024 * 1024)
            cpu_after = psutil.cpu_percent()
            
            execution_times.append((end_time - start_time) * 1000)  # Convert to ms
            memory_usages.append(max(0, memory_after - memory_before))
            cpu_utilizations.append(max(0, cpu_after - cpu_before))
        
        # Calculate averages
        avg_execution_time = sum(execution_times) / len(execution_times)
        avg_memory_usage = sum(memory_usages) / len(memory_usages)
        avg_cpu_utilization = sum(cpu_utilizations) / len(cpu_utilizations)
        
        # Calculate FLOPs
        with torch.no_grad():
            sample_output = module(input_tensor)
        flops = self._calculate_flops(module, input_tensor, sample_output)
        
        # Count parameters
        parameters = sum(p.numel() for p in module.parameters())
        
        return LayerProfile(
            layer_name=layer_name,
            layer_type=type(module).__name__,
            execution_time_ms=avg_execution_time,
            memory_usage_mb=avg_memory_usage,
            flops=flops,
            input_shape=tuple(input_tensor.shape),
            output_shape=tuple(sample_output.shape),
            parameters=parameters,
            cpu_utilization=avg_cpu_utilization
        )
    
    def profile_functional_operation(self, operation_func: Callable, input_tensor: torch.Tensor, 
                                   operation_name: str, layer_name: str, flops_calc: Callable = None) -> LayerProfile:
        """Profile a functional operation."""
        input_tensor = input_tensor.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_iterations):
                _ = operation_func(input_tensor)
        
        # Profile execution time
        execution_times = []
        memory_usages = []
        
        for _ in range(self.profile_iterations):
            # Measure memory before (for consistency with layer profiling)
            memory_before = psutil.virtual_memory().used / (1024 * 1024)
            
            # Ensure GPU operations are complete before timing
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = operation_func(input_tensor)
            
            # Ensure GPU operations are complete after execution
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            # Measure memory after
            memory_after = psutil.virtual_memory().used / (1024 * 1024)
            
            execution_times.append((end_time - start_time) * 1000)  # Convert to ms
            memory_usages.append(max(0, memory_after - memory_before))
        
        avg_execution_time = sum(execution_times) / len(execution_times)
        avg_memory_usage = sum(memory_usages) / len(memory_usages) if memory_usages else 0.0
        
        # Calculate FLOPs
        if flops_calc:
            flops = flops_calc(input_tensor, output)
        else:
            flops = int(output.numel())  # Conservative estimate
        
        return LayerProfile(
            layer_name=layer_name,
            layer_type=operation_name,
            execution_time_ms=avg_execution_time,
            memory_usage_mb=avg_memory_usage,  # Now tracking memory for functional ops too
            flops=flops,
            input_shape=tuple(input_tensor.shape),
            output_shape=tuple(output.shape),
            parameters=0,  # No learnable parameters
            cpu_utilization=0.0  # Skip CPU measurement for functional ops
        )
    
    def profile_model(self, model: nn.Module, sample_input: torch.Tensor, model_name: str) -> ModelProfile:
        """Profile all layers and functional operations in a model."""
        self.logger.info(f"Starting enhanced profiling for model: {model_name}")
        
        model.eval()
        layer_profiles = []
        
        # Step 1: Profile all named modules with hooks
        intermediate_outputs = self._collect_intermediate_outputs(model, sample_input)
        
        # Step 2: Profile each named module
        for name, module in model.named_modules():
            if len(list(module.children())) == 0 and name in intermediate_outputs:  # Leaf modules only
                input_tensor, output_tensor = intermediate_outputs[name]
                
                try:
                    profile = self.profile_layer(module, input_tensor, name)
                    layer_profiles.append(profile)
                    self.logger.info(f"Profiled layer {name}: {profile.execution_time_ms:.2f}ms, {profile.memory_usage_mb:.2f}MB")
                except Exception as e:
                    self.logger.warning(f"Failed to profile layer {name}: {e}")
        
        # Step 3: Add missing functional operations
        layer_profiles = self._add_functional_operations(model, layer_profiles, sample_input, model_name)
        
        # Step 4: Aggregate block-level profiles
        import re
        block_profiles = {}
        current_block_id = -1
        current_block_time = 0.0
        current_block_memory = 0.0
        current_block_flops = 0
        current_block_params = 0
        
        for lp in layer_profiles:
            block_match = re.search(r'features\.(\d+)', lp.layer_name)
            block_id = int(block_match.group(1)) if block_match else -1
            
            if block_id != current_block_id and current_block_id != -1:
                # Save the previous block
                block_profiles[current_block_id] = {
                    'execution_time_ms': current_block_time,
                    'memory_usage_mb': current_block_memory,
                    'flops': current_block_flops,
                    'parameters': current_block_params
                }
                # Reset for new block
                current_block_time = 0.0
                current_block_memory = 0.0
                current_block_flops = 0
                current_block_params = 0
            
            current_block_id = block_id
            current_block_time += lp.execution_time_ms
            current_block_memory += lp.memory_usage_mb
            current_block_flops += lp.flops
            current_block_params += lp.parameters
        
        # Don't forget the last block
        if current_block_id != -1:
            block_profiles[current_block_id] = {
                'execution_time_ms': current_block_time,
                'memory_usage_mb': current_block_memory,
                'flops': current_block_flops,
                'parameters': current_block_params
            }
        
        # Log block-level summary
        if block_profiles:
            self.logger.info("\nBlock-level execution summary:")
            for block_id in sorted(block_profiles.keys()):
                bp = block_profiles[block_id]
                self.logger.info(f"  Block {block_id}: {bp['execution_time_ms']:.2f}ms, "
                               f"{bp['parameters']:,} params")
        
        # Calculate totals
        total_time = sum(lp.execution_time_ms for lp in layer_profiles)
        total_memory = sum(lp.memory_usage_mb for lp in layer_profiles)
        total_flops = sum(lp.flops for lp in layer_profiles)
        total_parameters = sum(lp.parameters for lp in layer_profiles)
        
        self.logger.info(f"Completed enhanced profiling for {model_name}: {len(layer_profiles)} operations profiled")
        
        # Create enhanced ModelProfile with block data
        profile = ModelProfile(
            model_name=model_name,
            layer_profiles=layer_profiles,
            total_time_ms=total_time,
            total_memory_mb=total_memory,
            total_flops=total_flops,
            total_parameters=total_parameters
        )
        
        # Add block profiles as an attribute
        profile.block_profiles = block_profiles
        
        return profile
    
    def _collect_intermediate_outputs(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Collect intermediate outputs from all named modules."""
        intermediate_outputs = {}
        handles = []
        
        def make_hook(name):
            def hook_fn(module, input, output):
                intermediate_outputs[name] = (input[0].clone(), output.clone())
            return hook_fn
        
        # Register hooks for all named modules  
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                handle = module.register_forward_hook(make_hook(name))
                handles.append(handle)
        
        # Run forward pass to collect intermediate outputs
        with torch.no_grad():
            _ = model(sample_input)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        return intermediate_outputs
    
    def save_layer_timings_gapp_style(self, profile: ModelProfile, output_dir: str, device: str):
        """Save individual layer timings in GAPP-style format."""
        for layer_profile in profile.layer_profiles:
            # Create GAPP-style identifier
            shape_suffix = "x".join(map(str, layer_profile.input_shape[1:]))  # Skip batch dimension
            block_identifier = f"{profile.model_name}_{layer_profile.layer_name.replace('.', '_')}_input{shape_suffix}"
            
            # Create timing data in GAPP format
            timing_data = {
                "block_identifier": block_identifier,
                "target_execution_time_ms": layer_profile.execution_time_ms,
                "target_device": device,
                "example_input_shape": list(layer_profile.input_shape),
                # Additional metrics beyond GAPP
                "memory_usage_mb": layer_profile.memory_usage_mb,
                "flops": layer_profile.flops,
                "parameters": layer_profile.parameters
            }
            
            # Save to file
            filename = f"{block_identifier}_time.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(timing_data, f, indent=2)
    
    def _add_functional_operations(self, model: nn.Module, layer_profiles: List[LayerProfile], 
                                 sample_input: torch.Tensor, model_name: str) -> List[LayerProfile]:
        """Add missing functional operations based on model architecture."""
        model_key = model_name.lower()
        
        # Find matching functional operations definition
        functional_ops_getter = None
        for key, getter in self.model_functional_ops.items():
            if key in model_key:
                functional_ops_getter = getter
                break
        
        if not functional_ops_getter:
            self.logger.info(f"No functional operations defined for {model_name}")
            return layer_profiles
        
        functional_ops = functional_ops_getter()
        
        # Build a map of layer names to indices
        layer_name_to_idx = {profile.layer_name: i for i, profile in enumerate(layer_profiles)}
        
        # Insert functional operations in reverse order to maintain indices
        for func_op in reversed(functional_ops):
            insert_after = func_op['insert_after']
            
            # Find insertion point
            insert_idx = -1
            if insert_after in layer_name_to_idx:
                insert_idx = layer_name_to_idx[insert_after] + 1
            else:
                # Try to find by partial match
                for layer_name, idx in layer_name_to_idx.items():
                    if insert_after in layer_name:
                        insert_idx = idx + 1
                        break
            
            if insert_idx > 0:
                # Get input tensor for this operation
                if insert_after.startswith('functional.'):
                    # Previous functional operation - use its output shape
                    prev_layer = layer_profiles[insert_idx - 1]
                    input_tensor = torch.randn(*prev_layer.output_shape)
                else:
                    # Named module - use its output
                    prev_layer = layer_profiles[insert_idx - 1]
                    input_tensor = torch.randn(*prev_layer.output_shape)
                
                # Profile the functional operation
                try:
                    func_profile = self.profile_functional_operation(
                        operation_func=func_op['operation'],
                        input_tensor=input_tensor,
                        operation_name=func_op['name'],
                        layer_name=func_op['layer_name'],
                        flops_calc=func_op['flops_calc']
                    )
                    
                    # Insert the functional operation
                    layer_profiles.insert(insert_idx, func_profile)
                    
                    # Update the layer name to index mapping
                    layer_name_to_idx = {profile.layer_name: i for i, profile in enumerate(layer_profiles)}
                    
                    self.logger.info(f"Added functional operation: {func_op['layer_name']} after {insert_after}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to profile functional operation {func_op['name']}: {e}")
        
        return layer_profiles


def profile_model_enhanced(model: nn.Module, sample_input: torch.Tensor, model_name: str,
                         device: str = "cpu", save_path: str = None, output_dir: str = None,
                         save_block_csv: bool = True) -> ModelProfile:
    """
    Convenience function to profile a model with enhanced functional operation detection.
    
    Args:
        model: PyTorch model to profile
        sample_input: Sample input tensor
        model_name: Name of the model
        device: Device to run profiling on
        save_path: Optional path to save the profile (overrides output_dir)
        output_dir: Optional directory for GAPP-style output naming
        save_block_csv: Whether to save block execution times to CSV (default: True)
    
    Returns:
        ModelProfile with complete profiling data
    """
    profiler = LayerProfiler(device=device)
    profile = profiler.profile_model(model, sample_input, model_name)
    
    if save_path:
        profile.save_to_file(save_path)
        # Also save block times CSV in the same directory
        if save_block_csv and profile.block_profiles:
            csv_path = save_path.replace('.json', '_block_times.csv')
            profile.save_block_times_to_csv(csv_path)
            print(f"Block execution times saved to: {csv_path}")
    elif output_dir:
        # Use GAPP-style naming convention
        os.makedirs(output_dir, exist_ok=True)
        # Save overall profile
        profile_path = os.path.join(output_dir, f"{model_name}_profile_{device}.json")
        profile.save_to_file(profile_path)
        
        # Save block times CSV
        if save_block_csv and profile.block_profiles:
            csv_path = os.path.join(output_dir, f"{model_name}_block_times_{device}.csv")
            profile.save_block_times_to_csv(csv_path)
            print(f"Block execution times saved to: {csv_path}")
        
        # Save individual layer timings in GAPP format
        profiler.save_layer_timings_gapp_style(profile, output_dir, device)
    
    return profile


def profile_for_pi(model: nn.Module, sample_input: torch.Tensor, model_name: str,
                   save_dir: str = "pi_profiles", batch_sizes: List[int] = [1, 4, 8]):
    """
    Profile a model specifically for Raspberry Pi deployment.
    
    Args:
        model: PyTorch model to profile
        sample_input: Sample input tensor (batch size will be adjusted)
        model_name: Name of the model
        save_dir: Directory to save Pi-specific profiles
        batch_sizes: List of batch sizes to profile
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Force CPU for Pi profiling
    profiler = LayerProfiler(device="cpu", warmup_iterations=5, profile_iterations=10)
    
    for batch_size in batch_sizes:
        # Adjust input for batch size
        input_shape = list(sample_input.shape)
        input_shape[0] = batch_size
        batch_input = torch.randn(input_shape)
        
        print(f"\nProfiling {model_name} with batch size {batch_size} for Raspberry Pi...")
        profile = profiler.profile_model(model, batch_input, model_name)
        
        # Save with Pi-specific naming
        filename = f"{model_name}_pi_profile_batch{batch_size}.json"
        filepath = os.path.join(save_dir, filename)
        profile.save_to_file(filepath)
        
        # Save block times CSV
        if profile.block_profiles:
            csv_filename = f"{model_name}_pi_block_times_batch{batch_size}.csv"
            csv_filepath = os.path.join(save_dir, csv_filename)
            profile.save_block_times_to_csv(csv_filepath)
            print(f"Saved block times to: {csv_filepath}")
        
        print(f"Saved Pi profile to: {filepath}")
        print(f"Total execution time: {profile.total_time_ms:.2f}ms")
        print(f"Throughput: {batch_size * 1000 / profile.total_time_ms:.2f} images/sec")


if __name__ == "__main__":
    # Example usage
    import torchvision.models as models
    
    logging.basicConfig(level=logging.INFO)
    
    # Test with MobileNetV2
    model = models.mobilenet_v2(weights=None)
    sample_input = torch.randn(1, 3, 224, 224)
    
    # Standard profiling
    profile = profile_model_enhanced(model, sample_input, "mobilenetv2", 
                                   output_dir="profiling_results", save_block_csv=True)
    print(f"Enhanced profiling complete: {len(profile.layer_profiles)} operations profiled")
    
    # Display block execution times
    if profile.block_profiles:
        print("\nBlock execution times:")
        print(f"{'Block':>6} | {'Time (ms)':>12} | {'Memory (MB)':>12} | {'Parameters':>12}")
        print("-" * 50)
        for block_id in sorted(profile.block_profiles.keys()):
            bp = profile.block_profiles[block_id]
            print(f"{block_id:>6} | {bp['execution_time_ms']:>12.4f} | {bp['memory_usage_mb']:>12.4f} | {bp['parameters']:>12,}")
    
    # Pi-specific profiling
    profile_for_pi(model, sample_input, "mobilenetv2")