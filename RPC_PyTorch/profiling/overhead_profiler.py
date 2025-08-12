#!/usr/bin/env python3
"""
Overhead Profiler - Measures inter-layer overhead during model execution.

This module extends the layer profiling to capture the overhead between layers,
including memory allocation, data movement, and framework dispatch costs.
"""

import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
import numpy as np

from .layer_profiler import LayerProfiler, ModelProfile, LayerProfile


@dataclass
class OverheadProfile:
    """Profile data including overhead measurements."""
    layer_profile: LayerProfile
    overhead_before_ms: float  # Overhead before this layer executes
    overhead_after_ms: float   # Overhead after this layer executes
    sequential_time_ms: float  # Time when executed in sequence
    isolated_time_ms: float    # Time when executed in isolation
    
    @property
    def total_overhead_ms(self) -> float:
        """Total overhead for this layer."""
        return self.overhead_before_ms + self.overhead_after_ms
    
    @property
    def overhead_ratio(self) -> float:
        """Ratio of overhead to computation time."""
        if self.isolated_time_ms > 0:
            return self.total_overhead_ms / self.isolated_time_ms
        return 0.0


class OverheadProfiler:
    """
    Profiles model execution including inter-layer overhead.
    
    Measures both isolated layer execution and sequential execution to
    quantify the overhead between layers.
    """
    
    def __init__(self, device: str = "cpu", warmup_iterations: int = 5, 
                 profile_iterations: int = 20):
        self.device = device
        self.warmup_iterations = warmup_iterations
        self.profile_iterations = profile_iterations
        self.logger = logging.getLogger(__name__)
        
        # Use existing layer profiler for isolated measurements
        self.layer_profiler = LayerProfiler(
            device=device,
            warmup_iterations=warmup_iterations,
            profile_iterations=profile_iterations
        )
    
    def profile_with_overhead(self, model: nn.Module, input_shape: Tuple[int, ...],
                             batch_size: int = 1) -> List[OverheadProfile]:
        """
        Profile model with overhead measurements.
        
        Args:
            model: PyTorch model to profile
            input_shape: Input tensor shape (excluding batch dimension)
            batch_size: Batch size for profiling
            
        Returns:
            List of overhead profiles for each layer
        """
        model = model.to(self.device).eval()
        
        # First, get isolated layer profiles
        self.logger.info("Profiling layers in isolation...")
        isolated_profile = self.layer_profiler.profile_model(
            model, input_shape, batch_size
        )
        
        # Then, measure sequential execution with detailed timing
        self.logger.info("Profiling sequential execution with overhead tracking...")
        sequential_profiles = self._profile_sequential_execution(
            model, input_shape, batch_size, isolated_profile
        )
        
        # Combine results
        overhead_profiles = self._compute_overhead_profiles(
            isolated_profile, sequential_profiles
        )
        
        return overhead_profiles
    
    def _profile_sequential_execution(self, model: nn.Module, 
                                    input_shape: Tuple[int, ...],
                                    batch_size: int,
                                    isolated_profile: ModelProfile) -> Dict[str, Dict[str, float]]:
        """
        Profile sequential execution with fine-grained timing.
        
        Returns dict mapping layer names to timing information.
        """
        # Create hooks to measure timing at each layer
        timing_data = {}
        hook_handles = []
        
        def create_timing_hook(layer_name: str):
            def hook(module, input, output):
                if not hasattr(hook, 'timings'):
                    hook.timings = []
                
                # Record pre-execution time
                pre_time = time.perf_counter()
                
                # Force synchronization if using CUDA
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                # The actual forward pass happens here (between pre and post)
                
                # Record post-execution time
                post_time = time.perf_counter()
                
                hook.timings.append({
                    'pre_time': pre_time,
                    'post_time': post_time,
                    'layer_name': layer_name
                })
                
                timing_data[layer_name] = hook.timings
            
            return hook
        
        # Register hooks for all named modules
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                handle = module.register_forward_hook(create_timing_hook(name))
                hook_handles.append(handle)
        
        # Create test input
        test_input = torch.randn(batch_size, *input_shape).to(self.device)
        
        # Warmup runs
        for _ in range(self.warmup_iterations):
            with torch.no_grad():
                _ = model(test_input)
        
        # Clear timing data from warmup
        timing_data.clear()
        
        # Profile runs
        all_sequential_times = []
        
        for _ in range(self.profile_iterations):
            # Clear previous iteration data
            for key in timing_data:
                timing_data[key] = []
            
            # Measure total execution time
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model(test_input)
            
            if self.device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            total_time = (end_time - start_time) * 1000  # Convert to ms
            all_sequential_times.append(total_time)
        
        # Remove hooks
        for handle in hook_handles:
            handle.remove()
        
        # Process timing data to compute per-layer sequential times
        sequential_profiles = self._process_sequential_timing(
            timing_data, isolated_profile
        )
        
        # Add total execution time info
        sequential_profiles['_total_time_ms'] = {
            'mean': np.mean(all_sequential_times),
            'std': np.std(all_sequential_times)
        }
        
        return sequential_profiles
    
    def _process_sequential_timing(self, timing_data: Dict[str, List[Dict]],
                                  isolated_profile: ModelProfile) -> Dict[str, Dict[str, float]]:
        """Process raw timing data to compute overhead."""
        processed = {}
        
        # Get ordered list of layers from isolated profile
        layer_order = [lp.layer_name for lp in isolated_profile.layer_profiles]
        
        # For each iteration, compute inter-layer gaps
        for layer_name in layer_order:
            if layer_name not in timing_data:
                continue
            
            layer_timings = timing_data[layer_name]
            if not layer_timings:
                continue
            
            # Compute execution time in sequential context
            exec_times = []
            for timing in layer_timings:
                # This is a simplified calculation
                # In practice, we'd need more sophisticated hook placement
                exec_time = (timing['post_time'] - timing['pre_time']) * 1000
                exec_times.append(exec_time)
            
            processed[layer_name] = {
                'sequential_time_ms': np.mean(exec_times),
                'sequential_std_ms': np.std(exec_times)
            }
        
        return processed
    
    def _compute_overhead_profiles(self, isolated_profile: ModelProfile,
                                  sequential_profiles: Dict[str, Dict[str, float]]) -> List[OverheadProfile]:
        """Combine isolated and sequential profiles to compute overhead."""
        overhead_profiles = []
        
        total_isolated_time = sum(lp.execution_time_ms for lp in isolated_profile.layer_profiles)
        total_sequential_time = sequential_profiles.get('_total_time_ms', {}).get('mean', total_isolated_time)
        total_overhead = total_sequential_time - total_isolated_time
        
        # Distribute overhead proportionally (simplified approach)
        for layer_prof in isolated_profile.layer_profiles:
            sequential_time = sequential_profiles.get(
                layer_prof.layer_name, {}
            ).get('sequential_time_ms', layer_prof.execution_time_ms)
            
            # Simple overhead distribution - can be made more sophisticated
            layer_overhead_ratio = layer_prof.execution_time_ms / total_isolated_time
            layer_overhead = total_overhead * layer_overhead_ratio
            
            overhead_profile = OverheadProfile(
                layer_profile=layer_prof,
                overhead_before_ms=layer_overhead * 0.5,  # Split overhead
                overhead_after_ms=layer_overhead * 0.5,   # before/after
                sequential_time_ms=sequential_time,
                isolated_time_ms=layer_prof.execution_time_ms
            )
            
            overhead_profiles.append(overhead_profile)
        
        return overhead_profiles
    
    def analyze_overhead(self, overhead_profiles: List[OverheadProfile]) -> Dict[str, Any]:
        """Analyze overhead patterns from profiles."""
        total_computation = sum(op.isolated_time_ms for op in overhead_profiles)
        total_overhead = sum(op.total_overhead_ms for op in overhead_profiles)
        total_sequential = sum(op.sequential_time_ms for op in overhead_profiles)
        
        # Find layers with highest overhead
        sorted_by_overhead = sorted(
            overhead_profiles,
            key=lambda x: x.total_overhead_ms,
            reverse=True
        )
        
        analysis = {
            'total_computation_ms': total_computation,
            'total_overhead_ms': total_overhead,
            'total_sequential_ms': total_sequential,
            'overhead_percentage': (total_overhead / total_computation) * 100,
            'top_overhead_layers': [
                {
                    'layer_name': op.layer_profile.layer_name,
                    'overhead_ms': op.total_overhead_ms,
                    'overhead_ratio': op.overhead_ratio * 100
                }
                for op in sorted_by_overhead[:5]
            ]
        }
        
        return analysis


def measure_split_overhead(model: nn.Module, split_point: int,
                          input_shape: Tuple[int, ...], 
                          device: str = "cpu",
                          batch_size: int = 1) -> Dict[str, float]:
    """
    Measure overhead for a specific model split.
    
    Compares sum of isolated shard times vs actual split execution time.
    """
    model = model.to(device).eval()
    
    # Create shards
    layers = list(model.children())
    shard1 = nn.Sequential(*layers[:split_point])
    shard2 = nn.Sequential(*layers[split_point:])
    
    # Profile shards individually
    profiler = LayerProfiler(device=device)
    
    shard1_profile = profiler.profile_model(shard1, input_shape, batch_size)
    
    # Get intermediate shape for shard2
    test_input = torch.randn(batch_size, *input_shape).to(device)
    with torch.no_grad():
        intermediate = shard1(test_input)
    intermediate_shape = intermediate.shape[1:]  # Remove batch dimension
    
    shard2_profile = profiler.profile_model(shard2, intermediate_shape, batch_size)
    
    # Calculate theoretical times (sum of layers)
    shard1_theoretical = sum(lp.execution_time_ms for lp in shard1_profile.layer_profiles)
    shard2_theoretical = sum(lp.execution_time_ms for lp in shard2_profile.layer_profiles)
    
    # Measure actual execution times
    warmup = 5
    iterations = 20
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            intermediate = shard1(test_input)
            _ = shard2(intermediate)
    
    # Measure shard1
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()
    
    for _ in range(iterations):
        with torch.no_grad():
            intermediate = shard1(test_input)
    
    torch.cuda.synchronize() if device == "cuda" else None
    shard1_actual = (time.perf_counter() - start) / iterations * 1000
    
    # Measure shard2
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()
    
    for _ in range(iterations):
        with torch.no_grad():
            _ = shard2(intermediate)
    
    torch.cuda.synchronize() if device == "cuda" else None
    shard2_actual = (time.perf_counter() - start) / iterations * 1000
    
    return {
        'shard1_theoretical_ms': shard1_theoretical,
        'shard1_actual_ms': shard1_actual,
        'shard1_overhead_ms': shard1_actual - shard1_theoretical,
        'shard1_overhead_percent': ((shard1_actual - shard1_theoretical) / shard1_theoretical) * 100,
        'shard2_theoretical_ms': shard2_theoretical,
        'shard2_actual_ms': shard2_actual,
        'shard2_overhead_ms': shard2_actual - shard2_theoretical,
        'shard2_overhead_percent': ((shard2_actual - shard2_theoretical) / shard2_theoretical) * 100,
    }


if __name__ == "__main__":
    # Example usage
    import torchvision.models as models
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    model = models.mobilenet_v2(weights=None)
    
    # Profile with overhead
    profiler = OverheadProfiler(device="cpu")
    overhead_profiles = profiler.profile_with_overhead(
        model, 
        input_shape=(3, 224, 224),
        batch_size=1
    )
    
    # Analyze results
    analysis = profiler.analyze_overhead(overhead_profiles)
    
    print("\n=== Overhead Analysis ===")
    print(f"Total computation time: {analysis['total_computation_ms']:.2f} ms")
    print(f"Total overhead: {analysis['total_overhead_ms']:.2f} ms")
    print(f"Overhead percentage: {analysis['overhead_percentage']:.1f}%")
    
    print("\nTop 5 layers by overhead:")
    for layer in analysis['top_overhead_layers']:
        print(f"  {layer['layer_name']}: {layer['overhead_ms']:.3f} ms ({layer['overhead_ratio']:.1f}% overhead)")
    
    # Test split overhead
    print("\n=== Split Overhead Analysis ===")
    split_overhead = measure_split_overhead(
        model, 
        split_point=10,
        input_shape=(3, 224, 224),
        device="cpu"
    )
    
    print(f"Shard 1: {split_overhead['shard1_theoretical_ms']:.2f} ms (theoretical) vs "
          f"{split_overhead['shard1_actual_ms']:.2f} ms (actual) = "
          f"{split_overhead['shard1_overhead_percent']:.1f}% overhead")
    print(f"Shard 2: {split_overhead['shard2_theoretical_ms']:.2f} ms (theoretical) vs "
          f"{split_overhead['shard2_actual_ms']:.2f} ms (actual) = "
          f"{split_overhead['shard2_overhead_percent']:.1f}% overhead")