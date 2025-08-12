#!/usr/bin/env python3
"""
Payload Size Measurement Utility

This utility measures the actual serialized payload sizes for network transmission,
implementing the same approach as your labmate's research but for PyTorch RPC.

Usage:
    from utils.payload_measurement import PayloadMeasurer
    
    # Measure actual payload size before RPC transmission
    size_info = PayloadMeasurer.measure_rpc_payload_size(tensor, batch_id)
    actual_bytes = size_info['torch_serialized_bytes']
"""

import torch
import pickle
import io
from typing import Any, Dict, Optional


class PayloadMeasurer:
    """Utility class for measuring actual serialized payload sizes."""
    
    @staticmethod
    def measure_torch_serialization_size(data: Any) -> int:
        """
        Measure size using PyTorch's serialization (what RPC uses internally).
        
        Args:
            data: The data to serialize (tensor, dict, etc.)
            
        Returns:
            Size in bytes of the serialized data
        """
        buffer = io.BytesIO()
        torch.save(data, buffer)
        return buffer.tell()
    
    @staticmethod 
    def measure_pickle_serialization_size(data: Any) -> int:
        """
        Measure size using pickle serialization (like your labmate's approach).
        
        Args:
            data: The data to serialize
            
        Returns:
            Size in bytes of the pickled data
        """
        return len(pickle.dumps(data))
    
    @staticmethod
    def measure_rpc_payload_size(tensor: torch.Tensor, batch_id: Optional[int] = None) -> Dict[str, int]:
        """
        Measure the actual size of RPC payload that will be transmitted.
        This is the research-grade measurement approach.
        
        Args:
            tensor: The tensor being transmitted
            batch_id: Optional batch identifier
            
        Returns:
            Dictionary with different size measurements:
            - torch_serialized_bytes: PyTorch serialization size (most accurate for RPC)
            - pickle_serialized_bytes: Pickle serialization size (for comparison)
            - raw_tensor_bytes: Raw tensor memory size (current buggy approach)
        """
        # Create the payload that would be transmitted via RPC
        payload = {
            'tensor': tensor,
            'batch_id': batch_id
        }
        
        # Measure with different serialization methods
        torch_size = PayloadMeasurer.measure_torch_serialization_size(payload)
        pickle_size = PayloadMeasurer.measure_pickle_serialization_size(payload)
        
        # Also measure raw tensor size for comparison with current approach
        raw_size = tensor.numel() * tensor.element_size()
        
        return {
            'torch_serialized_bytes': torch_size,
            'pickle_serialized_bytes': pickle_size, 
            'raw_tensor_bytes': raw_size,
            'serialization_overhead': torch_size / raw_size if raw_size > 0 else 1.0
        }
    
    @staticmethod
    def calculate_network_throughput_mbps(payload_size_bytes: int, network_time_ms: float) -> float:
        """
        Calculate network throughput like your labmate's approach.
        
        Args:
            payload_size_bytes: Actual bytes transmitted
            network_time_ms: Network transmission time in milliseconds
            
        Returns:
            Network throughput in Mbps
        """
        if network_time_ms <= 0:
            return 0.0
        
        # Convert to MB and seconds, then to Mbps
        size_mb = payload_size_bytes / (1024 * 1024)
        time_s = network_time_ms / 1000.0
        return (size_mb * 8) / time_s


def create_enhanced_forward_method(original_forward_method):
    """
    Decorator to enhance a forward method with payload size measurement.
    
    Args:
        original_forward_method: The original forward method to enhance
        
    Returns:
        Enhanced method that returns both result and payload size info
    """
    def enhanced_forward(self, x: torch.Tensor, batch_id: Optional[int] = None):
        import time
        import logging
        import socket
        
        logger = logging.getLogger(__name__)
        
        # Measure input payload size (what was actually transmitted to this shard)
        input_size_info = PayloadMeasurer.measure_rpc_payload_size(x, batch_id)
        
        logger.info(f"[PAYLOAD_MEASUREMENT] [{socket.gethostname()}] Shard {getattr(self, 'shard_id', 'unknown')} "
                   f"input payload: torch_serialized={input_size_info['torch_serialized_bytes']} bytes "
                   f"({input_size_info['torch_serialized_bytes']/(1024*1024):.3f} MB), "
                   f"overhead={input_size_info['serialization_overhead']:.2f}x")
        
        # Run the original forward pass and measure compute time
        start_time = time.time()
        output = original_forward_method(self, x, batch_id)
        end_time = time.time()
        compute_time_ms = (end_time - start_time) * 1000
        
        # Measure output payload size
        output_size_info = PayloadMeasurer.measure_rpc_payload_size(output, batch_id)
        
        # Return enhanced result with size information
        return {
            'result': output,
            'input_payload_sizes': input_size_info,
            'output_payload_sizes': output_size_info,
            'compute_time_ms': compute_time_ms,
            'shard_id': getattr(self, 'shard_id', None)
        }
    
    return enhanced_forward


# Example usage and testing functions
if __name__ == "__main__":
    def test_payload_measurement():
        """Test the payload measurement functionality."""
        print("=== Payload Measurement Test ===")
        
        # Create test tensors of different sizes
        test_cases = [
            ("Small tensor (8, 3, 32, 32)", torch.randn(8, 3, 32, 32)),
            ("Medium tensor (8, 64, 56, 56)", torch.randn(8, 64, 56, 56)),
            ("Large tensor (8, 256, 28, 28)", torch.randn(8, 256, 28, 28)),
        ]
        
        for name, tensor in test_cases:
            print(f"\n{name}:")
            print(f"  Tensor shape: {tensor.shape}")
            
            size_info = PayloadMeasurer.measure_rpc_payload_size(tensor, batch_id=0)
            
            print(f"  Raw tensor size: {size_info['raw_tensor_bytes']:,} bytes")
            print(f"  PyTorch serialized: {size_info['torch_serialized_bytes']:,} bytes")
            print(f"  Pickle serialized: {size_info['pickle_serialized_bytes']:,} bytes")
            print(f"  Serialization overhead: {size_info['serialization_overhead']:.2f}x")
            
            # Test throughput calculation
            network_time_ms = 100  # Example network time
            throughput = PayloadMeasurer.calculate_network_throughput_mbps(
                size_info['torch_serialized_bytes'], network_time_ms
            )
            print(f"  Throughput (100ms network): {throughput:.2f} Mbps")
    
    test_payload_measurement()