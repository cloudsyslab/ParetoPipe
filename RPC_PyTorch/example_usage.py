#!/usr/bin/env python3
"""
Example usage of the enhanced distributed inference system.
This script demonstrates key features and provides templates for research use.
"""

import sys
import os
sys.path.append('.')

from profiling import LayerProfiler, IntelligentSplitter, split_model_intelligently
from metrics import EnhancedMetricsCollector
from core import ModelLoader
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_model_profiling():
    """Example: Profile a model to understand computational costs."""
    logger.info("=== Model Profiling Example ===")
    
    # Load a model
    model_loader = ModelLoader("./models")  # Models directory for weight files
    model = model_loader.load_model("mobilenetv2", num_classes=10)
    sample_input = model_loader.get_sample_input("mobilenetv2", batch_size=1)
    
    # Profile the model
    profiler = LayerProfiler(device="cpu", warmup_iterations=2, profile_iterations=5)
    profile = profiler.profile_model(model, sample_input, "mobilenetv2_example")
    
    # Print profiling results
    logger.info(f"Model: {profile.model_name}")
    logger.info(f"Total execution time: {profile.total_time_ms:.2f}ms")
    logger.info(f"Total parameters: {profile.total_parameters:,}")
    logger.info(f"Total FLOPs: {profile.total_flops:,}")
    
    # Show top 5 most expensive layers
    sorted_layers = sorted(profile.layer_profiles, key=lambda x: x.execution_time_ms, reverse=True)
    logger.info("\nTop 5 most expensive layers:")
    for i, layer in enumerate(sorted_layers[:5]):
        logger.info(f"  {i+1}. {layer.layer_name}: {layer.execution_time_ms:.2f}ms ({layer.layer_type})")
    
    return profile


def example_intelligent_splitting(profile):
    """Example: Use intelligent splitting to find optimal split points."""
    logger.info("\n=== Intelligent Splitting Example ===")
    
    # Create intelligent splitter
    splitter = IntelligentSplitter(
        communication_latency_ms=5.0,    # Typical Pi network latency
        network_bandwidth_mbps=100.0,    # WiFi bandwidth
        load_balance_weight=0.7,         # Prioritize load balancing
        communication_weight=0.3         # But consider communication costs
    )
    
    # Find optimal splits for different numbers of partitions
    for num_splits in [2, 3, 4]:
        logger.info(f"\nOptimal split for {num_splits} partitions:")
        
        split_config = splitter.find_optimal_splits(profile, num_splits)
        
        logger.info(f"  Load balance score: {split_config.load_balance_score:.4f}")
        logger.info(f"  Estimated total time: {split_config.estimated_total_time_ms:.2f}ms")
        logger.info(f"  Communication overhead: {split_config.estimated_communication_overhead_ms:.2f}ms")
        
        # Show split points
        if split_config.split_points:
            logger.info("  Split points:")
            for i, sp in enumerate(split_config.split_points):
                logger.info(f"    Split {i+1}: after {sp.layer_name} (cost: {sp.split_cost:.2f}ms)")


def example_metrics_collection():
    """Example: Use enhanced metrics collection."""
    logger.info("\n=== Enhanced Metrics Collection Example ===")
    
    # Create metrics collector
    metrics = EnhancedMetricsCollector(rank=0, output_dir="./example_metrics")
    
    # Simulate processing batches
    for batch_id in range(3):
        # Start batch
        metrics.start_batch(batch_id, batch_size=8)
        
        # Simulate pipeline stages
        import time
        for stage_id in range(2):
            stage_start = time.time()
            time.sleep(0.1)  # Simulate processing
            stage_end = time.time()
            
            metrics.record_pipeline_stage(
                batch_id=batch_id,
                stage_id=stage_id,
                stage_name=f"example_stage_{stage_id}",
                start_time=stage_start,
                end_time=stage_end,
                input_size_bytes=1024 * 1024,  # 1MB
                output_size_bytes=1024 * 1024
            )
        
        # Simulate network metrics
        metrics.record_network_metrics(latency_ms=5.2, throughput_mbps=95.3)
        
        # End batch
        metrics.end_batch(batch_id, accuracy=85.5 + batch_id * 2.1)
    
    # Get summary
    device_summary = metrics.get_device_summary()
    pipeline_stats = metrics.get_pipeline_efficiency_stats()
    
    logger.info(f"Device IPS: {device_summary['images_per_second']:.2f}")
    logger.info(f"Average processing time: {device_summary['average_processing_time_ms']:.2f}ms")
    logger.info(f"Pipeline utilization: {pipeline_stats['average_pipeline_utilization']:.2f}")
    
    # Finalize
    results = metrics.finalize("example_model")
    logger.info(f"Metrics saved to: {results['csv_files']['device_metrics']}")


def example_compare_splitting_methods():
    """Example: Compare traditional vs intelligent splitting."""
    logger.info("\n=== Splitting Methods Comparison ===")
    
    # This would require implementing traditional splitting comparison
    # For now, we'll show the concept
    
    model_loader = ModelLoader("./models")
    model = model_loader.load_model("resnet18", num_classes=10)
    sample_input = model_loader.get_sample_input("resnet18", batch_size=1)
    
    # Profile model
    profiler = LayerProfiler(device="cpu", warmup_iterations=1, profile_iterations=3)
    profile = profiler.profile_model(model, sample_input, "resnet18_comparison")
    
    # Traditional splitting (even layer distribution)
    traditional_splits = []
    n_layers = len(profile.layer_profiles)
    num_splits = 3
    
    for i in range(1, num_splits + 1):
        split_idx = i * n_layers // (num_splits + 1)
        traditional_splits.append(split_idx)
    
    # Intelligent splitting
    splitter = IntelligentSplitter()
    intelligent_split = splitter.find_optimal_splits(profile, num_splits)
    
    # Compare results
    logger.info("Traditional splitting (even layers):")
    logger.info(f"  Split points at layers: {traditional_splits}")
    
    logger.info("Intelligent splitting:")
    logger.info(f"  Load balance score: {intelligent_split.load_balance_score:.4f}")
    logger.info(f"  Estimated total time: {intelligent_split.estimated_total_time_ms:.2f}ms")
    
    if intelligent_split.split_points:
        intelligent_indices = [sp.layer_index for sp in intelligent_split.split_points]
        logger.info(f"  Split points at layers: {intelligent_indices}")


def main():
    """Run all examples."""
    logger.info("Starting Enhanced Distributed Inference Examples")
    
    try:
        # Example 1: Model profiling
        profile = example_model_profiling()
        
        # Example 2: Intelligent splitting
        example_intelligent_splitting(profile)
        
        # Example 3: Enhanced metrics
        example_metrics_collection()
        
        # Example 4: Compare methods
        example_compare_splitting_methods()
        
        logger.info("\n=== All Examples Completed Successfully ===")
        
    except Exception as e:
        logger.error(f"Error in examples: {e}", exc_info=True)


if __name__ == "__main__":
    main()