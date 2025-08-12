#!/usr/bin/env python3
"""
Final summary test and demonstration of the enhanced distributed inference system.
"""

import sys
import os
sys.path.append('.')
sys.path.append('..')

import logging
from profiling import LayerProfiler, IntelligentSplitter
from metrics import EnhancedMetricsCollector  
from core import ModelLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demonstrate_enhanced_features():
    """Demonstrate all enhanced features working together."""
    logger.info("🚀 Enhanced Distributed Inference System - Feature Demonstration")
    logger.info("=" * 70)
    
    # 1. Load model
    logger.info("1️⃣  Loading MobileNetV2 model...")
    model_loader = ModelLoader("./models")
    model = model_loader.load_model("mobilenetv2", num_classes=10)
    sample_input = model_loader.get_sample_input("mobilenetv2", batch_size=1)
    logger.info(f"   ✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 2. Profile model
    logger.info("2️⃣  Profiling model layers for computational costs...")
    profiler = LayerProfiler(device="cpu", warmup_iterations=1, profile_iterations=3)
    profile = profiler.profile_model(model, sample_input, "demo_mobilenetv2")
    
    # Save profile
    os.makedirs("./demo_profiles", exist_ok=True)
    profile.save_to_json("./demo_profiles/mobilenetv2_profile.json")
    
    logger.info(f"   ✓ Profiled {len(profile.layer_profiles)} layers")
    logger.info(f"   ✓ Total execution time: {profile.total_time_ms:.2f}ms")
    logger.info(f"   ✓ Profile saved to: ./demo_profiles/mobilenetv2_profile.json")
    
    # Find most expensive layers
    sorted_layers = sorted(profile.layer_profiles, key=lambda x: x.execution_time_ms, reverse=True)
    logger.info("   📊 Top 3 most expensive layers:")
    for i, layer in enumerate(sorted_layers[:3]):
        logger.info(f"      {i+1}. {layer.layer_name}: {layer.execution_time_ms:.2f}ms ({layer.flops:,} FLOPs)")
    
    # 3. Intelligent splitting
    logger.info("3️⃣  Finding optimal split points...")
    splitter = IntelligentSplitter(
        communication_latency_ms=5.0,   # Pi network latency
        network_bandwidth_mbps=100.0,   # WiFi bandwidth
        load_balance_weight=0.7,        # Prioritize load balance
        communication_weight=0.3        # Consider communication cost
    )
    
    results = {}
    for num_splits in [2, 3, 4]:
        split_config = splitter.find_optimal_splits(profile, num_splits)
        results[num_splits] = split_config
        
        logger.info(f"   🔄 {num_splits} splits:")
        logger.info(f"      Load balance score: {split_config.load_balance_score:.4f}")
        logger.info(f"      Communication overhead: {split_config.estimated_communication_overhead_ms:.2f}ms")
        logger.info(f"      Estimated total time: {split_config.estimated_total_time_ms:.2f}ms")
    
    # Find best split configuration
    best_splits = min(results.keys(), key=lambda k: results[k].load_balance_score)
    logger.info(f"   🏆 Best configuration: {best_splits} splits (score: {results[best_splits].load_balance_score:.4f})")
    
    # 4. Enhanced metrics collection
    logger.info("4️⃣  Collecting enhanced metrics...")
    metrics = EnhancedMetricsCollector(rank=0, output_dir="./demo_metrics", enable_realtime=True)
    
    # Simulate inference batches
    import time
    import torch
    
    dummy_loader = model_loader.load_dataset("dummy", "mobilenetv2", batch_size=4)
    
    for batch_id, (images, labels) in enumerate(dummy_loader):
        if batch_id >= 3:  # Process 3 batches
            break
            
        # Start batch tracking
        metrics.start_batch(batch_id, len(images))
        
        # Simulate pipeline stages
        stage_start = time.time()
        with torch.no_grad():
            output = model(images)
        stage_end = time.time()
        
        # Record pipeline stage
        metrics.record_pipeline_stage(
            batch_id=batch_id,
            stage_id=0,
            stage_name="full_model",
            start_time=stage_start,
            end_time=stage_end,
            input_size_bytes=images.numel() * images.element_size(),
            output_size_bytes=output.numel() * output.element_size()
        )
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        accuracy = (predicted == labels).sum().item() / len(labels) * 100.0
        
        # Record network metrics (simulated)
        metrics.record_network_metrics(latency_ms=5.2, throughput_mbps=95.3)
        
        # End batch
        metrics.end_batch(batch_id, accuracy=accuracy)
        
        logger.info(f"   📊 Batch {batch_id + 1}: {accuracy:.1f}% accuracy")
    
    # Get final metrics
    device_summary = metrics.get_device_summary()
    pipeline_stats = metrics.get_pipeline_efficiency_stats()
    
    logger.info("5️⃣  Results Summary:")
    logger.info(f"   🎯 Images per second: {device_summary['images_per_second']:.2f}")
    logger.info(f"   ⏱️  Average processing time: {device_summary['average_processing_time_ms']:.2f}ms")
    logger.info(f"   🔄 Pipeline utilization: {pipeline_stats['average_pipeline_utilization']:.2f}")
    logger.info(f"   🌐 Network latency: {device_summary['avg_network_latency_ms']:.2f}ms")
    logger.info(f"   📈 Network throughput: {device_summary['avg_throughput_mbps']:.2f}Mbps")
    
    # Finalize metrics
    results = metrics.finalize("demo_mobilenetv2")
    
    logger.info("6️⃣  Output Files Generated:")
    for csv_type, csv_path in results['csv_files'].items():
        size = os.path.getsize(csv_path)
        logger.info(f"   📄 {csv_type}: {csv_path} ({size} bytes)")
    
    return True


def show_system_capabilities():
    """Show what the enhanced system can do."""
    logger.info("\n🎯 Enhanced Distributed Inference System Capabilities")
    logger.info("=" * 60)
    
    capabilities = [
        "🧠 Layer-by-layer profiling with execution time, memory, and FLOPs",
        "⚡ Intelligent model splitting based on computational costs",
        "🚀 Sequential pipelining for overlapping execution",
        "📊 Comprehensive metrics: per-device IPS, pipeline efficiency",
        "🌐 Network metrics: latency, throughput, communication overhead",
        "📈 Real-time system monitoring with configurable intervals",
        "💾 CSV export for research analysis and visualization",
        "🔧 Modular architecture for easy extension and customization",
        "🏗️  Support for 6 model architectures (MobileNetV2, ResNet18, etc.)",
        "🔄 Both local threading and distributed RPC execution modes"
    ]
    
    for capability in capabilities:
        logger.info(f"   {capability}")
    
    logger.info("\n🔬 Research Applications:")
    research_apps = [
        "📐 Optimal model partitioning across heterogeneous devices",
        "⚖️  Load balancing efficiency in distributed systems", 
        "🌉 Communication vs computation trade-off analysis",
        "🔀 Pipeline optimization and bottleneck identification",
        "📱 Edge computing performance characterization",
        "🔋 Energy efficiency analysis (with power monitoring)"
    ]
    
    for app in research_apps:
        logger.info(f"   {app}")


def main():
    """Run the demonstration."""
    logger.info("Enhanced Distributed DNN Inference System - Test Summary")
    logger.info("=" * 65)
    
    try:
        # Show capabilities
        show_system_capabilities()
        
        logger.info("\n" + "=" * 65)
        
        # Run demonstration
        success = demonstrate_enhanced_features()
        
        if success:
            logger.info("\n" + "=" * 65)
            logger.info("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 65)
            logger.info("The enhanced distributed inference system is ready for research!")
            logger.info("\nNext steps:")
            logger.info("1. Run with multiple nodes using RPC for distributed testing")
            logger.info("2. Compare intelligent vs traditional splitting performance")
            logger.info("3. Enable pipelining for overlapping execution")
            logger.info("4. Analyze the generated CSV files for insights")
            logger.info("=" * 65)
        
        return success
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)