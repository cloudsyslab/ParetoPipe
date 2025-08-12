#!/usr/bin/env python3
"""
Single Device Benchmark for comparing against distributed inference.
Runs the same models and datasets on a single device to establish baseline performance.
"""

import torch
import torch.nn as nn
import time
import logging
import argparse
import os
import socket
from dotenv import load_dotenv
from typing import Dict, Any

from core.model_loader import ModelLoader
from metrics.enhanced_metrics import EnhancedMetricsCollector


class SingleDeviceBenchmark:
    """Single device benchmark runner."""
    
    def __init__(self, model_type: str, batch_size: int, num_classes: int = 10, 
                 models_dir: str = ".", metrics_dir: str = "./single_device_metrics"):
        """
        Initialize single device benchmark.
        
        Args:
            model_type: Type of model to benchmark
            batch_size: Batch size for inference
            num_classes: Number of output classes
            models_dir: Directory containing model weights
            metrics_dir: Directory for metrics output
        """
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.models_dir = models_dir
        self.metrics_dir = metrics_dir
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics collector (rank 0 for single device)
        self.metrics_collector = EnhancedMetricsCollector(0, metrics_dir, enable_realtime=True)
        
        # Load model
        self.model_loader = ModelLoader(models_dir)
        self.model = self.model_loader.load_model(model_type, num_classes)
        self.model.eval()
        
        self.logger.info(f"Single device benchmark initialized for {model_type}")
    
    def run_benchmark(self, dataset: str = "cifar10", num_test_samples: int = 64,
                     device: str = "cpu") -> Dict[str, Any]:
        """
        Run single device benchmark.
        
        Args:
            dataset: Dataset to use ("cifar10" or "dummy")
            num_test_samples: Number of samples to test
            device: Device to run on ("cpu" or "cuda")
            
        Returns:
            Dictionary with benchmark results
        """
        hostname = socket.gethostname()
        self.logger.info(f"=== Single Device Benchmark on {hostname} ===")
        self.logger.info(f"Model: {self.model_type}")
        self.logger.info(f"Loading dataset: {dataset}")
        self.logger.info(f"Device: {device}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Test samples: {num_test_samples}")
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Load dataset
        test_loader = self.model_loader.load_dataset(dataset, self.model_type, self.batch_size)
        self.logger.info(f"Dataset successfully loaded: {dataset} (batch_size={self.batch_size})")
        
        # Run inference
        self.logger.info("Starting single device inference...")
        start_time = time.time()
        
        total_images = 0
        num_correct = 0
        batch_count = 0
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_loader):
                if total_images >= num_test_samples:
                    break
                
                # Trim batch if necessary
                remaining = num_test_samples - total_images
                if images.size(0) > remaining:
                    images = images[:remaining]
                    labels = labels[:remaining]
                
                # Move data to device
                images = images.to(device)
                labels = labels.to(device)
                
                # Start batch tracking
                batch_start_time = self.metrics_collector.start_batch(batch_count, len(images))
                
                self.logger.info(f"Processing batch {batch_count + 1} with {len(images)} images")
                
                # Run inference (single device - no partitioning)
                batch_inference_start = time.time()
                
                # Record pipeline stage for consistency with distributed version
                stage_start = time.time()
                output = self.model(images)
                stage_end = time.time()
                
                # Record single stage metrics for consistency
                self.metrics_collector.record_pipeline_stage(
                    batch_id=batch_count,
                    stage_id=0,
                    stage_name="single_device_full_model",
                    start_time=stage_start,
                    end_time=stage_end,
                    input_size_bytes=images.numel() * images.element_size(),
                    output_size_bytes=output.numel() * output.element_size(),
                    queue_wait_time_ms=0.0
                )
                
                batch_inference_end = time.time()
                
                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                batch_correct = (predicted == labels).sum().item()
                num_correct += batch_correct
                total_images += len(images)
                
                batch_accuracy = (batch_correct / len(labels)) * 100.0
                
                # End batch tracking
                self.metrics_collector.end_batch(batch_count, accuracy=batch_accuracy)
                
                # Log batch performance
                batch_time_ms = (batch_inference_end - batch_inference_start) * 1000
                batch_ips = len(images) / (batch_time_ms / 1000) if batch_time_ms > 0 else 0.0
                
                self.logger.info(f"Batch {batch_count + 1} - Time: {batch_time_ms:.2f}ms, "
                               f"IPS: {batch_ips:.2f}, Accuracy: {batch_accuracy:.2f}%")
                
                batch_count += 1
        
        elapsed_time = time.time() - start_time
        final_accuracy = (num_correct / total_images) * 100.0 if total_images > 0 else 0.0
        overall_ips = total_images / elapsed_time if elapsed_time > 0 else 0.0
        
        # Log results
        self.logger.info(f"=== Single Device Results ===")
        self.logger.info(f"Total images processed: {total_images}")
        self.logger.info(f"Total time: {elapsed_time:.2f}s")
        self.logger.info(f"Final accuracy: {final_accuracy:.2f}%")
        self.logger.info(f"Overall throughput: {overall_ips:.2f} images/sec")
        
        # Finalize metrics
        final_results = self.metrics_collector.finalize(f"{self.model_type}_single_device")
        
        self.logger.info("=== Final Single Device Metrics Summary ===")
        device_summary = final_results['device_summary']
        efficiency_stats = final_results['efficiency_stats']
        
        self.logger.info(f"Images per second: {device_summary.get('images_per_second', 0):.2f}")
        self.logger.info(f"NEW Throughput (inter-batch): {efficiency_stats.get('new_pipeline_throughput_ips', 0):.2f} images/sec")
        self.logger.info(f"Average processing time: {device_summary.get('average_processing_time_ms', 0):.2f}ms")
        self.logger.info(f"Metrics saved to: {self.metrics_dir}")
        
        return {
            'hostname': hostname,
            'model_type': self.model_type,
            'dataset': dataset,
            'device': device,
            'batch_size': self.batch_size,
            'total_images': total_images,
            'elapsed_time': elapsed_time,
            'overall_ips': overall_ips,
            'final_accuracy': final_accuracy,
            'device_summary': device_summary,
            'efficiency_stats': efficiency_stats
        }


def main():
    """Main function for single device benchmark."""
    parser = argparse.ArgumentParser(description="Single Device Benchmark for DNN Inference")
    parser.add_argument("--model", type=str, default="mobilenetv2", 
                       choices=ModelLoader.list_supported_models(),
                       help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of output classes")
    parser.add_argument("--dataset", type=str, default="cifar10", 
                       choices=["cifar10", "dummy"], help="Dataset to use")
    parser.add_argument("--num-test-samples", type=int, default=64, help="Number of images to test")
    parser.add_argument("--metrics-dir", type=str, default="./single_device_metrics", 
                       help="Directory for metrics output")
    parser.add_argument("--models-dir", type=str, default="./models", 
                       help="Directory containing model weight files")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                       help="Device to run on")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(hostname)s:single_device] - %(message)s'
    )
    
    # Set hostname in log records
    hostname = socket.gethostname()
    old_factory = logging.getLogRecordFactory()
    
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.hostname = hostname
        return record
    
    logging.setLogRecordFactory(record_factory)
    
    # Load environment variables
    load_dotenv()
    
    # Run benchmark
    benchmark = SingleDeviceBenchmark(
        model_type=args.model,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        models_dir=args.models_dir,
        metrics_dir=args.metrics_dir
    )
    
    results = benchmark.run_benchmark(
        dataset=args.dataset,
        num_test_samples=args.num_test_samples,
        device=args.device
    )
    
    print(f"\n=== BENCHMARK COMPLETE ===")
    print(f"Host: {results['hostname']}")
    print(f"Model: {results['model_type']}")
    print(f"Dataset: {results['dataset']}")
    print(f"Device: {results['device']}")
    print(f"Throughput: {results['overall_ips']:.2f} images/sec")
    print(f"Accuracy: {results['final_accuracy']:.2f}%")
    print(f"Total Time: {results['elapsed_time']:.2f}s")


if __name__ == "__main__":
    main()