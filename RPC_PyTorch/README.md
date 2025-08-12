# Enhanced Distributed DNN Inference System

This directory contains an advanced distributed deep neural network inference system designed for heterogeneous edge devices. The system features intelligent model splitting based on computational profiling, true sequential pipelining, and comprehensive metrics collection.

## Performance Results

Our pipelined implementation achieves **13.85 images/sec** on a Raspberry Pi cluster, which is **6.38x faster** than our sequential baseline (2.17 images/sec). The key breakthrough was implementing true async RPC pipelining with multiple batches in flight and finding that balanced workload splits (60/40) significantly outperform heavily imbalanced splits. See [experiment_results.md](experiment_results.md) for detailed performance analysis.

## Features

### 🧠 Intelligent Model Splitting
- **Layer-by-layer profiling**: Measures actual computational costs, memory usage, and FLOPs for each layer
- **Optimal split point detection**: Uses dynamic programming and greedy algorithms to find splits that balance load and minimize communication overhead
- **Hardware-aware splitting**: Considers device capabilities and network characteristics

### 🚀 Sequential Pipelining
- **Overlapping execution**: Multiple batches can be processed simultaneously across different pipeline stages
- **Configurable pipeline depth**: Control the number of concurrent batches in the pipeline
- **Both local and RPC modes**: Support for threaded local processing or distributed RPC-based execution

### 📊 Comprehensive Metrics Collection
- **Per-device performance**: Images per second, processing time, resource utilization
- **Pipeline efficiency**: Utilization, bubble time, stage-wise throughput
- **Network metrics**: Latency, throughput, communication overhead
- **Real-time monitoring**: Continuous system monitoring with configurable intervals

## Directory Structure

```
optimized_distributed_inference/
├── core/
│   ├── __init__.py
│   └── model_loader.py          # Model loading and dataset utilities
├── profiling/
│   ├── __init__.py
│   ├── layer_profiler.py        # Layer-by-layer profiling system
│   └── intelligent_splitter.py  # Intelligent splitting algorithms
├── metrics/
│   ├── __init__.py
│   └── enhanced_metrics.py      # Comprehensive metrics collection
├── pipelining/
│   ├── __init__.py
│   └── pipeline_manager.py      # Pipeline management system
├── models/                      # Pre-trained model weights
│   ├── mobilenetv2_cifar10.pth
│   ├── resnet18_cifar10.pth
│   ├── inception_cifar10.pth
│   └── resnet50_cifar10.pth
├── distributed_runner.py       # Main distributed script
├── single_device.py             # Single device baseline
├── test_*.py                    # Test scripts
├── example_usage.py             # Usage examples
├── .env.example                 # Environment configuration template
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Usage

### Basic Usage

```bash
# Master node (rank 0)
python distributed_runner.py \
    --rank 0 \
    --world-size 3 \
    --model mobilenetv2 \
    --batch-size 8 \
    --num-partitions 2 \
    --use-intelligent-splitting \
    --metrics-dir ./enhanced_metrics

# Worker nodes (rank 1, 2, ...)
python distributed_runner.py \
    --rank 1 \
    --world-size 3 \
    --model mobilenetv2
```

### Advanced Features

```bash
# Enable pipelining for overlapping execution
python distributed_runner.py \
    --rank 0 \
    --world-size 3 \
    --model resnet18 \
    --use-intelligent-splitting \
    --use-pipelining \
    --num-test-samples 100

# Use traditional splitting (disable intelligent features)
python distributed_runner.py \
    --rank 0 \
    --world-size 3 \
    --model alexnet \
    --disable-intelligent-splitting

# Test with dummy data for development
python distributed_runner.py \
    --rank 0 \
    --world-size 2 \
    --model resnet50 \
    --dataset dummy \
    --num-test-samples 50
```

## Key Improvements Over Original System

### 1. Intelligent Splitting
- **Before**: Split models by equal layer count, leading to unbalanced computational loads
- **After**: Profile actual layer costs and split based on computational complexity and communication overhead

### 2. Pipeline Processing
- **Before**: Sequential processing - each image waits for complete pipeline traversal
- **After**: Overlapping execution - multiple images processed simultaneously across pipeline stages

### 3. Enhanced Metrics
- **Before**: Basic timing and accuracy metrics
- **After**: Comprehensive per-device IPS, pipeline utilization, network metrics, and real-time monitoring

### 4. Modular Architecture
- **Before**: Monolithic script with tightly coupled components
- **After**: Modular design with separate profiling, metrics, and pipelining systems

## Metrics Output

The system generates several CSV files with detailed metrics:

- `batch_metrics_*.csv`: Per-batch performance and pipeline utilization
- `device_metrics_*.csv`: Per-device performance summaries
- `pipeline_metrics_*.csv`: Detailed pipeline stage metrics
- `profiles/*.json`: Layer-by-layer profiling data

## Research Applications

This system is ideal for research into:

1. **Optimal Model Partitioning**: Finding the best split points for different model architectures
2. **Edge Computing Efficiency**: Measuring performance across heterogeneous devices
3. **Communication vs Computation Trade-offs**: Analyzing the balance between processing and network costs
4. **Pipeline Optimization**: Studying the effectiveness of pipelined execution in distributed systems

## Supported Models

- MobileNetV2
- InceptionV3  
- AlexNet
- ResNet18
- ResNet50
- VGG16

All models are pre-configured for CIFAR-10 classification but can be easily adapted for other datasets.

## Dependencies

The system requires the same dependencies as the original system plus:
- `numpy` (for mathematical operations in profiling)
- `psutil` (for system monitoring)

## Configuration

Environment variables (edit `.env` file):
```bash
MASTER_ADDR=192.168.1.100    # IP of master node
MASTER_PORT=29500            # RPC communication port  
GLOO_SOCKET_IFNAME=eth0      # Network interface name
TP_SOCKET_IFNAME=eth0        # TensorPipe socket interface
CIFAR10_PATH=/path/to/cifar10  # Optional: CIFAR-10 dataset path
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the root `optimized_distributed_inference` directory
2. **RPC Connection Failures**: Check network connectivity and firewall settings
3. **Memory Issues**: Reduce batch size or number of concurrent pipeline batches
4. **Profiling Errors**: Ensure model weights are available in the models directory

### Performance Tips

1. **For CPU-only devices**: Use smaller batch sizes (4-8) and fewer pipeline stages
2. **For heterogeneous setups**: Enable intelligent splitting to balance load across different device capabilities
3. **For high-latency networks**: Increase pipeline depth to hide communication costs
4. **For debugging**: Start with dummy dataset and small test samples

## Contributing

When adding new features:

1. Follow the modular architecture
2. Add comprehensive logging
3. Include metrics collection for new components
4. Update this README with usage examples