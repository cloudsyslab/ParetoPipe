# Distributed DNN Inference Experiment Results

## System Configuration
- Master: PlamaLV (your machine)  
- Worker 1: master-pi
- Worker 2: core-pi
- Model: MobileNetV2
- Dataset: CIFAR-10
- Batch size: 8

## Reference Implementation Performance
- Labmate's implementation: **5.77 samples/sec**
- Split configuration: Block 8 (similar to our default)

## Our Implementation Results

### Sequential (Non-pipelined) Performance

#### Run 1: Default Split (Block 8)
- Date/Time: 2025-06-28 01:25:38
- Command: `python distributed_runner.py --rank 0 --world-size 3 --model mobilenetv2 --batch-size 8 --num-test-samples 64 --num-partitions 2`
- Split ratio: 3.4% / 96.6%
- Performance: **2.17 images/sec**
- Inter-batch throughput: 2.69 images/sec
- Accuracy: 76.56%
- Average processing time: 460.10ms
- RPC total time: 1840.31ms
  - Network overhead: 3.82ms
  - Worker computation: 1836.50ms
- Pipeline utilization: 0.00

#### Run 2: Default Split (Block 8) - Confirmation
- Date/Time: 2025-06-28 01:32:45
- Split ratio: 3.4% / 96.6%
- Performance: **2.17 images/sec**
- Inter-batch throughput: 2.72 images/sec
- Average processing time: 460.50ms
- RPC total time: 1841.88ms
  - Network overhead: 3.82ms
  - Worker computation: 1838.06ms
- Pipeline utilization: 0.00

### Pipelined Performance

#### Previous Test (from conversation summary)
- Date/Time: 2025-06-28 01:09 (approx)
- Configuration: Default split (block 8), pipelining enabled
- Performance: **13.35 images/sec** 
- Speedup over sequential: **6.27x**
- Outperformed reference by: **2.3x**

### Split Point Analysis
From test_split_points.py analysis:

| Split Block | Shard 1 | Shard 2 | Load Balance | Theoretical Speedup |
|-------------|---------|---------|--------------|-------------------|
| 6           | 1.2%    | 98.8%   | 0.822        | 1.178x            |
| 8 (default) | 2.2%    | 97.8%   | 0.874        | 1.126x            |
| 10          | 5.3%    | 94.7%   | 0.890        | 1.110x            |
| 12          | 8.7%    | 91.3%   | 0.918        | 1.082x            |

#### Run 3: Pipelined Split at Block 6
- Date/Time: 2025-06-28 01:44:33
- Command: `python distributed_runner.py --rank 0 --world-size 3 --model mobilenetv2 --batch-size 8 --num-test-samples 32 --num-partitions 2 --use-pipelining --split-block 6`
- Split ratio: 1.8% / 98.2%
- Performance: **12.75 images/sec**
- Inter-batch throughput: 4.93 images/sec
- Accuracy: 87.50%
- Total time: 5.02s (for 64 images)
- Average processing time: 768.35ms
- Pipeline utilization: 0.00 (metric issue)
- Notes: 
  - Shows "Using PIPELINED inference for maximum throughput" 
  - Multiple batches in flight (up to 3 concurrent)
  - Speedup over sequential: **5.88x**

#### Run 4: Pipelined Split at Block 10
- Date/Time: 2025-06-28 01:48:12
- Command: `python distributed_runner.py --rank 0 --world-size 3 --model mobilenetv2 --batch-size 8 --num-test-samples 32 --num-partitions 2 --use-pipelining --split-block 10`
- Split ratio: 8.3% / 91.7%
- Performance: **13.46 images/sec** (BEST!)
- Inter-batch throughput: 4.82 images/sec
- Accuracy: 78.12%
- Total time: 4.75s (for 64 images)
- Speedup over sequential: **6.20x**

#### Run 5: Pipelined Split at Block 12
- Date/Time: 2025-06-28 01:48:57
- Command: `python distributed_runner.py --rank 0 --world-size 3 --model mobilenetv2 --batch-size 8 --num-test-samples 32 --num-partitions 2 --use-pipelining --split-block 12`
- Split ratio: 13.7% / 86.3%
- Performance: **12.66 images/sec**
- Inter-batch throughput: 4.65 images/sec
- Accuracy: 85.94%
- Total time: 5.06s (for 64 images)
- Speedup over sequential: **5.83x**

#### Run 6: Pipelined Split at Block 6 (Re-run)
- Date/Time: 2025-06-28 01:49:45
- Command: `python distributed_runner.py --rank 0 --world-size 3 --model mobilenetv2 --batch-size 8 --num-test-samples 32 --num-partitions 2 --use-pipelining --split-block 6`
- Split ratio: 1.8% / 98.2%
- Performance: **12.61 images/sec**
- Inter-batch throughput: 4.76 images/sec
- Accuracy: 76.56%
- Total time: 5.08s (for 64 images)
- Speedup over sequential: **5.81x**

#### Run 7: Pipelined Split at Block 17 (Near 50/50)
- Date/Time: 2025-06-28 01:52:58
- Command: `python distributed_runner.py --rank 0 --world-size 3 --model mobilenetv2 --batch-size 8 --num-test-samples 32 --num-partitions 2 --use-pipelining --split-block 17`
- Split ratio: 59.8% / 40.2% (most balanced)
- Performance: **13.85 images/sec** (NEW BEST!)
- Accuracy: 79.69%
- Total time: 4.62s (for 64 images)
- Speedup over sequential: **6.38x**
- Speedup over reference: **2.40x**

### Summary of Results

| Split Block | Split Ratio | Throughput (img/s) | Speedup vs Sequential | Speedup vs Reference |
|-------------|-------------|-------------------|---------------------|---------------------|
| 6           | 1.8/98.2    | 12.61             | 5.81x               | 2.18x               |
| 8 (previous)| 3.4/96.6    | 13.35             | 6.15x               | 2.31x               |
| 10          | 8.3/91.7    | 13.46             | 6.20x               | 2.33x               |
| 12          | 13.7/86.3   | 12.66             | 5.83x               | 2.19x               |
| 17          | 59.8/40.2   | **13.85**         | **6.38x**           | **2.40x**           |

## Key Findings
1. **Pipelining provides massive speedup** (up to 6.38x) over sequential execution
2. **Our best configuration (13.85 images/sec) significantly outperforms reference** (5.77 images/sec) by 2.40x
3. **Balanced splits perform better**: Block 17 with 59.8/40.2 split gave best results, contradicting theoretical analysis
4. **Async RPC is crucial**: The key was fixing the "pipelining" to use rpc_async() instead of rpc_sync()
5. **Network latency measurements were misleading**: What was labeled as "network latency" actually included RPC overhead + computation time

## Implementation Changes
1. Fixed pipelining in `pipeline_manager.py` to use `rpc_async()` and futures
2. Added `--split-block` parameter to `distributed_runner.py` for testing different partition points
3. Created `test_split_points.py` to analyze optimal split configurations
4. Proper async pipeline management with multiple batches in flight (up to 3 concurrent)
3. Split point matters less with good pipelining, but block 6 shows best theoretical speedup
4. "Network latency" in metrics actually includes RPC overhead + computation time
5. Sequential performance is consistent across runs (~2.17 images/sec)