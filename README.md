### ParetoPipe 📈🛤️

ParetoPipe is an open-source framework designed to systematically benchmark and analyze the performance of distributed Deep Neural Network (DNN) inference across heterogeneous edge devices. This tool was developed for the research paper, "Where to Split? A Pareto-Front Analysis of DNN Partitioning for Edge Inference". The core mission of ParetoPipe is to reframe DNN partitioning as a multi-objective optimization problem. Instead of focusing on a single metric like latency or throughput, our framework allows researchers and practitioners to explore the complex trade-off between these competing objectives. By identifying the Pareto frontier, ParetoPipe helps find all optimal partitioning strategies where one metric cannot be improved without degrading another.

### Key Features:

1. Pipeline Parallelism: Implements pipeline parallelism to distribute sequential segments of a DNN across a network of devices, making it ideal for edge topologies;
2. Heterogeneous Device Support: Explicitly designed to benchmark performance on heterogeneous testbeds, such as configurations between two Raspberry Pis (edge-to-edge) or a Raspberry Pi and a GPU-equipped server (edge-to-server);
3. Pareto Front Analysis: Systematically tests every valid partition point for a given model to collect latency and throughput data, then performs a Pareto analysis to identify the optimal set of trade-offs;
4. Network Condition Simulation: Integrates with tools like Linux tc to simulate real-world network imperfections, such as added latency and throttled bandwidth, allowing for robust;\
5. performance analysis under duress;
6. Dual Communication Backends: Features two communication backends to enable fine-grained analysis of runtime overhead: i) PyTorch RPC: A high-level abstraction using PyTorch's built-in distributed communication framework; ii) Custom TCP Sockets: A lightweight, low-level implementation to minimize overhead and provide finer execution control.

---
### Author/Contact Information 📞

- **Dr. Palden Lama** – [palden.lama@utsa.edu](mailto:palden.lama@utsa.edu) – *(Current Contributor)*
- **Adiba Masud** – [adiba.masud@my.utsa.edu](mailto:adiba.masud@my.utsa.edu) – *(Current Contributor)*
- **Nicholas Foley** – [nicholas.foley@my.utsa.edu](mailto:nicholas.foley@my.utsa.edu) – *(Current Contributor)*
- **Pragathi Durga Rajarajan** – [durga.rajarajan@my.utsa.edu](mailto:durga.rajarajan@my.utsa.edu) – *(Current Contributor)*

---
### Instructions to Run (Custom Implementation) 🧪⚙️
1) Get the code & enter the folder

   git clone https://github.com/cloudsyslab/ParetoPipe.git\
   cd ParetoPipe/custom_implementation/Pareto

3) Create environments

   A. GPU server\
   conda create -n pareto python=3.10 -y\
   conda activate pareto\
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
   pip install -r requirements.txt \
   pip install numpy pillow psutil pynvml pandas matplotlib\
   B. Raspberry Pi\
   python3 -m venv ~/venvs/pareto\
   source ~/venvs/pareto/bin/activate\

   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\
   pip install -r requirements.txt\
   pip install numpy pillow psutil pandas\

4) Network & paths (set once)

   export PI1_IP= your ip\
   export PI2/GPU_IP= your ip\
   export PORT= 111(your wish)\
   export OUT_BASE=~/pareto/results (any writable folder)\
   mkdir -p $OUT_BASE\

5) Connection pi1 to pi2 or gpu by running ./pareto.sh
6) (Optional) Sweep multiple splits : For getting best result you run 5 times and collecting the data.
7) Simulate network delay and bandwidth
   ```bash
   #Run on the Pi (change eth0 to your NIC if needed)
   sudo tc qdisc del dev eth0 root 2>/dev/null || true #clear any existing rules
   sudo tc qdisc add dev eth0 root handle 1: netem delay 200ms #add delay & limit bandwidth (example: 200ms delay, 5mbit)
   sudo tc qdisc add dev eth0 parent 1: handle 10: tbf rate 5mbit burst 32kbit latency 400ms
   tc qdisc show dev eth0 #verify
8) Troubleshooting
   ->Client hangs / “connection refused” → start server first; check $SERVER_IP & $PORT; wait 2–5s before client;\
   ->GPU util ~0% → ensure part1 and its inputs are on CUDA (.to('cuda')); try larger --batch-size;\
   ->Shape mismatch → confirm identical transforms on both sides; print tensor shapes right after the split;\
   ->Port busy → kill old server/tmux or fuser -k $PORT/tcp;\
   ->InceptionV3 errors → use --input-size 299 and aux_logits=False in the model init.
9) Follow the chart:

                           [ GPU Server ]
                           |
                           |--> Executes partition.sh
                           |
                           +=======================+
                           |                       |
                           (SSH)                   (SSH)
                           |                       |
                           v                       v
                           [ Raspberry Pi 1 ] ----> [ Raspberry Pi 2 ]
                           - Runs partition1.py     - Runs partition2.py
                           - Executes Model Part 1  - Executes Model Part 2

10) Finally, just wait to see the output results.

---

### Instructions to Run (PyTorch RPC Implementation)

1. Clone repository on all machines

2. Set your `.env` files for each machine (example below)

   ```
   # Distributed inference configuration
   # Copy this file to .env and modify the values for your setup

   # Master node IP address (where rank 0 runs)
   MASTER_ADDR=192.168.1.1

   # Master node port for RPC communication
   MASTER_PORT=123456

   # Network interface name (use `ip a` to find yours)
   GLOO_SOCKET_IFNAME=eth0

   # TensorPipe socket interface (usually same as GLOO)
   TP_SOCKET_IFNAME=eth0

   # Optional: Dataset path
   # CIFAR10_PATH=/path/to/cifar10
   ```

3. Download requirements
   `pip install -r RPC_PyTorch/requirements.txt`

4. Running (2 Options)

- **`distributed_runner.py`** - Main distributed inference runner with the following flags:

  - `--rank` (int, default=0): Rank of current process (0 for master, 1+ for workers)
  - `--world-size` (int, default=3): Total number of processes (1 master + N workers)
  - `--model` (str, default="mobilenetv2"): Model architecture to use (mobilenetv2, resnet18, resnet50, vgg16, alexnet, inceptionv3)
  - `--batch-size` (int, default=8): Batch size for inference
  - `--num-classes` (int, default=10): Number of output classes
  - `--dataset` (str, default="cifar10"): Dataset to use (cifar10 or dummy)
  - `--num-test-samples` (int, default=64): Number of images to test
  - `--num-partitions` (int, default=2): Number of model partitions to split across devices
  - `--metrics-dir` (str, default="./enhanced_metrics"): Directory for saving performance metrics
  - `--models-dir` (str, default="./models"): Directory containing model weight files
  - `--use-intelligent-splitting` (flag, default=True): Use profiling-based intelligent splitting
  - `--use-pipelining` (flag, default=False): Enable pipelined execution for improved throughput
  - `--num-threads` (int, default=4): Number of RPC threads for communication
  - `--disable-intelligent-splitting` (flag): Disable intelligent splitting, use traditional method
  - `--split-block` (int): Specific block number to split at (for MobileNetV2)
  - `--use-local-loading` (flag, default=True): Load model weights locally on workers from pre-split files
  - `--shards-dir` (str, default="~/datasets/model_shards"): Directory containing pre-split model shards
  - `--enable-prefetch` (flag, default=False): Enable data prefetching for improved throughput
  - `--prefetch-batches` (int, default=2): Number of batches to prefetch

- **`automated_split_tester.py`** - Automated testing tool for evaluating different split points:
  - `--splits` (list of ints): Specific split blocks to test (default: all 0-18 for MobileNetV2)
  - `--runs` (int, default=3): Number of runs per split for averaging results
  - `--wait-time` (int, default=60): Seconds to wait for workers to be ready
  - `--cleanup` (flag): Clean up individual output files after consolidation
  - `--no-optimizations` (flag): Disable optimization features (local loading, caching, prefetching)
  - `--model` (str, default="mobilenetv2"): Model to test (mobilenetv2, resnet18, resnet50, vgg16, alexnet, inceptionv3)
1
