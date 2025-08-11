# ParetoPipe 📈🛤️

ParetoPipe is an open-source framework designed to systematically benchmark and analyze the performance of distributed Deep Neural Network (DNN) inference across heterogeneous edge devices. This tool was developed for the research paper, "Where to Split? A Pareto-Front Analysis of DNN Partitioning for Edge Inference". The core mission of ParetoPipe is to reframe DNN partitioning as a multi-objective optimization problem. Instead of focusing on a single metric like latency or throughput, our framework allows researchers and practitioners to explore the complex trade-off between these competing objectives. By identifying the Pareto frontier, ParetoPipe helps find all optimal partitioning strategies where one metric cannot be improved without degrading another.

# Key Features:

1. Pipeline Parallelism: Implements pipeline parallelism to distribute sequential segments of a DNN across a network of devices, making it ideal for edge topologies.

2. Heterogeneous Device Support: Explicitly designed to benchmark performance on heterogeneous testbeds, such as configurations between two Raspberry Pis (edge-to-edge) or a Raspberry Pi and a GPU-equipped server (edge-to-server).

3. Pareto Front Analysis: Systematically tests every valid partition point for a given model to collect latency and throughput data, then performs a Pareto analysis to identify the optimal set of trade-offs.

4. Network Condition Simulation: Integrates with tools like Linux tc to simulate real-world network imperfections, such as added latency and throttled bandwidth, allowing for robust
5. performance analysis under duress.
   
6. Dual Communication Backends: Features two communication backends to enable fine-grained analysis of runtime overhead: i) PyTorch RPC: A high-level abstraction using PyTorch's built-in distributed communication framework; ii) Custom TCP Sockets: A lightweight, low-level implementation to minimize overhead and provide finer execution control.

---
# Instructions to Run (Custom Implementation) 🧪⚙️
1) Get the code & enter the folder

   git clone https://github.com/cloudsyslab/ParetoPipe.git\
   cd ParetoPipe/custom_implementation/Pareto

3) Create environments

   A. GPU server (Lambda)\
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
7) (Optional) Sweep multiple splits : For getting best result you run 5 times and collecting the data.
8) Simulate network delay and bandwidth
   Run on the Pi (change eth0 to your NIC if needed):\
   sudo tc qdisc del dev eth0 root 2>/dev/null || true (clear any existing rules)\
   sudo tc qdisc add dev eth0 root handle 1: netem delay 200ms (add delay & limit bandwidth (example: 200ms delay, 5mbit))\
   sudo tc qdisc add dev eth0 parent 1: handle 10: tbf rate 5mbit burst 32kbit latency 400ms\
   tc qdisc show dev eth0(verify)\
10) Troubleshooting
   ->Client hangs / “connection refused” → start server first; check $SERVER_IP & $PORT; wait 2–5s before client;\
   ->GPU util ~0% → ensure part1 and its inputs are on CUDA (.to('cuda')); try larger --batch-size;\
   ->Shape mismatch → confirm identical transforms on both sides; print tensor shapes right after the split;\
   ->Port busy → kill old server/tmux or fuser -k $PORT/tcp;\
   ->InceptionV3 errors → use --input-size 299 and aux_logits=False in the model init;\
11) One-shot example (MobileNetV2 + 200 ms delay) for test
      Simply navigate and follow the above instruction as well as run ./pareto.sh remember that to have the file mobilenet_pi1.py and mobilenet_pi2.py, save the results on json files.
