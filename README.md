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

5) Connection pi1 to pi2 or gpu by running 
7) (Optional) Sweep multiple splits
8) Simulate network delay and bandwidth
9) Troubleshooting
10) One-shot example (MobileNetV2 + 200 ms delay)
