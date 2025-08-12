#!/usr/bin/env python3
"""
Simple RPC Diagnostic Script
Tests basic PyTorch RPC connectivity between nodes 0, 1, 2
"""

import os
import sys
import time
import torch
import torch.distributed.rpc as rpc
import logging
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [Rank %(rank)s] %(message)s'
)

class RankFilter(logging.Filter):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
    
    def filter(self, record):
        record.rank = self.rank
        return True

def simple_add(a, b):
    """Simple function to test RPC calls"""
    result = a + b
    print(f"simple_add({a}, {b}) = {result}")
    return result

def hello_from_worker(rank):
    """Simple greeting function"""
    message = f"Hello from worker {rank}!"
    print(message)
    return message

def test_rpc_master(rank, world_size):
    """Master node (rank 0) test logic"""
    logger = logging.getLogger(__name__)
    logger.addFilter(RankFilter(rank))
    
    logger.info("=== MASTER NODE STARTING ===")
    logger.info(f"World size: {world_size}")
    
    # Setup environment
    master_addr = os.getenv('MASTER_ADDR', 'PlamaLV')
    master_port = os.getenv('MASTER_PORT', '30000')
    
    logger.info(f"Master address: {master_addr}")
    logger.info(f"Master port: {master_port}")
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    worker_name = f"worker_{rank}"
    logger.info(f"Initializing RPC as: {worker_name}")
    
    try:
        # Initialize RPC with TensorPipe backend
        rpc.init_rpc(
            name=worker_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                num_worker_threads=4,
                rpc_timeout=30  # 30 second timeout
            )
        )
        logger.info("✅ RPC initialization successful!")
        
        # Wait a bit for workers to be ready
        logger.info("Waiting 5 seconds for workers to initialize...")
        time.sleep(5)
        
        # Test RPC calls to each worker
        for worker_rank in range(1, world_size):
            worker_name = f"worker_{worker_rank}"
            logger.info(f"Testing RPC call to {worker_name}...")
            
            try:
                # Test simple addition
                result = rpc.rpc_sync(worker_name, simple_add, args=(10, worker_rank))
                logger.info(f"✅ RPC call successful! Result: {result}")
                
                # Test greeting function
                greeting = rpc.rpc_sync(worker_name, hello_from_worker, args=(worker_rank,))
                logger.info(f"✅ Greeting received: {greeting}")
                
            except Exception as e:
                logger.error(f"❌ RPC call to {worker_name} failed: {e}")
        
        logger.info("All RPC tests completed!")
        
    except Exception as e:
        logger.error(f"❌ RPC initialization failed: {e}")
        return False
    
    finally:
        logger.info("Shutting down RPC...")
        rpc.shutdown()
        logger.info("✅ RPC shutdown complete")
    
    return True

def test_rpc_worker(rank, world_size):
    """Worker node test logic"""
    logger = logging.getLogger(__name__)
    logger.addFilter(RankFilter(rank))
    
    logger.info("=== WORKER NODE STARTING ===")
    
    # Setup environment
    master_addr = os.getenv('MASTER_ADDR', 'PlamaLV')
    master_port = os.getenv('MASTER_PORT', '30000')
    
    logger.info(f"Master address: {master_addr}")
    logger.info(f"Master port: {master_port}")
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    worker_name = f"worker_{rank}"
    logger.info(f"Initializing RPC as: {worker_name}")
    
    try:
        # Initialize RPC with TensorPipe backend
        rpc.init_rpc(
            name=worker_name,
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
                num_worker_threads=4,
                rpc_timeout=30  # 30 second timeout
            )
        )
        logger.info("✅ RPC initialization successful!")
        
        # Worker just waits to serve RPC requests
        logger.info("Worker ready to serve RPC requests...")
        
        # Keep alive for testing (master will shut us down)
        time.sleep(60)  # Wait up to 1 minute
        
    except Exception as e:
        logger.error(f"❌ RPC initialization failed: {e}")
        return False
    
    finally:
        logger.info("Shutting down RPC...")
        rpc.shutdown()
        logger.info("✅ RPC shutdown complete")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Simple RPC Diagnostic Test')
    parser.add_argument('--rank', type=int, required=True, help='Node rank (0=master, 1,2=workers)')
    parser.add_argument('--world-size', type=int, default=3, help='Total number of nodes')
    
    args = parser.parse_args()
    
    print(f"Starting RPC diagnostic test - Rank: {args.rank}, World Size: {args.world_size}")
    
    if args.rank == 0:
        # Master node
        success = test_rpc_master(args.rank, args.world_size)
    else:
        # Worker node
        success = test_rpc_worker(args.rank, args.world_size)
    
    if success:
        print("✅ RPC diagnostic test completed successfully!")
        sys.exit(0)
    else:
        print("❌ RPC diagnostic test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()