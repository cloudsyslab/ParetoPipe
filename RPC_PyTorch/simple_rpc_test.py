#!/usr/bin/env python3
"""
Dead simple RPC test using multiprocessing to verify basic functionality
"""
import os
import time
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

def worker_main(rank, world_size):
    """Worker process main function"""
    print(f"[WORKER{rank}] Starting...")
    
    # Environment setup
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '54321'
    
    try:
        if rank == 0:
            # Master
            print("[MASTER] Initializing RPC...")
            rpc.init_rpc("master", rank=0, world_size=world_size)
            print("[MASTER] RPC initialized!")
            
            # Test RPC calls
            print("[MASTER] Testing RPC call to worker1...")
            result = rpc.rpc_sync("worker1", torch.add, args=(torch.tensor([1, 2]), torch.tensor([3, 4])))
            print(f"[MASTER] Result: {result}")
            
            print("[MASTER] SUCCESS - RPC call worked!")
            
        else:
            # Worker
            worker_name = f"worker{rank}"
            print(f"[{worker_name.upper()}] Initializing RPC...")
            rpc.init_rpc(worker_name, rank=rank, world_size=world_size)
            print(f"[{worker_name.upper()}] RPC initialized, waiting for calls...")
            
            # Keep worker alive for a bit
            time.sleep(10)
            
    except Exception as e:
        print(f"[RANK{rank}] ERROR: {e}")
    finally:
        print(f"[RANK{rank}] Shutting down...")
        try:
            rpc.shutdown()
        except:
            pass

def main():
    """Main test function"""
    print("=== Simple Single-Machine RPC Test ===")
    
    world_size = 2  # Just master + 1 worker for simplicity
    
    # Use multiprocessing to spawn processes
    mp.spawn(worker_main, args=(world_size,), nprocs=world_size, join=True)
    
    print("=== Test Complete ===")

if __name__ == "__main__":
    main()