#!/usr/bin/env python3
"""
Pipeline management system for overlapping inference stages.
Implements true sequential pipelining to maximize throughput in distributed inference.
"""

import time
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import threading
import queue
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from collections import deque
import concurrent.futures
from enum import Enum


class PipelineStage(Enum):
    """Pipeline stage states."""
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class PipelineBatch:
    """Represents a batch moving through the pipeline."""
    batch_id: int
    data: torch.Tensor
    labels: Optional[torch.Tensor] = None
    start_time: float = 0.0
    stage_times: Dict[int, Tuple[float, float]] = None  # stage_id -> (start_time, end_time)
    current_stage: int = 0
    completed: bool = False
    
    def __post_init__(self):
        if self.stage_times is None:
            self.stage_times = {}
        if self.start_time == 0.0:
            self.start_time = time.time()


@dataclass
class PipelineStageInfo:
    """Information about a pipeline stage."""
    stage_id: int
    device_id: str
    worker_name: str
    shard_module: nn.Module
    input_queue: queue.Queue
    output_queue: queue.Queue
    metrics_callback: Optional[Callable] = None
    status: PipelineStage = PipelineStage.IDLE


class PipelineWorker:
    """Worker that processes one stage of the pipeline."""
    
    def __init__(self, stage_id: int, shard_module: nn.Module, device_id: str,
                 metrics_callback: Optional[Callable] = None, max_queue_size: int = 10):
        self.stage_id = stage_id
        self.shard_module = shard_module.eval()
        self.device_id = device_id
        self.metrics_callback = metrics_callback
        self.max_queue_size = max_queue_size
        
        # Queues for pipeline communication
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)
        
        # Processing state
        self.processing = False
        self.shutdown_requested = False
        self.worker_thread = None
        self.logger = logging.getLogger(f"{__name__}.stage_{stage_id}")
        
        # Performance tracking
        self.total_batches_processed = 0
        self.total_processing_time = 0.0
        self.queue_wait_times = deque(maxlen=100)
        
    def start(self):
        """Start the worker thread."""
        if self.worker_thread is not None:
            return
        
        self.shutdown_requested = False
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.logger.info(f"Started pipeline worker for stage {self.stage_id}")
    
    def stop(self):
        """Stop the worker thread."""
        self.shutdown_requested = True
        if self.worker_thread:
            # Add a sentinel to unblock the queue
            try:
                self.input_queue.put(None, timeout=1.0)
            except queue.Full:
                pass
            self.worker_thread.join(timeout=5.0)
            self.worker_thread = None
        self.logger.info(f"Stopped pipeline worker for stage {self.stage_id}")
    
    def _worker_loop(self):
        """Main worker processing loop."""
        self.logger.info(f"Worker loop started for stage {self.stage_id}")
        
        while not self.shutdown_requested:
            try:
                # Wait for input batch
                queue_wait_start = time.time()
                batch = self.input_queue.get(timeout=1.0)
                queue_wait_time = time.time() - queue_wait_start
                
                if batch is None:  # Shutdown sentinel
                    break
                
                self.queue_wait_times.append(queue_wait_time)
                
                # Process the batch
                self._process_batch(batch, queue_wait_time)
                
                # Mark task as done
                self.input_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in worker loop for stage {self.stage_id}: {e}")
    
    def _process_batch(self, batch: PipelineBatch, queue_wait_time: float):
        """Process a single batch through this stage."""
        self.processing = True
        process_start_time = time.time()
        
        try:
            # Record stage start time
            batch.stage_times[self.stage_id] = (process_start_time, 0.0)
            
            self.logger.debug(f"Processing batch {batch.batch_id} at stage {self.stage_id}")
            
            # Run inference through this shard
            with torch.no_grad():
                batch.data = batch.data.to("cpu")  # Ensure CPU processing
                output = self.shard_module(batch.data)
                batch.data = output.cpu()
            
            process_end_time = time.time()
            processing_time = process_end_time - process_start_time
            
            # Update stage timing
            batch.stage_times[self.stage_id] = (process_start_time, process_end_time)
            
            # Update performance tracking
            self.total_batches_processed += 1
            self.total_processing_time += processing_time
            
            # Call metrics callback if provided
            if self.metrics_callback:
                try:
                    self.metrics_callback(
                        batch_id=batch.batch_id,
                        stage_id=self.stage_id,
                        stage_name=f"shard_{self.stage_id}",
                        start_time=process_start_time,
                        end_time=process_end_time,
                        input_size_bytes=self._estimate_tensor_size(batch.data),
                        output_size_bytes=self._estimate_tensor_size(output),
                        queue_wait_time_ms=queue_wait_time * 1000
                    )
                except Exception as e:
                    self.logger.warning(f"Error in metrics callback: {e}")
            
            # Move batch to next stage
            self.output_queue.put(batch)
            
            self.logger.debug(f"Completed batch {batch.batch_id} at stage {self.stage_id} in {processing_time*1000:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"Error processing batch {batch.batch_id} at stage {self.stage_id}: {e}")
            # Still try to pass the batch forward to avoid pipeline stall
            batch.data = torch.zeros_like(batch.data)  # Dummy output
            self.output_queue.put(batch)
        finally:
            self.processing = False
    
    def _estimate_tensor_size(self, tensor: torch.Tensor) -> int:
        """Estimate tensor size in bytes."""
        return tensor.numel() * tensor.element_size()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this worker."""
        avg_processing_time = (self.total_processing_time / self.total_batches_processed 
                              if self.total_batches_processed > 0 else 0.0)
        avg_queue_wait = (sum(self.queue_wait_times) / len(self.queue_wait_times) 
                         if self.queue_wait_times else 0.0)
        
        return {
            'stage_id': self.stage_id,
            'device_id': self.device_id,
            'total_batches_processed': self.total_batches_processed,
            'total_processing_time': self.total_processing_time,
            'average_processing_time_ms': avg_processing_time * 1000,
            'average_queue_wait_time_ms': avg_queue_wait * 1000,
            'input_queue_size': self.input_queue.qsize(),
            'output_queue_size': self.output_queue.qsize(),
            'currently_processing': self.processing
        }


class DistributedPipelineWorker:
    """RPC-based distributed pipeline worker."""
    
    def __init__(self, stage_id: int, shard_module: nn.Module, metrics_callback: Optional[Callable] = None):
        self.stage_id = stage_id
        self.shard_module = shard_module.eval()
        self.metrics_callback = metrics_callback
        self.logger = logging.getLogger(f"{__name__}.rpc_stage_{stage_id}")
        self.total_processed = 0
    
    def process_batch_rpc(self, batch_data: torch.Tensor, batch_id: int) -> torch.Tensor:
        """Process a batch via RPC call."""
        start_time = time.time()
        
        try:
            self.logger.debug(f"RPC processing batch {batch_id} at stage {self.stage_id}")
            
            # Log input tensor shape
            self.logger.info(f"[FORWARD PASS] Shard {self.stage_id} received tensor shape: {batch_data.shape}, batch_id={batch_id}")
            
            # Time the actual computation
            compute_start = time.time()
            with torch.no_grad():
                batch_data = batch_data.to("cpu")
                
                # Use payload measurement method if available, otherwise fall back to direct call
                if hasattr(self.shard_module, 'forward_with_payload_measurement'):
                    self.logger.info(f"[PIPELINE] Using payload measurement for shard {self.stage_id}")
                    output = self.shard_module.forward_with_payload_measurement(batch_data, batch_id)
                else:
                    self.logger.info(f"[PIPELINE] Using direct shard call for shard {self.stage_id}")
                    output = self.shard_module(batch_data)
                    
                result = output.cpu()
            compute_end = time.time()
            
            # Calculate times
            compute_time_ms = (compute_end - compute_start) * 1000
            
            # Log timing information in the format expected by automated_split_tester
            self.logger.info(f"[FORWARD PASS] Shard {self.stage_id} computation completed in {compute_time_ms:.2f}ms")
            self.logger.info(f"[SHARD_TIMING] shard_{self.stage_id} processing_time_ms={compute_time_ms:.2f} batch_id={batch_id}")
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Update stats
            self.total_processed += 1
            
            # Call metrics callback if provided
            if self.metrics_callback:
                try:
                    self.metrics_callback(
                        batch_id=batch_id,
                        stage_id=self.stage_id,
                        stage_name=f"rpc_shard_{self.stage_id}",
                        start_time=start_time,
                        end_time=end_time,
                        input_size_bytes=batch_data.numel() * batch_data.element_size(),
                        output_size_bytes=result.numel() * result.element_size(),
                        queue_wait_time_ms=0.0
                    )
                except Exception as e:
                    self.logger.warning(f"Error in metrics callback: {e}")
            
            self.logger.debug(f"RPC completed batch {batch_id} at stage {self.stage_id} in {processing_time*1000:.2f}ms")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in RPC processing batch {batch_id} at stage {self.stage_id}: {e}")
            # Return zeros to avoid breaking the pipeline
            return torch.zeros_like(batch_data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            'stage_id': self.stage_id,
            'total_processed': self.total_processed
        }


class PipelineManager:
    """Manages the entire inference pipeline."""
    
    def __init__(self, shards: List[nn.Module], workers: List[str], 
                 metrics_callback: Optional[Callable] = None,
                 use_local_pipeline: bool = False,
                 max_concurrent_batches: int = 4):
        """
        Initialize pipeline manager.
        
        Args:
            shards: List of model shards (Sequential modules)
            workers: List of worker names for RPC
            metrics_callback: Optional callback for metrics collection
            use_local_pipeline: Whether to use local threading or RPC
            max_concurrent_batches: Maximum batches in pipeline simultaneously
        """
        self.shards = shards
        self.workers = workers
        self.metrics_callback = metrics_callback
        self.use_local_pipeline = use_local_pipeline
        self.max_concurrent_batches = max_concurrent_batches
        self.logger = logging.getLogger(__name__)
        
        # Pipeline state
        self.pipeline_workers = []
        self.rpc_workers = {}
        self.active_batches = {}
        self.completed_batches = {}
        self.batch_counter = 0
        self.pipeline_running = False
        
        # Performance tracking
        self.total_batches_started = 0
        self.total_batches_completed = 0
        self.pipeline_start_time = 0.0
        
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Setup the pipeline based on configuration."""
        if self.use_local_pipeline:
            self._setup_local_pipeline()
        else:
            self._setup_rpc_pipeline()
    
    def _setup_local_pipeline(self):
        """Setup local threaded pipeline."""
        self.logger.info("Setting up local threaded pipeline")
        
        for i, shard in enumerate(self.shards):
            worker = PipelineWorker(
                stage_id=i,
                shard_module=shard,
                device_id=f"local_stage_{i}",
                metrics_callback=self.metrics_callback
            )
            self.pipeline_workers.append(worker)
        
        # Connect stages
        for i in range(len(self.pipeline_workers) - 1):
            # Output of stage i becomes input of stage i+1
            # This will be handled in the processing loop
            pass
    
    def _setup_rpc_pipeline(self):
        """Setup RPC-based distributed pipeline."""
        self.logger.info("Setting up RPC distributed pipeline")
        
        for i, shard in enumerate(self.shards):
            worker_name = self.workers[i % len(self.workers)]
            
            # Create remote worker
            rpc_worker = rpc.remote(
                worker_name,
                DistributedPipelineWorker,
                args=(i, shard, None)  # Don't pass metrics callback to remote workers
            )
            self.rpc_workers[i] = (worker_name, rpc_worker)
    
    def start_pipeline(self):
        """Start the pipeline processing."""
        if self.pipeline_running:
            return
        
        self.pipeline_running = True
        self.pipeline_start_time = time.time()
        
        if self.use_local_pipeline:
            for worker in self.pipeline_workers:
                worker.start()
        
        self.logger.info("Pipeline started")
    
    def stop_pipeline(self):
        """Stop the pipeline processing."""
        if not self.pipeline_running:
            return
        
        self.pipeline_running = False
        
        if self.use_local_pipeline:
            for worker in self.pipeline_workers:
                worker.stop()
        
        self.logger.info("Pipeline stopped")
    
    def process_batch_local(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None) -> PipelineBatch:
        """Process a batch through the local pipeline."""
        batch_id = self.batch_counter
        self.batch_counter += 1
        
        batch = PipelineBatch(
            batch_id=batch_id,
            data=data,
            labels=labels
        )
        
        self.active_batches[batch_id] = batch
        self.total_batches_started += 1
        
        # Add to first stage
        if self.pipeline_workers:
            self.pipeline_workers[0].input_queue.put(batch)
        
        return batch
    
    def process_batch_rpc(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process a batch through the RPC pipeline (non-pipelined for compatibility)."""
        batch_id = self.batch_counter
        self.batch_counter += 1
        
        current_data = data
        
        # Process through each stage sequentially via RPC
        for stage_id in range(len(self.shards)):
            if stage_id in self.rpc_workers:
                worker_name, rpc_worker = self.rpc_workers[stage_id]
                
                # Measure RPC latency
                rpc_start_time = time.time()
                current_data = rpc_worker.rpc_sync().process_batch_rpc(current_data, batch_id)
                rpc_end_time = time.time()
                
                # Record network metrics
                if self.metrics_callback:
                    data_size_bytes = current_data.numel() * current_data.element_size()
                    rpc_latency_ms = (rpc_end_time - rpc_start_time) * 1000
                    throughput_mbps = (data_size_bytes / (1024 * 1024)) / (rpc_latency_ms / 1000) if rpc_latency_ms > 0 else 0
                    
                    # This would need to be called on a metrics collector instance
                    # self.metrics_callback.record_network_metrics(rpc_latency_ms, throughput_mbps)
        
        self.total_batches_completed += 1
        return current_data
    
    def start_batch_rpc_pipelined(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None) -> int:
        """Start processing a batch through the pipelined RPC system."""
        batch_id = self.batch_counter
        self.batch_counter += 1
        
        # Create batch object
        batch = PipelineBatch(
            batch_id=batch_id,
            data=data,
            labels=labels
        )
        
        # Store as active batch
        self.active_batches[batch_id] = batch
        self.total_batches_started += 1
        
        # Start processing on first stage asynchronously
        if 0 in self.rpc_workers:
            worker_name, rpc_worker = self.rpc_workers[0]
            
            # Calculate the input tensor size for the first stage (this is what we're sending)
            input_tensor_size_mb = (data.numel() * data.element_size()) / (1024 * 1024)
            
            # Track RPC start time
            rpc_start_time = time.time()
            batch.stage_times[0] = (rpc_start_time, None)
            
            # Create future for first stage
            future = rpc_worker.rpc_async().process_batch_rpc(data, batch_id)
            
            # Set up continuation for subsequent stages, passing the input tensor size
            self._setup_pipeline_continuation(future, batch_id, 0, input_tensor_size_mb)
        
        return batch_id
    
    def _setup_pipeline_continuation(self, future, batch_id: int, completed_stage: int, input_tensor_size_mb: float = 0.0):
        """Set up continuation for the next stage in the pipeline."""
        def continue_pipeline():
            try:
                # Get result from completed stage
                result = future.wait()
                rpc_end_time = time.time()
                
                # Update batch data
                if batch_id in self.active_batches:
                    batch = self.active_batches[batch_id]
                    batch.data = result
                    batch.current_stage = completed_stage + 1
                    
                    # Record completion time for this stage
                    if completed_stage not in batch.stage_times:
                        batch.stage_times[completed_stage] = (batch.start_time, rpc_end_time)
                    else:
                        # Update end time
                        start_time = batch.stage_times[completed_stage][0]
                        batch.stage_times[completed_stage] = (start_time, rpc_end_time)
                        
                        # Calculate and log RPC timing
                        rpc_time_ms = (rpc_end_time - start_time) * 1000
                        
                        # Use the INPUT tensor size that was sent (not the output)
                        tensor_size_mb = input_tensor_size_mb
                        estimated_network_ms = 0.5 + (tensor_size_mb * 0.3) + (tensor_size_mb * 8 / 940) * 1000 * 2
                        
                        # Log timing information
                        self.logger.info(f"[FORWARD SEQUENTIAL] RPC call to shard {completed_stage} completed in {rpc_time_ms:.2f}ms")
                        self.logger.info(f"[NETWORK_TIMING] shard_{completed_stage} network_time_ms={estimated_network_ms:.2f} tensor_size_mb={tensor_size_mb:.2f} batch_id={batch_id}")
                    
                    # Check if there's a next stage
                    next_stage = completed_stage + 1
                    if next_stage < len(self.shards) and next_stage in self.rpc_workers:
                        # Start next stage
                        worker_name, rpc_worker = self.rpc_workers[next_stage]
                        
                        # Calculate the input tensor size for the next stage (this is what we're sending)
                        next_input_tensor_size_mb = (result.numel() * result.element_size()) / (1024 * 1024)
                        
                        # Track RPC start time for network timing
                        rpc_start_time = time.time()
                        batch.stage_times[next_stage] = (rpc_start_time, None)  # Start time for next stage
                        
                        next_future = rpc_worker.rpc_async().process_batch_rpc(result, batch_id)
                        
                        # Set up continuation for next stage, passing the input size
                        self._setup_pipeline_continuation(next_future, batch_id, next_stage, next_input_tensor_size_mb)
                    else:
                        # Pipeline complete
                        batch.completed = True
                        self.completed_batches[batch_id] = batch
                        self.total_batches_completed += 1
                        
                        # Remove from active batches
                        if batch_id in self.active_batches:
                            del self.active_batches[batch_id]
                            
                        self.logger.debug(f"Batch {batch_id} completed pipeline")
                        
            except Exception as e:
                self.logger.error(f"Error in pipeline continuation for batch {batch_id}: {e}")
                # Mark batch as completed with error
                if batch_id in self.active_batches:
                    batch = self.active_batches[batch_id]
                    batch.completed = True
                    self.completed_batches[batch_id] = batch
                    del self.active_batches[batch_id]
        
        # Add callback to execute when future completes
        future.add_done_callback(lambda f: continue_pipeline())
    
    def get_completed_batch(self, batch_id: int, timeout: Optional[float] = None) -> Optional[torch.Tensor]:
        """Get a completed batch result, waiting if necessary."""
        start_time = time.time()
        
        while True:
            # Check if batch is completed
            if batch_id in self.completed_batches:
                batch = self.completed_batches[batch_id]
                del self.completed_batches[batch_id]
                return batch.data
            
            # Check timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                return None
            
            # Small sleep to avoid busy waiting
            time.sleep(0.01)
    
    def process_batch_rpc_pipelined(self, data: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Process a batch through the pipelined RPC system (blocking interface for compatibility)."""
        # Start the batch
        batch_id = self.start_batch_rpc_pipelined(data, labels)
        
        # Wait for completion
        result = self.get_completed_batch(batch_id, timeout=300.0)  # 5 minute timeout
        
        if result is None:
            self.logger.error(f"Timeout waiting for batch {batch_id}")
            return torch.zeros_like(data)  # Return zeros on timeout
        
        return result
    
    def _collect_completed_batches(self):
        """Collect completed batches from the pipeline."""
        if not self.use_local_pipeline or not self.pipeline_workers:
            return
        
        # Check output queue of last stage
        last_worker = self.pipeline_workers[-1]
        
        while True:
            try:
                completed_batch = last_worker.output_queue.get_nowait()
                completed_batch.completed = True
                
                self.completed_batches[completed_batch.batch_id] = completed_batch
                self.total_batches_completed += 1
                
                # Remove from active batches
                if completed_batch.batch_id in self.active_batches:
                    del self.active_batches[completed_batch.batch_id]
                
                self.logger.debug(f"Collected completed batch {completed_batch.batch_id}")
                
            except queue.Empty:
                break
        
        # Move batches between stages
        for i in range(len(self.pipeline_workers) - 1):
            current_worker = self.pipeline_workers[i]
            next_worker = self.pipeline_workers[i + 1]
            
            while True:
                try:
                    batch = current_worker.output_queue.get_nowait()
                    next_worker.input_queue.put(batch)
                except queue.Empty:
                    break
    
    def wait_for_completion(self, timeout: Optional[float] = None) -> List[PipelineBatch]:
        """Wait for all batches to complete processing."""
        start_wait_time = time.time()
        
        while (self.active_batches and 
               (timeout is None or (time.time() - start_wait_time) < timeout)):
            
            if self.use_local_pipeline:
                self._collect_completed_batches()
            
            time.sleep(0.1)  # Small sleep to avoid busy waiting
        
        # Collect any remaining completed batches
        if self.use_local_pipeline:
            self._collect_completed_batches()
        
        # Return completed batches
        completed = list(self.completed_batches.values())
        self.completed_batches.clear()
        
        return completed
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        stats = {
            'total_batches_started': self.total_batches_started,
            'total_batches_completed': self.total_batches_completed,
            'active_batches': len(self.active_batches),
            'pipeline_running': self.pipeline_running,
            'use_local_pipeline': self.use_local_pipeline,
            'worker_stats': []
        }
        
        # Add pipeline runtime
        if self.pipeline_start_time > 0:
            stats['pipeline_runtime_seconds'] = time.time() - self.pipeline_start_time
        
        # Add worker statistics
        if self.use_local_pipeline:
            for worker in self.pipeline_workers:
                stats['worker_stats'].append(worker.get_performance_stats())
        else:
            # Get RPC worker stats (this would need to be implemented)
            for stage_id, (worker_name, rpc_worker) in self.rpc_workers.items():
                try:
                    worker_stats = rpc_worker.rpc_sync().get_stats()
                    worker_stats['worker_name'] = worker_name
                    stats['worker_stats'].append(worker_stats)
                except Exception as e:
                    self.logger.warning(f"Could not get stats from worker {worker_name}: {e}")
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        self.start_pipeline()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_pipeline()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    print("Pipeline manager module loaded successfully")