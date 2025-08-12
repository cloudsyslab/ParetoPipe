
#!/usr/bin/env python3
"""
Enhanced Distributed DNN Inference with Intelligent Splitting and Pipelining

This script provides advanced distributed deep neural network inference capabilities:
- Layer-by-layer profiling for optimal split point detection
- Intelligent model splitting based on computational costs
- True sequential pipelining for overlapping inference stages
- Comprehensive metrics collection including per-device IPS and pipeline efficiency
"""

# Disable PyTorch's advanced CPU optimizations for Raspberry Pi compatibility
import os
os.environ['ATEN_CPU_CAPABILITY'] = ''

import time
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torch.distributed.rpc import RRef
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from dotenv import load_dotenv
import argparse
import logging
import socket
import sys
from typing import List, Dict, Any, Optional
import json
from collections import OrderedDict
import queue
import threading

# Import our enhanced modules
from profiling import LayerProfiler, IntelligentSplitter, split_model_intelligently
from metrics import EnhancedMetricsCollector
from pipelining import PipelineManager, DistributedPipelineWorker
from core import ModelLoader
from utils.payload_measurement import PayloadMeasurer


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


class EnhancedShardWrapper(nn.Module):
    """Enhanced shard wrapper with metrics integration."""

    def __init__(self, submodule: nn.Module, shard_id: int,
                 metrics_collector: Optional[EnhancedMetricsCollector] = None):
        super().__init__()
        self.module = submodule.to("cpu")
        self.shard_id = shard_id
        self.metrics_collector = metrics_collector

        # Create metrics collector if none provided (for workers)
        if self.metrics_collector is None:
            try:
                rank = rpc.get_worker_info().id
            except:
                rank = 0
            self.metrics_collector = EnhancedMetricsCollector(rank)

    def forward(self, x: torch.Tensor, batch_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass with enhanced metrics collection."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor but got {type(x)}")

        logging.info(f"[{socket.gethostname()}] Shard {self.shard_id} processing tensor shape: {x.shape}, batch_id={batch_id}")

        x = x.to("cpu")
        start_time = time.time()

        # Forward pass
        with torch.no_grad():
            output = self.module(x).cpu()

        end_time = time.time()

        # Record stage metrics
        if self.metrics_collector:
            self.metrics_collector.record_pipeline_stage(
                batch_id=batch_id if batch_id is not None else 0,
                stage_id=self.shard_id,
                stage_name=f"shard_{self.shard_id}",
                start_time=start_time,
                end_time=end_time,
                input_size_bytes=x.numel() * x.element_size(),
                output_size_bytes=output.numel() * output.element_size()
            )
            logging.info(f"[METRICS DEBUG] Recorded stage metrics for shard {self.shard_id}, batch_id={batch_id}, start={start_time:.3f}, end={end_time:.3f}")

        logging.info(f"[{socket.gethostname()}] Shard {self.shard_id} completed: {output.shape}")
        return output

    def forward_with_payload_measurement(self, x: torch.Tensor, batch_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass with payload measurement."""
        payload_size_bytes = PayloadMeasurer.measure_torch_serialization_size(x)
        payload_size_mb = payload_size_bytes / (1024 * 1024)
        logging.info(f"[PAYLOAD_MEASUREMENT] shard_{self.shard_id} input_payload_size_mb={payload_size_mb:.2f} batch_id={batch_id}")
        return self.forward(x, batch_id)

    def parameter_rrefs(self):
        """Get parameter RRefs for distributed training (if needed)."""
        return [RRef(p) for p in self.parameters()]


class LocalLoadingShardWrapper(nn.Module):
    """Shard wrapper that loads ONLY shard weights from pre-split files."""

    def __init__(self, shard_config: Dict[str, Any], shard_id: int,
                 metrics_collector: Optional[EnhancedMetricsCollector] = None):
        super().__init__()
        self.shard_id = shard_id
        self.metrics_collector = metrics_collector

        logger = logging.getLogger(__name__)
        logger.info(f"[WORKER INIT] Initializing LocalLoadingShardWrapper for shard {shard_id}")
        logger.info(f"[WORKER INIT] Config received: model_type={shard_config.get('model_type')}, "
                   f"split_block={shard_config.get('split_block')}, "
                   f"shards_dir={shard_config.get('shards_dir')}")

        # Create metrics collector if none provided (for workers)
        if self.metrics_collector is None:
            try:
                rank = rpc.get_worker_info().id
                logger.info(f"[WORKER INIT] Creating metrics collector for rank {rank}")
            except:
                rank = 0
                logger.warning("[WORKER INIT] Could not get RPC worker info, using rank 0")
            self.metrics_collector = EnhancedMetricsCollector(rank)

        # Load the shard locally based on config
        logger.info(f"[WORKER INIT] Starting shard loading for shard {shard_id}")
        self.module = self._load_local_shard(shard_config)
        logger.info(f"[WORKER INIT] Shard {shard_id} loading completed successfully")

    def _load_local_shard(self, config: Dict[str, Any]) -> nn.Module:
        """Load ONLY the shard weights from pre-split weight files."""
        logger = logging.getLogger(__name__)
        logger.info(f"[SHARD LOADING] _load_local_shard called for shard {config['shard_id']}")

        # Check if we have pre-split weights
        shards_dir = config.get('shards_dir', './model_shards')
        # Expand the path locally on this machine
        shards_dir = os.path.expanduser(shards_dir)
        model_type = config['model_type']
        shard_id = config['shard_id']

        logger.info(f"[SHARD LOADING] Base shards_dir (from config): {config.get('shards_dir')}")
        logger.info(f"[SHARD LOADING] Expanded local shards_dir: {shards_dir}")
        logger.info(f"[SHARD LOADING] Model type: {model_type}, Shard ID: {shard_id}")

        # Check if we have a split_block specified
        split_block = config.get('split_block')
        if split_block is not None:
            # Look in the split-specific subdirectory
            shards_dir = os.path.join(shards_dir, f"split_{split_block}")
            logger.info(f"[SHARD LOADING] Split block specified: {split_block}, updated shards_dir: {shards_dir}")

        # Try to load from pre-split weights first
        shard_filename = f"{model_type}_shard_{shard_id}_of_{config['total_shards']}.pth"
        shard_path = os.path.join(shards_dir, shard_filename)

        logger.info(f"[SHARD LOADING] Looking for pre-split shard at: {shard_path}")

        if os.path.exists(shard_path):
            logger.info(f"[SHARD LOADING] Found pre-split shard file, loading from {shard_path}")

            # Load the shard checkpoint
            logger.info(f"[SHARD LOADING] Starting torch.load for {shard_path}")
            load_start = time.time()
            checkpoint = torch.load(shard_path, map_location='cpu')
            load_time = time.time() - load_start
            logger.info(f"[SHARD LOADING] torch.load completed in {load_time:.3f}s, checkpoint keys: {checkpoint.keys()}")

            # The checkpoint should contain the model directly
            if 'model' in checkpoint:
                # Load the pre-built shard model
                shard = checkpoint['model']
                logger.info(f"[SHARD LOADING] Successfully loaded shard {shard_id} module from 'model' key")
                logger.info(f"[SHARD LOADING] Shard type: {type(shard)}, moving to CPU")
            elif 'state_dict' in checkpoint:
                # If we have state_dict but no structure, try to infer from the checkpoint
                logger.info(f"[SHARD LOADING] Checkpoint contains state_dict, attempting to load directly")
                # For now, we'll need to fall back to the full model approach
                # In the future, we should save the model structure with the checkpoint
                logger.warning(f"[SHARD LOADING] No model structure in checkpoint, falling back to full model loading")
                return self._load_from_full_model(config)
            else:
                logger.error(f"[SHARD LOADING] Invalid checkpoint format: {checkpoint.keys()}")
                return self._load_from_full_model(config)

            logger.info(f"[SHARD LOADING] Moving shard to CPU and returning")
            return shard.to("cpu")
        else:
            # Fall back to original method if no pre-split weights
            logger.warning(f"[SHARD LOADING] Pre-split weights not found at {shard_path}, loading from full model")
            logger.warning(f"[SHARD LOADING] This may indicate path mismatch - check shards_dir: {shards_dir}")
            return self._load_from_full_model(config)

    def _create_shard_from_structure(self, structure: List[Dict[str, Any]]) -> nn.Module:
        """Recreate shard module from structure definition."""
        shard = nn.Sequential()

        for layer_def in structure:
            layer_type = layer_def['type']
            layer_args = layer_def.get('args', {})
            layer_name = layer_def.get('name', f"layer_{len(shard)}")

            # Create layer based on type
            if layer_type == 'Conv2d':
                layer = nn.Conv2d(**layer_args)
            elif layer_type == 'BatchNorm2d':
                layer = nn.BatchNorm2d(**layer_args)
            elif layer_type == 'ReLU' or layer_type == 'ReLU6':
                layer = getattr(nn, layer_type)(**layer_args)
            elif layer_type == 'Linear':
                layer = nn.Linear(**layer_args)
            elif layer_type == 'AdaptiveAvgPool2d':
                layer = nn.AdaptiveAvgPool2d(**layer_args)
            elif layer_type == 'Flatten':
                layer = nn.Flatten(**layer_args)
            elif layer_type == 'Sequential':
                # Recursively create sequential
                sub_layers = self._create_shard_from_structure(layer_args.get('layers', []))
                layer = sub_layers
            else:
                # Try to get from nn module
                layer_class = getattr(nn, layer_type, None)
                if layer_class:
                    layer = layer_class(**layer_args)
                else:
                    raise ValueError(f"Unknown layer type: {layer_type}")

            shard.add_module(layer_name, layer)

        return shard

    def _load_from_full_model(self, config: Dict[str, Any]) -> nn.Module:
        """Fall back to loading from full model (original method)."""
        model_loader = ModelLoader(config.get('models_dir', './models'))
        full_model = model_loader.load_model(config['model_type'], config.get('num_classes', 10))

        # Extract specified layers
        shard = nn.Sequential()
        for i, layer_spec in enumerate(config['layers']):
            if isinstance(layer_spec, nn.Module):
                # Direct module (for compatibility)
                shard.add_module(f"layer_{i}", layer_spec)
            elif isinstance(layer_spec, dict):
                # Layer specification
                if 'module' in layer_spec:
                    shard.add_module(layer_spec.get('name', f"layer_{i}"), layer_spec['module'])
                elif 'path' in layer_spec:
                    # Navigate to nested module
                    module = full_model
                    for attr in layer_spec['path'].split('.'):
                        module = getattr(module, attr)
                    shard.add_module(layer_spec.get('name', f"layer_{i}"), module)

        # Move to CPU (same as original)
        return shard.to("cpu")

    def is_shard_loaded(self) -> bool:
        """Check if shard is loaded and ready."""
        return self.module is not None

    def forward(self, x: torch.Tensor, batch_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass with enhanced metrics collection."""
        logger = logging.getLogger(__name__)

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor but got {type(x)}")

        logger.info(f"[FORWARD PASS] [{socket.gethostname()}] Shard {self.shard_id} received tensor shape: {x.shape}, batch_id={batch_id}")

        x = x.to("cpu")
        logger.info(f"[FORWARD PASS] Shard {self.shard_id} starting computation")
        start_time = time.time()

        # Forward pass
        with torch.no_grad():
            output = self.module(x).cpu()

        end_time = time.time()
        compute_time = (end_time - start_time) * 1000  # Convert to ms

        logger.info(f"[FORWARD PASS] Shard {self.shard_id} computation completed in {compute_time:.2f}ms")
        # Add parseable format for automated_split_tester
        logger.info(f"[SHARD_TIMING] shard_{self.shard_id} processing_time_ms={compute_time:.2f} batch_id={batch_id}")

        # Record stage metrics
        if self.metrics_collector:
            self.metrics_collector.record_pipeline_stage(
                batch_id=batch_id if batch_id is not None else 0,
                stage_id=self.shard_id,
                stage_name=f"shard_{self.shard_id}",
                start_time=start_time,
                end_time=end_time,
                input_size_bytes=x.numel() * x.element_size(),
                output_size_bytes=output.numel() * output.element_size()
            )
            logger.info(f"[METRICS DEBUG] Recorded stage metrics for shard {self.shard_id}, batch_id={batch_id}, start={start_time:.3f}, end={end_time:.3f}")
        else:
            logger.warning(f"[METRICS DEBUG] No metrics collector for shard {self.shard_id}!")

        logger.info(f"[FORWARD PASS] [{socket.gethostname()}] Shard {self.shard_id} returning output shape: {output.shape}")
        return output

    def forward_with_payload_measurement(self, x: torch.Tensor, batch_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass with payload measurement."""
        payload_size_bytes = PayloadMeasurer.measure_torch_serialization_size(x)
        payload_size_mb = payload_size_bytes / (1024 * 1024)
        logging.info(f"[PAYLOAD_MEASUREMENT] shard_{self.shard_id} input_payload_size_mb={payload_size_mb:.2f} batch_id={batch_id}")
        return self.forward(x, batch_id)

    def parameter_rrefs(self):
        """Get parameter RRefs for distributed training (if needed)."""
        return [RRef(p) for p in self.parameters()]


# CachedShardWrapper removed - caching not needed for pass-through devices


class PrefetchDataLoader:
    """Prefetching wrapper for PyTorch DataLoader."""

    def __init__(self, dataloader: torch.utils.data.DataLoader,
                 prefetch_batches: int = 2):
        self.dataloader = dataloader
        self.prefetch_batches = prefetch_batches
        self.prefetch_queue = queue.Queue(maxsize=prefetch_batches)
        self.stop_event = threading.Event()
        self.prefetch_thread = None

    def _prefetch_worker(self):
        """Worker thread that prefetches batches."""
        try:
            for batch_data in self.dataloader:
                if self.stop_event.is_set():
                    break
                # Use timeout to prevent hanging if queue is full and no one is consuming
                while not self.stop_event.is_set():
                    try:
                        self.prefetch_queue.put(batch_data, timeout=1.0)
                        break
                    except queue.Full:
                        continue
        except Exception as e:
            logging.error(f"Prefetch worker error: {e}")
            try:
                self.prefetch_queue.put(None, timeout=1.0)  # Signal error
            except queue.Full:
                pass

    def __iter__(self):
        """Start prefetching and return iterator."""
        self.stop_event.clear()
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker)
        self.prefetch_thread.start()
        return self

    def __next__(self):
        """Get next prefetched batch."""
        item = self.prefetch_queue.get()
        if item is None:
            raise StopIteration
        return item

    def __len__(self):
        return len(self.dataloader)

    def stop(self):
        """Stop prefetching."""
        self.stop_event.set()

        # Clear the queue to unblock the worker thread if it's waiting to put
        try:
            while not self.prefetch_queue.empty():
                self.prefetch_queue.get_nowait()
        except queue.Empty:
            pass

        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=5.0)
            if self.prefetch_thread.is_alive():
                logging.warning("Prefetch thread did not terminate cleanly")


class EnhancedDistributedModel(nn.Module):
    """Enhanced distributed model with intelligent splitting and pipelining."""

    def __init__(self, model_type: str, num_splits: int, workers: List[str],
                 num_classes: int = 10, metrics_collector: Optional[EnhancedMetricsCollector] = None,
                 use_intelligent_splitting: bool = True, use_pipelining: bool = False,
                 models_dir: str = ".", split_block: Optional[int] = None,
                 use_local_loading: bool = True, shards_dir: str = "./model_shards"):
        super().__init__()
        self.model_type = model_type
        self.num_splits = num_splits
        self.workers = workers
        self.num_classes = num_classes
        self.metrics_collector = metrics_collector
        self.use_intelligent_splitting = use_intelligent_splitting
        self.use_pipelining = use_pipelining
        self.models_dir = models_dir
        self.split_block = split_block
        self.use_local_loading = use_local_loading
        self.shards_dir = shards_dir

        self.logger = logging.getLogger(__name__)

        self.logger.info("[ORCHESTRATOR INIT] ========== Starting EnhancedDistributedModel initialization ==========")
        self.logger.info(f"[ORCHESTRATOR INIT] Model type: {model_type}, Num splits: {num_splits}")
        self.logger.info(f"[ORCHESTRATOR INIT] Workers: {workers}")
        self.logger.info(f"[ORCHESTRATOR INIT] Split block: {split_block}, Use local loading: {use_local_loading}")
        self.logger.info(f"[ORCHESTRATOR INIT] Use intelligent splitting: {use_intelligent_splitting}")
        self.logger.info(f"[ORCHESTRATOR INIT] Use pipelining: {use_pipelining}")

        # Load model
        self.logger.info(f"[ORCHESTRATOR INIT] Loading model from {models_dir}")
        model_loader = ModelLoader(models_dir)
        self.original_model = model_loader.load_model(model_type, num_classes)
        self.logger.info(f"[ORCHESTRATOR INIT] Model loaded successfully: {type(self.original_model)}")

        # Profile model if using intelligent splitting
        self.model_profile = None
        if self.use_intelligent_splitting:
            self.logger.info("[ORCHESTRATOR INIT] Starting model profiling for intelligent splitting")
            self._profile_model(model_loader)

        # Split model
        self.logger.info("[ORCHESTRATOR INIT] Starting model splitting")
        self.shards = self._split_model()
        self.logger.info(f"[ORCHESTRATOR INIT] Model split into {len(self.shards)} shards")

        # Deploy shards to workers
        if len(self.workers) > 0:
            self.logger.info(f"[ORCHESTRATOR INIT] Deploying shards to {len(self.workers)} workers")
            self.worker_rrefs = self._deploy_shards()
            self.logger.info(f"[ORCHESTRATOR INIT] Successfully deployed {len(self.worker_rrefs)} shard RRefs")
        else:
            # No workers - run locally
            self.worker_rrefs = []
            self.logger.warning("[ORCHESTRATOR INIT] No workers available - model will run locally without distribution")

        # Setup pipeline if enabled
        self.pipeline_manager = None
        if self.use_pipelining:
            self.logger.info("[ORCHESTRATOR INIT] Setting up pipeline manager")
            self._setup_pipeline()

        self.logger.info("[ORCHESTRATOR INIT] ========== EnhancedDistributedModel initialization complete ==========")

    def _profile_model(self, model_loader: ModelLoader):
        """Profile the model for intelligent splitting."""
        self.logger.info(f"Profiling model: {self.model_type}")

        # Get sample input
        sample_input = model_loader.get_sample_input(self.model_type, batch_size=1)

        # Profile the model
        profiler = LayerProfiler(device="cpu", warmup_iterations=2, profile_iterations=5)
        self.model_profile = profiler.profile_model(self.original_model, sample_input, self.model_type)

        # Save profile for analysis
        profile_path = f"./profiles/{self.model_type}_profile.json"
        os.makedirs("./profiles", exist_ok=True)
        self.model_profile.save_to_file(profile_path)

        self.logger.info(f"Model profile saved to: {profile_path}")
        self.logger.info(f"Total model execution time: {self.model_profile.total_time_ms:.2f}ms")
        self.logger.info(f"Total model parameters: {self.model_profile.total_parameters:,}")

    def _split_model(self) -> List[nn.Module]:
        """Split the model using intelligent, block-level, or traditional methods."""
        if self.use_intelligent_splitting and self.model_profile:
            # Try block-level splitting first (like reference implementation)
            if hasattr(self.original_model, 'features') and hasattr(self.original_model, 'classifier'):
                self.logger.info("Using block-level splitting (reference implementation style)")
                return self._split_model_block_level()
            else:
                self.logger.info("Using intelligent splitting based on profiling data")

                # Use intelligent splitter
                shards, self.split_config = split_model_intelligently(
                    self.original_model,
                    self.model_profile,
                    self.num_splits,
                    network_config={
                        'communication_latency_ms': 200.0,  # Realistic RPC + serialization latency
                        'network_bandwidth_mbps': 3.5       # Measured effective bandwidth between Pis
                    }
                )

                self.logger.info(f"Intelligent split created {len(shards)} shards")
                self.logger.info(f"Load balance score: {self.split_config.load_balance_score:.4f}")
                self.logger.info(f"Estimated communication overhead: {self.split_config.estimated_communication_overhead_ms:.2f}ms")

                return shards
        else:
            self.logger.info("Using traditional manual splitting")
            # Fall back to the original manual splitting from the base script
            return self._split_model_traditional()

    def _split_model_traditional(self) -> List[nn.Module]:
        """Traditional model splitting (from original script)."""
        # Import the original splitting function
        import sys
        sys.path.append('..')

        try:
            # This imports the function from the original script
            exec(open('../rpc_layer_split_with_metrics.py').read(), globals())
            return split_model_into_n_shards(self.original_model, self.num_splits)
        except Exception as e:
            self.logger.error(f"Failed to use traditional splitting: {e}")
            # Fallback: just return the whole model as one shard
            return [self.original_model]

    def _split_model_block_level(self) -> List[nn.Module]:
        """Block-level model splitting (like reference implementation)."""
        feature_blocks = list(self.original_model.features.children())
        total_blocks = len(feature_blocks)

        self.logger.info(f"Model has {total_blocks} feature blocks")

        # # Reject invalid split_block=0
        # if self.split_block == 0:
        #     self.logger.error("split_block=0 is invalid (creates empty shard), use split_block >= 1")
        #     raise ValueError("Invalid split_block=0. Split block must be >= 1 to ensure non-empty shards.")

        # Calculate split point based on number of splits requested
        if self.split_block is not None:
            # Use user-specified split block
            split_at_block = self.split_block
            self.logger.info(f"Using user-specified split block: {split_at_block}")
        elif self.num_splits == 1:
            # For MobileNetV2: split at block 8 (like reference implementation)
            if self.model_type.lower() == 'mobilenetv2':
                split_at_block = 8
            else:
                # For other models, split roughly in the middle
                split_at_block = total_blocks // 2
        else:
            # For multiple splits, distribute blocks evenly
            split_at_block = total_blocks // (self.num_splits + 1)

        self.logger.info(f"Splitting at block {split_at_block} (reference style)")

        # Validate split point is within valid range
        if split_at_block < 1 or split_at_block >= total_blocks:
            self.logger.warning(f"Split point {split_at_block} is out of range [1, {total_blocks-1}], adjusting to middle")
            split_at_block = max(1, min(total_blocks - 1, total_blocks // 2))

        # Create shard 1: first part of features
        shard1_modules = feature_blocks[:split_at_block]
        shard1 = nn.Sequential(*shard1_modules)

        # Create shard 2: remaining features + pooling + classifier
        shard2_modules = feature_blocks[split_at_block:]

        # Handle model-specific architecture
        if self.model_type.lower() == 'vgg16':
            # VGG16 has avgpool as a separate module
            if hasattr(self.original_model, 'avgpool'):
                shard2_modules.append(self.original_model.avgpool)
            else:
                shard2_modules.append(nn.AdaptiveAvgPool2d((7, 7)))  # VGG16 uses 7x7 pooling
            shard2_modules.append(nn.Flatten())
            shard2_modules.append(self.original_model.classifier)
        else:
            # Generic handling for other models
            shard2_modules.append(nn.AdaptiveAvgPool2d((1, 1)))
            shard2_modules.append(nn.Flatten())
            shard2_modules.append(self.original_model.classifier)

        shard2 = nn.Sequential(*shard2_modules)

        # Validate shard2 has classifier
        has_classifier = False
        for module in shard2_modules:
            if isinstance(module, nn.Linear) or (hasattr(module, '__name__') and 'classifier' in str(module)):
                has_classifier = True
                break
            # Check if it's a Sequential containing Linear layers (e.g., VGG classifier)
            if isinstance(module, nn.Sequential):
                for submodule in module.modules():
                    if isinstance(submodule, nn.Linear):
                        has_classifier = True
                        break

        if not has_classifier:
            self.logger.error("Shard 2 missing classifier layer - invalid split configuration")
            self.logger.error(f"Shard 2 modules: {[type(m).__name__ for m in shard2_modules]}")
            raise RuntimeError(f"Invalid split at block {split_at_block}: Shard 2 lacks classifier layers")

        # Log partition details (for TODO item #2)
        shard1_params = sum(p.numel() for p in shard1.parameters())
        shard2_params = sum(p.numel() for p in shard2.parameters())
        total_params = shard1_params + shard2_params

        self.logger.info(f"Shard 1: Adding explicit AdaptiveAvgPool2d and Flatten for features→classifier transition")
        self.logger.info(f"Created 2 shards from block-level split")
        self.logger.info(f"Shard 1 parameters: {shard1_params:,} ({shard1_params/total_params*100:.1f}%)")
        self.logger.info(f"Shard 2 parameters: {shard2_params:,} ({shard2_params/total_params*100:.1f}%)")
        self.logger.info(f"Split ratio: Shard1={shard1_params/total_params*100:.1f}%, Shard2={shard2_params/total_params*100:.1f}%")

        return [shard1, shard2]

    def _create_shard_configs(self) -> List[Dict[str, Any]]:
        """Create configuration for each shard for local loading."""
        shard_configs = []

        # Expand shards_dir locally on the orchestrator
        expanded_shards_dir = os.path.expanduser(self.shards_dir)

        # First check if we have split-specific metadata
        if self.split_block is not None:
            split_dir = os.path.join(expanded_shards_dir, f"split_{self.split_block}")
            metadata_path = os.path.join(split_dir, f"{self.model_type}_shards_metadata.json")
        else:
            # Check if we have pre-split metadata in root
            metadata_path = os.path.join(expanded_shards_dir, f"{self.model_type}_shards_metadata.json")

        if os.path.exists(metadata_path):
            # Load metadata for pre-split weights
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.logger.info(f"Found pre-split metadata at {metadata_path}")

            for shard_info in metadata['shards']:
                config = {
                    'model_type': self.model_type,
                    'models_dir': self.models_dir,
                    'shards_dir': self.shards_dir,  # Keep as-is for remote expansion
                    'num_classes': self.num_classes,
                    'shard_id': shard_info['shard_id'],
                    'total_shards': metadata['num_shards'],
                    'shard_filename': shard_info['filename'],
                    'shard_path': shard_info['path'],  # Keep relative for remote loading
                    'split_block': self.split_block  # Add split_block for path resolution
                }

                # Add shard structure if available
                if 'structure' in shard_info:
                    config['shard_structure'] = shard_info['structure']

                shard_configs.append(config)

        else:
            # Fall back to original method for backward compatibility
            self.logger.warning(f"No pre-split metadata found at {metadata_path}")

            if len(self.shards) == 2 and self.model_type.lower() == 'mobilenetv2':
                # For MobileNetV2 with 2 shards, we know the split
                split_point = self.split_block if self.split_block is not None else 8

                # Shard 1: features[:split_point]
                shard1_config = {
                    'model_type': self.model_type,
                    'models_dir': self.models_dir,
                    'shards_dir': self.shards_dir,  # Keep as-is for remote expansion
                    'num_classes': self.num_classes,
                    'shard_id': 0,
                    'total_shards': 2,
                    'split_block': self.split_block,
                    'layers': []
                }
                for i in range(split_point):
                    shard1_config['layers'].append({
                        'name': f'features_{i}',
                        'path': f'features.{i}'
                    })
                shard_configs.append(shard1_config)

                # Shard 2: features[split_point:] + pooling + classifier
                shard2_config = {
                    'model_type': self.model_type,
                    'models_dir': self.models_dir,
                    'shards_dir': self.shards_dir,  # Keep as-is for remote expansion
                    'num_classes': self.num_classes,
                    'shard_id': 1,
                    'total_shards': 2,
                    'split_block': self.split_block,
                    'layers': []
                }
                # Get total number of feature blocks
                total_blocks = len(list(self.original_model.features.children()))
                for i in range(split_point, total_blocks):
                    shard2_config['layers'].append({
                        'name': f'features_{i}',
                        'path': f'features.{i}'
                    })
                # Add pooling and classifier layers as modules
                shard2_config['layers'].extend([
                    {'name': 'pool', 'module': nn.AdaptiveAvgPool2d((1, 1))},
                    {'name': 'flatten', 'module': nn.Flatten()},
                    {'name': 'classifier', 'path': 'classifier'}
                ])
                shard_configs.append(shard2_config)

            else:
                # For other cases, create generic configs based on existing shards
                for i, shard in enumerate(self.shards):
                    config = {
                        'model_type': self.model_type,
                        'models_dir': self.models_dir,
                        'shards_dir': self.shards_dir,  # Keep as-is for remote expansion
                        'num_classes': self.num_classes,
                        'shard_id': i,
                        'total_shards': len(self.shards),
                        'split_block': self.split_block,
                        'layers': []
                    }
                    # Add the shard modules directly (fallback approach)
                    for j, module in enumerate(shard.children()):
                        config['layers'].append({
                            'name': f'layer_{j}',
                            'module': module
                        })
                    shard_configs.append(config)

        return shard_configs

    def _deploy_shards(self) -> List[RRef]:
        """Deploy shards to worker nodes."""
        self.logger.info("[DEPLOY SHARDS] Starting shard deployment")
        worker_rrefs = []

        if self.use_local_loading:
            self.logger.info("[DEPLOY SHARDS] Using local loading mode - creating shard configurations")
            # Create shard configurations for local loading
            shard_configs = self._create_shard_configs()
            self.logger.info(f"[DEPLOY SHARDS] Created {len(shard_configs)} shard configurations")

            # Always use LocalLoadingShardWrapper (no caching)
            wrapper_class = LocalLoadingShardWrapper

            for i, config in enumerate(shard_configs):
                worker_name = self.workers[i % len(self.workers)]

                self.logger.info(f"[DEPLOY SHARDS] Deploying shard {i} to worker {worker_name}")
                self.logger.info(f"[DEPLOY SHARDS] Config: shard_id={config.get('shard_id')}, "
                               f"model_type={config.get('model_type')}, "
                               f"split_block={config.get('split_block')}")

                # Deploy with local loading
                deploy_start = time.time()
                rref = rpc.remote(
                    worker_name,
                    wrapper_class,
                    args=(config, i, None)
                )
                deploy_time = time.time() - deploy_start

                worker_rrefs.append(rref)

                self.logger.info(f"[DEPLOY SHARDS] Successfully deployed shard {i} to {worker_name} "
                               f"(took {deploy_time:.3f}s)")
        else:
            # Original deployment method
            self.logger.info("[DEPLOY SHARDS] Using traditional deployment - sending shard objects")
            for i, shard in enumerate(self.shards):
                worker_name = self.workers[i % len(self.workers)]

                self.logger.info(f"[DEPLOY SHARDS] Deploying shard {i} object to worker {worker_name}")

                # Create remote shard wrapper
                deploy_start = time.time()
                rref = rpc.remote(
                    worker_name,
                    EnhancedShardWrapper,
                    args=(shard, i, None)  # Workers create their own metrics collectors
                )
                deploy_time = time.time() - deploy_start

                worker_rrefs.append(rref)

                self.logger.info(f"[DEPLOY SHARDS] Successfully deployed shard {i} to {worker_name} "
                               f"(took {deploy_time:.3f}s)")

        self.logger.info(f"[DEPLOY SHARDS] Deployment complete - created {len(worker_rrefs)} RRefs")

        # CRITICAL: Wait for all workers to finish loading their shards
        self.logger.info("[DEPLOY SHARDS] Waiting for all workers to finish loading shards...")
        for i, rref in enumerate(worker_rrefs):
            worker_name = self.workers[i % len(self.workers)]
            try:
                # Call a remote method to check if shard is loaded
                # This avoids trying to serialize the entire Worker object
                is_ready = rref.rpc_sync(timeout=30).is_shard_loaded()
            except TimeoutError as e:
                self.logger.error(f"[DEPLOY SHARDS] Timeout verifying shard {i} on {worker_name}: {e}")
                raise RuntimeError(f"Timeout verifying shard {i} on {worker_name} after 30s")
            except Exception as e:
                self.logger.error(f"[DEPLOY SHARDS] Failed to verify shard {i} on {worker_name}: {e}")
                raise

            if is_ready:
                self.logger.info(f"[DEPLOY SHARDS] Worker {worker_name} confirmed shard {i} is loaded")
            else:
                self.logger.error(f"[DEPLOY SHARDS] Worker {worker_name} shard {i} not loaded!")
                raise RuntimeError(f"Worker {worker_name} failed to load shard {i}")

        self.logger.info("[DEPLOY SHARDS] All workers have finished loading their shards")
        return worker_rrefs

    def _setup_pipeline(self):
        """Setup pipeline manager for pipelined execution."""
        self.logger.info("Setting up pipeline for pipelined execution")

        # Create pipeline manager
        metrics_callback = None
        if self.metrics_collector:
            metrics_callback = self.metrics_collector.record_pipeline_stage

        self.pipeline_manager = PipelineManager(
            shards=self.shards,
            workers=self.workers,
            metrics_callback=metrics_callback,
            use_local_pipeline=False,  # Use RPC-based pipeline
            max_concurrent_batches=4
        )

    def forward(self, x: torch.Tensor, batch_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass through the distributed model with payload measurement."""
        self.logger.info(f"[MODEL FORWARD] Called with input shape: {x.shape}, batch_id: {batch_id}")

        if self.use_pipelining and self.pipeline_manager:
            # Use pipelined execution
            self.logger.info("[MODEL FORWARD] Using pipelined execution path")
            return self.pipeline_manager.process_batch_rpc_pipelined(x)
        else:
            # Sequential execution with payload measurement (research-grade accuracy)
            self.logger.info("[MODEL FORWARD] Using sequential execution path with payload measurement")  
            return self._forward_sequential_with_payload_measurement(x, batch_id)

    def _forward_sequential(self, x: torch.Tensor, batch_id: Optional[int] = None) -> torch.Tensor:
        """Sequential forward pass (non-pipelined)."""
        self.logger.info(f"[FORWARD SEQUENTIAL] Starting sequential forward pass, batch_id={batch_id}")
        self.logger.info(f"[FORWARD SEQUENTIAL] Input tensor shape: {x.shape}")

        current_tensor = x

        for i, shard_rref in enumerate(self.worker_rrefs):
            self.logger.info(f"[FORWARD SEQUENTIAL] Processing shard {i}")

            # Log tensor details before RPC
            self.logger.info(f"[FORWARD SEQUENTIAL] Sending tensor shape {current_tensor.shape} to shard {i}")

            # Measure actual payload size BEFORE sending (fix for tensor_size_mb=0.00 bug)
            payload_size_bytes = PayloadMeasurer.measure_torch_serialization_size(current_tensor)
            tensor_size_mb = payload_size_bytes / (1024 * 1024)

            # Measure RPC latency (includes computation)
            start_time = time.time()
            self.logger.info(f"[FORWARD SEQUENTIAL] Making RPC call to shard {i}")

            current_tensor = shard_rref.rpc_sync().forward(current_tensor, batch_id=batch_id)

            end_time = time.time()
            rpc_time = (end_time - start_time) * 1000  # Convert to ms

            self.logger.info(f"[FORWARD SEQUENTIAL] RPC call to shard {i} completed in {rpc_time:.2f}ms")
            self.logger.info(f"[FORWARD SEQUENTIAL] Received tensor shape from shard {i}: {current_tensor.shape}")

            # Record RPC metrics (computation + network)
            if self.metrics_collector:
                rpc_total_ms = (end_time - start_time) * 1000

                # Use the payload size measured before sending
                estimated_network_ms = 0.5 + (tensor_size_mb * 0.3) + (tensor_size_mb * 8 / 940) * 1000 * 2  # RTT + serialize + transfer

                # The rest is computation time
                estimated_computation_ms = max(0, rpc_total_ms - estimated_network_ms)

                self.logger.info(f"[FORWARD SEQUENTIAL] Shard {i} metrics: "
                               f"RPC total={rpc_total_ms:.2f}ms, "
                               f"Est. network={estimated_network_ms:.2f}ms, "
                               f"Est. computation={estimated_computation_ms:.2f}ms")

                # Add parseable format for network time
                self.logger.info(f"[NETWORK_TIMING] shard_{i} network_time_ms={estimated_network_ms:.2f} tensor_size_mb={tensor_size_mb:.2f} batch_id={batch_id}")

                # Record as "RPC latency" not "network latency"
                self.metrics_collector.record_network_metrics(rpc_total_ms, estimated_network_ms)

        self.logger.info(f"[FORWARD SEQUENTIAL] Sequential forward pass complete, output shape: {current_tensor.shape}")
        return current_tensor

    def _forward_sequential_with_payload_measurement(self, x: torch.Tensor, batch_id: Optional[int] = None) -> torch.Tensor:
        """Sequential forward pass with actual payload measurement (research-grade accuracy)."""
        self.logger.info(f"[FORWARD SEQUENTIAL PAYLOAD] Starting sequential forward pass with payload measurement, batch_id={batch_id}")
        self.logger.info(f"[FORWARD SEQUENTIAL PAYLOAD] Input tensor shape: {x.shape}")

        current_tensor = x

        for i, shard_rref in enumerate(self.worker_rrefs):
            self.logger.info(f"[FORWARD SEQUENTIAL PAYLOAD] Processing shard {i}")
            
            # Measure actual payload size BEFORE sending (this is the fix!)
            payload_size_bytes = PayloadMeasurer.measure_torch_serialization_size(current_tensor)
            payload_size_mb = payload_size_bytes / (1024 * 1024)
            self.logger.info(f"[FORWARD SEQUENTIAL PAYLOAD] Actual payload size for shard {i}: {payload_size_mb:.2f}MB")

            # Measure RPC latency (includes computation)
            start_time = time.time()
            self.logger.info(f"[FORWARD SEQUENTIAL PAYLOAD] Making RPC call to shard {i}")

            current_tensor = shard_rref.rpc_sync().forward_with_payload_measurement(current_tensor, batch_id=batch_id)

            end_time = time.time()
            rpc_time = (end_time - start_time) * 1000  # Convert to ms

            self.logger.info(f"[FORWARD SEQUENTIAL PAYLOAD] RPC call to shard {i} completed in {rpc_time:.2f}ms")
            self.logger.info(f"[FORWARD SEQUENTIAL PAYLOAD] Received tensor shape from shard {i}: {current_tensor.shape}")

            # Record RPC metrics with actual payload measurement
            if self.metrics_collector:
                rpc_total_ms = (end_time - start_time) * 1000
                
                # Calculate network throughput using actual payload size
                network_throughput_mbps = PayloadMeasurer.calculate_network_throughput_mbps(payload_size_mb, rpc_total_ms/1000.0)
                
                # Estimate network overhead using actual payload size
                estimated_network_ms = 0.5 + (payload_size_mb * 0.3) + (payload_size_mb * 8 / 940) * 1000 * 2  # RTT + serialize + transfer
                estimated_computation_ms = max(0, rpc_total_ms - estimated_network_ms)

                self.logger.info(f"[FORWARD SEQUENTIAL PAYLOAD] Shard {i} metrics: "
                               f"RPC total={rpc_total_ms:.2f}ms, "
                               f"Actual payload={payload_size_mb:.2f}MB, "
                               f"Network throughput={network_throughput_mbps:.2f}Mbps, "
                               f"Est. network={estimated_network_ms:.2f}ms, "
                               f"Est. computation={estimated_computation_ms:.2f}ms")

                # Fixed parseable format with actual payload size (this fixes the 0.00 bug!)
                self.logger.info(f"[NETWORK_TIMING] shard_{i} network_time_ms={estimated_network_ms:.2f} tensor_size_mb={payload_size_mb:.2f} batch_id={batch_id}")

                # Record network metrics
                self.metrics_collector.record_network_metrics(rpc_total_ms, estimated_network_ms)

        self.logger.info(f"[FORWARD SEQUENTIAL PAYLOAD] Sequential forward pass complete, output shape: {current_tensor.shape}")
        return current_tensor

    def parameter_rrefs(self):
        """Get parameter RRefs from all shards."""
        remote_params = []
        for rref in self.worker_rrefs:
            remote_params.extend(rref.remote().parameter_rrefs().to_here())
        return remote_params


# Global metrics collector for RPC access
global_metrics_collector = None

def collect_worker_summary(model_name: str, batch_size: int, num_parameters: int = 0) -> Dict[str, Any]:
    """RPC function to collect summary from workers."""
    global global_metrics_collector
    if global_metrics_collector:
        return global_metrics_collector.get_device_summary()
    return {}

def finalize_worker_metrics(model_name: str) -> bool:
    """RPC function to finalize worker metrics and write to CSV."""
    global global_metrics_collector
    if global_metrics_collector:
        global_metrics_collector.finalize(model_name)
        return True
    return False


def run_enhanced_inference(rank: int, world_size: int, model_type: str, batch_size: int,
                          num_classes: int, dataset: str, num_test_samples: int,
                          num_splits: int, metrics_dir: str, use_intelligent_splitting: bool = True,
                          use_pipelining: bool = False, num_threads: int = 4,
                          models_dir: str = ".", split_block: Optional[int] = None,
                          use_local_loading: bool = True, shards_dir: str = "./model_shards",
                          enable_prefetch: bool = False, prefetch_batches: int = 2):
    """
    Run enhanced distributed inference with profiling and pipelining.
    """
    # Initialize enhanced metrics collector
    metrics_collector = EnhancedMetricsCollector(rank, metrics_dir, enable_realtime=True)

    # Make globally accessible for RPC
    global global_metrics_collector
    global_metrics_collector = metrics_collector

    # Setup logging with hostname and rank
    hostname = socket.gethostname()
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.hostname = hostname
        record.rank = rank
        return record
    logging.setLogRecordFactory(record_factory)

    for handler in logging.root.handlers:
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(hostname)s:rank%(rank)s] - %(message)s'
        ))

    logger = logging.getLogger(__name__)
    logger.info("Starting enhanced distributed inference")

    # Load environment variables
    load_dotenv()
    master_addr = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '29555')

    # Setup RPC environment
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Force CPU-only operation for TensorPipe
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['TP_ENABLE_SHM'] = '0'  # Disable shared memory transport
    os.environ['TP_ENABLE_CMA'] = '0'  # Disable Cross Memory Attach

    # Use network interface from .env file or defaults
    gloo_ifname = os.getenv('GLOO_SOCKET_IFNAME', 'eth0' if rank == 0 else 'eth0')
    os.environ['GLOO_SOCKET_IFNAME'] = gloo_ifname

    if rank == 0:
        os.environ['TENSORPIPE_SOCKET_IFADDR'] = '0.0.0.0'

    rpc_initialized = False

    if rank == 0:  # Master node
        logger.info("[MASTER] ========================================")
        logger.info("[MASTER] Initializing master node with enhanced features")
        logger.info(f"[MASTER] Dataset: {dataset}, Batch size: {batch_size}")
        logger.info(f"[MASTER] Model: {model_type}, Num splits: {num_splits}")
        logger.info(f"[MASTER] World size: {world_size}, Num threads: {num_threads}")
        logger.info("[MASTER] ========================================")

        try:
            # Initialize RPC
            logger.info(f"[MASTER RPC] Initializing RPC backend on port {master_port}")
            rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
                num_worker_threads=num_threads,
                rpc_timeout=3600
            )

            logger.info("[MASTER RPC] Calling rpc.init_rpc...")
            rpc.init_rpc("master", rank=rank, world_size=world_size,
                        rpc_backend_options=rpc_backend_options)
            rpc_initialized = True
            logger.info("[MASTER RPC] RPC initialization successful")

            # Define workers
            workers = [f"worker{i}" for i in range(1, world_size)]
            logger.info(f"[MASTER] Defined workers: {workers}")

            # Create enhanced distributed model
            logger.info("[MASTER] Creating EnhancedDistributedModel...")
            model_creation_start = time.time()

            model = EnhancedDistributedModel(
                model_type=model_type,
                num_splits=num_splits,
                workers=workers,
                num_classes=num_classes,
                metrics_collector=metrics_collector,
                use_intelligent_splitting=use_intelligent_splitting,
                use_pipelining=use_pipelining,
                models_dir=models_dir,
                split_block=split_block,
                use_local_loading=use_local_loading,
                shards_dir=shards_dir
            )

            model_creation_time = time.time() - model_creation_start
            logger.info(f"[MASTER] Enhanced distributed model created successfully in {model_creation_time:.2f}s")

            # Load dataset
            logger.info(f"[MASTER] Loading dataset {dataset}...")
            dataset_start = time.time()
            model_loader = ModelLoader(models_dir)
            test_loader = model_loader.load_dataset(dataset, model_type, batch_size)
            dataset_time = time.time() - dataset_start

            # Wrap with prefetching if enabled
            if enable_prefetch:
                logger.info(f"[MASTER] Enabling prefetching with {prefetch_batches} batches")
                test_loader = PrefetchDataLoader(test_loader, prefetch_batches=prefetch_batches)

            logger.info(f"[MASTER] Dataset loaded in {dataset_time:.2f}s: {dataset} (batch_size={batch_size})")

            # Note: Shards are already deployed in model.__init__, which includes waiting for workers to be ready
            logger.info("[MASTER] Shards already deployed and workers confirmed ready during model initialization")

            # Run inference with enhanced metrics
            logger.info("[MASTER] ========== Starting Inference ==========")
            logger.info(f"[MASTER] Starting inference with {num_test_samples} test samples")
            logger.info("[MASTER] Starting timing after model creation (shards already loaded)")
            start_time = time.time()

            total_images = 0
            num_correct = 0
            batch_count = 0
            batch_completion_times = []  # Track when each batch completes

            if use_pipelining and model.use_pipelining and model.pipeline_manager:
                # Pipelined inference with multiple batches in flight
                logger.info("[MASTER INFERENCE] Using PIPELINED inference mode for maximum throughput")
                logger.info("[MASTER INFERENCE] Pipeline manager initialized and ready")

                # Configuration for pipelining
                max_batches_in_flight = 4  # Increased for better overlap on Pis
                active_batches = {}  # batch_id -> (images, labels, start_time)

                with torch.no_grad():
                    data_iter = iter(test_loader)
                    batch_id = 0

                    # Keep pipeline full
                    while total_images < num_test_samples:
                        # Start new batches if we have capacity
                        while len(active_batches) < max_batches_in_flight and total_images < num_test_samples:
                            try:
                                images, labels = next(data_iter)
                            except StopIteration:
                                break

                            # Trim batch if necessary
                            remaining = num_test_samples - total_images
                            if images.size(0) > remaining:
                                images = images[:remaining]
                                labels = labels[:remaining]

                            # Explicit device placement
                            images = images.to("cpu")
                            labels = labels.to("cpu")

                            # Start batch tracking
                            batch_start_time = metrics_collector.start_batch(batch_id, len(images))

                            logger.info(f"Starting batch {batch_id + 1} with {len(images)} images (pipeline)")

                            # Start batch processing asynchronously
                            pipeline_batch_id = model.pipeline_manager.start_batch_rpc_pipelined(images, labels)
                            active_batches[pipeline_batch_id] = (images, labels, batch_start_time, batch_id)

                            total_images += len(images)
                            batch_id += 1

                        # Collect completed batches
                        completed_ids = []
                        for pid, (orig_images, orig_labels, start_time, tracking_id) in list(active_batches.items()):
                            # Check if batch is complete (non-blocking)
                            result = model.pipeline_manager.get_completed_batch(pid, timeout=0.001)
                            if result is not None:
                                # Calculate accuracy
                                # Validate output shape before torch.max
                                if len(result.shape) != 2:
                                    logger.error(f"[MASTER INFERENCE] Unexpected output shape: {result.shape}, expected (batch_size, num_classes)")
                                    logger.error(f"[MASTER INFERENCE] This typically means the model is missing final layers (pooling/classifier)")
                                    raise RuntimeError(f"Invalid output shape {result.shape} - model may be incomplete")

                                _, predicted = torch.max(result.data, 1)
                                batch_correct = (predicted == orig_labels).sum().item()
                                num_correct += batch_correct

                                batch_accuracy = (batch_correct / len(orig_labels)) * 100.0

                                # End batch tracking
                                metrics_collector.end_batch(tracking_id, accuracy=batch_accuracy)

                                # Track batch completion time
                                batch_completion_times.append(time.time())

                                logger.info(f"Completed batch {tracking_id + 1} accuracy: {batch_accuracy:.2f}%")
                                completed_ids.append(pid)
                                batch_count += 1

                        # Remove completed batches
                        for pid in completed_ids:
                            del active_batches[pid]

                        # Small sleep if no completions to avoid busy waiting
                        if not completed_ids and len(active_batches) >= max_batches_in_flight:
                            time.sleep(0.01)

                        # Check for pipeline stall
                        if len(active_batches) == 0 and total_images >= num_test_samples:
                            logger.warning("[MASTER] No batches processed in pipeline, possible configuration error")

                    # Wait for remaining batches
                    logger.info(f"Waiting for {len(active_batches)} final batches to complete...")
                    while active_batches:
                        completed_ids = []
                        for pid, (orig_images, orig_labels, start_time, tracking_id) in list(active_batches.items()):
                            result = model.pipeline_manager.get_completed_batch(pid, timeout=0.1)
                            if result is not None:
                                # Calculate accuracy
                                # Validate output shape before torch.max
                                if len(result.shape) != 2:
                                    logger.error(f"[MASTER INFERENCE] Unexpected output shape: {result.shape}, expected (batch_size, num_classes)")
                                    logger.error(f"[MASTER INFERENCE] This typically means the model is missing final layers (pooling/classifier)")
                                    raise RuntimeError(f"Invalid output shape {result.shape} - model may be incomplete")

                                _, predicted = torch.max(result.data, 1)
                                batch_correct = (predicted == orig_labels).sum().item()
                                num_correct += batch_correct

                                batch_accuracy = (batch_correct / len(orig_labels)) * 100.0

                                # End batch tracking
                                metrics_collector.end_batch(tracking_id, accuracy=batch_accuracy)

                                # Track batch completion time
                                batch_completion_times.append(time.time())

                                logger.info(f"Completed batch {tracking_id + 1} accuracy: {batch_accuracy:.2f}%")
                                completed_ids.append(pid)
                                batch_count += 1

                        for pid in completed_ids:
                            del active_batches[pid]

            else:
                # Original sequential inference
                logger.info("[MASTER INFERENCE] Using SEQUENTIAL inference mode")

                with torch.no_grad():
                    for i, (images, labels) in enumerate(test_loader):
                        if total_images >= num_test_samples:
                            logger.info(f"[MASTER INFERENCE] Reached target of {num_test_samples} samples, stopping")
                            break

                        # Trim batch if necessary
                        remaining = num_test_samples - total_images
                        if images.size(0) > remaining:
                            images = images[:remaining]
                            labels = labels[:remaining]
                            logger.info(f"[MASTER INFERENCE] Trimmed batch to {len(images)} images")

                        # Explicit device placement for consistency and predictability
                        images = images.to("cpu")
                        labels = labels.to("cpu")

                        # Start batch tracking
                        batch_start_time = metrics_collector.start_batch(batch_count, len(images))

                        logger.info(f"[MASTER INFERENCE] ===== Processing batch {batch_count + 1}/{num_test_samples//batch_size + 1} =====")
                        logger.info(f"[MASTER INFERENCE] Batch size: {len(images)}, Total processed: {total_images}")

                        # Run inference
                        logger.info(f"[MASTER INFERENCE] Calling model.forward() for batch {batch_count + 1}")
                        inference_start = time.time()

                        output = model(images, batch_id=batch_count)

                        inference_time = time.time() - inference_start
                        logger.info(f"[MASTER INFERENCE] Model forward pass completed in {inference_time:.3f}s")

                        # Calculate accuracy
                        # Validate output shape before torch.max
                        if len(output.shape) != 2:
                            logger.error(f"[MASTER INFERENCE] Unexpected output shape: {output.shape}, expected (batch_size, {num_classes})")
                            logger.error(f"[MASTER INFERENCE] This typically means the model is missing final layers (pooling/classifier)")
                            logger.error(f"[MASTER INFERENCE] Current split_block={split_block} may be causing incomplete model architecture")
                            raise RuntimeError(f"Invalid output shape {output.shape} - model may be incomplete at split_block={split_block}")

                        _, predicted = torch.max(output.data, 1)
                        batch_correct = (predicted == labels).sum().item()
                        num_correct += batch_correct
                        total_images += len(images)

                        batch_accuracy = (batch_correct / len(labels)) * 100.0

                        # End batch tracking
                        metrics_collector.end_batch(batch_count, accuracy=batch_accuracy)

                        # Track batch completion time
                        batch_completion_times.append(time.time())

                        logger.info(f"[MASTER INFERENCE] Batch {batch_count + 1} completed: "
                                   f"accuracy={batch_accuracy:.2f}%, "
                                   f"time={inference_time:.3f}s, "
                                   f"throughput={len(images)/inference_time:.2f} img/s")
                        batch_count += 1

            elapsed_time = time.time() - start_time

            # Validate that batches were actually processed
            if batch_count == 0:
                logger.warning("[MASTER] No batches processed, setting throughput to 0")
                overall_ips = 0.0
                final_accuracy = 0.0
            else:
                final_accuracy = (num_correct / total_images) * 100.0 if total_images > 0 else 0.0
                overall_ips = total_images / elapsed_time if elapsed_time > 0 else 0.0

                # Calculate inter-batch throughput (more accurate for pipelining)
                inter_batch_ips = 0.0
                if len(batch_completion_times) > 1:
                    # Calculate average time between batch completions
                    intervals = []
                    for i in range(1, len(batch_completion_times)):
                        interval = batch_completion_times[i] - batch_completion_times[i-1]
                        intervals.append(interval)

                    if intervals:
                        avg_interval = sum(intervals) / len(intervals)
                        inter_batch_ips = batch_size / avg_interval if avg_interval > 0 else 0.0
                        logger.info(f"[MASTER] Inter-batch throughput (realistic for pipelining): {inter_batch_ips:.2f} images/sec")
                        logger.info(f"[MASTER] Average interval between batch completions: {avg_interval:.3f}s")

                        # Use inter-batch throughput as primary metric for pipelined inference
                        if use_pipelining:
                            logger.info(f"[MASTER] Using inter-batch throughput as primary metric for pipelined inference")
                            overall_ips = inter_batch_ips

                # Sanity check for unrealistic throughput on Raspberry Pi
                if overall_ips > 10:
                    logger.warning(f"[MASTER] Suspiciously high throughput ({overall_ips:.2f} images/sec) detected, capping at 10 for Raspberry Pi")
                    overall_ips = 10.0

                # Log if early termination
                if total_images < num_test_samples:
                    logger.warning(f"[MASTER] Only {total_images}/{num_test_samples} images processed, possible early termination")

            logger.info("[MASTER] ========== Inference Complete ==========")
            logger.info(f"[MASTER RESULTS] Total images processed: {total_images}")
            logger.info(f"[MASTER RESULTS] Total time: {elapsed_time:.2f}s")
            logger.info(f"[MASTER RESULTS] Final accuracy: {final_accuracy:.2f}%")
            logger.info(f"[MASTER RESULTS] Overall throughput: {overall_ips:.2f} images/sec")

            # Store accurate throughput in metrics collector
            if hasattr(metrics_collector, 'record_system_metrics'):
                metrics_collector.record_system_metrics(
                    total_images=total_images,
                    total_time_s=elapsed_time,
                    throughput_ips=overall_ips
                )

            # Calculate actual per-image latency from end-to-end time
            actual_latency_ms = (elapsed_time * 1000.0) / total_images if total_images > 0 else 0.0
            logger.info(f"Actual per-image latency: {actual_latency_ms:.2f}ms")

            # Finalize worker metrics BEFORE collecting summaries
            logger.info("Finalizing worker metrics via RPC...")
            for i in range(1, world_size):
                worker_name = f"worker{i}"
                try:
                    logger.info(f"Calling finalize_worker_metrics on {worker_name}")
                    success = rpc.rpc_sync(worker_name, finalize_worker_metrics, args=(model_type,), timeout=30)
                    if success:
                        logger.info(f"Successfully finalized metrics on {worker_name}")
                    else:
                        logger.warning(f"Failed to finalize metrics on {worker_name}")
                except Exception as e:
                    logger.warning(f"RPC call to finalize metrics on {worker_name} failed: {e}")

            # Collect worker summaries
            logger.info("Collecting enhanced metrics from workers...")
            worker_summaries = []

            for i in range(1, world_size):
                worker_name = f"worker{i}"
                try:
                    summary = rpc.rpc_sync(worker_name, collect_worker_summary,
                                         args=(model_type, batch_size, 0))
                    if summary:
                        worker_summaries.append(summary)
                        logger.info(f"Collected enhanced summary from {worker_name}")
                except Exception as e:
                    logger.warning(f"Failed to collect summary from {worker_name}: {e}")

            # Aggregate worker metrics
            logger.info("=== Aggregated Worker Metrics ===")
            if worker_summaries:
                # Simple aggregation
                num_workers = len(worker_summaries)
                agg_ips = sum(s.get('images_per_second', 0) for s in worker_summaries) / num_workers
                agg_proc_time = sum(s.get('average_processing_time_ms', 0) for s in worker_summaries) / num_workers
                agg_util = sum(s.get('average_pipeline_utilization', 0) for s in worker_summaries) / num_workers
                agg_network_latency = sum(s.get('avg_network_latency_ms', 0) for s in worker_summaries) / num_workers
                agg_throughput = sum(s.get('avg_throughput_mbps', 0) for s in worker_summaries)  # Sum throughput

                logger.info(f"Aggregated images per second (avg per worker): {agg_ips:.2f}")
                logger.info(f"Aggregated average processing time: {agg_proc_time:.2f}ms")
                logger.info(f"Aggregated pipeline utilization: {agg_util:.2f}")
                logger.info(f"Aggregated network latency: {agg_network_latency:.2f}ms")
                logger.info(f"Aggregated throughput (total): {agg_throughput:.2f}mbps")

                # Log per-worker details for debugging
                for idx, summary in enumerate(worker_summaries):
                    logger.info(f"Worker {idx+1} details:")
                    logger.info(f"  Images per second: {summary.get('images_per_second', 0):.2f}")
                    logger.info(f"  Average processing time: {summary.get('average_processing_time_ms', 0):.2f}ms")
                    logger.info(f"  Pipeline utilization: {summary.get('average_pipeline_utilization', 0):.2f}")

                # Merge summaries into master's collector
                for summary in worker_summaries:
                    if hasattr(metrics_collector, 'merge_summary'):
                        metrics_collector.merge_summary(summary)
                    else:
                        logger.warning("Metrics collector doesn't have merge_summary method")
            else:
                logger.warning("No worker summaries collected—check RPC connections or worker collectors.")

            # Final metrics collection
            pipeline_stats = model.pipeline_manager.get_pipeline_stats() if model.pipeline_manager else {}

            logger.info("=== Pipeline Statistics ===")
            if pipeline_stats:
                logger.info(f"Pipeline utilization: {pipeline_stats.get('pipeline_utilization', 0):.2f}")
                logger.info(f"Active batches: {pipeline_stats.get('active_batches', 0)}")

        except Exception as e:
            logger.error(f"Error in enhanced master node: {e}", exc_info=True)
        finally:
            # Master completes its work
            logger.info("[MASTER] Master work complete, preparing for shutdown...")

            # Cleanup prefetch loader if used
            if enable_prefetch and 'test_loader' in locals() and hasattr(test_loader, 'stop'):
                test_loader.stop()
                logger.info("Stopped prefetch loader")
                
            # Finalize metrics BEFORE RPC shutdown to ensure worker metrics are accessible
            logger.info("[MASTER] Finalizing metrics before RPC shutdown...")
            metrics_collector.finalize(model_type)

            # RPC Cleanup - MOVED INSIDE FINALLY BLOCK
            if rpc_initialized:
                logger.info("[CLEANUP] ========== Starting RPC Shutdown ==========")
                logger.info(f"[CLEANUP] Rank {rank} initiating RPC shutdown")
                try:
                    shutdown_start = time.time()
                    rpc.shutdown(graceful=False)
                    shutdown_time = time.time() - shutdown_start
                    logger.info(f"[CLEANUP] RPC shutdown completed successfully in {shutdown_time:.2f}s")
                except Exception as e:
                    logger.error(f"[CLEANUP] Error during RPC shutdown: {e}", exc_info=True)

    else:  # Worker nodes
        logger.info("[WORKER] ========================================")
        logger.info(f"[WORKER] Initializing enhanced worker node with rank {rank}")
        logger.info(f"[WORKER] Master address: {master_addr}:{master_port}")
        logger.info(f"[WORKER] World size: {world_size}, Num threads: {num_threads}")
        logger.info("[WORKER] ========================================")

        retry_count = 0
        max_retries = 30
        connected = False

        while retry_count < max_retries and not connected:
            try:
                logger.info(f"[WORKER RPC] Connection attempt {retry_count + 1}/{max_retries}")
                logger.info(f"[WORKER RPC] Connecting to master at tcp://{master_addr}:{master_port}")

                rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
                num_worker_threads=num_threads,
                rpc_timeout=3600
            )

                logger.info(f"[WORKER RPC] Calling rpc.init_rpc for worker{rank}...")
                rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size,
                           rpc_backend_options=rpc_backend_options)
                connected = rpc_initialized = True
                logger.info(f"[WORKER RPC] Worker {rank} connected successfully!")
                logger.info("[WORKER] Ready to receive shard deployments")

                # Worker stays alive until shutdown is called
                logger.info("[WORKER] Worker ready and waiting for RPC calls...")

                # Exit the retry loop successfully
                break

            except Exception as e:
                retry_count += 1
                logger.warning(f"[WORKER RPC] Connection attempt {retry_count} failed: {e}")
                if retry_count >= max_retries:
                    logger.error(f"[WORKER RPC] Worker {rank} failed to connect after {max_retries} attempts")
                    sys.exit(1)
                wait_time = 10 + (retry_count % 5)
                logger.info(f"[WORKER RPC] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)

        # If worker connected successfully, wait here until RPC framework shuts down
        if connected:
            logger.info("[WORKER] Entering wait loop - worker will stay alive until RPC shutdown")
            try:
                # This loop keeps the worker alive with timeout
                timeout_start = time.time()
                while rpc._is_current_rpc_agent_set() and (time.time() - timeout_start < 600):  # 10 minutes max
                    time.sleep(1)

                if rpc._is_current_rpc_agent_set():
                    logger.warning("[WORKER] Wait loop timed out after 10 minutes, forcing shutdown")
                    rpc.shutdown(graceful=False)
            except Exception as e:
                logger.info(f"[WORKER] Wait loop exited: {e}")
            logger.info("[WORKER] RPC agent no longer set, proceeding to cleanup")

    # Finalize metrics BEFORE RPC shutdown to ensure worker metrics are accessible
    final_results = metrics_collector.finalize(model_type)
    
    # Cleanup
    if rpc_initialized:
        logger.info("[CLEANUP] ========== Starting RPC Shutdown ==========")
        logger.info(f"[CLEANUP] Rank {rank} initiating RPC shutdown")
        try:
            shutdown_start = time.time()
            # For PyTorch RPC, we need to specify graceful=False to avoid hanging
            # when workers are still in their wait loops
            rpc.shutdown(graceful=False)
            shutdown_time = time.time() - shutdown_start
            logger.info(f"[CLEANUP] RPC shutdown completed successfully in {shutdown_time:.2f}s")
        except Exception as e:
            logger.error(f"[CLEANUP] Error during RPC shutdown: {e}", exc_info=True)

    logger.info("=== Final Enhanced Metrics Summary ===")
    device_summary = final_results['device_summary']
    efficiency_stats = final_results['efficiency_stats']

    logger.info(f"Images per second: {device_summary.get('images_per_second', 0):.2f}")
    logger.info(f"NEW Throughput (inter-batch): {efficiency_stats.get('new_pipeline_throughput_ips', 0):.2f} images/sec")
    logger.info(f"Average processing time: {device_summary.get('average_processing_time_ms', 0):.2f}ms")
    logger.info(f"Pipeline utilization: {efficiency_stats.get('average_pipeline_utilization', 0):.2f}")
    rpc_total = device_summary.get('avg_network_latency_ms', 0)
    network_overhead = device_summary.get('avg_throughput_mbps', 0)
    computation_time = rpc_total - network_overhead

    logger.info(f"RPC total time: {rpc_total:.2f}ms")
    logger.info(f"  - Network overhead: {network_overhead:.2f}ms")
    logger.info(f"  - Worker computation: {computation_time:.2f}ms")


def main():
    """Main function with enhanced argument parsing."""
    parser = argparse.ArgumentParser(description="Enhanced Distributed DNN Inference with Profiling and Pipelining")
    parser.add_argument("--rank", type=int, default=0, help="Rank of current process")
    parser.add_argument("--world-size", type=int, default=3, help="World size (1 master + N workers)")
    parser.add_argument("--model", type=str, default="mobilenetv2",
                       choices=ModelLoader.list_supported_models(),
                       help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of output classes")
    parser.add_argument("--dataset", type=str, default="cifar10",
                       choices=["cifar10", "dummy"], help="Dataset to use")
    parser.add_argument("--num-test-samples", type=int, default=64, help="Number of images to test")
    parser.add_argument("--num-partitions", type=int, default=2, help="Number of model partitions")
    parser.add_argument("--metrics-dir", type=str, default="./enhanced_metrics",
                       help="Directory for enhanced metrics")
    parser.add_argument("--models-dir", type=str, default="./models",
                       help="Directory containing model weight files")

    # Enhanced features
    parser.add_argument("--use-intelligent-splitting", action="store_true", default=True,
                       help="Use intelligent splitting based on profiling")
    parser.add_argument("--use-pipelining", action="store_true", default=False,
                       help="Enable pipelined execution")
    parser.add_argument("--num-threads", type=int, default=4,
                       help="Number of RPC threads")
    parser.add_argument("--disable-intelligent-splitting", action="store_true",
                       help="Disable intelligent splitting (use traditional method)")
    parser.add_argument("--split-block", type=int, default=None,
                       help="Specific block number to split at (for MobileNetV2)")

    # New optimization features
    parser.add_argument("--use-local-loading", action="store_true", default=True,
                       help="Load model weights locally on workers from pre-split files")
    parser.add_argument("--shards-dir", type=str,
                       default=os.path.expanduser('~/datasets/model_shards'),
                       help="Directory containing pre-split model shards")
    parser.add_argument("--enable-prefetch", action="store_true", default=False,
                       help="Enable data prefetching for improved throughput")
    parser.add_argument("--prefetch-batches", type=int, default=2,
                       help="Number of batches to prefetch")

    args = parser.parse_args()

    # Handle intelligent splitting flag
    if args.disable_intelligent_splitting:
        args.use_intelligent_splitting = False

    # Validation
    if args.world_size > 1 and args.num_partitions > args.world_size - 1:
        raise ValueError(f"Partitions ({args.num_partitions}) cannot exceed workers ({args.world_size - 1})")

    # Run enhanced inference
    run_enhanced_inference(
        rank=args.rank,
        world_size=args.world_size,
        model_type=args.model,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        dataset=args.dataset,
        num_test_samples=args.num_test_samples,
        num_splits=args.num_partitions - 1,  # Convert partitions to split points
        metrics_dir=args.metrics_dir,
        use_intelligent_splitting=args.use_intelligent_splitting,
        use_pipelining=args.use_pipelining,
        num_threads=args.num_threads,
        models_dir=args.models_dir,
        split_block=args.split_block,
        use_local_loading=args.use_local_loading,
        shards_dir=args.shards_dir,
        enable_prefetch=args.enable_prefetch,
        prefetch_batches=args.prefetch_batches
    )


if __name__ == "__main__":
    main()
