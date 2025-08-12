#!/usr/bin/env python3
"""
Generate shard files for all supported models at their natural split points.
This creates pre-split weight files for every possible split configuration.
"""

import os
import sys
import time
import logging
import argparse
from utils.model_split_info import get_model_split_info, get_default_split_ranges
from prepare_shards import main as prepare_shards_main

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_shards_for_model(model_type: str, splits_to_test: list = None):
    """Generate shard files for a specific model at all specified split points."""
    
    # Get model split information
    try:
        split_info = get_model_split_info(model_type)
    except ValueError as e:
        logger.error(f"Error: {e}")
        return [], []
    
    max_splits = split_info['max_splits']
    
    # Determine which splits to generate
    if splits_to_test is None:
        # Generate ALL splits for every model
        splits_to_test = list(range(max_splits + 1))
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating shards for {model_type.upper()}")
    logger.info(f"Model type: {split_info['split_type']} ({split_info['description']})")
    logger.info(f"Max splits: {max_splits}")
    logger.info(f"Splits to generate: {splits_to_test}")
    logger.info(f"{'='*60}")
    
    # Use the hardcoded shards directory
    base_shard_dir = os.path.expanduser("~/datasets/model_shards")
    os.makedirs(base_shard_dir, exist_ok=True)
    
    successful_splits = []
    failed_splits = []
    
    for num_splits in splits_to_test:
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing {num_splits} split(s) for {model_type}")
        logger.info(f"{'='*40}")
        
        try:
            # Create directory for this split configuration
            split_dir = os.path.join(base_shard_dir, f"split_{num_splits}")
            os.makedirs(split_dir, exist_ok=True)
            
            # Prepare arguments for prepare_shards
            # Always use num-splits=1 to create exactly 2 shards for our 2-worker system
            args = [
                'prepare_shards.py',
                '--model', model_type,
                '--num-splits', '1',
                '--num-classes', '10',
                '--shards-dir', split_dir
            ]
            
            # For MobileNetV2, specify which block to split at
            if model_type.lower() == 'mobilenetv2' and num_splits > 0:
                # num_splits represents the block number to split at
                args.extend(['--split-block', str(num_splits)])
            
            sys.argv = args
            
            # Run the shard preparation
            prepare_shards_main()
            
            successful_splits.append(num_splits)
            logger.info(f"✓ Successfully generated shards for {num_splits} split(s)")
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.5)
            
        except Exception as e:
            failed_splits.append(num_splits)
            logger.error(f"✗ Failed to generate shards for {num_splits} split(s): {e}")
            import traceback
            traceback.print_exc()
    
    return successful_splits, failed_splits


def generate_all_model_shards(models: list = None):
    """Generate shards for all specified models."""
    
    if models is None:
        # Default to all supported models
        models = ['mobilenetv2', 'resnet18', 'resnet50', 'vgg16', 'alexnet', 'inceptionv3']
    
    start_time = time.time()
    
    # Summary tracking
    all_results = {}
    
    for model in models:
        successful, failed = generate_shards_for_model(model)
        all_results[model] = {
            'successful': successful,
            'failed': failed
        }
    
    elapsed_time = time.time() - start_time
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SHARD GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    
    for model, results in all_results.items():
        logger.info(f"\n{model.upper()}:")
        logger.info(f"  Successful: {len(results['successful'])} splits")
        logger.info(f"  Failed: {len(results['failed'])} splits")
        if results['successful']:
            logger.info(f"  Generated splits: {results['successful']}")
        if results['failed']:
            logger.info(f"  Failed splits: {results['failed']}")
    
    # Calculate total disk space
    total_size = 0
    shard_base = os.path.expanduser("~/datasets/model_shards")
    if os.path.exists(shard_base):
        for root, dirs, files in os.walk(shard_base):
            for file in files:
                if file.endswith('.pth'):
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)
    
    logger.info(f"\nTotal disk space used: {total_size / (1024**2):.2f} MB")
    
    # Create deployment script
    create_deployment_script(models)


def create_deployment_script(models: list):
    """Create a script for copying shards to Pi nodes."""
    
    script_content = """#!/bin/bash
# Copy model shard files to Pi nodes

echo "Creating shard directories on Pi nodes..."
ssh cc@master-pi "mkdir -p ~/projects/distributed-inference/model_shards"
ssh cc@core-pi "mkdir -p ~/projects/distributed-inference/model_shards"

echo "Copying all shard files to master-pi..."
scp -r model_shards/* cc@master-pi:~/projects/distributed-inference/model_shards/

echo "Copying all shard files to core-pi..."
scp -r model_shards/* cc@core-pi:~/projects/distributed-inference/model_shards/
"""
    
    script_content += """
echo "Done! All shard files copied to Pi nodes."
"""
    
    with open('copy_model_shards_to_pis.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('copy_model_shards_to_pis.sh', 0o755)
    logger.info("\nCreated copy_model_shards_to_pis.sh script for easy deployment")


def main():
    parser = argparse.ArgumentParser(description="Generate shard files for models")
    parser.add_argument("--models", nargs='+', 
                       help="Models to generate shards for (default: all)")
    parser.add_argument("--model", type=str,
                       help="Single model to generate shards for")
    parser.add_argument("--splits", nargs='+', type=int,
                       help="Specific split numbers to generate (default: model-specific)")
    
    args = parser.parse_args()
    
    if args.model:
        # Generate for a single model
        generate_shards_for_model(args.model, args.splits)
    else:
        # Generate for multiple models
        models = args.models
        generate_all_model_shards(models)


if __name__ == "__main__":
    main()