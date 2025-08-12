#!/usr/bin/env python3
"""
Generate shard files for all possible split points in MobileNetV2.
This creates pre-split weight files for every possible block split (0-18).
"""

import os
import sys
import time
import logging
from prepare_shards import main as prepare_shards_main

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_all_shards():
    """Generate shard files for all 19 possible split points."""
    
    # MobileNetV2 has 19 blocks (0-18)
    total_blocks = 19
    
    logger.info("=" * 60)
    logger.info("Generating shards for all MobileNetV2 split points")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Create a subdirectory for each split configuration
    base_shard_dir = "./model_shards"
    os.makedirs(base_shard_dir, exist_ok=True)
    
    successful_splits = []
    failed_splits = []
    
    for split_block in range(total_blocks):
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing split at block {split_block}")
        logger.info(f"{'='*40}")
        
        try:
            # Create directory for this split
            split_dir = os.path.join(base_shard_dir, f"split_{split_block}")
            os.makedirs(split_dir, exist_ok=True)
            
            # Prepare arguments for prepare_shards
            sys.argv = [
                'prepare_shards.py',
                '--model', 'mobilenetv2',
                '--num-splits', '1',  # Always 1 split = 2 shards
                '--split-block', str(split_block),
                '--num-classes', '10',
                '--shards-dir', split_dir
            ]
            
            # Run the shard preparation
            prepare_shards_main()
            
            successful_splits.append(split_block)
            logger.info(f"✓ Successfully generated shards for split block {split_block}")
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.5)
            
        except Exception as e:
            failed_splits.append(split_block)
            logger.error(f"✗ Failed to generate shards for split block {split_block}: {e}")
            import traceback
            traceback.print_exc()
    
    elapsed_time = time.time() - start_time
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SHARD GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed_time:.2f} seconds")
    logger.info(f"Successful: {len(successful_splits)} splits")
    logger.info(f"Failed: {len(failed_splits)} splits")
    
    if successful_splits:
        logger.info(f"\nSuccessful splits: {successful_splits}")
    if failed_splits:
        logger.info(f"\nFailed splits: {failed_splits}")
    
    # Calculate total size
    total_size = 0
    for split_block in range(total_blocks):
        split_dir = os.path.join(base_shard_dir, f"split_{split_block}")
        if os.path.exists(split_dir):
            for file in os.listdir(split_dir):
                if file.endswith('.pth'):
                    file_path = os.path.join(split_dir, file)
                    total_size += os.path.getsize(file_path)
    
    logger.info(f"\nTotal disk space used: {total_size / (1024**2):.2f} MB")
    
    # Create a convenience script for copying to Pis
    with open('copy_shards_to_pis.sh', 'w') as f:
        f.write("""#!/bin/bash
# Copy all shard files to Pi nodes

echo "Creating shard directories on Pi nodes..."
ssh cc@master-pi "mkdir -p ~/projects/distributed-inference/model_shards"
ssh cc@core-pi "mkdir -p ~/projects/distributed-inference/model_shards"

echo "Copying shard files to master-pi..."
scp -r model_shards/* cc@master-pi:~/projects/distributed-inference/model_shards/

echo "Copying shard files to core-pi..."
scp -r model_shards/* cc@core-pi:~/projects/distributed-inference/model_shards/

echo "Done! Shard files copied to all Pi nodes."
""")
    
    os.chmod('copy_shards_to_pis.sh', 0o755)
    logger.info("\nCreated copy_shards_to_pis.sh script for easy deployment")


if __name__ == "__main__":
    generate_all_shards()