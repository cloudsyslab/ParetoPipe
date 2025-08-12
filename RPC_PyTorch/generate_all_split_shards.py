#!/usr/bin/env python3
"""
Generate compatible shards for all split directories.
Since we're using 2 shards for all models, we just need to copy the split_0 shards to all other splits.
"""

import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def copy_shards_to_all_splits():
    """Copy split_0 shards to all other split directories."""
    
    base_dir = "/home/xfu601/datasets/model_shards"
    models = ['resnet18', 'resnet50', 'alexnet', 'vgg16', 'inceptionv3']
    
    # For each model
    for model in models:
        logger.info(f"Processing {model}...")
        
        # Check if split_0 has the shards
        split0_dir = os.path.join(base_dir, "split_0")
        shard_files = [
            f"{model}_shard_0_of_2.pth",
            f"{model}_shard_1_of_2.pth",
            f"{model}_shards_metadata.json"
        ]
        
        # Verify all files exist in split_0
        all_exist = all(os.path.exists(os.path.join(split0_dir, f)) for f in shard_files)
        
        if not all_exist:
            logger.warning(f"Missing shard files for {model} in split_0, skipping...")
            continue
            
        # Copy to all other split directories
        for split_num in range(1, 32):  # split_1 through split_31
            split_dir = os.path.join(base_dir, f"split_{split_num}")
            
            # Create directory if it doesn't exist
            os.makedirs(split_dir, exist_ok=True)
            
            # Copy each shard file
            for shard_file in shard_files:
                src = os.path.join(split0_dir, shard_file)
                dst = os.path.join(split_dir, shard_file)
                
                try:
                    shutil.copy2(src, dst)
                    logger.debug(f"Copied {shard_file} to split_{split_num}")
                except Exception as e:
                    logger.error(f"Failed to copy {shard_file} to split_{split_num}: {e}")
        
        logger.info(f"Completed copying {model} shards to all splits")
    
    logger.info("Done! All models copied to all split directories.")

if __name__ == "__main__":
    copy_shards_to_all_splits()