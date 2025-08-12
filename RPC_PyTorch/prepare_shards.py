#!/usr/bin/env python3
"""
Script to pre-split models and save individual shards for distributed inference.
This should be run once before distributed inference to prepare the shard files.
"""

import os
import sys
import argparse
import logging

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from model_splitter import ModelSplitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pre-split models for distributed inference")
    parser.add_argument("--model", type=str, default="mobilenetv2", 
                       help="Model type to split (e.g., mobilenetv2, resnet18)")
    parser.add_argument("--num-splits", type=int, default=1, 
                       help="Number of split points (resulting in num_splits+1 shards)")
    parser.add_argument("--split-block", type=int, default=None, 
                       help="Specific block to split at (for MobileNetV2)")
    parser.add_argument("--num-classes", type=int, default=10, 
                       help="Number of output classes")
    parser.add_argument("--models-dir", type=str, default="./models", 
                       help="Directory containing model weights")
    parser.add_argument("--shards-dir", type=str, default="./model_shards", 
                       help="Directory to save shard files")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Model Pre-Splitting Tool")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Number of splits: {args.num_splits} (will create {args.num_splits + 1} shards)")
    logger.info(f"Split block: {args.split_block if args.split_block else 'auto'}")
    logger.info(f"Number of classes: {args.num_classes}")
    logger.info(f"Models directory: {args.models_dir}")
    logger.info(f"Shards directory: {args.shards_dir}")
    logger.info("=" * 60)
    
    # Create splitter
    splitter = ModelSplitter(args.models_dir, args.shards_dir)
    
    # Split and save model
    try:
        metadata = splitter.split_and_save_model(
            args.model,
            args.num_splits,
            args.split_block,
            args.num_classes
        )
        
        logger.info("\n" + "=" * 60)
        logger.info("SUCCESS: Model split and saved")
        logger.info("=" * 60)
        
        # Print shard summary
        total_params = sum(s['num_parameters'] for s in metadata['shards'])
        logger.info(f"\nShard Summary:")
        for shard in metadata['shards']:
            pct = (shard['num_parameters'] / total_params) * 100
            logger.info(f"  Shard {shard['shard_id']}: {shard['num_parameters']:,} params ({pct:.1f}%)")
            logger.info(f"    File: {shard['filename']}")
        
        logger.info(f"\nMetadata saved to: {args.shards_dir}/{args.model}_shards_metadata.json")
        logger.info("\nYou can now run distributed inference with --use-local-loading flag")
        
    except Exception as e:
        logger.error(f"Failed to split model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()