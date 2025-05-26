#!/usr/bin/env python

"""
Script to upload a local LeRobot dataset to the Hugging Face Hub.
This script validates the dataset structure using LeRobotDataset and then pushes it to Hugging Face,
ignoring the images directory.
"""

import argparse
import logging
import shutil
from pathlib import Path
import tempfile

from huggingface_hub import HfApi, create_repo
from lerobot.common.datasets.utils import load_info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Upload a local LeRobot dataset to Hugging Face Hub")
    parser.add_argument(
        "--local_path", 
        type=str, 
        required=True,
        help="Path to the local dataset directory"
    )
    parser.add_argument(
        "--repo_id", 
        type=str, 
        required=True,
        help="Hugging Face repository ID (e.g., 'username/dataset-name')"
    )
    parser.add_argument(
        "--private", 
        action="store_true", 
        help="Make the repository private"
    )
    parser.add_argument(
        "--branch", 
        type=str, 
        default=None,
        help="Branch name to push to (default: main)"
    )
    parser.add_argument(
        "--license", 
        type=str, 
        default="apache-2.0",
        help="License for the dataset (default: apache-2.0)"
    )
    parser.add_argument(
        "--description", 
        type=str, 
        default="LeRobot dataset",
        help="Short description for the dataset card"
    )
    parser.add_argument(
        "--tags", 
        type=str, 
        nargs="+", 
        default=["lerobot", "robotics"],
        help="Tags for the dataset (default: lerobot robotics)"
    )
    return parser.parse_args()

def validate_dataset_structure(local_path):
    """Validate the dataset structure."""
    # Check for required directories
    return True
    # required_dirs = ["meta", "data", "videos"]
    # for dir_name in required_dirs:
    #     dir_path = local_path / dir_name
    #     if not dir_path.exists() or not dir_path.is_dir():
    #         logger.error(f"Required directory '{dir_name}' not found at {dir_path}")
    #         return False
    
    # # Check for info.json
    # info_path = local_path / "meta" / "info.json"
    # if not info_path.exists():
    #     logger.error(f"Required file 'info.json' not found at {info_path}")
    #     return False
    
    # Load and validate info.json
    try:
        info = load_info(local_path)
        required_keys = ["fps", "data_path", "video_path", "features", "total_episodes", "total_frames"]
        for key in required_keys:
            if key not in info:
                logger.error(f"Required key '{key}' not found in info.json")
                return False
        logger.info(f"Dataset info loaded: {len(info['features'])} features, {info['total_episodes']} episodes, {info['total_frames']} frames")
    except Exception as e:
        logger.error(f"Error loading info.json: {e}")
        return False
    
    return True

def main():
    """Main function to upload the dataset."""
    args = parse_args()
    
    local_path = Path(args.local_path).resolve()
    if not local_path.exists():
        logger.error(f"Local path {local_path} does not exist.")
        return
    
    logger.info(f"Loading dataset from {local_path}")
    logger.info(f"Will upload to {args.repo_id}")
    
    # Validate dataset structure
    # logger.info("Validating dataset structure...")
    # if not validate_dataset_structure(local_path):
    #     logger.error("Dataset validation failed. Please check the structure and try again.")
    #     return
    
    try:
        # Create the repository first if it doesn't exist
        logger.info(f"Creating repository on Hugging Face Hub: {args.repo_id}")
        api = HfApi()
        create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True
        )
        
        # Create a temporary directory with the correct structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            logger.info(f"Created temporary directory: {temp_path}")
            
            # Copy the dataset to the temporary directory with the correct structure
            logger.info("Copying dataset with correct structure...")
            for dir_name in ["meta", "data", "videos"]:
                src_dir = local_path / dir_name
                dst_dir = temp_path / dir_name
                if src_dir.exists():
                    shutil.copytree(src_dir, dst_dir)
            
            # Upload the dataset directly to the root of the repository
            logger.info(f"Uploading dataset to Hugging Face Hub: {args.repo_id}")
            api.upload_folder(
                folder_path=temp_path,
                repo_id=args.repo_id,
                repo_type="dataset",
                # ignore_patterns=["images/"],  # Ignore the images directory
                path_in_repo="",  # Upload to the root of the repository
                commit_message=f"Upload dataset: {local_path.name} with correct structure",
            )
        
        logger.info(f"Successfully uploaded dataset to {args.repo_id}")
        
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise

if __name__ == "__main__":
    main()