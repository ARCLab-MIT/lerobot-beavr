#!/usr/bin/env python

import os
import json
import shutil
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import re
from PIL import Image
from tqdm import tqdm
import huggingface_hub

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleMoonLanderConverter:
    """
    A simplified converter for Moon Lander dataset.
    """
    
    def __init__(
        self,
        csv_dir: str,
        image_dir: str,
        output_dir: str,
        repo_id: str,
        fps: int = 5,
    ):
        self.csv_dir = Path(csv_dir)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.repo_id = repo_id
        self.fps = fps
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "meta").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        self.image_files = list(self.image_dir.glob("*.png"))
        logger.info(f"Found {len(self.image_files)} image files")
        
        # Sample a few image files
        sample_images = self.image_files[:5]
        logger.info(f"Sample image files: {[img.name for img in sample_images]}")
        
        # Find all CSV files
        self.csv_files = list(self.csv_dir.glob("*.csv"))
        logger.info(f"Found {len(self.csv_files)} CSV files")
    
    def convert(self):
        """
        Convert the dataset.
        """
        # Process each CSV file
        episode_lengths = {}
        
        for csv_file in tqdm(self.csv_files, desc="Processing episodes"):
            # Extract episode number from filename
            match = re.search(r'trajectory_(\d+)\.csv', csv_file.name)
            if not match:
                logger.warning(f"Could not extract episode number from {csv_file.name}, skipping")
                continue
            
            episode_idx = int(match.group(1))
            
            # Read CSV file
            try:
                df = pd.read_csv(csv_file)
                episode_lengths[episode_idx] = len(df)
            except Exception as e:
                logger.error(f"Error reading CSV file {csv_file}: {e}")
                continue
            
            # Create episode directory in images
            episode_dir = self.output_dir / "images" / f"episode_{episode_idx:06d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images for this episode
            for frame_idx in range(len(df)):
                # Construct image filename
                image_filename = f"img_traj_{episode_idx}_step_{frame_idx}.png"
                src_path = self.image_dir / image_filename
                
                if src_path.exists():
                    # Create destination path
                    dst_path = episode_dir / f"frame_{frame_idx:06d}.png"
                    
                    # Copy the image
                    shutil.copy2(src_path, dst_path)
                else:
                    logger.warning(f"Image not found: {src_path}")
        
        # Create metadata files
        self._create_metadata(episode_lengths)
        
        logger.info(f"Dataset conversion complete. Output directory: {self.output_dir}")
    
    def _create_metadata(self, episode_lengths):
        """
        Create metadata files.
        
        Args:
            episode_lengths: Dictionary mapping episode indices to lengths
        """
        # Create info.json
        info = {
            "dataset_type": "lerobot_dataset",
            "version": "2.1",
            "fps": self.fps,
            "num_episodes": len(episode_lengths),
            "num_frames": sum(episode_lengths.values()),
            "image_keys": ["observation.images"],
            "camera_keys": ["observation.images"],
            "state_keys": ["position", "velocity", "attitude", "angular_velocity"],
            "action_key": "action",
            "task_key": "task",
            "features": {
                "position": {
                    "dtype": "float32",
                    "shape": [3],
                    "names": ["x", "y", "z"]
                },
                "velocity": {
                    "dtype": "float32",
                    "shape": [3],
                    "names": ["v_x", "v_y", "v_z"]
                },
                "attitude": {
                    "dtype": "float32",
                    "shape": [4],
                    "names": ["q0", "q1", "q2", "q3"]
                },
                "angular_velocity": {
                    "dtype": "float32",
                    "shape": [3],
                    "names": ["w1", "w2", "w3"]
                },
                "action": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": ["T_x", "T_y", "T_z", "L_x", "L_y", "L_z"]
                },
                "timestamp": {
                    "dtype": "float32",
                    "shape": [1],
                    "names": ["timestamp"]
                },
                "frame_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": ["frame_index"]
                },
                "episode_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": ["episode_index"]
                },
                "task_index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": ["task_index"]
                },
                "index": {
                    "dtype": "int64",
                    "shape": [1],
                    "names": ["index"]
                },
                "observation.images": {
                    "dtype": "image",
                    "shape": [3, 256, 256],
                    "names": ["channels", "height", "width"]
                }
            }
        }
        
        with open(self.output_dir / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        # Create episodes.json
        episodes = []
        for episode_idx, length in episode_lengths.items():
            episodes.append({
                "episode_index": episode_idx,
                "length": length,
                "chunk_index": 0,  # All episodes in one chunk
                "task": "moon_landing"
            })
        
        with open(self.output_dir / "meta" / "episodes.json", "w") as f:
            json.dump(episodes, f, indent=2)
        
        # Create tasks.json
        tasks = [
            {
                "task_index": 0,
                "task": "moon_landing",
                "description": "Moon landing simulation"
            }
        ]
        
        with open(self.output_dir / "meta" / "tasks.json", "w") as f:
            json.dump(tasks, f, indent=2)
        
        # Create image_keys.json
        with open(self.output_dir / "meta" / "image_keys.json", "w") as f:
            json.dump(["observation.images"], f, indent=2)
    
    def push_to_hub(self, private=False):
        """
        Push the dataset to the Hugging Face Hub.
        
        Args:
            private: Whether to make the dataset private
        """
        try:
            # Create the repository
            api = huggingface_hub.HfApi()
            api.create_repo(repo_id=self.repo_id, repo_type="dataset", private=private, exist_ok=True)
            
            # Upload the dataset files
            logger.info(f"Uploading dataset to {self.repo_id}...")
            api.upload_folder(
                folder_path=str(self.output_dir),
                repo_id=self.repo_id,
                repo_type="dataset",
            )
            logger.info(f"Dataset pushed to {self.repo_id}")
        except Exception as e:
            logger.error(f"Failed to push dataset to Hub: {str(e)}")
            raise

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Moon Lander data to LeRobotDataset format")
    parser.add_argument("--csv-dir", type=str, required=True, help="Directory containing CSV files")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing image files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the dataset")
    parser.add_argument("--repo-id", type=str, required=True, help="Repository ID for the dataset")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second")
    parser.add_argument("--push", action="store_true", help="Push dataset to HuggingFace Hub")
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    
    args = parser.parse_args()
    
    # Create converter
    converter = SimpleMoonLanderConverter(
        csv_dir=args.csv_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
    )
    
    # Convert dataset
    converter.convert()
    
    # Push to HuggingFace Hub if requested
    if args.push:
        converter.push_to_hub(private=args.private)

if __name__ == "__main__":
    main() 