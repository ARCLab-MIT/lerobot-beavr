#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import shutil
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
from PIL import Image
import datasets
from tqdm import tqdm
import huggingface_hub

def _ensure_minimum_ndim(obj):
    """Recursively ensure that every numeric value is at least a 1-D NumPy array."""
    if isinstance(obj, dict):
        return {k: _ensure_minimum_ndim(v) for k, v in obj.items()}
    # If it is already an array, use np.atleast_1d
    elif isinstance(obj, np.ndarray):
        return np.atleast_1d(obj)
    # If it is a numeric type, wrap it.
    elif isinstance(obj, (int, float, complex, np.integer, np.floating, np.complexfloating)):
        return np.array([obj])
    # If it is a list, try to convert to a NumPy array (and then ensure at least 1-D)
    elif isinstance(obj, list):
        try:
            return np.atleast_1d(np.array(obj))
        except Exception:
            return obj
    # Fallback: try to force it with np.atleast_1d
    try:
        return np.atleast_1d(obj)
    except Exception:
        return obj


from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import (
    INFO_PATH, 
    EPISODES_PATH, 
    STATS_PATH, 
    EPISODES_STATS_PATH,
    TASKS_PATH,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PARQUET_PATH,
    DEFAULT_VIDEO_PATH,
    DEFAULT_IMAGE_PATH,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CSVToLeRobotDatasetConverter:
    """
    Converts CSV trajectory data and associated images to LeRobotDataset v2.1 format.
    
    This class handles the conversion of CSV files containing trajectory data (position, velocity, 
    attitude, etc.) and associated images into the LeRobotDataset v2.1 format, which can be used
    for training and evaluation with the LeRobot framework.
    """
    
    def __init__(
        self,
        csv_dir: Union[str, Path],
        image_dir: Union[str, Path],
        output_dir: Union[str, Path],
        repo_id: str,
        fps: Optional[int] = None,
        task_name: str = "meta-RL policy rollout",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        image_key: str = "observation.images",
        image_pattern: str = "img_traj_{episode}_step_{frame}",
        image_extension: str = ".png",
        csv_pattern: str = "trajectory_{episode}.csv",
        use_videos: bool = False,
        debug: bool = False,
        image_height: int = 256,
        image_width: int = 256,
    ):
        """
        Initialize the converter.
        
        Args:
            csv_dir: Directory containing CSV files
            image_dir: Directory containing image files
            output_dir: Directory to save the dataset
            repo_id: Repository ID for the dataset
            fps: Frames per second (if None, calculated from data)
            task_name: Name of the task
            chunk_size: Number of episodes per chunk
            image_key: Key for image feature
            image_pattern: Pattern for image filenames
            image_extension: File extension for images
            csv_pattern: Pattern for CSV filenames
            use_videos: Whether to encode images as videos
            debug: Whether to enable debug logging
            image_height: Height of images in pixels
            image_width: Width of images in pixels
        """
        self.csv_dir = Path(csv_dir)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.repo_id = repo_id
        self.fps = fps
        self.task_name = task_name
        self.chunk_size = chunk_size
        self.image_key = image_key
        self.image_pattern = image_pattern
        self.image_extension = image_extension
        self.csv_pattern = csv_pattern
        self.use_videos = use_videos
        self.debug = debug
        self.image_height = image_height
        self.image_width = image_width
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize features and stats
        self.features = {}
        self.stats = {}
        self.episodes_stats = {}
        
        # Set up logging
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Verify that CSV and image directories exist
        if not self.csv_dir.exists():
            raise ValueError(f"CSV directory does not exist: {self.csv_dir}")
        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {self.image_dir}")
        
        # Check for image files
        image_files = list(self.image_dir.glob(f"*{self.image_extension}"))
        if not image_files:
            logger.warning(f"No image files found in {self.image_dir} with extension {self.image_extension}")
        else:
            logger.info(f"Found {len(image_files)} image files")
            
            # Log a few sample image filenames
            sample_images = image_files[:5]
            logger.info(f"Sample image files: {[img.name for img in sample_images]}")

    def _copy_image(self, episode_index: int, frame_index: int, dest_path: Path) -> None:
        """
        Copy an image file from the source directory to the destination path.
        
        Args:
            episode_index: Episode index
            frame_index: Frame index
            dest_path: Destination path
        """
        # Create the destination directory if it doesn't exist
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format the source image filename using the pattern
        source_filename = self.image_pattern.format(episode=episode_index, frame=frame_index) + self.image_extension
        source_path = self.image_dir / source_filename
        
        # Check if the source image exists
        if not source_path.exists():
            logger.warning(f"Image file not found: {source_path}")
            # Create a blank image as a fallback
            blank_image = Image.new('RGB', (self.image_height, self.image_width), color='black')
            blank_image.save(dest_path)
            return
        
        # Copy the image
        try:
            shutil.copy2(source_path, dest_path)
            if self.debug:
                logger.debug(f"Copied image from {source_path} to {dest_path}")
        except Exception as e:
            logger.error(f"Failed to copy image: {e}")
            # Create a blank image as a fallback
            blank_image = Image.new('RGB', (self.image_height, self.image_width), color='black')
            blank_image.save(dest_path)

    def _create_video(self, episode_index: int, frame_count: int, chunk_dir: Path) -> None:
        """
        Create a video from a sequence of images.
        
        Args:
            episode_index: Episode index
            frame_count: Number of frames
            chunk_dir: Directory to save the video
        """
        # Import cv2 here to avoid dependency issues
        try:
            import cv2
        except ImportError:
            logger.error("OpenCV (cv2) is required for video encoding. Please install it with 'pip install opencv-python'.")
            return
        
        # Create the video directory
        video_dir = chunk_dir / self.image_key
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Create the video file path
        video_path = video_dir / f"episode_{episode_index:06d}.mp4"
        
        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(str(video_path), fourcc, self.fps, (self.image_width, self.image_height))
        
        # Add each frame to the video
        for frame_index in range(frame_count):
            # Format the source image filename using the pattern
            source_filename = self.image_pattern.format(episode=episode_index, frame=frame_index) + self.image_extension
            source_path = self.image_dir / source_filename
            
            # Check if the source image exists
            if not source_path.exists():
                logger.warning(f"Image file not found: {source_path}")
                # Create a blank frame as a fallback
                blank_frame = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                video.write(blank_frame)
                continue
            
            # Read the image and add it to the video
            try:
                frame = cv2.imread(str(source_path))
                if frame is None:
                    logger.warning(f"Failed to read image: {source_path}")
                    blank_frame = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                    video.write(blank_frame)
                    continue
                
                # Resize the frame if necessary
                if frame.shape[:2] != (self.image_height, self.image_width):
                    frame = cv2.resize(frame, (self.image_width, self.image_height))
                
                video.write(frame)
            except Exception as e:
                logger.error(f"Failed to process image for video: {e}")
                blank_frame = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
                video.write(blank_frame)
        
        # Release the video
        video.release()
        
        if self.debug:
            logger.debug(f"Created video: {video_path}")

    def _find_csv_files(self) -> List[Tuple[int, Path]]:
        """
        Find all CSV files in the CSV directory that match the pattern.
        
        Returns:
            List of tuples containing episode index and path to CSV file
        """
        csv_files = []
        
        # Check if the pattern contains a placeholder for the episode index
        if "{episode}" in self.csv_pattern:
            # Find all CSV files that match the pattern
            for csv_path in self.csv_dir.glob("*.csv"):
                try:
                    # Extract the episode index from the filename
                    filename = csv_path.name
                    pattern_parts = self.csv_pattern.split("{episode}")
                    if len(pattern_parts) != 2:
                        logger.warning(f"Invalid CSV pattern: {self.csv_pattern}")
                        continue
                    
                    prefix, suffix = pattern_parts
                    if not filename.startswith(prefix) or not filename.endswith(suffix):
                        continue
                    
                    episode_str = filename[len(prefix):-len(suffix)]
                    episode_index = int(episode_str)
                    
                    csv_files.append((episode_index, csv_path))
                except Exception as e:
                    logger.warning(f"Failed to parse episode index from {csv_path}: {e}")
        else:
            # If no pattern is specified, use all CSV files and assign sequential indices
            for i, csv_path in enumerate(sorted(self.csv_dir.glob("*.csv"))):
                csv_files.append((i, csv_path))
        
        # Sort by episode index
        csv_files.sort(key=lambda x: x[0])
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.csv_dir} that match the pattern {self.csv_pattern}")
        
        logger.info(f"Found {len(csv_files)} CSV files")
        
        return csv_files

    def _process_csv_file(self, episode_index: int, csv_path: Path) -> pd.DataFrame:
        """
        Process a CSV file to add frame index and episode index.
        
        Args:
            episode_index: Episode index
            csv_path: Path to CSV file
            
        Returns:
            Processed DataFrame
        """
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Add frame index
        df["frame_index"] = np.arange(len(df))
        
        # Add episode index
        df["episode_index"] = episode_index
        
        # Add task index (always 0 for now)
        df["task_index"] = 0
        
        # Add index (unique identifier for each row)
        df["index"] = np.arange(len(df)) + episode_index * 1000000  # Ensure uniqueness across episodes
        
        # Add task (string identifier for the task)
        df["task"] = self.task_name
        
        return df

    def _restructure_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Restructure the DataFrame to match the expected schema.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Restructured DataFrame
        """
        # Create a new dictionary to hold the restructured data
        restructured_data = {}
        
        # Process each feature
        for feature_name, feature_info in self.features.items():
            # Skip the image feature, as it will be handled separately
            if feature_name == self.image_key:
                continue
            
            # Get the column names for this feature
            column_names = feature_info.get("names", [])
            
            # Check if all columns exist in the DataFrame
            if all(col in df.columns for col in column_names):
                # Extract the columns for this feature
                feature_data = df[column_names].values
                
                # Ensure the feature data is properly shaped
                if len(column_names) == 1:
                    # For single-column features, ensure they're 1D arrays
                    restructured_data[feature_name] = feature_data.flatten()
                else:
                    # For multi-column features, ensure each row is a separate entry
                    restructured_data[feature_name] = [row for row in feature_data]
            else:
                # Feature not found in CSV, create placeholder
                logger.warning(f"Feature {feature_name} not found in CSV file. Creating placeholder.")
                
                # Create placeholder data based on feature shape
                shape = feature_info.get("shape", (1,))
                
                if len(shape) == 1 and shape[0] == 1:
                    # For scalar features, create a 1D array of zeros
                    restructured_data[feature_name] = np.zeros(len(df), dtype=np.float32)
                else:
                    # For vector features, create a list of zero arrays
                    restructured_data[feature_name] = [np.zeros(shape, dtype=np.float32) for _ in range(len(df))]
        
        # Add task information
        restructured_data["task"] = [self.task_name] * len(df)
        
        # Create the restructured DataFrame
        restructured_df = pd.DataFrame({
            k: v for k, v in restructured_data.items()
        })
        
        return restructured_df

    def _compute_episode_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute statistics for an episode.
        
        Args:
            df: DataFrame for the episode
            
        Returns:
            Dictionary of statistics
        """
        stats = {}
        
        # Compute statistics for each numeric feature
        for column in df.columns:
            if column == "task":
                continue  # Skip non-numeric columns
            
            try:
                values = df[column].values
                if isinstance(values, np.ndarray) and values.size > 0:
                    stats[column] = {
                        "mean": np.mean(values, axis=0),
                        "std": np.std(values, axis=0),
                        "min": np.min(values, axis=0),
                        "max": np.max(values, axis=0),
                    }
            except Exception as e:
                logger.warning(f"Failed to compute statistics for {column}: {e}")
        
        return stats

    def _aggregate_stats(self, episode_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate statistics across all episodes.
        
        Args:
            episode_stats: List of episode statistics
            
        Returns:
            Aggregated statistics
        """
        if not episode_stats:
            return {}
        
        # Initialize aggregated stats with the first episode's stats
        aggregated_stats = {}
        
        # Get all keys from all episodes
        all_keys = set()
        for stats in episode_stats:
            all_keys.update(stats.keys())
        
        # Aggregate statistics for each key
        for key in all_keys:
            # Collect statistics for this key from all episodes
            key_stats = []
            for stats in episode_stats:
                if key in stats:
                    key_stats.append(stats[key])
            
            if not key_stats:
                continue
            
            # Initialize aggregated stats for this key
            aggregated_stats[key] = {}
            
            # Aggregate each statistic
            for stat_name in ["mean", "std", "min", "max"]:
                try:
                    # Collect values for this statistic from all episodes
                    values = [stats[stat_name] for stats in key_stats if stat_name in stats]
                    if not values:
                        continue
                    
                    # Convert to numpy arrays
                    values = [np.array(value) for value in values]
                    
                    # Aggregate
                    if stat_name == "mean":
                        # Weighted average of means
                        weights = np.array([value.size for value in values])
                        aggregated_stats[key][stat_name] = np.average(values, weights=weights, axis=0)
                    elif stat_name == "std":
                        # Pooled standard deviation
                        weights = np.array([value.size for value in values])
                        aggregated_stats[key][stat_name] = np.sqrt(np.average(np.square(values), weights=weights, axis=0))
                    elif stat_name == "min":
                        # Minimum of minimums
                        aggregated_stats[key][stat_name] = np.min(values, axis=0)
                    elif stat_name == "max":
                        # Maximum of maximums
                        aggregated_stats[key][stat_name] = np.max(values, axis=0)
                except Exception as e:
                    logger.warning(f"Failed to aggregate {stat_name} for {key}: {e}")
        
        return aggregated_stats

    def _create_hf_features(self) -> Dict[str, Any]:
        """
        Create HuggingFace features for the dataset.
        
        Returns:
            Dictionary of HuggingFace features
        """
        # Define the features
        hf_features = {
            "position": datasets.Features.Sequence(datasets.Value("float32"), length=3),
            "velocity": datasets.Features.Sequence(datasets.Value("float32"), length=3),
            "attitude": datasets.Features.Sequence(datasets.Value("float32"), length=4),
            "angular_velocity": datasets.Features.Sequence(datasets.Value("float32"), length=3),
            "action": datasets.Features.Sequence(datasets.Value("float32"), length=6),
            "timestamp": datasets.Value("float32"),
            "frame_index": datasets.Value("int64"),
            "episode_index": datasets.Value("int64"),
            "task_index": datasets.Value("int64"),
            "index": datasets.Value("int64"),
            "task": datasets.Value("string"),
        }
        
        # Update the features dictionary
        self.features = {
            "position": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
            "velocity": {"dtype": "float32", "shape": (3,), "names": ["v_x", "v_y", "v_z"]},
            "attitude": {"dtype": "float32", "shape": (4,), "names": ["q0", "q1", "q2", "q3"]},
            "angular_velocity": {"dtype": "float32", "shape": (3,), "names": ["w1", "w2", "w3"]},
            "action": {"dtype": "float32", "shape": (6,), "names": ["T_x", "T_y", "T_z", "L_x", "L_y", "L_z"]},
            "timestamp": {"dtype": "float32", "shape": (1,), "names": ["timestamp"]},
            "frame_index": {"dtype": "int64", "shape": (1,), "names": ["frame_index"]},
            "episode_index": {"dtype": "int64", "shape": (1,), "names": ["episode_index"]},
            "task_index": {"dtype": "int64", "shape": (1,), "names": ["task_index"]},
            "index": {"dtype": "int64", "shape": (1,), "names": ["index"]},
        }
        
        # Add image feature if using images
        if self.image_key:
            self.features[self.image_key] = {
                "dtype": "image",
                "shape": (3, self.image_height, self.image_width),
                "names": ["channels", "height", "width"]
            }
        
        return hf_features

    def _create_metadata(self, csv_files: List[Tuple[int, Path]], episode_lengths: List[int]) -> None:
        """Create metadata files for the dataset."""
        # Create meta directory
        meta_dir = self.output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute statistics for all episodes
        all_episode_stats = {}
        for episode_index, csv_path in tqdm(csv_files, desc="Computing episode statistics"):
            # Read the CSV file and process it first to add frame_index
            df = pd.read_csv(csv_path)
            df = self._process_csv_file(episode_index, csv_path)
            
            # Now restructure the processed dataframe
            restructured_df = self._restructure_dataframe(df)
            episode_stats = self._compute_episode_stats(restructured_df)
            all_episode_stats[episode_index] = episode_stats
            self.episodes_stats[episode_index] = episode_stats
        
        # Aggregate statistics across all episodes
        self.stats = self._aggregate_stats(list(all_episode_stats.values()))
        
        # Create info.json
        info = {
            "codebase_version": "v2.1",
            "data_path": DEFAULT_PARQUET_PATH.replace("{chunk_index}", "{episode_chunk:03d}").replace("{episode_id}", "{episode_index:06d}"),
            "video_path": DEFAULT_VIDEO_PATH.replace("{chunk_index}", "{episode_chunk:03d}").replace("{episode_id}", "{episode_index:06d}") if self.use_videos else None,
            "features": self.features,
            "fps": self.fps,
            "robot_type": "unknown",  # Set to unknown for generic datasets
            "total_episodes": len(csv_files),
            "total_frames": sum(episode_lengths),
            "total_tasks": 1,  # Only one task for all episodes
            "total_chunks": (len(csv_files) + self.chunk_size - 1) // self.chunk_size,
            "chunks_size": self.chunk_size,
            "total_videos": len(csv_files) if self.use_videos else 0,
            "splits": {"train": f"0:{len(csv_files)}"},
        }
        
        # Add video info if using videos
        if self.use_videos and self.image_key in self.features:
            self.features[self.image_key]["video_info"] = {
                "video.fps": float(self.fps),
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        
        # Helper function to convert numpy arrays to Python types
        def convert_numpy_to_python(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_python(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_python(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj
        
        # Write metadata files directly to the meta directory
        with open(meta_dir / "info.json", "w") as f:
            json.dump(convert_numpy_to_python(info), f, indent=2)
        
        # Create tasks.jsonl
        with open(meta_dir / "tasks.jsonl", "w") as f:
            task_dict = {
                "task_index": 0,
                "task": self.task_name,
            }
            f.write(json.dumps(task_dict) + "\n")
        
        # Create episodes.jsonl
        with open(meta_dir / "episodes.jsonl", "w") as f:
            for episode_index, _ in csv_files:
                episode_dict = {
                    "episode_index": episode_index,
                    "tasks": [self.task_name],
                    "length": episode_lengths[episode_index],
                }
                f.write(json.dumps(episode_dict) + "\n")
        
        # Create stats.json
        with open(meta_dir / "stats.json", "w") as f:
            json.dump(convert_numpy_to_python(self.stats), f, indent=2)
        
        # Create episodes_stats.jsonl
        with open(meta_dir / "episodes_stats.jsonl", "w") as f:
            for episode_index, stats in self.episodes_stats.items():
                episode_stats_dict = {
                    "episode_index": episode_index,
                    "stats": convert_numpy_to_python(stats),
                }
                f.write(json.dumps(episode_stats_dict) + "\n")
        
        # Create image_keys.json to explicitly define image keys
        if self.image_key:
            with open(meta_dir / "image_keys.json", "w") as f:
                json.dump([self.image_key], f, indent=2)

    def _embed_images(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Embed images in the dataset.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Dataset with embedded images
        """
        # This is a placeholder for future implementation
        # Currently, images are stored as separate files and not embedded in the dataset
        return dataset

    def convert(self) -> None:
        """
        Convert CSV files and images to LeRobotDataset format.
        """
        # Find all CSV files
        csv_files = self._find_csv_files()
        
        # Process all CSV files first to get the data
        all_dfs = {}
        episode_lengths = {}
        
        for episode_index, csv_path in tqdm(csv_files, desc="Processing CSV files"):
            # Process the CSV file
            df = self._process_csv_file(episode_index, csv_path)
            all_dfs[episode_index] = df
            episode_lengths[episode_index] = len(df)
            
            # Calculate FPS if not provided
            if self.fps is None and "timestamp" in df.columns:
                # Calculate FPS from timestamps
                timestamps = df["timestamp"].values
                if len(timestamps) > 1:
                    time_diff = timestamps[-1] - timestamps[0]
                    if time_diff > 0:
                        self.fps = int(len(timestamps) / time_diff)
                        logger.info(f"Calculated FPS: {self.fps}")
        
        # Set default FPS if not calculated
        if self.fps is None:
            self.fps = 5  # Default FPS
            logger.info(f"Using default FPS: {self.fps}")
        
        # Create chunks of episodes
        chunks = []
        current_chunk = []
        
        for episode_index, csv_path in csv_files:
            current_chunk.append((episode_index, csv_path))
            
            if len(current_chunk) >= self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(current_chunk)
        
        # Process each chunk
        for chunk_index, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            # Create chunk directory
            chunk_dir = self.output_dir / "data" / f"chunk-{chunk_index:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each episode in the chunk
            for episode_index, csv_path in chunk:
                # Get the DataFrame for this episode
                df = all_dfs[episode_index]
                
                # Restructure the DataFrame to match the expected schema
                restructured_df = self._restructure_dataframe(df)
                
                # Create HuggingFace features
                hf_features = self._create_hf_features()
                
                # Create HuggingFace dataset
                hf_dataset = datasets.Dataset.from_pandas(restructured_df, features=hf_features)
                
                # Save to parquet
                episode_path = chunk_dir / f"episode_{episode_index:06d}.parquet"
                hf_dataset.to_parquet(episode_path)
                
                # Copy images if not using videos
                if not self.use_videos:
                    for frame_index in range(len(df)):
                        # Format episode and frame indices as strings with leading zeros
                        episode_str = f"{episode_index:06d}"
                        frame_str = f"{frame_index:06d}"
                        # Copy images if not using videos
                if not self.use_videos:
                    # Create image directory structure
                    image_dir = self.output_dir / "images" / f"chunk-{chunk_index:03d}" / self.image_key
                    image_dir.mkdir(parents=True, exist_ok=True)
                    
                    for frame_index in range(len(df)):
                        # Format episode and frame indices as strings with leading zeros
                        episode_str = f"{episode_index}"
                        frame_str = f"{frame_index}"
                        
                        # Construct image filename based on pattern
                        image_filename = self.image_pattern.format(episode=episode_str, frame=frame_str) + self.image_extension
                        
                        # Source and destination paths
                        src_path = self.image_dir / image_filename
                        dst_path = image_dir / f"episode_{episode_index:06d}_frame_{frame_index:06d}{self.image_extension}"
                        
                        # Check if source image exists
                        if src_path.exists():
                            # Copy the image
                            shutil.copy2(src_path, dst_path)
                        else:
                            logger.warning(f"Image not found: {src_path}")
                else:
                    # Create videos from images
                    self._create_videos(episode_index, chunk_index, df)
        
        # Calculate statistics
        self._calculate_statistics(all_dfs)
        
        # Create metadata files
        self._create_metadata(episode_lengths)
        
        logger.info(f"Dataset conversion complete. Output directory: {self.output_dir}")
    
    def _create_videos(self, episode_index: int, chunk_index: int, df: pd.DataFrame) -> None:
        """
        Create videos from images for an episode.
        
        Args:
            episode_index: Episode index
            chunk_index: Chunk index
            df: DataFrame for the episode
        """
        import cv2
        
        # Create video directory structure
        video_dir = self.output_dir / "videos" / f"chunk-{chunk_index:03d}" / self.image_key
        video_dir.mkdir(parents=True, exist_ok=True)
        
        # Video path
        video_path = video_dir / f"episode_{episode_index:06d}.mp4"
        
        # Collect all images for this episode
        images = []
        for frame_index in range(len(df)):
            # Format episode and frame indices
            episode_str = f"{episode_index}"
            frame_str = f"{frame_index}"
            
            # Construct image filename based on pattern
            image_filename = self.image_pattern.format(episode=episode_str, frame=frame_str) + self.image_extension
            
            # Image path
            image_path = self.image_dir / image_filename
            
            # Check if image exists
            if image_path.exists():
                # Read the image
                img = cv2.imread(str(image_path))
                if img is not None:
                    images.append(img)
                else:
                    logger.warning(f"Failed to read image: {image_path}")
            else:
                logger.warning(f"Image not found: {image_path}")
        
        # Create video if we have images
        if images:
            # Get image dimensions
            height, width = images[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(str(video_path), fourcc, self.fps, (width, height))
            
            # Write images to video
            for img in images:
                video_writer.write(img)
            
            # Release video writer
            video_writer.release()
            
            logger.info(f"Created video for episode {episode_index} with {len(images)} frames")
        else:
            logger.warning(f"No images found for episode {episode_index}")
    
    def load_dataset(self) -> LeRobotDataset:
        """Load the converted dataset using LeRobotDataset."""
        return LeRobotDataset(
            repo_id=self.repo_id,
            root=self.output_dir,
        )
    
    def push_to_hub(self, private: bool = False) -> None:
        """
        Push the dataset to the Hugging Face Hub.
        
        Args:
            private: Whether to make the dataset private
        """
        # Make sure repo_id has the correct format (username/repo-name)
        if "/" not in self.repo_id:
            logger.warning(f"Repository ID '{self.repo_id}' does not contain a slash. "
                          f"Using '{self.repo_id}/moon_lander' instead.")
            self.repo_id = f"{self.repo_id}/moon_lander"
        
        try:
            # Try to load the dataset (this might fail if it's a new repository)
            dataset = self.load_dataset()
            logger.info(f"Pushing dataset to {self.repo_id}")
            dataset.push_to_hub(private=private)
        except Exception as e:
            # Repository doesn't exist yet or there was another error
            logger.info(f"Creating new repository {self.repo_id}")
            logger.debug(f"Original error: {str(e)}")
            
            try:
                # Create the repository
                api = huggingface_hub.HfApi()
                api.create_repo(repo_id=self.repo_id, repo_type="dataset", private=private)
                
                # Upload the dataset files
                logger.info("Uploading dataset files...")
                api.upload_folder(
                    folder_path=str(self.output_dir),
                    repo_id=self.repo_id,
                    repo_type="dataset",
                )
                logger.info(f"Dataset pushed to {self.repo_id}")
            except Exception as upload_error:
                logger.error(f"Failed to push dataset to Hub: {str(upload_error)}")
                raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert CSV trajectory data to LeRobotDataset format")
    parser.add_argument("--csv-dir", type=str, required=True, help="Directory containing CSV files")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing image files")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the dataset")
    parser.add_argument("--repo-id", type=str, required=True, help="Repository ID for the dataset")
    parser.add_argument("--fps", type=int, default=None, help="Frames per second (if None, calculated from data)")
    parser.add_argument("--task-name", type=str, default="meta-RL policy rollout", help="Name of the task")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Number of episodes per chunk")
    parser.add_argument("--image-key", type=str, default="observation.images", help="Key for image feature")
    parser.add_argument("--image-pattern", type=str, default="img_traj_{episode}_step_{frame}", help="Pattern for image filenames")
    parser.add_argument("--image-extension", type=str, default=".png", help="File extension for images")
    parser.add_argument("--csv-pattern", type=str, default="trajectory_{episode}.csv", help="Pattern for CSV filenames")
    parser.add_argument("--use-videos", action="store_true", help="Encode images as videos")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--push", action="store_true", help="Push dataset to HuggingFace Hub")
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    
    # Add image dimension arguments
    parser.add_argument("--image-height", type=int, default=256, help="Height of images in pixels")
    parser.add_argument("--image-width", type=int, default=256, help="Width of images in pixels")
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create converter
    converter = CSVToLeRobotDatasetConverter(
        csv_dir=args.csv_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        fps=args.fps,
        task_name=args.task_name,
        chunk_size=args.chunk_size,
        image_key=args.image_key,
        image_pattern=args.image_pattern,
        image_extension=args.image_extension,
        csv_pattern=args.csv_pattern,
        use_videos=args.use_videos,
        debug=args.debug,
        image_height=args.image_height,
        image_width=args.image_width,
    )
    
    # Convert dataset
    converter.convert()
    
    # Push to HuggingFace Hub if requested
    if args.push:
        converter.push_to_hub(private=args.private)


if __name__ == "__main__":
    main()