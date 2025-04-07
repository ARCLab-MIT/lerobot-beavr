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

"""
This module provides the LeRobotDatasetHandler class for loading, processing, and extracting
frames from robotic datasets from Hugging Face.

Features:
- Loading datasets from the Hugging Face hub
- Exploring dataset metadata and properties
- Extracting and saving frames from specified episodes
- Accessing frames by episode number
- Using timestamp-based frame selection
- Compatibility with PyTorch DataLoader for batch processing
"""

import os
from pprint import pprint
from typing import List, Dict, Union, Optional, Any

import torch
import numpy as np
from PIL import Image
from huggingface_hub import HfApi
from tqdm import tqdm

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


class LeRobotDatasetHandler:
    """
    A class to handle LeRobot datasets, providing functionality to load, explore,
    and extract frames from robotic datasets.
    """
    
    def __init__(self, repo_id: str, episodes: Optional[List[int]] = None, 
                 delta_timestamps: Optional[Dict[str, List[float]]] = None,
                 verbose: bool = False):
        """
        Initialize the LeRobotDatasetHandler.
        
        Args:
            repo_id (str): The Hugging Face repository ID for the dataset.
            episodes (List[int], optional): Specific episodes to load. If None, loads all episodes.
            delta_timestamps (Dict[str, List[float]], optional): Timestamps for loading history/future frames.
            verbose (bool, optional): Whether to print detailed information. Defaults to False.
        """
        self.repo_id = repo_id
        self.verbose = verbose
        
        # Load metadata
        self.meta = LeRobotDatasetMetadata(repo_id)
        
        if self.verbose:
            self._print_metadata_summary()
        
        # Load dataset
        self.dataset = LeRobotDataset(repo_id, episodes=episodes, delta_timestamps=delta_timestamps)
        
        if self.verbose:
            self._print_dataset_summary()
    
    def _print_metadata_summary(self) -> None:
        """Print a summary of the dataset metadata."""
        print(f"Total number of episodes: {self.meta.total_episodes}")
        print(f"Average number of frames per episode: {self.meta.total_frames / self.meta.total_episodes:.3f}")
        print(f"Frames per second used during data collection: {self.meta.fps}")
        print(f"Robot type: {self.meta.robot_type}")
        print(f"Keys to access images from cameras: {self.meta.camera_keys}\n")
        
        print("Tasks:")
        print(self.meta.tasks)
        print("Features:")
        pprint(self.meta.features)
        
        print(self.meta)
    
    def _print_dataset_summary(self) -> None:
        """Print a summary of the loaded dataset."""
        print(f"Selected episodes: {self.dataset.episodes}")
        print(f"Number of episodes selected: {self.dataset.num_episodes}")
        print(f"Number of frames selected: {self.dataset.num_frames}")
    
    def get_episode_frame_indices(self, episode_index: int) -> tuple:
        """
        Get the frame indices for a specific episode.
        
        Args:
            episode_index (int): The index of the episode.
            
        Returns:
            tuple: A tuple containing the start and end indices for the episode.
        """
        from_idx = self.dataset.episode_data_index["from"][episode_index].item()
        to_idx = self.dataset.episode_data_index["to"][episode_index].item()
        return from_idx, to_idx
    
    def get_episode_frames(self, episode_index: int, camera_key: Optional[str] = None) -> List[torch.Tensor]:
        """
        Get all frames from a specific episode.
        
        Args:
            episode_index (int): The index of the episode.
            camera_key (str, optional): The camera key to use. If None, uses the first camera.
            
        Returns:
            List[torch.Tensor]: A list of frames from the episode.
        """
        if camera_key is None:
            camera_key = self.meta.camera_keys[0]
        
        from_idx, to_idx = self.get_episode_frame_indices(episode_index)
        frames = [self.dataset[idx][camera_key] for idx in range(from_idx, to_idx)]
        return frames
    
    def save_episode_frames(self, episode_indices: Union[int, List[int]], 
                           output_dir: str, camera_key: Optional[str] = None,
                           file_prefix: str = "frame") -> None:
        """
        Save frames from specified episodes as JPG images.
        
        Args:
            episode_indices (Union[int, List[int]]): Episode index or list of indices to extract frames from.
            output_dir (str): Directory to save the frames to.
            camera_key (str, optional): The camera key to use. If None, uses the first camera.
            file_prefix (str, optional): Prefix for the saved image files. Defaults to "frame".
        """
        if camera_key is None:
            camera_key = self.meta.camera_keys[0]
        
        # Convert single episode to list
        if isinstance(episode_indices, int):
            episode_indices = [episode_indices]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for episode_idx in episode_indices:
            # Create episode directory
            episode_dir = os.path.join(output_dir, f"episode_{episode_idx}")
            os.makedirs(episode_dir, exist_ok=True)
            
            from_idx, to_idx = self.get_episode_frame_indices(episode_idx)
            
            # Save frames with progress bar
            for i, frame_idx in enumerate(tqdm(range(from_idx, to_idx), 
                                              desc=f"Saving frames from episode {episode_idx}")):
                frame = self.dataset[frame_idx][camera_key]
                
                # Handle different tensor shapes based on delta_timestamps
                if len(frame.shape) == 4:  # (t, c, h, w)
                    # If we have multiple timestamps, save only the current frame (last one)
                    frame = frame[-1]
                
                # Convert from tensor to PIL Image and save
                # Convert from (c, h, w) to (h, w, c) and from [0,1] to [0,255]
                frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                
                # Handle different channel formats
                if frame_np.shape[2] == 1:  # Grayscale
                    frame_np = frame_np.squeeze(2)
                    img = Image.fromarray(frame_np, mode='L')
                else:  # RGB or RGBA
                    img = Image.fromarray(frame_np)
                
                img.save(os.path.join(episode_dir, f"{file_prefix}_{i:04d}.jpg"))
    
    def create_dataloader(self, batch_size: int = 32, shuffle: bool = True, 
                         num_workers: int = 0) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader for the dataset.
        
        Args:
            batch_size (int, optional): Batch size for the DataLoader. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
            num_workers (int, optional): Number of worker processes. Defaults to 0.
            
        Returns:
            torch.utils.data.DataLoader: A DataLoader for the dataset.
        """
        return torch.utils.data.DataLoader(
            self.dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=shuffle,
        )


# Example usage
if __name__ == "__main__":
    # Initialize the dataset handler
    repo_id = "arclabmit/koch_gear_and_bin"
    
    # Example with delta timestamps for history/future frames
    delta_timestamps = {
        "observation.images.nexigo_webcam": [-1, -0.5, -0.20, 0],  # 4 frames: past and current
        "observation.state": [-1.5, -1, -0.5, -0.20, -0.10, 0],    # 6 state vectors
        "action": [t / 30 for t in range(64)],                     # 64 future actions
    }
    
    # Create handler with verbose output
    handler = LeRobotDatasetHandler(repo_id, verbose=True)
    
    # Save frames from episodes 0 and 1
    handler.save_episode_frames([0, 1], output_dir="./extracted_frames")
    
    # Create and use a dataloader
    dataloader = handler.create_dataloader(batch_size=16)
    
    # Example of processing a batch
    for batch in dataloader:
        camera_key = handler.meta.camera_keys[0]
        print(f"{batch[camera_key].shape=}")
        break
