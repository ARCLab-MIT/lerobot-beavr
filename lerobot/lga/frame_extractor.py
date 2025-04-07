#!/usr/bin/env python3
"""
Script to extract and save frames from the Koch gear and bin dataset.
This script can extract a specific number of frames from each camera or
all frames from all episodes.
"""

import os
import argparse
from typing import List, Optional, Dict, Union

from lerobot.lga.load_dataset import LeRobotDatasetHandler


class MultiCameraFrameExtractor:
    """
    A class to extract and save frames from multiple cameras in a LeRobot dataset.
    """
    
    def __init__(self, repo_id: str, output_dir: str, verbose: bool = False):
        """
        Initialize the MultiCameraFrameExtractor.
        
        Args:
            repo_id (str): The Hugging Face repository ID for the dataset.
            output_dir (str): Directory to save the extracted frames.
            verbose (bool, optional): Whether to print detailed information. Defaults to False.
        """
        self.repo_id = repo_id
        self.output_dir = output_dir
        self.verbose = verbose
        
        # Create the dataset handler
        self.handler = LeRobotDatasetHandler(repo_id, verbose=verbose)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get available cameras
        self.cameras = self.handler.meta.camera_keys
        if self.verbose:
            print(f"Available cameras: {self.cameras}")
    
    def extract_all_frames(self, camera_keys: Optional[List[str]] = None) -> None:
        """
        Extract and save all frames from all episodes for specified cameras.
        
        Args:
            camera_keys (List[str], optional): List of camera keys to extract frames from.
                                              If None, uses all available cameras.
        """
        if camera_keys is None:
            camera_keys = self.cameras
        
        # Validate camera keys
        valid_camera_keys = [key for key in camera_keys if key in self.cameras]
        if len(valid_camera_keys) < len(camera_keys):
            invalid_keys = set(camera_keys) - set(valid_camera_keys)
            if self.verbose:
                print(f"Warning: Some camera keys not found in dataset: {invalid_keys}")
                print(f"Available cameras: {self.cameras}")
        
        if self.verbose:
            print(f"Extracting all frames from all episodes")
            print(f"Cameras: {valid_camera_keys}")
            print(f"Saving to: {self.output_dir}")
        
        # Get all episodes
        episodes = list(range(self.handler.meta.total_episodes))
        
        # Process each camera
        for camera_key in valid_camera_keys:
            # Create camera directory
            camera_dir = os.path.join(self.output_dir, camera_key.split('.')[-1])
            os.makedirs(camera_dir, exist_ok=True)
            
            if self.verbose:
                print(f"Extracting all frames from camera: {camera_key}")
            
            # Process each episode
            for episode_idx in episodes:
                # Create episode directory
                episode_dir = os.path.join(camera_dir, f"episode_{episode_idx}")
                os.makedirs(episode_dir, exist_ok=True)
                
                # Get frame indices for this episode
                from_idx, to_idx = self.handler.get_episode_frame_indices(episode_idx)
                
                # Extract and save frames
                import numpy as np
                from PIL import Image
                from tqdm import tqdm
                
                for frame_offset, frame_idx in enumerate(tqdm(range(from_idx, to_idx), 
                                                  desc=f"Saving {camera_key} frames from episode {episode_idx}")):
                    frame = self.handler.dataset[frame_idx][camera_key]
                    
                    # Handle different tensor shapes based on delta_timestamps
                    if len(frame.shape) == 4:  # (t, c, h, w)
                        frame = frame[-1]  # Use the current frame (last one)
                    
                    # Convert from (c, h, w) to (h, w, c) and from [0,1] to [0,255]
                    frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    
                    # Handle different channel formats
                    if frame_np.shape[2] == 1:  # Grayscale
                        frame_np = frame_np.squeeze(2)
                        img = Image.fromarray(frame_np, mode='L')
                    else:  # RGB or RGBA
                        img = Image.fromarray(frame_np)
                    
                    # Save the image
                    img.save(os.path.join(episode_dir, f"frame_{frame_offset:04d}.jpg"))
                
                if self.verbose:
                    print(f"Extracted {to_idx - from_idx} frames from episode {episode_idx} for camera {camera_key}")
    
    def extract_frames_from_cameras(self, 
                                   frames_per_camera: Dict[str, int],
                                   episodes: Optional[List[int]] = None) -> None:
        """
        Extract and save a specific number of frames from each camera.
        
        Args:
            frames_per_camera (Dict[str, int]): Dictionary mapping camera keys to number of frames to extract.
            episodes (List[int], optional): Episodes to extract frames from. If None, uses episode 0.
        """
        if episodes is None:
            episodes = [0]  # Default to first episode
        
        if self.verbose:
            print(f"Extracting frames from episodes {episodes}")
            print(f"Frames per camera: {frames_per_camera}")
            print(f"Saving to: {self.output_dir}")
        
        # Process each camera
        for camera_key, num_frames in frames_per_camera.items():
            if camera_key not in self.cameras:
                if self.verbose:
                    print(f"Warning: Camera {camera_key} not found in dataset. Available cameras: {self.cameras}")
                continue
            
            # Create camera directory
            camera_dir = os.path.join(self.output_dir, camera_key.split('.')[-1])
            os.makedirs(camera_dir, exist_ok=True)
            
            if self.verbose:
                print(f"Extracting {num_frames} frames from camera: {camera_key}")
            
            # Calculate frames per episode
            frames_per_episode = num_frames // len(episodes)
            remaining_frames = num_frames % len(episodes)
            
            for i, episode_idx in enumerate(episodes):
                # Get frame indices for this episode
                from_idx, to_idx = self.handler.get_episode_frame_indices(episode_idx)
                total_episode_frames = to_idx - from_idx
                
                # Calculate how many frames to extract from this episode
                frames_to_extract = frames_per_episode
                if i < remaining_frames:
                    frames_to_extract += 1
                
                if frames_to_extract > total_episode_frames:
                    if self.verbose:
                        print(f"Warning: Episode {episode_idx} only has {total_episode_frames} frames. "
                              f"Extracting all available frames.")
                    frames_to_extract = total_episode_frames
                
                # Calculate frame indices to extract (evenly spaced)
                if frames_to_extract == 1:
                    indices = [from_idx + total_episode_frames // 2]  # Middle frame
                else:
                    step = total_episode_frames / (frames_to_extract - 1) if frames_to_extract > 1 else 0
                    indices = [from_idx + int(i * step) for i in range(frames_to_extract)]
                
                # Create episode directory
                episode_dir = os.path.join(camera_dir, f"episode_{episode_idx}")
                os.makedirs(episode_dir, exist_ok=True)
                
                # Extract and save frames
                import numpy as np
                from PIL import Image
                from tqdm import tqdm
                
                for j, frame_idx in enumerate(tqdm(indices, desc=f"Saving {camera_key} frames from episode {episode_idx}")):
                    frame = self.handler.dataset[frame_idx][camera_key]
                    
                    # Handle different tensor shapes based on delta_timestamps
                    if len(frame.shape) == 4:  # (t, c, h, w)
                        frame = frame[-1]  # Use the current frame (last one)
                    
                    # Convert from (c, h, w) to (h, w, c) and from [0,1] to [0,255]
                    frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    
                    # Handle different channel formats
                    if frame_np.shape[2] == 1:  # Grayscale
                        frame_np = frame_np.squeeze(2)
                        img = Image.fromarray(frame_np, mode='L')
                    else:  # RGB or RGBA
                        img = Image.fromarray(frame_np)
                    
                    # Save the image
                    img.save(os.path.join(episode_dir, f"frame_{j:04d}.jpg"))
                
                if self.verbose:
                    print(f"Extracted {len(indices)} frames from episode {episode_idx} for camera {camera_key}")


def main():
    """Main function to parse arguments and extract frames."""
    parser = argparse.ArgumentParser(description="Extract frames from multiple cameras in a LeRobot dataset")
    parser.add_argument("--repo-id", type=str, default="arclabmit/koch_gear_and_bin",
                        help="Hugging Face repository ID")
    parser.add_argument("--output-dir", type=str, default="./extracted_frames",
                        help="Directory to save extracted frames")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1],
                        help="Episodes to extract frames from")
    parser.add_argument("--nexigo-frames", type=int, default=25,
                        help="Number of frames to extract from nexigo_webcam")
    parser.add_argument("--realsense-frames", type=int, default=25,
                        help="Number of frames to extract from realsense")
    parser.add_argument("--all-frames", action="store_true",
                        help="Extract all frames from all episodes")
    parser.add_argument("--cameras", type=str, nargs="+", 
                        help="Camera keys to extract frames from (used with --all-frames)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed information")
    
    args = parser.parse_args()
    
    # Create extractor
    extractor = MultiCameraFrameExtractor(args.repo_id, args.output_dir, args.verbose)
    
    if args.all_frames:
        # Extract all frames from all episodes
        camera_keys = args.cameras
        if camera_keys is None:
            # Default camera keys if not specified
            camera_keys = [
                "observation.images.nexigo_webcam",
                "observation.images.realsense"
            ]
        extractor.extract_all_frames(camera_keys=camera_keys)
    else:
        # Extract specific number of frames
        frames_per_camera = {
            "observation.images.nexigo_webcam": args.nexigo_frames,
            "observation.images.realsense": args.realsense_frames
        }
        
        extractor.extract_frames_from_cameras(
            frames_per_camera=frames_per_camera,
            episodes=args.episodes
        )


if __name__ == "__main__":
    main()
