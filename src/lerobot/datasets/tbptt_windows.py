"""
TBPTT Window Streaming for LeRobot Datasets

This module provides an IterableDataset that streams fixed-length windows from LeRobot datasets
for efficient Truncated Backpropagation Through Time (TBPTT) training.

Key features:
- Works directly with LeRobot datasets (no conversion needed)
- Pre-shapes windows to (K, ...) format for zero-copy collation
- Provides *_last views for efficient last-step processing
- Supports persistent workers and video decoder reuse
- Includes masks for proper gradient computation
"""

from typing import Iterator, Dict, Any
import torch
from torch.utils.data import IterableDataset, get_worker_info

from lerobot.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_ENV_STATE


class TBPTTWindowStreamer(IterableDataset):
    """
    Streams fixed-length windows from a LeRobot dataset for TBPTT training.
    
    This dataset wrapper:
    1. Iterates through episodes sequentially
    2. Slides a fixed-size window (K frames) across each episode
    3. Pre-shapes each window to (K, ...) format
    4. Provides *_last views for efficient ACT-style last-step processing
    5. Includes masks for handling variable-length episodes
    
    Args:
        dataset: LeRobot dataset instance
        window_size: Number of timesteps per window (K)
        stride: Step size for sliding window (default: window_size for non-overlapping)
        clamp_pad: If True, pad with last valid frame; if False, pad with zeros
        shuffle_episodes: If True, shuffle episode order each epoch
    """
    
    def __init__(
        self,
        dataset,
        window_size: int,
        stride: int | None = None,
        clamp_pad: bool = True,
        shuffle_episodes: bool = True,
    ):
        super().__init__()
        self.dataset = dataset
        self.K = window_size
        self.S = stride if stride is not None else window_size
        self.clamp_pad = clamp_pad
        self.shuffle_episodes = shuffle_episodes
        
        # Extract episode boundaries from dataset
        self.episode_data_index = dataset.episode_data_index
        self.num_episodes = dataset.num_episodes
        
        # Determine which features are present
        self.has_images = len(dataset.meta.camera_keys) > 0
        self.has_state = OBS_STATE in dataset.features
        self.has_env_state = OBS_ENV_STATE in dataset.features
        self.has_action = ACTION in dataset.features
        
    def _get_worker_episodes(self) -> list[int]:
        """Shard episodes across workers for parallel loading."""
        episode_indices = list(range(self.num_episodes))
        
        if self.shuffle_episodes:
            # Use worker-specific seed for reproducibility
            wi = get_worker_info()
            seed = torch.initial_seed() if wi is None else wi.seed
            rng = torch.Generator().manual_seed(seed)
            perm = torch.randperm(self.num_episodes, generator=rng).tolist()
            episode_indices = [episode_indices[i] for i in perm]
        
        wi = get_worker_info()
        if wi is None or wi.num_workers <= 1:
            return episode_indices
        
        # Round-robin sharding
        return [ep for i, ep in enumerate(episode_indices) if i % wi.num_workers == wi.id]
    
    def _get_episode_range(self, ep_idx: int) -> tuple[int, int, int]:
        """Get (start_frame, end_frame, length) for an episode."""
        start = self.episode_data_index["from"][ep_idx].item()
        end = self.episode_data_index["to"][ep_idx].item()
        return start, end, end - start
    
    def _fetch_frames(self, frame_indices: list[int]) -> list[dict]:
        """Fetch multiple frames from the dataset."""
        return [self.dataset[idx] for idx in frame_indices]
    
    def _build_window_dict(
        self,
        frames: list[dict],
        ep_start: int,
        ep_end: int,
        window_start: int,
        ep_length: int,
    ) -> Dict[str, Any]:
        """
        Build a window dictionary from fetched frames.
        
        Returns dict with:
        - *_seq keys: (K, ...) tensors for the full window
        - *_last keys: (...) tensors for the last frame (view, not copy)
        - valid_mask: (K,) bool indicating valid timesteps
        - ended_mask: bool indicating if episode ended in this window
        """
        actual_K = len(frames)
        
        # Build output dict
        out = {}
        
        # Stack image sequences if present
        if self.has_images:
            # Group images by camera
            camera_keys = self.dataset.meta.camera_keys
            images_list = []
            for cam_key in camera_keys:
                # Stack frames for this camera: (K, C, H, W)
                cam_frames = torch.stack([f[cam_key] for f in frames], dim=0)
                images_list.append(cam_frames)
            out[f"{OBS_IMAGES}_seq"] = images_list  # List of (K, C, H, W)
            out[f"{OBS_IMAGES}_last"] = [img[-1] for img in images_list]  # List of (C, H, W)
        
        # Stack state sequence if present
        if self.has_state:
            state_seq = torch.stack([f[OBS_STATE] for f in frames], dim=0)  # (K, D_state)
            out[f"{OBS_STATE}_seq"] = state_seq
            out[f"{OBS_STATE}_last"] = state_seq[-1]  # (D_state,)
        
        # Stack env state sequence if present
        if self.has_env_state:
            env_state_seq = torch.stack([f[OBS_ENV_STATE] for f in frames], dim=0)
            out[f"{OBS_ENV_STATE}_seq"] = env_state_seq
            out[f"{OBS_ENV_STATE}_last"] = env_state_seq[-1]
        
        # Stack action sequence if present
        if self.has_action:
            action_seq = torch.stack([f[ACTION] for f in frames], dim=0)  # (K, S, A)
            out[f"{ACTION}_seq"] = action_seq
            out[f"{ACTION}_last"] = action_seq[-1]  # (S, A)
            
            # Build action padding mask
            if f"{ACTION}_is_pad" in frames[0]:
                pad_seq = torch.stack([f[f"{ACTION}_is_pad"] for f in frames], dim=0)  # (K, S)
            else:
                # If no padding info, assume all valid
                S = action_seq.shape[1]
                pad_seq = torch.zeros(actual_K, S, dtype=torch.bool)
            out[f"{ACTION}_is_pad"] = pad_seq
        
        # Create valid mask: True for valid timesteps, False for padding
        valid_mask = torch.ones(actual_K, dtype=torch.bool)
        
        # Episode ended if this window extends beyond episode length
        ended_mask = torch.tensor(window_start + actual_K >= ep_length, dtype=torch.bool)
        
        out["valid_mask"] = valid_mask
        out["ended_mask"] = ended_mask
        out["window_start"] = torch.tensor(window_start, dtype=torch.long)
        
        return out
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over windows from all episodes assigned to this worker."""
        worker_episodes = self._get_worker_episodes()
        
        for ep_idx in worker_episodes:
            ep_start, ep_end, ep_length = self._get_episode_range(ep_idx)
            
            # Slide window across episode
            window_start = 0
            while window_start < ep_length:
                # Calculate actual window size (may be smaller at episode end)
                actual_K = min(self.K, ep_length - window_start)
                
                # Build frame indices for this window
                frame_indices = [ep_start + window_start + t for t in range(actual_K)]
                
                # Handle padding if window is shorter than K
                if actual_K < self.K and self.clamp_pad:
                    # Repeat last frame to fill window
                    last_frame_idx = frame_indices[-1]
                    frame_indices.extend([last_frame_idx] * (self.K - actual_K))
                
                # Fetch frames
                frames = self._fetch_frames(frame_indices)
                
                # Build window dict
                window = self._build_window_dict(
                    frames=frames[:actual_K],  # Only use actual frames for construction
                    ep_start=ep_start,
                    ep_end=ep_end,
                    window_start=window_start,
                    ep_length=ep_length,
                )
                
                yield window
                
                # Advance window
                window_start += self.S


def collate_tbptt(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for TBPTT windows - simply stacks along batch dimension.
    
    This is a zero-copy collation that just stacks pre-shaped windows from (K, ...) to (B, K, ...).
    Preserves *_last views for efficient last-step processing.
    
    Args:
        batch: List of window dicts from TBPTTWindowStreamer
        
    Returns:
        Dict with (B, K, ...) tensors and (B, ...) last-step views
    """
    if len(batch) == 0:
        return {}
    
    out = {}
    
    # Get all keys from first item
    first_item = batch[0]
    
    for key in first_item.keys():
        values = [item[key] for item in batch]
        
        # Handle image sequences (list of tensors per camera)
        if key.endswith("_seq") and isinstance(values[0], list):
            # Stack each camera separately: List[List[(K, C, H, W)]] -> List[(B, K, C, H, W)]
            num_cameras = len(values[0])
            stacked_cams = []
            for cam_idx in range(num_cameras):
                cam_windows = [v[cam_idx] for v in values]  # List of (K, C, H, W)
                stacked_cams.append(torch.stack(cam_windows, dim=0))  # (B, K, C, H, W)
            out[key] = stacked_cams
        
        # Handle image last views (list of tensors per camera)
        elif key.endswith("_last") and isinstance(values[0], list):
            # Stack each camera: List[List[(C, H, W)]] -> List[(B, C, H, W)]
            num_cameras = len(values[0])
            stacked_cams = []
            for cam_idx in range(num_cameras):
                cam_frames = [v[cam_idx] for v in values]
                stacked_cams.append(torch.stack(cam_frames, dim=0))  # (B, C, H, W)
            out[key] = stacked_cams
        
        # Handle regular tensors
        elif torch.is_tensor(values[0]):
            out[key] = torch.stack(values, dim=0)
        
        # Pass through other types (scalars, etc.)
        else:
            out[key] = values
    
    return out
