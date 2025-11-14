#!/usr/bin/env python3
"""
Preprocess LeRobot parquet dataset into HDF5 format for efficient training.

This script converts the parquet dataset format to the HDF5 format expected
by the MambaSequenceDataset class, enabling fast loading during training.
"""

import os
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from pathlib import Path
from tqdm import tqdm
import json


def load_parquet_dataset(parquet_dir):
    """Load all parquet data into memory for processing."""
    print(f"Loading parquet dataset from: {parquet_dir}")

    data_dir = os.path.join(parquet_dir, "data")
    meta_dir = os.path.join(parquet_dir, "meta")

    # Load metadata
    with open(os.path.join(meta_dir, "info.json"), 'r') as f:
        info = json.load(f)

    print(f"Dataset info: {info['total_episodes']} episodes, {info['total_frames']} frames")

    # Load all parquet files
    data_frames = []
    for chunk_dir in sorted(os.listdir(data_dir)):
        chunk_path = os.path.join(data_dir, chunk_dir)
        if os.path.isdir(chunk_path):
            for file in sorted(os.listdir(chunk_path)):
                if file.endswith('.parquet'):
                    file_path = os.path.join(chunk_path, file)
                    print(f"Loading {file_path}")
                    table = pq.read_table(file_path)
                    df = table.to_pandas()
                    data_frames.append(df)

    if data_frames:
        all_data = pd.concat(data_frames, ignore_index=True)
        print(f"Loaded {len(all_data)} total frames")
        return all_data, info
    else:
        raise ValueError("No parquet files found")


def extract_episodes(data_df, info):
    """Extract individual episodes from the concatenated data."""
    episodes = []
    current_episode = None
    current_frames = []

    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Extracting episodes"):
        episode_idx = row['episode_index']

        if episode_idx != current_episode:
            if current_frames:
                episodes.append((current_episode, current_frames))
            current_frames = []
            current_episode = episode_idx

        current_frames.append(row)

    # Add the last episode
    if current_frames:
        episodes.append((current_episode, current_frames))

    print(f"Extracted {len(episodes)} episodes")
    return episodes


def process_episode_to_hdf5(episode_idx, episode_frames, output_dir, future_steps=16):
    """Convert a single episode to HDF5 format."""
    if len(episode_frames) < future_steps:
        print(f"Skipping episode {episode_idx}: too short ({len(episode_frames)} < {future_steps})")
        return

    # Extract data
    states = []
    actions = []

    for frame in episode_frames:
        states.append(frame['observation.state'])
        actions.append(frame['action'])

    states = np.array(states)
    actions = np.array(actions)

    # Create output file
    output_file = os.path.join(output_dir, "train", "episode", f"episode_{episode_idx:06d}.hdf5")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with h5py.File(output_file, 'w') as f:
        # Save states and actions
        f.create_dataset('/observations/qpos', data=states, dtype=np.float32)
        f.create_dataset('/action', data=actions, dtype=np.float32)

        # Create dummy image data (since we don't have actual images)
        # The dataset expects images in the format /observations/images/camera_name
        # Each should be an array of shape (num_frames, H, W, C)
        dummy_image = np.zeros((640, 480, 3), dtype=np.uint8)  # BGR format expected by cv2
        images_array = np.tile(dummy_image, (len(states), 1, 1, 1))  # Repeat for each frame

        # Create images group and datasets for each camera
        images_group = f.create_group('/observations/images')
        images_group.create_dataset('top', data=images_array, dtype=np.uint8)
        # Note: We could add 'angle' camera too if needed, but for now just 'top'

    print(f"Saved episode {episode_idx} with {len(episode_frames)} frames to {output_file}")


def create_scaler_data(episodes, output_dir):
    """Create scaler fitting data and scaler parameters."""
    print("Creating scaler data...")

    all_states = []
    all_actions = []

    for episode_idx, frames in episodes:
        for frame in frames:
            all_states.append(frame['observation.state'])
            all_actions.append(frame['action'])

    all_states = np.array(all_states)
    all_actions = np.array(all_actions)

    # Save scaler data (states and actions for fitting normalization)
    scaler_file = os.path.join(output_dir, "scaler_data.hdf5")
    with h5py.File(scaler_file, 'w') as f:
        f.create_dataset('states', data=all_states, dtype=np.float32)
        f.create_dataset('actions', data=all_actions, dtype=np.float32)

    print(f"Saved scaler data: {len(all_states)} frames to {scaler_file}")

    # Create scaler parameters in the format expected by Scaler class
    # The Scaler expects mean_dict and std_dict ParameterDict objects
    print("Computing scaler parameters...")

    # Define the lowdim_dict structure (same as in training)
    lowdim_dict = {
        'agl_1': 1, 'agl_2': 1, 'agl_3': 1, 'agl_4': 1, 'agl_5': 1, 'agl_6': 1,
        'agl2_1': 1, 'agl2_2': 1, 'agl2_3': 1, 'agl2_4': 1, 'agl2_5': 1, 'agl2_6': 1,
        'gripper_pos': 1,
        'gripper_pos2': 1,
        'agl_1_act': (16,1), 'agl_2_act': (16,1), 'agl_3_act': (16,1),
        'agl_4_act': (16,1), 'agl_5_act': (16,1), 'agl_6_act': (16,1),
        'agl2_1_act': (16,1), 'agl2_2_act': (16,1), 'agl2_3_act': (16,1),
        'agl2_4_act': (16,1), 'agl2_5_act': (16,1), 'agl2_6_act': (16,1),
        'gripper_act':(16,1), 'gripper_act2':(16,1)
    }

    # Create mock data for fitting
    mock_data = {}

    # Joint mapping from parquet names to training names
    joint_mapping = {
        'left_waist': 'agl_1', 'left_shoulder': 'agl_2', 'left_elbow': 'agl_3',
        'left_forearm_roll': 'agl_4', 'left_wrist_angle': 'agl_5', 'left_wrist_rotate': 'agl_6',
        'right_waist': 'agl2_1', 'right_shoulder': 'agl2_2', 'right_elbow': 'agl2_3',
        'right_forearm_roll': 'agl2_4', 'right_wrist_angle': 'agl2_5', 'right_wrist_rotate': 'agl2_6',
        'left_gripper': 'gripper_pos', 'right_gripper': 'gripper_pos2'
    }

    # For states (shape [N, 1])
    for joint_name, training_name in joint_mapping.items():
        joint_idx = ['left_waist', 'left_shoulder', 'left_elbow', 'left_forearm_roll',
                    'left_wrist_angle', 'left_wrist_rotate', 'right_waist', 'right_shoulder',
                    'right_elbow', 'right_forearm_roll', 'right_wrist_angle', 'right_wrist_rotate',
                    'left_gripper', 'right_gripper'].index(joint_name)
        mock_data[training_name] = torch.tensor(all_states[:, joint_idx:joint_idx+1], dtype=torch.float32)

    # For actions (shape [N, 16, 1])
    for joint_name, training_name in joint_mapping.items():
        joint_idx = ['left_waist', 'left_shoulder', 'left_elbow', 'left_forearm_roll',
                    'left_wrist_angle', 'left_wrist_rotate', 'right_waist', 'right_shoulder',
                    'right_elbow', 'right_forearm_roll', 'right_wrist_angle', 'right_wrist_rotate',
                    'left_gripper', 'right_gripper'].index(joint_name)
        # Reshape actions to [N, 16, 1]
        action_data = all_actions[:, joint_idx]  # [N]
        # Create 16-step sequences (this is a simplification - in reality we'd need future actions)
        action_sequences = np.tile(action_data[:, None], (1, 16))  # [N, 16]
        mock_data[f"{training_name}_act"] = torch.tensor(action_sequences[:, :, None], dtype=torch.float32)

    # Create a temporary scaler and fit it
    import sys
    sys.path.append('/home/aposadasn/lerobot-beavr/src/lerobot/policies/dact/mtil_paper')
    from scaler_M import Scaler

    scaler = Scaler(lowdim_dict=lowdim_dict)
    scaler.fit(mock_data)

    # Save the fitted scaler parameters
    scaler_params_file = os.path.join(output_dir, "scaler_params.pth")
    scaler.save(scaler_params_file)
    print(f"Saved fitted scaler parameters to {scaler_params_file}")


def main():
    # Configuration
    parquet_dir = "/mnt/data/aposadasn/.cache/huggingface/hub/datasets--lerobot--aloha_sim_insertion_scripted/snapshots/4d34dbebc14c30a71c09a86618b0c1531304a842"
    output_dir = "/home/aposadasn/lerobot-beavr/outputs/datasets/aloha_sim_insertion_processed"
    future_steps = 16

    print("Starting parquet preprocessing...")
    print(f"Input: {parquet_dir}")
    print(f"Output: {output_dir}")

    # Load data
    data_df, info = load_parquet_dataset(parquet_dir)

    # Extract episodes
    episodes = extract_episodes(data_df, info)

    # Process each episode
    print("Converting episodes to HDF5...")
    for episode_idx, frames in tqdm(episodes, desc="Processing episodes"):
        process_episode_to_hdf5(episode_idx, frames, output_dir, future_steps)

    # Create scaler data
    create_scaler_data(episodes, output_dir)

    print("Preprocessing complete!")
    print(f"Processed {len(episodes)} episodes")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
