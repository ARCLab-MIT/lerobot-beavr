#!/usr/bin/env python

"""
Script to restructure the moon lander dataset to match the expected format.
This script combines separate state components into a single observation.state vector.
"""

import os
import logging
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def restructure_episode_file(file_path):
    """
    Restructure a single episode file to combine state components into observation.state
    and remove individual components
    
    Args:
        file_path: Path to the parquet file
    """
    logger.info(f"Processing {file_path}")
    
    # Read the parquet file
    df = pd.read_parquet(file_path)
    
    # Create a backup of the original file
    backup_path = file_path.with_suffix('.parquet.bak')
    if not backup_path.exists():
        logger.info(f"Creating backup at {backup_path}")
        shutil.copy(file_path, backup_path)
    
    # Check if we need to restructure
    if 'observation.state' in df.columns:
        # Check if it's already in the right format and individual components are removed
        if isinstance(df['observation.state'].iloc[0], np.ndarray) and len(df['observation.state'].iloc[0]) == 13:
            # Check if individual components are already removed
            components = ['position', 'velocity', 'attitude', 'angular_velocity']
            if not any(comp in df.columns for comp in components):
                logger.info(f"File {file_path} already has the correct structure")
                return
    
    # Check if we have the required columns
    required_columns = ['position', 'velocity', 'attitude', 'angular_velocity']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"Missing required columns in {file_path}. Found: {df.columns}")
        return
    
    # Combine the components into a single state vector
    logger.info("Creating observation.state from position, velocity, attitude, and angular_velocity")
    
    # Create the combined state vector
    # Based on the inspection, we have:
    # position: [x, y, z] (3 values)
    # velocity: [vx, vy, vz] (3 values)
    # attitude: [q0, q1, q2, q3] (4 values)
    # angular_velocity: [wx, wy, wz] (3 values)
    # Total: 13 values, which matches the expected shape in info.json
    df['observation.state'] = df.apply(
        lambda row: np.concatenate([
            row['position'],       # x, y, z (3 values)
            row['velocity'],       # vx, vy, vz (3 values)
            row['attitude'],       # q0, q1, q2, q3 (4 values)
            row['angular_velocity'] # wx, wy, wz (3 values)
        ]),
        axis=1
    )
    
    # Verify the shape of the new column
    first_state = df['observation.state'].iloc[0]
    logger.info(f"Created observation.state with shape {first_state.shape} and values {first_state}")
    
    # Remove individual components
    logger.info("Removing individual state components")
    for component in ['position', 'velocity', 'attitude', 'angular_velocity']:
        if component in df.columns:
            df = df.drop(columns=[component])
    
    # Reorder columns as specified
    desired_order = ['action', 'observation.state', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
    
    # Get all columns that exist in the dataframe
    existing_columns = [col for col in desired_order if col in df.columns]
    
    # Add any remaining columns that weren't in the desired order
    remaining_columns = [col for col in df.columns if col not in existing_columns]
    final_column_order = existing_columns + remaining_columns
    
    # Reorder the dataframe
    df = df[final_column_order]
    
    logger.info(f"Final columns: {df.columns.tolist()}")
    
    # Save the updated dataframe
    logger.info(f"Saving updated file to {file_path}")
    df.to_parquet(file_path, index=False)

def main():
    """Main function to restructure all episode files"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Restructure moon lander dataset")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="/home/demo/lerobot-beavr/lerobot/inav/datasets/moon_lander_lerobot/data",
        help="Path to the data directory containing chunk folders"
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    logger.info(f"Processing data in {data_dir}")
    
    # Find all chunk directories
    chunk_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith("chunk-")]
    
    for chunk_dir in chunk_dirs:
        logger.info(f"Processing chunk directory: {chunk_dir}")
        
        # Find all parquet files in this chunk
        parquet_files = list(chunk_dir.glob("*.parquet"))
        
        # Process each file
        for file_path in tqdm(parquet_files, desc=f"Processing {chunk_dir.name}"):
            restructure_episode_file(file_path)
    
    logger.info("Dataset restructuring complete")

if __name__ == "__main__":
    main() 