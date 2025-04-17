#!/usr/bin/env python

"""
Script to calculate statistics for each episode and output them in the required format.
This script calculates statistics for observation.state and action, and combines them
with the existing observation.images stats.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

def load_existing_image_stats(stats_file):
    """
    Load existing image statistics from a jsonl file
    
    Args:
        stats_file: Path to the stats file
    
    Returns:
        dict: Dictionary mapping episode_index to image stats
    """
    image_stats = {}
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                episode_index = data["episode_index"]
                if "stats" in data and "observation.images" in data["stats"]:
                    image_stats[episode_index] = data["stats"]["observation.images"]
    
    return image_stats

def calculate_stats(data, column_name):
    """
    Calculate statistics for a column in the dataframe
    
    Args:
        data: Pandas dataframe
        column_name: Name of the column to calculate statistics for
    
    Returns:
        dict: Dictionary containing min, max, mean, std, and count
    """
    if column_name not in data.columns:
        return None
    
    values = data[column_name].values
    
    # Handle different data types
    if len(values) == 0:
        return None
    
    # For numpy arrays (state, action vectors)
    if isinstance(values[0], np.ndarray):
        # Stack arrays to get shape (n_samples, n_features)
        stacked = np.stack(values)
        
        return {
            "min": stacked.min(axis=0).tolist(),
            "max": stacked.max(axis=0).tolist(),
            "mean": stacked.mean(axis=0).tolist(),
            "std": stacked.std(axis=0).tolist(),
            "count": [len(values)]
        }
    
    # For scalar values
    elif np.issubdtype(type(values[0]), np.number):
        return {
            "min": [float(np.min(values))],
            "max": [float(np.max(values))],
            "mean": [float(np.mean(values))],
            "std": [float(np.std(values))],
            "count": [len(values)]
        }
    
    # For boolean values
    elif isinstance(values[0], (bool, np.bool_)):
        return {
            "min": [bool(np.min(values))],
            "max": [bool(np.max(values))],
            "mean": [float(np.mean(values))],
            "std": [float(np.std(values))],
            "count": [len(values)]
        }
    
    # For other types (like strings)
    else:
        return {
            "min": [str(min(values))],
            "max": [str(max(values))],
            "mean": [0.0],  # Placeholder
            "std": [0.0],   # Placeholder
            "count": [len(values)]
        }

def process_episode_file(file_path):
    """
    Process a single episode file and calculate statistics
    
    Args:
        file_path: Path to the parquet file
    
    Returns:
        tuple: (episode_index, stats_dict)
    """
    try:
        df = pd.read_parquet(file_path)
        
        # Extract episode index
        if 'episode_index' in df.columns:
            episode_index = df['episode_index'].iloc[0]
        else:
            # Try to extract from filename
            filename = os.path.basename(file_path)
            if filename.startswith('episode_') and filename.endswith('.parquet'):
                try:
                    episode_index = int(filename[8:-8])  # Extract number from episode_XXXXX.parquet
                except ValueError:
                    logger.warning(f"Could not extract episode index from {filename}")
                    return None, None
            else:
                logger.warning(f"Could not determine episode index for {file_path}")
                return None, None
        
        # Calculate statistics for each column
        stats = {}
        
        # Process observation.state
        if 'observation.state' in df.columns:
            stats['observation.state'] = calculate_stats(df, 'observation.state')
        
        # Process action
        if 'action' in df.columns:
            stats['action'] = calculate_stats(df, 'action')
        
        # Process metadata columns
        for col in ['episode_index', 'frame_index', 'timestamp', 'index', 'task_index']:
            if col in df.columns:
                stats[col] = calculate_stats(df, col)
        
        # Add next.done if available
        if 'next.done' in df.columns:
            stats['next.done'] = calculate_stats(df, 'next.done')
        
        return episode_index, stats
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description='Calculate statistics for each episode')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--existing_stats', type=str, help='Path to existing stats file')
    parser.add_argument('--output_file', type=str, help='Path to output file')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_file = args.output_file or data_dir / "episode_stats.jsonl"
    
    # Load existing image stats if provided
    image_stats = {}
    if args.existing_stats:
        image_stats = load_existing_image_stats(args.existing_stats)
        logger.info(f"Loaded image stats for {len(image_stats)} episodes")
    
    # Find all parquet files
    parquet_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(Path(root) / file)
    
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    # Process each file
    all_stats = {}
    for file_path in tqdm(parquet_files, desc="Processing files"):
        episode_index, stats = process_episode_file(file_path)
        if episode_index is not None and stats is not None:
            all_stats[episode_index] = stats
    
    logger.info(f"Calculated stats for {len(all_stats)} episodes")
    
    # Combine with image stats
    for episode_index, stats in all_stats.items():
        if episode_index in image_stats:
            stats["observation.images"] = image_stats[episode_index]
    
    # Write to output file
    with open(output_file, 'w') as f:
        for episode_index in sorted(all_stats.keys()):
            stats = all_stats[episode_index]
            f.write(json.dumps({"episode_index": episode_index, "stats": stats}, cls=NumpyEncoder) + "\n")
    
    logger.info(f"Wrote stats to {output_file}")
    
    # Also create a combined stats.json file
    combined_stats = {}
    
    # Combine all episode stats
    for episode_stats in all_stats.values():
        for key, value in episode_stats.items():
            if key not in combined_stats:
                combined_stats[key] = value
    
    # Write combined stats
    with open(data_dir / "stats.json", 'w') as f:
        json.dump(combined_stats, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"Wrote combined stats to {data_dir / 'stats.json'}")

if __name__ == "__main__":
    main() 