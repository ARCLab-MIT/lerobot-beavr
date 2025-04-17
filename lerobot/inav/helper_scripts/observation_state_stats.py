#!/usr/bin/env python

"""
Script to compute statistics for all parquet files in the moon_lander_lerobot dataset
and generate a fixed stats.json file
"""

import argparse
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import glob

def main():
    parser = argparse.ArgumentParser(description="Compute statistics for moon_lander_lerobot dataset")
    parser.add_argument("--data_dir", type=str, default="lerobot/inav/datasets/moon_lander_lerobot/data",
                        help="Directory containing the parquet files")
    parser.add_argument("--output", type=str, default="lerobot/inav/fixed_stats.json",
                        help="Output file for the computed statistics")
    args = parser.parse_args()
    
    # Find all parquet files
    parquet_files = glob.glob(f"{args.data_dir}/**/*.parquet", recursive=True)
    print(f"Found {len(parquet_files)} parquet files")
    
    if len(parquet_files) == 0:
        print(f"No parquet files found in {args.data_dir}")
        return
    
    # Initialize dictionaries to store stats
    all_values = {}
    
    # Process each parquet file
    for file_path in tqdm(parquet_files, desc="Processing parquet files"):
        try:
            df = pd.read_parquet(file_path)
            
            # Process each column
            for col in df.columns:
                if col not in all_values:
                    all_values[col] = []
                
                # For array columns, we need to stack all arrays
                if isinstance(df[col].iloc[0], np.ndarray):
                    for val in df[col]:
                        all_values[col].append(val)
                else:
                    all_values[col].extend(df[col].tolist())
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Compute statistics
    stats = {}
    
    for col, values in all_values.items():
        if col in ['timestamp', 'frame_index', 'episode_index', 'task_index', 'index']:
            # For scalar columns
            values_array = np.array(values)
            stats[col] = {
                "mean": [float(np.mean(values_array))],
                "std": [float(np.std(values_array))],
                "max": [float(np.max(values_array))],
                "min": [float(np.min(values_array))]
            }
        else:
            # For array columns
            values_array = np.stack(values)
            stats[col] = {
                "mean": np.mean(values_array, axis=0).tolist(),
                "std": np.std(values_array, axis=0).tolist(),
                "max": np.max(values_array, axis=0).tolist(),
                "min": np.min(values_array, axis=0).tolist()
            }
    
    # Ensure observation.state is included
    if "observation.state" not in stats and "position" in stats and "velocity" in stats and "attitude" in stats and "angular_velocity" in stats:
        # Create observation.state from its components
        pos_mean = np.array(stats["position"]["mean"])
        vel_mean = np.array(stats["velocity"]["mean"])
        att_mean = np.array(stats["attitude"]["mean"])
        ang_vel_mean = np.array(stats["angular_velocity"]["mean"])
        
        pos_std = np.array(stats["position"]["std"])
        vel_std = np.array(stats["velocity"]["std"])
        att_std = np.array(stats["attitude"]["std"])
        ang_vel_std = np.array(stats["angular_velocity"]["std"])
        
        pos_max = np.array(stats["position"]["max"])
        vel_max = np.array(stats["velocity"]["max"])
        att_max = np.array(stats["attitude"]["max"])
        ang_vel_max = np.array(stats["angular_velocity"]["max"])
        
        pos_min = np.array(stats["position"]["min"])
        vel_min = np.array(stats["velocity"]["min"])
        att_min = np.array(stats["attitude"]["min"])
        ang_vel_min = np.array(stats["angular_velocity"]["min"])
        
        stats["observation.state"] = {
            "mean": np.concatenate([pos_mean, vel_mean, att_mean, ang_vel_mean]).tolist(),
            "std": np.concatenate([pos_std, vel_std, att_std, ang_vel_std]).tolist(),
            "max": np.concatenate([pos_max, vel_max, att_max, ang_vel_max]).tolist(),
            "min": np.concatenate([pos_min, vel_min, att_min, ang_vel_min]).tolist()
        }
    
    # Save the stats
    with open(args.output, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Statistics saved to {args.output}")

if __name__ == "__main__":
    main()