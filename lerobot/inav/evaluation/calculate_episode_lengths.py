#!/usr/bin/env python3
"""
Script to calculate average episode lengths across all available datasets:
1. unique_episodes.pkl - IL dataset
2. unique_episodes_random.pkl - IL random dataset
3. unique_episodes_random2.pkl - IL random dataset 2
4. Parquet files - RL dataset
"""

import os
import pickle
import glob
import numpy as np
import pandas as pd

# Paths
PKL_UNIQUE_PATH = os.path.join(os.path.dirname(__file__), 'unique_episodes.pkl')
PKL_RANDOM_PATH = os.path.join(os.path.dirname(__file__), 'unique_episodes_random.pkl')
PKL_RANDOM2_PATH = os.path.join(os.path.dirname(__file__), 'unique_episodes_random2.pkl')
PARQUET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../datasets/iss_docking_images/data/chunk-000'))

def get_episode_length_pkl(episode):
    """Get length of a single episode from pickle data."""
    return len(episode['x'])

def get_episode_lengths_pkl(pkl_path):
    """Get lengths of all episodes in a pickle file."""
    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}")
        return []
        
    print(f"\nAnalyzing {os.path.basename(pkl_path)}...")
    with open(pkl_path, 'rb') as f:
        episodes = pickle.load(f)
    
    lengths = [get_episode_length_pkl(ep) for ep in episodes]
    return lengths

def get_episode_lengths_parquet(parquet_dir):
    """Get lengths of all episodes in parquet files."""
    if not os.path.exists(parquet_dir):
        print(f"Directory not found: {parquet_dir}")
        return []
        
    print(f"\nAnalyzing parquet files in {os.path.basename(parquet_dir)}...")
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "episode_*.parquet")))
    
    lengths = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            lengths.append(len(df))
        except Exception as e:
            print(f"Error processing {os.path.basename(pf)}: {e}")
            continue
            
    return lengths

def print_stats(lengths, dataset_name):
    """Print statistics about episode lengths."""
    if not lengths:
        print(f"{dataset_name}: No episodes found")
        return
        
    lengths = np.array(lengths)
    print(f"\n{dataset_name} Statistics:")
    print(f"Number of episodes: {len(lengths)}")
    print(f"Average length: {np.mean(lengths):.2f}")
    print(f"Std deviation: {np.std(lengths):.2f}")
    print(f"Min length: {np.min(lengths)}")
    print(f"Max length: {np.max(lengths)}")
    print(f"Median length: {np.median(lengths):.0f}")

def main():
    print("="*60)
    print("EPISODE LENGTH ANALYSIS")
    print("="*60)
    
    # Analyze pickle files
    unique_lengths = get_episode_lengths_pkl(PKL_UNIQUE_PATH)
    random_lengths = get_episode_lengths_pkl(PKL_RANDOM_PATH)
    random2_lengths = get_episode_lengths_pkl(PKL_RANDOM2_PATH)
    
    # Analyze parquet files
    parquet_lengths = get_episode_lengths_parquet(PARQUET_DIR)
    
    # Print statistics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print_stats(unique_lengths, "IL Dataset (unique_episodes.pkl)")
    print_stats(random_lengths, "IL Random Dataset (unique_episodes_random.pkl)")
    print_stats(random2_lengths, "IL Random Dataset 2 (unique_episodes_random2.pkl)")
    print_stats(parquet_lengths, "RL Dataset (parquet files)")
    
    # Print overall statistics
    all_lengths = unique_lengths + random_lengths + random2_lengths + parquet_lengths
    if all_lengths:
        print("\n" + "="*60)
        print("OVERALL STATISTICS")
        print("="*60)
        print(f"Total number of episodes across all datasets: {len(all_lengths)}")
        print(f"Overall average episode length: {np.mean(all_lengths):.2f}")
        print(f"Overall std deviation: {np.std(all_lengths):.2f}")

if __name__ == "__main__":
    main() 