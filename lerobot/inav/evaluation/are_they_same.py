#!/usr/bin/env python3
"""
Analysis script to compare initial position ranges across three datasets and calculate mass loss:
1. unique_episodes.pkl - IL dataset
2. unique_episodes_random.pkl - IL random dataset  
3. Parquet files - RL dataset

This script extracts initial x,y,z positions and compares ranges and initial conditions.
It also calculates average mass loss per episode.
"""

import glob
import os
import pickle
from typing import Dict, Tuple, List

import numpy as np

# Try to import pandas, but make it optional
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Parquet analysis will be skipped.")

import pyarrow.parquet as pq

# Configuration flags
ENABLE_LOGGING = True  # Set to False to remove debug prints
DECIMAL_PRECISION = 6  # For comparison precision

# Paths
PKL_UNIQUE_PATH = os.path.join(os.path.dirname(__file__), 'unique_episodes.pkl')
PKL_RANDOM_PATH = os.path.join(os.path.dirname(__file__), 'unique_episodes_random2.pkl')
PARQUET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../datasets/iss_docking_images/data/chunk-000'))

def log(message: str) -> None:
    """Conditional logging based on configuration flag"""
    if ENABLE_LOGGING:
        print(message)

def calculate_mass_loss(episode_data: np.ndarray) -> float:
    """
    Calculate mass loss for a single episode.
    
    Args:
        episode_data: Array containing state data for all timesteps
        
    Returns:
        Total mass loss during the episode
    """
    # Mass is typically the last element in the state vector
    initial_mass = episode_data[0]  # First timestep, last element
    final_mass = episode_data[-1]   # Last timestep, last element
    return initial_mass - final_mass

def extract_mass_loss_pkl(pkl_path: str) -> List[float]:
    """
    Extract mass loss from each episode in pickle file.
    
    Args:
        pkl_path: Path to pickle file
        
    Returns:
        List of mass loss values for each episode
    """
    log(f"Calculating mass loss from: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        episodes = pickle.load(f)
    
    mass_losses = []
    
    for idx, episode in enumerate(episodes):
        # Get state data from 'x' key
        mass_data = np.array(episode['mass'])
        mass_loss = calculate_mass_loss(mass_data)
        mass_losses.append(mass_loss)
        
        if ENABLE_LOGGING and idx < 3:  # Log first few episodes for verification
            log(f"  Episode {idx}: mass_loss={mass_loss:.6f}")
    
    return mass_losses

def extract_initial_positions_pkl(pkl_path: str) -> np.ndarray:
    """
    Extract initial positions (time=0) from pickle file.
    
    Args:
        pkl_path: Path to pickle file
        
    Returns:
        Array of shape (n_episodes, 3) containing initial x,y,z positions
    """
    log(f"Loading pickle file: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        episodes = pickle.load(f)
    
    log(f"Loaded {len(episodes)} episodes")
    
    initial_positions = []
    
    for idx, episode in enumerate(episodes):
        # Get position data from 'x' key
        x_data = np.array(episode['x'])
        
        # Extract initial position (first timestep, first 3 coordinates)
        initial_pos = x_data[0, :3]  # x, y, z
        initial_positions.append(initial_pos)
        
        if ENABLE_LOGGING and idx < 3:  # Log first few episodes for verification
            time_data = np.array(episode['time'])
            log(f"  Episode {idx}: time[0]={time_data[0]:.6f}, initial_pos={initial_pos}")
    
    return np.array(initial_positions)

def extract_initial_positions_parquet_pyarrow(parquet_dir: str) -> np.ndarray:
    """
    Extract initial positions from parquet files using pyarrow.
    
    Args:
        parquet_dir: Directory containing episode parquet files
        
    Returns:
        Array of shape (n_episodes, 3) containing initial x,y,z positions
    """
    log(f"Loading parquet files from: {parquet_dir}")
    
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "episode_*.parquet")))
    log(f"Found {len(parquet_files)} parquet files")
    
    initial_positions = []

    for idx, parquet_file in enumerate(parquet_files):
        try:
            # First, let's inspect the schema of the first file to understand structure
            if idx == 0:
                # Read schema to see available columns
                table_sample = pq.read_table(parquet_file)
                log(f"Available columns: {table_sample.column_names}")
            
            # Try to read with the correct column name based on the error message
            table = pq.read_table(parquet_file, columns=['observation.state', 'timestamp'])
            state_column = table['observation.state']
            timestamp_column = table['timestamp']

            # Find the row with the minimum timestamp (should be the initial step)
            min_time_idx = timestamp_column.to_numpy().argmin()
            state_data = state_column[min_time_idx].as_py()  # This should be a list or array

            # Extract position from state - assuming first 3 elements are x, y, z
            initial_pos = np.array(state_data)[:3]
            initial_positions.append(initial_pos)
            
            if ENABLE_LOGGING and idx < 3:  # Log first few episodes for verification
                timestamp_val = timestamp_column.to_numpy()[min_time_idx]
                log(f"  Episode {idx}: time={timestamp_val:.6f}, initial_pos={initial_pos}")
                
        except Exception as e:
            log(f"Error processing {parquet_file}: {e}")
            continue

    return np.array(initial_positions)

def extract_mass_loss_parquet_pyarrow(parquet_dir: str) -> List[float]:
    """
    Extract mass loss from parquet files using pyarrow.
    
    Args:
        parquet_dir: Directory containing episode parquet files
        
    Returns:
        List of mass loss values for each episode
    """
    log(f"Calculating mass loss from parquet files in: {parquet_dir}")
    
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "episode_*.parquet")))
    log(f"Found {len(parquet_files)} parquet files")
    
    if not parquet_files:
        log(f"No parquet files found in directory: {parquet_dir}")
        return []
        
    mass_losses = []

    for idx, parquet_file in enumerate(parquet_files):
        try:
            # First, let's inspect the schema of the first file to understand structure
            if idx == 0:
                table_sample = pq.read_table(parquet_file)
                log(f"Available columns in parquet file: {table_sample.column_names}")
            
            # Read the mass data from the parquet file
            table = pq.read_table(parquet_file)
            
            # Try to get mass data from different possible column names
            if 'mass' in table.column_names:
                mass_column = table['mass'].to_numpy()
            elif 'observation.mass' in table.column_names:
                mass_column = table['observation.mass'].to_numpy()
            else:
                log(f"Could not find mass column in file {parquet_file}. Available columns: {table.column_names}")
                continue
            
            # Calculate mass loss (initial - final)
            mass_loss = float(mass_column[0] - mass_column[-1])
            mass_losses.append(mass_loss)
            
            if ENABLE_LOGGING and idx < 3:  # Log first few episodes for verification
                log(f"  Episode {idx}: initial_mass={mass_column[0]:.6f}, final_mass={mass_column[-1]:.6f}, mass_loss={mass_loss:.6f}")
                
        except Exception as e:
            log(f"Error processing {parquet_file}: {str(e)}")
            continue

    if not mass_losses:
        log("No mass loss values were successfully calculated from parquet files")
    else:
        log(f"Successfully calculated mass loss for {len(mass_losses)} episodes")

    return mass_losses

def calculate_ranges(positions: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """
    Calculate min/max ranges for x, y, z coordinates.
    
    Args:
        positions: Array of shape (n_episodes, 3)
        
    Returns:
        Dictionary with 'x', 'y', 'z' keys and (min, max) tuples
    """
    ranges = {}
    coord_names = ['x', 'y', 'z']
    
    for i, coord in enumerate(coord_names):
        coord_values = positions[:, i]
        ranges[coord] = (float(np.min(coord_values)), float(np.max(coord_values)))
    
    return ranges

def print_ranges(ranges: Dict[str, Tuple[float, float]], dataset_name: str) -> None:
    """Print formatted ranges for a dataset"""
    print(f"\n{dataset_name} Initial Position Ranges:")
    print("-" * 50)
    for coord, (min_val, max_val) in ranges.items():
        range_val = max_val - min_val
        print(f"{coord.upper()}: [{min_val:.6f}, {max_val:.6f}] (range: {range_val:.6f})")

def compare_initial_conditions(pos1: np.ndarray, pos2: np.ndarray, 
                             name1: str, name2: str, tolerance: float = 1e-6) -> bool:
    """
    Compare if two sets of initial positions are the same within tolerance.
    
    Args:
        pos1, pos2: Arrays of initial positions to compare
        name1, name2: Names for logging
        tolerance: Numerical tolerance for comparison
        
    Returns:
        True if positions match within tolerance
    """
    log(f"\nComparing {name1} vs {name2}:")
    
    if pos1.shape != pos2.shape:
        log(f"  Shape mismatch: {pos1.shape} vs {pos2.shape}")
        return False
    
    # Sort both arrays by x, then y, then z for proper comparison
    sort_idx1 = np.lexsort((pos1[:, 2], pos1[:, 1], pos1[:, 0]))
    sort_idx2 = np.lexsort((pos2[:, 2], pos2[:, 1], pos2[:, 0]))
    
    sorted_pos1 = pos1[sort_idx1]
    sorted_pos2 = pos2[sort_idx2]
    
    # Calculate differences
    diff = np.abs(sorted_pos1 - sorted_pos2)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    log(f"  Max difference: {max_diff:.10f}")
    log(f"  Mean difference: {mean_diff:.10f}")
    log(f"  Tolerance: {tolerance}")
    
    # Check if all differences are within tolerance
    within_tolerance = np.all(diff <= tolerance)
    
    if within_tolerance:
        log(f"  ✓ Datasets match within tolerance!")
        return True
    else:
        # Find episodes with largest differences
        episode_diffs = np.max(diff, axis=1)
        worst_episodes = np.argsort(episode_diffs)[-5:]  # 5 worst episodes
        
        log(f"  ✗ Datasets differ beyond tolerance")
        log(f"  Episodes with largest differences:")
        for i, ep_idx in enumerate(worst_episodes):
            ep_diff = episode_diffs[ep_idx]
            log(f"    Episode {ep_idx}: max_diff = {ep_diff:.10f}")
        
        return False

def main():
    """Main analysis function"""
    print("="*60)
    print("INITIAL POSITION AND MASS LOSS ANALYSIS")
    print("="*60)
    
    # Extract initial positions and mass loss from datasets
    unique_positions = extract_initial_positions_pkl(PKL_UNIQUE_PATH) * 1000
    random_positions = extract_initial_positions_pkl(PKL_RANDOM_PATH) * 1000
    unique_mass_losses = extract_mass_loss_pkl(PKL_UNIQUE_PATH)
    random_mass_losses = extract_mass_loss_pkl(PKL_RANDOM_PATH)
        
    log(f"\nTrying to load parquet data...")
    if os.path.exists(PARQUET_DIR):
        parquet_positions = extract_initial_positions_parquet_pyarrow(PARQUET_DIR)
        parquet_mass_losses = extract_mass_loss_parquet_pyarrow(PARQUET_DIR)
    
    # Calculate ranges for each dataset
    unique_ranges = calculate_ranges(unique_positions)
    random_ranges = calculate_ranges(random_positions)
    
    # Print results
    print_ranges(unique_ranges, "UNIQUE EPISODES (IL)")
    print_ranges(random_ranges, "RANDOM EPISODES (IL)")
    
    # Print mass loss statistics
    print("\nMASS LOSS STATISTICS")
    print("="*60)
    print("UNIQUE EPISODES (IL):")
    print(f"• Average mass loss: {np.mean(unique_mass_losses):.6f}")
    print(f"• Max mass loss: {np.max(unique_mass_losses):.6f}")
    print(f"• Min mass loss: {np.min(unique_mass_losses):.6f}")
    print(f"• Std mass loss: {np.std(unique_mass_losses):.6f}")
    
    print("\nRANDOM EPISODES (IL):")
    print(f"• Average mass loss: {np.mean(random_mass_losses):.6f}")
    print(f"• Max mass loss: {np.max(random_mass_losses):.6f}")
    print(f"• Min mass loss: {np.min(random_mass_losses):.6f}")
    print(f"• Std mass loss: {np.std(random_mass_losses):.6f}")
    
    if parquet_positions is not None and len(parquet_positions) > 0:
        parquet_ranges = calculate_ranges(parquet_positions)
        print_ranges(parquet_ranges, "PARQUET EPISODES (RL)")
        
        if parquet_mass_losses and len(parquet_mass_losses) > 0:
            print("\nPARQUET EPISODES (RL):")
            print(f"• Average mass loss: {np.mean(parquet_mass_losses):.6f}")
            print(f"• Max mass loss: {np.max(parquet_mass_losses):.6f}")
            print(f"• Min mass loss: {np.min(parquet_mass_losses):.6f}")
            print(f"• Std mass loss: {np.std(parquet_mass_losses):.6f}")
        
        # Compare unique_episodes.pkl vs parquet data
        print(f"\n{'='*60}")
        print("VERIFICATION: Comparing unique_episodes.pkl vs parquet data")
        print(f"{'='*60}")
        
        match = compare_initial_conditions(
            unique_positions, parquet_positions,
            "unique_episodes.pkl", "parquet_data"
        )
        
        if match:
            print(f"\n✓ VERIFIED: unique_episodes.pkl and parquet data have the same initial conditions!")
        else:
            print(f"\n✗ DIFFERENCE: unique_episodes.pkl and parquet data have different initial conditions!")
    else:
        log("\nSkipping parquet comparison due to loading issues")
    
    # Compare unique vs random episodes
    print(f"\n{'='*60}")
    print("COMPARISON: unique_episodes.pkl vs unique_episodes_random.pkl")
    print(f"{'='*60}")
    
    match_pkl = compare_initial_conditions(
        unique_positions, random_positions,
        "unique_episodes.pkl", "unique_episodes_random.pkl"
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"• Unique episodes: {len(unique_positions)} episodes")
    print(f"• Random episodes: {len(random_positions)} episodes")
    if parquet_positions is not None:
        print(f"• Parquet episodes: {len(parquet_positions)} episodes")
    
    print(f"\nDataset comparisons:")
    if parquet_positions is not None and len(parquet_positions) > 0:
        print(f"• Unique vs Parquet: {'MATCH' if match else 'DIFFER'}")
    print(f"• Unique vs Random: {'MATCH' if match_pkl else 'DIFFER'}")
    
    print(f"\nMass Loss Summary:")
    print(f"• Unique Episodes Avg Mass Loss: {np.mean(unique_mass_losses):.6f}")
    print(f"• Random Episodes Avg Mass Loss: {np.mean(random_mass_losses):.6f}")
    if parquet_mass_losses and len(parquet_mass_losses) > 0:
        print(f"• Parquet Episodes Avg Mass Loss: {np.mean(parquet_mass_losses):.6f}")
    else:
        print(f"• Parquet Episodes Avg Mass Loss: N/A")

if __name__ == "__main__":
    main()
