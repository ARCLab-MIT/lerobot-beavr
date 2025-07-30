import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_episodes(file_path):
    """Load episodes from pickle file."""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def check_port_visibility(episode, rl=False):
    # If the last position is outside of 2m radius, return False
    if rl:
        return True
    return np.linalg.norm(episode['x'][-1][0:3]) <= 0.004


def compute_final_state_stats(valid_episodes):
    """Compute statistics for final states of valid episodes."""
    final_positions = []
    final_velocities = []
    for ep in valid_episodes:
        final_x = ep['x'][-1]
        final_positions.append(final_x[0:3])
        final_velocities.append(final_x[3:6])
    final_positions = np.array(final_positions)
    final_velocities = np.array(final_velocities)
    pos_magnitudes = np.linalg.norm(final_positions, axis=1)
    vel_magnitudes = np.linalg.norm(final_velocities, axis=1)

    stats = {
        'position': {
            'mean': np.mean(pos_magnitudes),
            'p75': np.percentile(pos_magnitudes, 75),
            'p95': np.percentile(pos_magnitudes, 95),
            'p99': np.percentile(pos_magnitudes, 99)
        },
        'velocity': {
            'mean': np.mean(vel_magnitudes),
            'p75': np.percentile(vel_magnitudes, 75),
            'p95': np.percentile(vel_magnitudes, 95),
            'p99': np.percentile(vel_magnitudes, 99)
        }
    }
    return stats

def evaluate_trajectories(episodes, rl=False):
    """Evaluate episodes according to benchmark criteria."""
    valid_episodes = []
    violation_count = 0
    for episode in episodes:
        if not check_port_visibility(episode, rl=rl):
            violation_count += 1
        else:
            valid_episodes.append(episode)
    return valid_episodes, violation_count

def print_results(stats, total_episodes, violation_count, dataset_name, final_positions, threshold=2):
    print(f"\nResults for {dataset_name}:")
    print("-" * 50)
    print(f"Total episodes: {total_episodes}")
    print(f"Violation rate: {(violation_count/total_episodes)*100:.2f}% ({violation_count} episodes)")
    print(f"Valid episodes: {total_episodes - violation_count}")
    print("\nPosition magnitude statistics (meters):")
    print(f"  Mean: {stats['position']['mean']:.6f}")
    print(f"  75th percentile: {stats['position']['p75']:.6f}")
    print(f"  95th percentile: {stats['position']['p95']:.6f}")
    print(f"  99th percentile: {stats['position']['p99']:.6f}")
    print("\nVelocity magnitude statistics (m/s):")
    print(f"  Mean: {stats['velocity']['mean']:.6f}")
    print(f"  75th percentile: {stats['velocity']['p75']:.6f}")
    print(f"  95th percentile: {stats['velocity']['p95']:.6f}")
    print(f"  99th percentile: {stats['velocity']['p99']:.6f}")
    # Calculate percentage within threshold
    if final_positions is not None:
        above_threshold = np.linalg.norm(final_positions, axis=1) >= threshold
        for traj in final_positions[above_threshold]:
            print(np.linalg.norm(traj))
        percent_above = 100.0 * np.sum(above_threshold) / len(final_positions) if len(final_positions) > 0 else 0.0
        print(f"\nPercentage of valid trajectories above {threshold} m of origin: {percent_above:.2f}% ({np.sum(above_threshold)}/{len(final_positions)})")
        # Print the lowest final position
        print(f"Lowest final position: {np.min(np.linalg.norm(final_positions, axis=1)):.6f} m")

def load_rl_parquet_episodes(parquet_dir):
    """Load RL episodes from parquet files, extracting final position and velocity from observation.state."""
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, 'episode_*.parquet')))
    episodes = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            state = np.stack(df['observation.state'].to_numpy())  # (N, 13)
            ep = {'x': state}
            episodes.append(ep)
        except Exception as e:
            print(f"Failed to process RL episode {pf}: {e}")
    return episodes

def main():
    files = {
        'Original Initial Conditions': 'lerobot/inav/evaluation/unique_episodes.pkl',
        'Random Initial Conditions': 'lerobot/inav/evaluation/unique_episodes_random2.pkl'
    }
    print("Docking Evaluation Analysis")
    print("=" * 50)
    for name, file_path in files.items():
        if not Path(file_path).exists():
            print(f"Warning: {file_path} not found!")
            continue
        episodes = load_episodes(file_path)
        valid_episodes, violation_count = evaluate_trajectories(episodes, rl=False)
        if valid_episodes:
            stats = compute_final_state_stats(valid_episodes)
            # Convert from km to m
            stats['position']['mean'] *= 1000
            stats['position']['p75'] *= 1000
            stats['position']['p95'] *= 1000
            stats['position']['p99'] *= 1000
            stats['velocity']['mean'] *= 1000
            stats['velocity']['p75'] *= 1000
            stats['velocity']['p95'] *= 1000
            stats['velocity']['p99'] *= 1000
            # Get final positions in meters
            final_positions = np.array([ep['x'][-1][0:3]*1000 for ep in valid_episodes])
            print_results(stats, len(episodes), violation_count, name, final_positions)
        else:
            print(f"\nNo valid trajectories found in {name}")

    # --- RL (parquet) analysis ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
    rl_dir = os.path.join(project_root, 'datasets/iss_docking_images/data/chunk-000')
    rl_episodes = load_rl_parquet_episodes(rl_dir)
    if rl_episodes:
        valid_episodes, violation_count = evaluate_trajectories(rl_episodes, rl=True)
        if valid_episodes:
            stats = compute_final_state_stats(valid_episodes)
            # Do NOT multiply by 1000 for RL
            final_positions = np.array([ep['x'][-1][0:3] for ep in valid_episodes])
            print_results(stats, len(rl_episodes), violation_count, 'RL (Parquet)', final_positions)
        else:
            print("\nNo valid RL (parquet) trajectories found")
    else:
        print("\nNo RL (parquet) episodes found")

if __name__ == "__main__":
    main()