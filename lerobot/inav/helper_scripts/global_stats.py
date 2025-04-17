# lerobot/inav/compute_global_image_stats.py
import json
import numpy as np
import argparse
from pathlib import Path

def compute_global_image_stats(
    episodes_stats_file,
    output_stats_file=None,
    old_key="observation.images",
    new_key="observation.images.cam"
):
    """
    Compute global image statistics from per-episode statistics
    
    Args:
        episodes_stats_file: Path to episodes_stats_fixed.jsonl
        output_stats_file: Path to save the updated stats.json (optional)
        old_key: Original image key in episode stats
        new_key: New image key to use in global stats
    """
    print(f"Computing global statistics for {new_key} from {episodes_stats_file}")
    
    # Load episode stats
    episode_stats = []
    with open(episodes_stats_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if old_key in data['stats']:
                # Add episode index and count for weighted averaging
                stats_entry = data['stats'][old_key]
                stats_entry['episode_index'] = data['episode_index']
                episode_stats.append(stats_entry)
    
    if not episode_stats:
        raise ValueError(f"No episodes with '{old_key}' found in the stats file")
    
    # Initialize arrays for aggregation
    total_count = 0
    weighted_means = []
    weighted_vars = []
    all_mins = []
    all_maxs = []
    
    # Process each episode's statistics
    for stats in episode_stats:
        count = stats['count'][0]
        total_count += count
        
        # Extract statistics (assuming they're in the format from the file)
        mean = np.array(stats['mean'])  # Shape: [3, 1, 1]
        std = np.array(stats['std'])    # Shape: [3, 1, 1]
        min_val = np.array(stats['min'])  # Shape: [3, 1, 1]
        max_val = np.array(stats['max'])  # Shape: [3, 1, 1]
        
        # Store weighted mean for later averaging
        weighted_means.append(mean * count)
        
        # For std, we need to convert back to sum of squares
        # Var = Std^2, and weighted variance needs special handling
        weighted_vars.append((std**2) * count)
        
        # Track min/max values
        all_mins.append(min_val)
        all_maxs.append(max_val)
    
    # Compute global statistics
    global_mean = sum(weighted_means) / total_count
    
    # For variance, we need the weighted average of variances
    global_var = sum(weighted_vars) / total_count
    global_std = np.sqrt(global_var)
    
    # Find global min/max
    global_min = np.min(np.stack(all_mins), axis=0)
    global_max = np.max(np.stack(all_maxs), axis=0)
    
    # Create the global stats dictionary
    global_stats = {
        "mean": global_mean.flatten().tolist(),
        "std": global_std.flatten().tolist(),
        "min": global_min.flatten().tolist(),
        "max": global_max.flatten().tolist()
    }
    
    # If output file is specified, update the stats.json file
    if output_stats_file:
        try:
            with open(output_stats_file, 'r') as f:
                stats = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            stats = {}
        
        # Add the new key with computed statistics
        stats[new_key] = global_stats
        
        # Save the updated stats
        with open(output_stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Updated stats saved to {output_stats_file}")
    
    # Return the computed statistics
    return global_stats

def main():
    parser = argparse.ArgumentParser(description="Compute global image statistics from episode stats")
    parser.add_argument("--episodes-stats", required=True, help="Path to episodes_stats_fixed.jsonl")
    parser.add_argument("--stats-json", help="Path to stats.json to update (optional)")
    parser.add_argument("--old-key", default="observation.images", help="Original image key in episode stats")
    parser.add_argument("--new-key", default="observation.images.cam", help="New image key for global stats")
    
    args = parser.parse_args()
    
    # Compute global statistics
    global_stats = compute_global_image_stats(
        episodes_stats_file=args.episodes_stats,
        output_stats_file=args.stats_json,
        old_key=args.old_key,
        new_key=args.new_key
    )
    
    # Print the computed statistics
    print(f"Global statistics for {args.new_key}:")
    print(json.dumps(global_stats, indent=2))

if __name__ == "__main__":
    main()