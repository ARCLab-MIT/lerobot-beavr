#!/usr/bin/env python

import json
import numpy as np
from pathlib import Path

# Path to the dataset
dataset_dir = Path("/home/demo/lerobot-beavr/lerobot/inav/datasets/moon_lander_lerobot")
stats_file = dataset_dir / "meta" / "stats.json"
episodes_stats_file = dataset_dir / "meta" / "episodes_stats.jsonl"

# Check if files exist
if not stats_file.exists():
    print(f"Stats file not found: {stats_file}")
    exit(1)

if not episodes_stats_file.exists():
    print(f"Episodes stats file not found: {episodes_stats_file}")
    exit(1)

# Load the stats
with open(stats_file, 'r') as f:
    stats = json.load(f)

# Check for scalar values in stats
def check_dimensions(obj, path=""):
    if isinstance(obj, (int, float, bool, str, type(None))):
        print(f"Found scalar value at {path}: {obj}")
        return True
    elif isinstance(obj, list):
        if len(obj) == 0:
            print(f"Empty list found at {path}")
            return False
        # Check if this is a scalar disguised as a list
        if not isinstance(obj[0], (list, dict)):
            print(f"Found 1D array at {path}: {obj}")
        # Recursively check elements
        for i, item in enumerate(obj):
            check_dimensions(item, f"{path}[{i}]")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            check_dimensions(value, f"{path}.{key}" if path else key)
    else:
        print(f"Unknown type at {path}: {type(obj)}")
        return True
    return False

# Check stats
print("Checking stats.json...")
check_dimensions(stats)

# Load and check episodes_stats
print("\nChecking episodes_stats.jsonl...")
with open(episodes_stats_file, 'r') as f:
    for i, line in enumerate(f):
        episode_stats = json.loads(line)
        print(f"\nChecking episode {i}...")
        check_dimensions(episode_stats["stats"])
        if i >= 2:  # Just check the first few episodes
            break