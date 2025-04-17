# lerobot/inav/trace_dataset_loading.py
import argparse
import os
import sys
import json
from pathlib import Path

def trace_dataset_loading(dataset_dir):
    """
    Trace the dataset loading process to identify where the error occurs
    """
    print(f"Tracing dataset loading for: {dataset_dir}")
    
    # Check directory structure
    print("\nDirectory structure:")
    for root, dirs, files in os.walk(dataset_dir):
        level = root.replace(dataset_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")
    
    # Check stats.json location
    stats_paths = [
        os.path.join(dataset_dir, "stats.json"),
        os.path.join(dataset_dir, "meta", "stats.json")
    ]
    
    for stats_path in stats_paths:
        if os.path.exists(stats_path):
            print(f"\nFound stats.json at: {stats_path}")
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            print(f"Keys in stats.json: {list(stats.keys())}")
            
            if 'observation.images.cam' in stats:
                print("'observation.images.cam' is present in this file")
            else:
                print("WARNING: 'observation.images.cam' is NOT present in this file")
    
    # Check if there's a dataset_info.json file
    info_paths = [
        os.path.join(dataset_dir, "info.json"),
        os.path.join(dataset_dir, "meta", "info.json"),
        os.path.join(dataset_dir, "dataset_info.json")
    ]
    
    for info_path in info_paths:
        if os.path.exists(info_path):
            print(f"\nFound info file at: {info_path}")
            with open(info_path, 'r') as f:
                info = json.load(f)
            print(f"Keys in info file: {list(info.keys())}")
            
            if 'features' in info:
                print(f"Features: {info['features']}")
                
                if 'observation.images.cam' in info['features']:
                    print("'observation.images.cam' is present in features")
                else:
                    print("WARNING: 'observation.images.cam' is NOT present in features")
    
    # Create a symlink to ensure stats.json is in the expected location
    meta_dir = os.path.join(dataset_dir, "meta")
    if not os.path.exists(meta_dir):
        os.makedirs(meta_dir)
    
    stats_in_root = os.path.join(dataset_dir, "stats.json")
    stats_in_meta = os.path.join(meta_dir, "stats.json")
    
    if os.path.exists(stats_in_root) and not os.path.exists(stats_in_meta):
        print(f"\nCreating symlink from {stats_in_root} to {stats_in_meta}")
        os.symlink(stats_in_root, stats_in_meta)
    elif os.path.exists(stats_in_meta) and not os.path.exists(stats_in_root):
        print(f"\nCreating symlink from {stats_in_meta} to {stats_in_root}")
        os.symlink(stats_in_meta, stats_in_root)

def main():
    parser = argparse.ArgumentParser(description="Trace dataset loading process")
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset directory")
    
    args = parser.parse_args()
    trace_dataset_loading(args.dataset_dir)

if __name__ == "__main__":
    main()