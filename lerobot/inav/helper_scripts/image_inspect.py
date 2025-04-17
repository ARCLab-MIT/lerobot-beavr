#!/usr/bin/env python

import argparse
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main():
    parser = argparse.ArgumentParser(description="Detailed dataset inspection")
    parser.add_argument("--repo_id", type=str, required=True, help="Dataset repo ID")
    args = parser.parse_args()
    
    # Load the dataset
    print(f"Loading dataset from {args.repo_id}")
    dataset = LeRobotDataset(args.repo_id)
    
    # Print dataset metadata
    print("\nDataset metadata:")
    print(f"  Number of episodes: {dataset.num_episodes}")
    print(f"  Number of frames: {dataset.num_frames}")
    print(f"  FPS: {dataset.fps}")
    
    # Print dataset features
    print("\nDataset features:")
    for key, feature in dataset.features.items():
        print(f"  {key}: {feature}")
    
    # Print dataset stats if available
    print("\nDataset stats:")
    if hasattr(dataset, 'stats'):
        for key, stat in dataset.stats.items():
            print(f"  {key}: {stat}")
    else:
        print("  No stats attribute found in dataset")
    
    # Check if dataset has image keys
    print("\nChecking for image keys:")
    if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'image_keys'):
        print(f"  Image keys from meta: {dataset.meta.image_keys}")
    else:
        print("  No image_keys attribute found in dataset.meta")
    
    # Try to access a single item directly
    print("\nAccessing a single item directly:")
    try:
        item = dataset[0]
        print("  Keys in item:")
        for key in item:
            if isinstance(item[key], list):
                print(f"    {key}: list with {len(item[key])} items")
                if len(item[key]) > 0:
                    if hasattr(item[key][0], 'shape'):
                        print(f"      - First item shape: {item[key][0].shape}")
                    else:
                        print(f"      - First item type: {type(item[key][0])}")
            elif torch.is_tensor(item[key]):
                print(f"    {key}: tensor with shape {item[key].shape}")
            else:
                print(f"    {key}: {type(item[key])}")
        
        # Check specifically for observation.images
        if 'observation.images' in item:
            print("\n  Found 'observation.images' in direct item access")
            if isinstance(item['observation.images'], list):
                print(f"    It's a list with {len(item['observation.images'])} items")
                if len(item['observation.images']) > 0:
                    print(f"    First item type: {type(item['observation.images'][0])}")
                    if hasattr(item['observation.images'][0], 'shape'):
                        print(f"    First item shape: {item['observation.images'][0].shape}")
            elif torch.is_tensor(item['observation.images']):
                print(f"    It's a tensor with shape {item['observation.images'].shape}")
        else:
            print("\n  'observation.images' not found in direct item access")
    except Exception as e:
        print(f"  Error accessing item: {e}")
    
    # Create a dataloader and check a batch
    print("\nCreating dataloader and checking a batch:")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    try:
        batch = next(iter(dataloader))
        print("  Keys in batch:")
        for key in batch:
            if isinstance(batch[key], list):
                print(f"    {key}: list with {len(batch[key])} items")
                if len(batch[key]) > 0:
                    if hasattr(batch[key][0], 'shape'):
                        print(f"      - First item shape: {batch[key][0].shape}")
                    else:
                        print(f"      - First item type: {type(batch[key][0])}")
            elif torch.is_tensor(batch[key]):
                print(f"    {key}: tensor with shape {batch[key].shape}")
            else:
                print(f"    {key}: {type(batch[key])}")
    except Exception as e:
        print(f"  Error accessing batch: {e}")
    
    # Try to find any keys that might contain images
    print("\nLooking for potential image keys:")
    image_related_keys = []
    for key in dataset.features:
        if 'image' in key.lower() or 'camera' in key.lower() or 'visual' in key.lower():
            image_related_keys.append(key)
    if image_related_keys:
        print(f"  Found potential image-related keys: {image_related_keys}")
    else:
        print("  No potential image-related keys found in features")

if __name__ == "__main__":
    main()