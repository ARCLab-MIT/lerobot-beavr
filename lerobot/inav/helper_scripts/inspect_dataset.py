#!/usr/bin/env python

import argparse
from datasets import load_dataset
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main():
    parser = argparse.ArgumentParser(description="Inspect dataset structure")
    parser.add_argument("--repo_id", type=str, required=True, help="Dataset repo ID")
    args = parser.parse_args()
    
    # Load the dataset
    dataset = LeRobotDataset(args.repo_id)
    
    # Print dataset features
    print("Dataset features:")
    for key, feature in dataset.features.items():
        print(f"  {key}: {feature}")
    
    # Print a sample batch
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    batch = next(iter(dataloader))
    print("\nSample batch keys:")
    for key in batch:
        if isinstance(batch[key], list):
            print(f"  {key}: list with {len(batch[key])} items")
            if len(batch[key]) > 0:
                if hasattr(batch[key][0], 'shape'):
                    print(f"    - First item shape: {batch[key][0].shape}")
                else:
                    print(f"    - First item type: {type(batch[key][0])}")
        else:
            print(f"  {key}: {batch[key].shape}")
    
    # Check if observation.images exists
    if "observation.images" in batch:
        print("\nFound 'observation.images' key")
        if isinstance(batch["observation.images"], list):
            print(f"  It's a list with {len(batch['observation.images'])} items")
            if len(batch["observation.images"]) > 0 and hasattr(batch["observation.images"][0], 'shape'):
                print(f"  First image shape: {batch['observation.images'][0].shape}")
    else:
        print("\nNo 'observation.images' key found")
        # Look for potential image keys
        image_keys = [k for k in batch.keys() if "image" in k.lower()]
        if image_keys:
            print(f"Potential image keys: {image_keys}")

if __name__ == "__main__":
    main()