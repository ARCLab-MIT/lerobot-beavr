#!/usr/bin/env python

"""
Script to print the dataset statistics to understand what keys are available
"""

import logging
from lerobot.common.datasets.factory import make_dataset

logging.basicConfig(level=logging.INFO)

# Create the dataset
dataset = make_dataset(
    {
        "repo_id": "aposadasn/lander18",
        "split": "train"
    }
)

# Print the dataset statistics
print("Dataset statistics keys:")
for key in dataset.stats:
    print(f"  - {key}")
    if isinstance(dataset.stats[key], dict) and "mean" in dataset.stats[key]:
        print(f"    - mean shape: {dataset.stats[key]['mean'].shape if hasattr(dataset.stats[key]['mean'], 'shape') else 'scalar'}")

# Print the dataset features
print("\nDataset features:")
for key in dataset.features:
    print(f"  - {key}")
    if hasattr(dataset.features[key], "shape"):
        print(f"    - shape: {dataset.features[key].shape}") 