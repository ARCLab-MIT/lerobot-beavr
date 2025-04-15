#!/usr/bin/env python3
"""
Script to tag a dataset with a codebase version.
"""

import os
import argparse
import json
from huggingface_hub import HfApi

def tag_dataset(repo_id, version=None, debug=True):
    """
    Tag a dataset with a codebase version.
    
    Args:
        repo_id (str): HuggingFace repository ID for the dataset
        version (str): Version to tag the dataset with (if None, uses v2.0)
        debug (bool): Whether to enable debug mode
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Use default version if not specified
    if version is None:
        version = "v2.0"
    
    if debug:
        print(f"Tagging dataset {repo_id} with version {version}")
    
    try:
        # Create HuggingFace API client
        hub_api = HfApi()
        
        # Create tag
        hub_api.create_tag(repo_id, tag=version, repo_type="dataset")
        
        if debug:
            print(f"Successfully tagged dataset {repo_id} with version {version}")
        
        return True
    except Exception as e:
        if debug:
            print(f"Error tagging dataset: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Tag a dataset with a codebase version")
    parser.add_argument("--repo-id", type=str, required=True,
                        help="HuggingFace repository ID for the dataset")
    parser.add_argument("--version", type=str, default="v2.0",
                        help="Version to tag the dataset with")
    parser.add_argument("--debug", action="store_true", default=True,
                        help="Enable debug mode with additional logging")
    
    args = parser.parse_args()
    
    success = tag_dataset(
        repo_id=args.repo_id,
        version=args.version,
        debug=args.debug
    )
    
    if success:
        print(f"Dataset {args.repo_id} successfully tagged with version {args.version}")
    else:
        print(f"Failed to tag dataset {args.repo_id}")

if __name__ == "__main__":
    main()