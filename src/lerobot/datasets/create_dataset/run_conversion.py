#!/usr/bin/env python3

"""Command-line interface for dataset conversion."""

import argparse
import sys

from lerobot.datasets.create_dataset.config.dataset_config import (
    DatasetConfig,
    create_sample_config,
    load_config,
)
from lerobot.datasets.create_dataset.config.defaults import DEFAULT_CONFIG
from lerobot.datasets.create_dataset.converter.convert_to_lerobot_dataset import DatasetConverter


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description="Convert custom datasets to LeRobotDataset format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert using config file
    python -m lerobot.datasets.create_dataset.cli --config my_config.yaml

    # Create sample config
    python -m lerobot.datasets.create_dataset.cli --create-sample-config sample_config.yaml

    # Quick conversion with minimal config
    python -m lerobot.datasets.create_dataset.cli --repo-id "user/dataset" --input-dir "/data" --fps 30
    
    # Convert with optimized settings for large datasets
    python -m lerobot.datasets.create_dataset.cli --config my_config.yaml --num-image-threads 16 --chunks-size 1000
    
    # Debug mode with detailed logging
    python -m lerobot.datasets.create_dataset.cli --config my_config.yaml --debug
        """,
    )

    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--create-sample-config", type=str, help="Create sample config file")

    # Quick setup arguments
    parser.add_argument("--repo-id", type=str, help="Repository ID for the dataset")
    parser.add_argument("--input-dir", type=str, help="Input directory containing raw data")
    parser.add_argument("--output-dir", type=str, help="Output directory for converted dataset")
    parser.add_argument("--fps", type=int, help="Frames per second")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--test-mode", action="store_true", help="Process only first few episodes")
    parser.add_argument("--parser", type=str, choices=["csv_image", "image_pair"], help="Select parser type")
    
    # Image processing arguments
    parser.add_argument("--num-image-threads", type=int, help="Number of threads for batch image loading (default: 8)")
    parser.add_argument("--image-batch-size", type=int, help="Number of images to load per batch (default: 32)")
    
    # Chunking and file size arguments (for large datasets)
    parser.add_argument("--chunks-size", type=int, help="Maximum number of files per chunk directory (default: 1000)")
    parser.add_argument("--data-files-size-mb", type=int, help="Maximum size for data parquet files in MB (default: 500)")
    parser.add_argument("--video-files-size-mb", type=int, help="Maximum size for video files in MB (default: 2000)")
    parser.add_argument("--batch-encoding-size", type=int, help="Number of episodes to batch before encoding videos (default: 4)")

    args = parser.parse_args()

    # Create sample config if requested
    if args.create_sample_config:
        create_sample_config(args.create_sample_config)
        print(f"Sample configuration created: {args.create_sample_config}")
        return 0

    # Load configuration
    try:
        if args.config:
            config = load_config(args.config)
        elif args.repo_id and args.input_dir and args.fps:
            # Quick setup
            config = DatasetConfig(
                repo_id=args.repo_id,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                fps=args.fps,
                debug=args.debug or False,
                test_mode=args.test_mode or False,
                parser_type=args.parser or DEFAULT_CONFIG.get("parser_type", "csv_image"),
            )
        else:
            # Use defaults from DEFAULT_CONFIG
            config = DatasetConfig(**DEFAULT_CONFIG)
            print("Using default configuration:")
            print(f"  Repo ID: {config.repo_id}")
            print(f"  Input dir: {config.input_dir}")
            print(f"  FPS: {config.fps}")
            print(f"  Use videos: {config.use_videos}")
            print(f"  Debug: {config.debug}")
            print(f"  Test mode: {config.test_mode}")
            print(f"  Push to hub: {config.push_to_hub}")

        # Override config with command line arguments
        if args.debug:
            config.debug = True
        if args.test_mode:
            config.test_mode = True
        if args.parser:
            config.parser_type = args.parser
        if args.num_image_threads:
            config.num_image_loading_threads = args.num_image_threads
        if args.image_batch_size:
            config.image_loading_batch_size = args.image_batch_size
        if args.chunks_size:
            config.chunks_size = args.chunks_size
        if args.data_files_size_mb:
            config.data_files_size_in_mb = args.data_files_size_mb
        if args.video_files_size_mb:
            config.video_files_size_in_mb = args.video_files_size_mb
        if args.batch_encoding_size:
            config.batch_encoding_size = args.batch_encoding_size

        # Run conversion
        converter = DatasetConverter(config)
        dataset = converter.convert()
        print("✅ Conversion completed successfully!")
        print(f"📁 Dataset saved to: {config.output_dir}")
        print(f"📊 Total episodes: {dataset.num_episodes}")
        print(f"🎞️  Total frames: {dataset.num_frames}")
        return 0

    except Exception as e:
        print(f"❌ Conversion failed: {e}", file=sys.stderr)
        if config.debug:
            raise
        return 1


if __name__ == "__main__":
    sys.exit(main())
