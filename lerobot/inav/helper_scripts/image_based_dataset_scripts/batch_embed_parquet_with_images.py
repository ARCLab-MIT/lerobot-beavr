# file: lerobot/inav/helper_scripts/embed_parquet_batch.py

import os
import glob
import logging
from lerobot.inav.helper_scripts.image_based_dataset_scripts.embed_parquet_with_images import add_images_to_parquet

def batch_embed_images(
    parquet_dir,
    images_root_dir,
    output_dir,
    test_mode=False,
    test_count=1,
    enable_logging=True
):
    """
    Batch process all Parquet files in a directory, embedding images.

    Args:
        parquet_dir: Directory containing input Parquet files.
        images_root_dir: Root directory containing image folders for episodes.
        output_dir: Directory to save output Parquet files.
        test_mode: If True, only process test_count files.
        test_count: Number of files to process in test mode.
        enable_logging: Enable logging output.
    """
    if enable_logging:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.ERROR)

    os.makedirs(output_dir, exist_ok=True)
    parquet_files = sorted(glob.glob(os.path.join(parquet_dir, "episode_*.parquet")))

    if test_mode:
        parquet_files = parquet_files[:test_count]
        logging.info(f"Test mode: Only processing {test_count} files.")

    for parquet_path in parquet_files:
        episode_name = os.path.splitext(os.path.basename(parquet_path))[0]
        images_dir = os.path.join(images_root_dir, episode_name)
        output_path = os.path.join(output_dir, f"{episode_name}.parquet")

        logging.info(f"Processing {parquet_path} with images from {images_dir}")
        add_images_to_parquet(
            parquet_path=parquet_path,
            images_dir=images_dir,
            output_path=output_path,
            enable_logging=enable_logging
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dir", required=True, help="Directory with input Parquet files")
    parser.add_argument("--images_root_dir", required=True, help="Root directory with image folders")
    parser.add_argument("--output_dir", required=True, help="Directory to save output Parquet files")
    parser.add_argument("--test_mode", action="store_true", help="Enable test mode (process only a few files)")
    parser.add_argument("--test_count", type=int, default=1, help="Number of files to process in test mode")
    parser.add_argument("--log", action="store_true", help="Enable logging")
    args = parser.parse_args()

    batch_embed_images(
        parquet_dir=args.parquet_dir,
        images_root_dir=args.images_root_dir,
        output_dir=args.output_dir,
        test_mode=args.test_mode,
        test_count=args.test_count,
        enable_logging=args.log
    )