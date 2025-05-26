# file: tools/add_images_to_parquet.py

import os
import logging
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO

def setup_logger(enable_logging: bool):
    """Set up logging based on config."""
    logging.basicConfig(
        level=logging.INFO if enable_logging else logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def load_image(image_path):
    """Load an image as a numpy RGB array."""
    img = Image.open(image_path).convert('RGB')
    return np.array(img)

def image_to_bytes(img_array):
    """Convert a numpy RGB array to PNG bytes."""
    buf = BytesIO()
    Image.fromarray(img_array).save(buf, format='PNG')
    return buf.getvalue()

def add_images_to_parquet(
    parquet_path: str,
    images_dir: str,
    output_path: str,
    enable_logging: bool = True
):
    """
    Adds images to each frame in the parquet file.

    Args:
        parquet_path: Path to the input parquet file.
        images_dir: Directory containing frame images for the episode.
        output_path: Path to save the new parquet file.
        enable_logging: Enable logging output.
    """
    setup_logger(enable_logging)
    df = pd.read_parquet(parquet_path)
    logging.info(f"Loaded parquet: {parquet_path} with {len(df)} rows.")

    images = []
    for idx, row in df.iterrows():
        frame_idx = row['frame_index']
        image_filename = f"frame_{frame_idx:06d}.png"
        image_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(image_path):
            logging.error(f"Image not found: {image_path}")
            images.append(None)
            continue
        img_array = load_image(image_path)
        img_bytes = image_to_bytes(img_array)
        images.append(img_bytes)
        logging.info(f"Loaded image for frame {frame_idx}: {image_path}")

    df['observation.image.cam'] = images
    df.to_parquet(output_path, index=False)
    logging.info(f"Saved updated parquet to: {output_path}")

# Example usage (for test mode):
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="Input parquet file")
    parser.add_argument("--images", required=True, help="Images directory for episode")
    parser.add_argument("--output", required=True, help="Output parquet file")
    parser.add_argument("--log", action="store_true", help="Enable logging")
    args = parser.parse_args()

    add_images_to_parquet(
        parquet_path=args.parquet,
        images_dir=args.images,
        output_path=args.output,
        enable_logging=args.log
    )