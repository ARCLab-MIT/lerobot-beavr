# file: lerobot/inav/helper_scripts/inspect_embedded_image.py

import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt

def show_embedded_image(parquet_path, row_idx=0):
    """
    Loads a Parquet file, decodes the embedded image bytes from the specified row,
    and displays the image.

    Args:
        parquet_path (str): Path to the Parquet file with embedded images.
        row_idx (int): Row index to inspect (default: 0).
    """
    # Load the Parquet file
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from {parquet_path}")
    print("Columns:", df.columns)

    # Get the image bytes from the specified row
    img_bytes = df.iloc[row_idx]['observation.image.cam']
    if img_bytes is None:
        print(f"No image found in row {row_idx}.")
        return

    # Decode the image bytes to a PIL Image
    img = Image.open(io.BytesIO(img_bytes))

    # Display the image
    plt.imshow(img)
    plt.title(f"Row {row_idx} - Frame {df.iloc[row_idx]['frame_index']}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="Path to the Parquet file")
    parser.add_argument("--row", type=int, default=0, help="Row index to inspect")
    args = parser.parse_args()

    show_embedded_image(args.parquet, args.row)