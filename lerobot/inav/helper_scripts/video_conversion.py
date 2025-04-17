#!/usr/bin/env python3

import os
import subprocess
import json
import re
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths
SOURCE_DIR = Path("/home/demo/lerobot-beavr/lerobot/inav/datasets/moon_lander_lerobot")
IMAGES_DIR = SOURCE_DIR / "images"
OUTPUT_DIR = SOURCE_DIR / "videos/chunk-000/observation.images"
META_FILE = SOURCE_DIR / "meta/info.json"

def load_metadata():
    """Load the dataset metadata from info.json"""
    try:
        with open(META_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Metadata file not found at {META_FILE}")
        raise

def get_episode_dirs():
    """Find all episode directories in the images folder"""
    if not IMAGES_DIR.exists():
        logging.error(f"Images directory not found at {IMAGES_DIR}")
        raise FileNotFoundError(f"Images directory not found at {IMAGES_DIR}")
    
    episode_dirs = []
    for item in IMAGES_DIR.iterdir():
        if item.is_dir() and item.name.startswith("episode_"):
            episode_dirs.append(item)
    
    return sorted(episode_dirs, key=lambda x: int(x.name.split('_')[1]))

def extract_frame_number(filename):
    """Extract frame number from filename like 'frame_000241.png'"""
    match = re.search(r'frame_(\d+)', filename.stem)
    if match:
        return int(match.group(1))
    return 0  # Fallback

def encode_video(episode_dir, output_path, fps):
    """Encode a video from images in the episode directory"""
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find all image files in the directory and sort them by frame number
    image_files = sorted(
        list(episode_dir.glob("*.png")), 
        key=extract_frame_number
    )
    
    if not image_files:
        logging.warning(f"No images found in {episode_dir}")
        return False
    
    logging.info(f"Found {len(image_files)} frames in {episode_dir}")
    
    # Create a temporary file with the list of images and their durations
    temp_list_file = episode_dir / "images_list.txt"
    with open(temp_list_file, 'w') as f:
        for img in image_files:
            # Each frame should last 1/fps seconds
            f.write(f"file '{img.absolute()}'\n")
            f.write(f"duration {1.0/fps}\n")
        # Add the last frame again to ensure it's displayed
        if image_files:
            f.write(f"file '{image_files[-1].absolute()}'\n")
    
    # Build the ffmpeg command - using the slideshow method
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file if it exists
        "-f", "concat",
        "-safe", "0",
        "-i", str(temp_list_file),
        "-vsync", "vfr",  # Variable frame rate
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-preset", "medium",
        str(output_path)
    ]
    
    try:
        # Run the ffmpeg command
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.debug(f"FFMPEG output: {result.stderr.decode()}")
        
        # Verify the video was created and has content
        if not output_path.exists() or output_path.stat().st_size < 1000:
            logging.error(f"Video file is too small or not created: {output_path}")
            return False
            
        # Clean up the temporary file
        temp_list_file.unlink()
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error encoding video for {episode_dir.name}: {e}")
        logging.error(f"FFMPEG stderr: {e.stderr.decode() if e.stderr else 'No error output'}")
        if temp_list_file.exists():
            temp_list_file.unlink()
        return False

def process_episode(episode_dir, fps=5):
    """Process a single episode directory"""
    episode_num = int(episode_dir.name.split('_')[1])
    output_path = OUTPUT_DIR / f"episode_{episode_num:06d}.mp4"
    
    logging.info(f"Processing {episode_dir.name} -> {output_path}")
    success = encode_video(episode_dir, output_path, fps)
    
    if success:
        logging.info(f"Successfully encoded {output_path}")
    else:
        logging.error(f"Failed to encode {output_path}")
    
    return success

def main():
    # Load metadata
    try:
        metadata = load_metadata()
        fps = metadata.get("fps", 5)
        logging.info(f"Loaded metadata: FPS={fps}")
    except Exception as e:
        logging.warning(f"Could not load metadata: {e}. Using default FPS=5")
        fps = 5
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get episode directories
    episode_dirs = get_episode_dirs()
    logging.info(f"Found {len(episode_dirs)} episode directories")
    
    # Process episodes in parallel
    successful = 0
    failed = 0
    
    # Use fewer workers to avoid overwhelming the system
    max_workers = min(os.cpu_count(), 4)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_episode, ep_dir, fps) for ep_dir in episode_dirs]
        
        for future in tqdm(futures, total=len(futures), desc="Converting episodes"):
            if future.result():
                successful += 1
            else:
                failed += 1
    
    logging.info(f"Conversion complete: {successful} successful, {failed} failed")

if __name__ == "__main__":
    main()