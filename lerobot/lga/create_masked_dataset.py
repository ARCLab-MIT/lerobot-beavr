import os
import cv2
import json
import shutil
import argparse
from tqdm import tqdm
from huggingface_hub import HfApi
import time
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def create_dataset_from_processed_data(processed_data_dir, original_repo_id, output_dir, debug=True):
    """
    Create a dataset from processed data by replacing original frames with masked frames.
    
    Args:
        processed_data_dir (str): Directory containing processed data (with episode_X folders)
        original_repo_id (str): HuggingFace repository ID for the original dataset
        output_dir (str): Directory to save the new dataset
        debug (bool): Whether to enable debug mode
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(processed_data_dir):
        print(f"Error: Processed data directory {processed_data_dir} does not exist.")
        return False
    
    # Load the original dataset
    delta_timestamps = {
        "observation.images.nexigo_webcam": [0],
        "observation.images.realsense": [0],
    }
    try:
        dataset = LeRobotDataset(original_repo_id, delta_timestamps=delta_timestamps)
        if debug:
            print(f"Loaded original dataset with {dataset.num_episodes} episodes and {dataset.num_frames} frames")
    except Exception as e:
        print(f"Error loading original dataset: {e}")
        return False
    
    # Create the output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Create meta directory and copy metadata
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    
    # Copy metadata files
    for file in os.listdir(os.path.join(dataset.root, "meta")):
        src_path = os.path.join(dataset.root, "meta", file)
        dst_path = os.path.join(meta_dir, file)
        shutil.copy2(src_path, dst_path)
        if debug:
            print(f"Copied {src_path} to {dst_path}")
    
    # Update info.json to indicate this is a masked dataset
    info_path = os.path.join(meta_dir, "info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        # Update the info
        info["dataset_type"] = "masked"
        info["original_dataset"] = original_repo_id
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=4)
    
    # Create data directory and copy data files
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Copy data files (parquet files)
    for chunk_dir in os.listdir(os.path.join(dataset.root, "data")):
        src_chunk_dir = os.path.join(dataset.root, "data", chunk_dir)
        dst_chunk_dir = os.path.join(data_dir, chunk_dir)
        
        if os.path.isdir(src_chunk_dir):
            os.makedirs(dst_chunk_dir, exist_ok=True)
            
            for file in os.listdir(src_chunk_dir):
                src_path = os.path.join(src_chunk_dir, file)
                dst_path = os.path.join(dst_chunk_dir, file)
                shutil.copy2(src_path, dst_path)
                if debug:
                    print(f"Copied {src_path} to {dst_path}")
    
    # Create videos directory
    videos_dir = os.path.join(output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # Get camera keys
    camera_keys = dataset.meta.camera_keys
    if debug:
        print(f"Processing camera feeds: {camera_keys}")
    
    # Process each episode directory in the processed data
    for episode_dir in tqdm(os.listdir(processed_data_dir), desc="Processing episodes"):
        if not episode_dir.startswith("episode_"):
            continue
        
        try:
            episode_idx = int(episode_dir.split("_")[1])
        except:
            if debug:
                print(f"Skipping invalid episode directory: {episode_dir}")
            continue
        
        # Determine which chunk this episode belongs to
        episode_chunk = episode_idx // 1000  # Assuming chunks of 1000 episodes
        
        # Create the destination directory for this episode's video
        dst_chunk_dir = os.path.join(videos_dir, f"chunk-{episode_chunk:03d}")
        os.makedirs(dst_chunk_dir, exist_ok=True)
        
        # Process each camera
        for camera_key in camera_keys:
            dst_camera_dir = os.path.join(dst_chunk_dir, camera_key)
            os.makedirs(dst_camera_dir, exist_ok=True)
            
            # Get all the masked object images for this episode and camera
            episode_path = os.path.join(processed_data_dir, episode_dir)
            camera_path = os.path.join(episode_path, camera_key)
            
            # Skip if camera directory doesn't exist
            if not os.path.exists(camera_path):
                if debug:
                    print(f"No camera directory found for {camera_key} in episode {episode_idx}")
                continue
            
            masked_frames = []
            
            # Find all object images and sort them by frame number
            object_images = []
            for file in os.listdir(camera_path):
                if file.endswith("_objects.jpg"):
                    frame_idx = int(file.split("_")[1])
                    object_images.append((frame_idx, os.path.join(camera_path, file)))
            
            # Sort by frame index
            object_images.sort(key=lambda x: x[0])
            
            if not object_images:
                if debug:
                    print(f"No masked images found for episode {episode_idx}, camera {camera_key}")
                continue
            
            # Load all masked frames
            for frame_idx, img_path in object_images:
                try:
                    masked_frame = cv2.imread(img_path)
                    if masked_frame is not None:
                        masked_frames.append(masked_frame)
                    else:
                        if debug:
                            print(f"Failed to load image: {img_path}")
                except Exception as e:
                    if debug:
                        print(f"Error loading image {img_path}: {e}")
            
            if not masked_frames:
                if debug:
                    print(f"No valid masked frames for episode {episode_idx}, camera {camera_key}")
                continue
            
            # Create a video from the masked frames
            video_path = os.path.join(dst_camera_dir, f"episode_{episode_idx:06d}.mp4")
            
            # Get the original video's FPS from the dataset info
            fps = dataset.meta.info.get("fps", 30)
            
            # Create a VideoWriter object
            height, width = masked_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            # Write each frame to the video
            for frame in masked_frames:
                video_writer.write(frame)
            
            # Release the video writer
            video_writer.release()
            
            if debug:
                print(f"Created video for episode {episode_idx}, camera {camera_key} with {len(masked_frames)} frames")
    
    if debug:
        print(f"Dataset creation complete. Saved to {output_dir}")
    
    return True

def upload_dataset(dataset_dir, repo_id, debug=True, max_retries=3, retry_delay=5):
    """
    Upload a dataset to HuggingFace.
    
    Args:
        dataset_dir (str): Path to the dataset directory
        repo_id (str): HuggingFace repository ID for the dataset
        debug (bool): Whether to enable debug mode
        max_retries (int): Maximum number of retries for uploading
        retry_delay (int): Delay in seconds between retries
        
    Returns:
        bool: True if upload was successful, False otherwise
    """
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory {dataset_dir} does not exist.")
        return False
    
    # Create a new repository name by appending "_masked" to the original repo_id if not already done
    if not repo_id.endswith("_masked"):
        new_repo_id = f"{repo_id}_masked"
    else:
        new_repo_id = repo_id
    
    if debug:
        print(f"Uploading dataset from {dataset_dir} to HuggingFace as {new_repo_id}...")
    
    # Use the HuggingFace API to upload the dataset with retry logic
    api = HfApi()
    
    # Create the repository if it doesn't exist
    try:
        api.create_repo(repo_id=new_repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        if debug:
            print(f"Error creating repository: {e}")
            print("Continuing with upload...")
    
    # Upload with retries
    for attempt in range(max_retries):
        try:
            if debug:
                print(f"Upload attempt {attempt + 1}/{max_retries}...")
            
            api.upload_folder(
                folder_path=dataset_dir,
                repo_id=new_repo_id,
                repo_type="dataset"
            )
            
            if debug:
                print(f"Dataset uploaded successfully to {new_repo_id}")
            return True
        except Exception as e:
            if debug:
                print(f"Upload attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                if debug:
                    print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                if debug:
                    print("All upload attempts failed. Dataset not uploaded to HuggingFace.")
                return False

def main():
    parser = argparse.ArgumentParser(description="Create a masked dataset from processed data")
    parser.add_argument("--processed-data-dir", type=str, default="processed_data",
                        help="Directory containing processed data (with episode_X folders)")
    parser.add_argument("--original-repo-id", type=str, default="arclabmit/koch_gear_and_bin",
                        help="HuggingFace repository ID for the original dataset")
    parser.add_argument("--output-dir", type=str, default="masked_dataset",
                        help="Directory to save the new dataset")
    parser.add_argument("--debug", action="store_true", default=True,
                        help="Enable debug mode with additional logging")
    parser.add_argument("--no-upload", action="store_true",
                        help="Disable uploading to HuggingFace")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Maximum number of retries for uploading to HuggingFace")
    parser.add_argument("--retry-delay", type=int, default=5,
                        help="Delay in seconds between retries")
    parser.add_argument("--upload-only", action="store_true",
                        help="Only upload an existing dataset without creating it")
    
    args = parser.parse_args()
    
    if args.upload_only:
        if not os.path.exists(args.output_dir):
            print(f"Error: Dataset directory {args.output_dir} does not exist.")
            return
        
        success = upload_dataset(
            dataset_dir=args.output_dir,
            repo_id=args.original_repo_id,
            debug=args.debug,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay
        )
        
        if success:
            print(f"Dataset uploaded successfully to {args.original_repo_id}_masked")
        else:
            print("Failed to upload dataset to HuggingFace.")
    else:
        success = create_dataset_from_processed_data(
            processed_data_dir=args.processed_data_dir,
            original_repo_id=args.original_repo_id,
            output_dir=args.output_dir,
            debug=args.debug
        )
        
        if success and not args.no_upload:
            success = upload_dataset(
                dataset_dir=args.output_dir,
                repo_id=args.original_repo_id,
                debug=args.debug,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay
            )
            
            if success:
                print(f"Dataset uploaded successfully to {args.original_repo_id}_masked")
            else:
                print("Failed to upload dataset to HuggingFace.")

if __name__ == "__main__":
    main()
