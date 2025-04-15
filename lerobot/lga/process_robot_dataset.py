import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
import shutil
import tempfile
import argparse
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.lga.frame_processor import FrameProcessor
from lerobot.lga.object_descriptor import ObjectDescriptor
from lerobot.lga.frame_masker import FrameMasker
from lerobot.lga.object_detector import ObjectDetector

def process_dataset(repo_id, output_dir="processed_data", debug=True, 
                   max_episodes=None, frame_step=1, process_all=False):
    """
    Process a robot dataset using the local inference server for detection and CLIP for object description.
    Creates a new dataset with masked images.
    
    Args:
        repo_id (str): HuggingFace repository ID for the dataset
        output_dir (str): Directory to save processed results
        debug (bool): Whether to enable debug mode
        max_episodes (int): Maximum number of episodes to process (None for all)
        frame_step (int): Process every Nth frame
        process_all (bool): If True, process all episodes and frames (overrides max_episodes and frame_step)
        
    Returns:
        dict: Processing statistics and results summary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the object detector to use the local inference server
    detector = ObjectDetector(
        debug=debug, 
        inference_server_url="http://localhost:9001",
        roboflow_api_key="gCqZi81mrMIb9yQDnfqK"
    )
    
    # Load dataset (using history for context)
    delta_timestamps = {
        "observation.images.nexigo_webcam": [0],
        "observation.images.realsense": [0],
    }
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
    if debug:
        print(f"Loaded dataset with {dataset.num_episodes} episodes and {dataset.num_frames} frames")
    
    # If process_all is True, override max_episodes and frame_step
    if process_all:
        max_episodes = None
        frame_step = 1
        if debug:
            print("Processing all episodes and frames")
    
    frame_processor = FrameProcessor(dataset, debug=debug)
    object_descriptor = ObjectDescriptor(method="clip", debug=debug)
    frame_masker = FrameMasker(debug=debug)
    
    # Get all camera keys from the dataset
    camera_keys = dataset.meta.camera_keys
    if debug:
        print(f"Processing camera feeds: {camera_keys}")
    
    num_episodes_to_process = min(max_episodes or dataset.num_episodes, dataset.num_episodes)
    
    results = {
        "processed_episodes": 0,
        "processed_frames": 0,
        "detected_objects": 0,
        "object_classes": {},
        "cameras_processed": {}
    }
    
    # Initialize camera-specific stats
    for camera_key in camera_keys:
        results["cameras_processed"][camera_key] = {
            "frames": 0,
            "objects": 0
        }
    
    # Create the dataset directory structure
    dataset_dir = os.path.join(output_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Copy the dataset metadata files
    for file in os.listdir(dataset.root):
        if file.endswith(".json") or file.endswith(".jsonl") or file.endswith(".parquet"):
            src_path = os.path.join(dataset.root, file)
            dst_path = os.path.join(dataset_dir, file)
            shutil.copy2(src_path, dst_path)
            if debug:
                print(f"Copied {src_path} to {dst_path}")
    
    # Create videos directory structure
    videos_dir = os.path.join(dataset_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)
    
    # Process each episode
    for episode_idx in tqdm(range(num_episodes_to_process), desc="Processing episodes"):
        if debug:
            print(f"Processing episode {episode_idx}")
        
        # Create episode directory in the output dir for debugging
        episode_dir = os.path.join(output_dir, f"episode_{episode_idx}")
        os.makedirs(episode_dir, exist_ok=True)
        
        # Create a directory for the episode's video frames in the new dataset
        episode_video_dir = os.path.join(videos_dir, f"episode_{episode_idx}")
        os.makedirs(episode_video_dir, exist_ok=True)
        
        # Process each camera feed
        for camera_key in camera_keys:
            if debug:
                print(f"Processing camera: {camera_key}")
            
            # Create camera-specific directory in the episode directory
            camera_dir = os.path.join(episode_video_dir, camera_key)
            os.makedirs(camera_dir, exist_ok=True)
            
            # Create camera-specific debug directory
            camera_debug_dir = os.path.join(episode_dir, camera_key)
            os.makedirs(camera_debug_dir, exist_ok=True)
            
            # Get the frames for this episode and camera
            frames = frame_processor.get_episode_frames(episode_idx, camera_key)
            
            # Process each frame
            for i, frame in enumerate(tqdm(frames[::frame_step], desc=f"Frames in episode {episode_idx}, camera {camera_key}", leave=False)):
                frame_idx = i * frame_step
                frame_np = frame_processor.tensor_to_numpy(frame)
                
                try:
                    # Use the object detector to get predictions
                    detection_result = detector.detect_and_segment(frame_np, box_threshold=0.25)
                    boxes = detection_result.get("boxes", [])
                    labels = detection_result.get("labels", [])
                    masks = detection_result.get("masks", [])
                    
                    # Create a single mask that combines all object masks
                    combined_mask = np.zeros_like(frame_np[:,:,0], dtype=bool)
                    object_descriptions = []
                    
                    # Process each detected object
                    for obj_idx, (mask, label) in enumerate(zip(masks, labels)):
                        # Add this object's mask to the combined mask
                        combined_mask = np.logical_or(combined_mask, mask)
                        
                        # Get object description
                        description = object_descriptor.describe(frame_np, mask=mask)
                        
                        # Store object information
                        object_info = {
                            "index": obj_idx,
                            "inference_label": label,
                            "clip_best_label": description.get('best_label', 'unknown'),
                            "clip_score": description.get('best_score', 0)
                        }
                        object_descriptions.append(object_info)
                        
                        results["detected_objects"] += 1
                        results["cameras_processed"][camera_key]["objects"] += 1
                        if "best_label" in description:
                            best_label = description["best_label"]
                            results["object_classes"][best_label] = results["object_classes"].get(best_label, 0) + 1
                    
                    # Create a masked image where only detected objects are visible
                    if len(masks) > 0:
                        masked_image = frame_masker.create_masked_image(frame_np, combined_mask) # Masked image is (480, 640, 3)
                        # if the masked image is (480, 640, 3), add a fourth dimension
                        if len(masked_image.shape) == 3:
                            masked_image = np.expand_dims(masked_image, axis=-1)
                        # Save the masked image to the output directory (for debugging)
                        cv2.imwrite(
                            os.path.join(camera_debug_dir, f"frame_{frame_idx}_objects.jpg"), 
                            masked_image
                        )
                        
                        # Save the masked image to the new dataset
                        cv2.imwrite(
                            os.path.join(camera_dir, f"frame_{frame_idx}.jpg"), 
                            masked_image
                        )
                        
                        # Save object information to a text file (for debugging)
                        with open(os.path.join(camera_debug_dir, f"frame_{frame_idx}_objects_info.txt"), "w") as f:
                            f.write(f"Frame {frame_idx} - {len(object_descriptions)} objects detected\n\n")
                            for obj in object_descriptions:
                                f.write(f"Object {obj['index']}:\n")
                                f.write(f"  Inference server label: {obj['inference_label']}\n")
                                f.write(f"  CLIP best label: {obj['clip_best_label']}\n")
                                f.write(f"  CLIP score: {obj['clip_score']:.4f}\n\n")
                    else:
                        # If no objects detected, save a black image
                        black_image = np.zeros_like(frame_np)
                        cv2.imwrite(
                            os.path.join(camera_dir, f"frame_{frame_idx}.jpg"), 
                            black_image
                        )
                        
                except Exception as e:
                    if debug:
                        print(f"Error processing frame {frame_idx} in episode {episode_idx}, camera {camera_key}: {e}")
                
                results["processed_frames"] += 1
                results["cameras_processed"][camera_key]["frames"] += 1
        results["processed_episodes"] += 1
    
    if debug:
        print(f"Processing complete. Results: {results}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Process a LeRobot dataset and create a masked version")
    parser.add_argument("--repo-id", type=str, default="arclabmit/koch_gear_and_bin",
                        help="HuggingFace repository ID for the dataset")
    parser.add_argument("--output-dir", type=str, default="processed_data",
                        help="Directory to save processed results")
    parser.add_argument("--debug", action="store_true", default=True,
                        help="Enable debug mode with additional logging")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Maximum number of episodes to process (None for all)")
    parser.add_argument("--frame-step", type=int, default=10,
                        help="Process every Nth frame")
    parser.add_argument("--process-all", action="store_true",
                        help="Process all episodes and frames (overrides max-episodes and frame-step)")
    
    args = parser.parse_args()
    
    results = process_dataset(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        debug=args.debug,
        max_episodes=args.max_episodes,
        frame_step=args.frame_step,
        process_all=args.process_all
    )
    
    print(f"Processed {results['processed_episodes']} episodes with {results['detected_objects']} objects")
    print(f"Object classes detected: {results['object_classes']}")
    
    # Print camera-specific stats
    for camera_key, stats in results["cameras_processed"].items():
        print(f"Camera {camera_key}: processed {stats['frames']} frames with {stats['objects']} objects")
    
    print(f"Processed dataset saved to {args.output_dir}")

if __name__ == "__main__":
    main()
