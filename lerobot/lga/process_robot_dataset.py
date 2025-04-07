import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.lga.frame_processor import FrameProcessor
from lerobot.lga.object_descriptor import ObjectDescriptor
from lerobot.lga.frame_masker import FrameMasker
from lerobot.lga.object_detector import ObjectDetector

def process_dataset(repo_id, output_dir="processed_data", debug=True, 
                   max_episodes=None, frame_step=10):
    """
    Process a robot dataset using the local inference server for detection and CLIP for object description.
    
    Args:
        repo_id (str): HuggingFace repository ID for the dataset
        output_dir (str): Directory to save processed results
        debug (bool): Whether to enable debug mode
        max_episodes (int): Maximum number of episodes to process (None for all)
        frame_step (int): Process every Nth frame
        
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
    }
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)
    if debug:
        print(f"Loaded dataset with {dataset.num_episodes} episodes and {dataset.num_frames} frames")
    
    frame_processor = FrameProcessor(dataset, debug=debug)
    object_descriptor = ObjectDescriptor(method="clip", debug=debug)
    frame_masker = FrameMasker(debug=debug)
    
    camera_key = dataset.meta.camera_keys[0]
    num_episodes_to_process = min(max_episodes or dataset.num_episodes, dataset.num_episodes)
    
    results = {
        "processed_episodes": 0,
        "processed_frames": 0,
        "detected_objects": 0,
        "object_classes": {}
    }
    
    for episode_idx in tqdm(range(num_episodes_to_process), desc="Processing episodes"):
        if debug:
            print(f"Processing episode {episode_idx}")
        episode_dir = os.path.join(output_dir, f"episode_{episode_idx}")
        os.makedirs(episode_dir, exist_ok=True)
        
        frames = frame_processor.get_episode_frames(episode_idx, camera_key)
        for i, frame in enumerate(tqdm(frames[::frame_step], desc=f"Frames in episode {episode_idx}", leave=False)):
            frame_idx = i * frame_step
            frame_np = frame_processor.tensor_to_numpy(frame)
            
            try:
                # Use the object detector to get predictions. Note that box_threshold is maintained for interface consistency.
                detection_result = detector.detect_and_segment(frame_np, box_threshold=0.25)
                boxes = detection_result.get("boxes", [])
                labels = detection_result.get("labels", [])
                masks = detection_result.get("masks", [])
                
                # Optionally, save a visualization image with detection boxes
                vis_image = frame_np.copy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                vis_path = os.path.join(episode_dir, f"frame_{frame_idx}_detections.jpg")
                cv2.imwrite(vis_path, vis_image)
                
                # Process each detected object
                for obj_idx, (mask, label) in enumerate(zip(masks, labels)):
                    masked_image = frame_masker.create_masked_image(frame_np, mask)
                    description = object_descriptor.describe(masked_image, mask=mask)
                    
                    cv2.imwrite(
                        os.path.join(episode_dir, f"frame_{frame_idx}_object_{obj_idx}.jpg"), 
                        masked_image
                    )
                    
                    with open(os.path.join(episode_dir, f"frame_{frame_idx}_object_{obj_idx}_info.txt"), "w") as f:
                        f.write(f"Inference server label: {label}\n")
                        f.write(f"CLIP best label: {description.get('best_label', 'unknown')}\n")
                        f.write(f"CLIP score: {description.get('best_score', 0):.4f}\n")
                        f.write(f"CLIP top labels: {description.get('top_labels', [])}\n")
                        f.write(f"CLIP top scores: {[f'{s:.4f}' for s in description.get('top_scores', [])]}\n")
                    
                    results["detected_objects"] += 1
                    if "best_label" in description:
                        best_label = description["best_label"]
                        results["object_classes"][best_label] = results["object_classes"].get(best_label, 0) + 1
                    
            except Exception as e:
                if debug:
                    print(f"Error processing frame {frame_idx} in episode {episode_idx}: {e}")
            
            results["processed_frames"] += 1
        results["processed_episodes"] += 1
    
    if debug:
        print(f"Processing complete. Results: {results}")
    return results

if __name__ == "__main__":
    repo_id = "arclabmit/koch_gear_and_bin"
    results = process_dataset(
        repo_id, 
        debug=True,
        max_episodes=5,
        frame_step=10
    )
    print(f"Processed {results['processed_episodes']} episodes with {results['detected_objects']} objects")
    print(f"Object classes detected: {results['object_classes']}")
