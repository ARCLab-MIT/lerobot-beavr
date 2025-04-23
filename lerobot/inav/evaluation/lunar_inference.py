from lerobot.common.policies.act.modeling_act import ACTPolicy
import torch
import time
import pandas as pd
import cv2
import numpy as np
import os

# Load the model from Hugging Face
model_id = "arclabmit/lunar_lander_act_model"
print(f"Loading model from Hugging Face: {model_id}")

try:
    # Load the model directly from Hugging Face
    policy = ACTPolicy.from_pretrained(model_id)
    policy.to("cuda")  # Move to appropriate device
    policy.eval()  # Set to evaluation mode
    print("Model loaded successfully from Hugging Face")
except Exception as e:
    print(f"Error loading model from Hugging Face: {e}")
    
    # Fallback to local checkpoint if available
    try:
        local_ckpt_path = "/home/demo/lerobot-beavr/outputs/train/lander/checkpoints/last/pretrained_model"
        print(f"Attempting to load from local checkpoint: {local_ckpt_path}")
        policy = ACTPolicy.from_pretrained(local_ckpt_path, local_files_only=True)
        policy.to("cuda")
        policy.eval()
        print("Model loaded successfully from local checkpoint")
    except Exception as local_e:
        print(f"Error loading from local checkpoint: {local_e}")
        raise ValueError("Failed to load model from both Hugging Face and local checkpoint")

# Function to load data from a local parquet file
def load_episode_data(episode_index, chunk=0):
    """
    Load episode data from local parquet files
    
    Args:
        episode_index (int): Index of the episode to load
        chunk (int): Chunk number (default: 0)
        
    Returns:
        pd.DataFrame: DataFrame containing the episode data
    """    
    # Format the path to the local parquet file
    data_path = f"/home/demo/lerobot-beavr/datasets/moon_lander_lerobot/data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
    
    try:
        # Load the parquet file using pandas
        df = pd.read_parquet(data_path)
        print(f"First column: {df.columns[0]}")
        return df
    except Exception as e:
        print(f"Error loading episode data from local file: {e}")
        raise ValueError(f"Failed to load episode data: {e}")

# Function to load image from local PNG files
def load_frame_image(episode_index, frame_index, chunk=0):
    """
    Load a specific frame from local PNG files
    
    Args:
        episode_index (int): Index of the episode
        frame_index (int): Index of the frame to load
        chunk (int): Chunk number (default: 0, not used for local files)
        
    Returns:
        np.ndarray: Image data in CHW format, normalized to [0, 1]
    """
    # Format the path to the local PNG file
    image_path = f"/home/demo/lerobot-beavr/datasets/moon_lander_lerobot/images/episode_{episode_index:06d}/frame_{frame_index:06d}.png"
    
    try:
        # Read the image using OpenCV
        frame = cv2.imread(image_path)
        
        if frame is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert BGR to RGB and normalize to [0, 1]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.transpose(2, 0, 1)  # HWC to CHW format
        frame = frame / 255.0  # Normalize to [0, 1]
        
        return frame
        
    except Exception as e:
        print(f"Error loading frame from local file: {e}")
        raise ValueError(f"Failed to load frame: {e}")

# For inference
def run_inference(episode_index, frame_index, chunk=0):
    """
    Run inference with the ACT policy model
    
    Args:
        episode_index (int): Index of the episode
        frame_index (int): Index of the frame
        chunk (int): Chunk number (default: 0)
        
    Returns:
        np.ndarray: Action predicted by the model
    """
    # Load episode data
    episode_data = load_episode_data(episode_index, chunk)
    print(f"Loaded episode data with {len(episode_data)} rows")
    
    # Get the row for the specific frame
    frame_row = episode_data[episode_data['frame_index'] == frame_index]
    if len(frame_row) == 0:
        raise ValueError(f"Frame index {frame_index} not found in episode {episode_index}")
    
    # Extract the state vector (which is a numpy array stored in the dataframe)
    state = frame_row['observation.state'].iloc[0]
    print(f"State shape: {state.shape}, State values: {state}")
    
    # Load the image for the specific frame
    image = load_frame_image(episode_index, frame_index, chunk)
    print(f"Image shape: {image.shape}")
    
    # Convert to tensors
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to("cuda")
    image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to("cuda")
    
    # Prepare observation dictionary with the correct key
    observation = {
        "observation.state": state_tensor,
        "observation.image.cam": image_tensor  # The model expects "observation.image.cam"
    }
    
    with torch.no_grad():
        # Get action from policy
        action = policy.select_action(observation)
        
        # Process the action as needed
        action = action.squeeze(0).cpu().numpy()
        print(f"Model output action: {action}")
        
        return action

# Function to run inference on an entire episode
def run_episode_inference(episode_index, chunk=0, output_dir="inference_results"):
    """
    Run inference on an entire episode and save results to a CSV file
    
    Args:
        episode_index (int): Index of the episode
        chunk (int): Chunk number (default: 0)
        output_dir (str): Directory to save results (default: "inference_results")
        
    Returns:
        str: Path to the saved CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load episode data
    episode_data = load_episode_data(episode_index, chunk)
    print(f"Loaded episode data with {len(episode_data)} rows")
    
    # Prepare results dataframe
    results = []
    
    # Process each frame in the episode
    total_frames = len(episode_data)
    start_time = time.time()
    
    for i, (_, row) in enumerate(episode_data.iterrows()):
        frame_index = row['frame_index']
        print(f"Processing frame {i+1}/{total_frames}: frame_index={frame_index}")
        
        try:
            # Extract the state vector
            state = row['observation.state']
            
            # Load the image for the specific frame
            image = load_frame_image(episode_index, frame_index, chunk)
            
            # Convert to tensors
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to("cuda")
            image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to("cuda")
            
            # Prepare observation dictionary
            observation = {
                "observation.state": state_tensor,
                "observation.image.cam": image_tensor
            }
            
            with torch.no_grad():
                # Get action from policy
                action = policy.select_action(observation)
                action = action.squeeze(0).cpu().numpy()
            
            # Store results
            results.append({
                'frame_index': frame_index,
                'action': action.tolist()
            })
            
        except Exception as e:
            print(f"Error processing frame {frame_index}: {e}")
    
    # Calculate processing statistics
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_frame = total_time / total_frames
    print(f"Processed {total_frames} frames in {total_time:.2f} seconds")
    print(f"Average time per frame: {avg_time_per_frame:.4f} seconds")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    output_path = os.path.join(output_dir, f"episode_{episode_index:06d}_inference.csv")
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    return output_path

# Example usage
if __name__ == "__main__":
    episode_index = 0  # First episode
    
    # Run inference on a single frame (original functionality)
    # frame_index = 1    # First frame
    # action = run_inference(episode_index, frame_index)
    
    # Run inference on the entire episode and save results
    output_path = run_episode_inference(episode_index)
    print(f"Inference results saved to: {output_path}")