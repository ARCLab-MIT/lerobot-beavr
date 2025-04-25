#!/usr/bin/env python3
"""
Camera diagnostic tool for LeRobot.
This script helps identify camera issues by testing each camera individually.
"""

import argparse
import cv2
import os
import time
from pathlib import Path

def test_camera(camera_index, width=640, height=480, fps=30, num_frames=10, save_dir=None):
    """Test a camera by capturing frames and optionally saving them."""
    print(f"Testing camera at index {camera_index}...")
    
    # Create capture object
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera at index {camera_index}")
        return False
    
    # Set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Get actual properties
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Requested: {width}x{height} at {fps} FPS")
    print(f"Actual: {actual_width}x{actual_height} at {actual_fps} FPS")
    
    # Create save directory if needed
    if save_dir:
        save_path = Path(save_dir) / f"camera_{camera_index}"
        os.makedirs(save_path, exist_ok=True)
    
    # Capture frames
    frames_captured = 0
    start_time = time.time()
    
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"ERROR: Failed to capture frame {i}")
            break
        
        frames_captured += 1
        
        if save_dir:
            cv2.imwrite(str(save_path / f"frame_{i:03d}.jpg"), frame)
            
        # Display frame
        cv2.imshow(f"Camera {camera_index}", frame)
        cv2.waitKey(1)  # Small delay
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    if frames_captured > 0:
        print(f"Successfully captured {frames_captured} frames in {elapsed:.2f} seconds")
        print(f"Effective FPS: {frames_captured / elapsed:.2f}")
        return True
    else:
        print("Failed to capture any frames")
        return False

def scan_available_cameras(max_index=10):
    """Scan for available cameras up to max_index."""
    print(f"Scanning for available cameras (0-{max_index})...")
    available = []
    
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    
    return available

def main():
    parser = argparse.ArgumentParser(description="Camera diagnostic tool for LeRobot")
    parser.add_argument("--scan", action="store_true", help="Scan for available cameras")
    parser.add_argument("--index", type=int, help="Test specific camera index")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")
    parser.add_argument("--frames", type=int, default=10, help="Number of frames to capture")
    parser.add_argument("--save-dir", type=str, help="Directory to save captured frames")
    
    args = parser.parse_args()
    
    if args.scan:
        available = scan_available_cameras()
        if available:
            print(f"Found {len(available)} available cameras at indices: {available}")
        else:
            print("No cameras found")
    
    if args.index is not None:
        test_camera(
            args.index, 
            width=args.width, 
            height=args.height, 
            fps=args.fps, 
            num_frames=args.frames,
            save_dir=args.save_dir
        )

if __name__ == "__main__":
    main()