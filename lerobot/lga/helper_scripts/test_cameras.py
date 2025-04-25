#!/usr/bin/env python3
"""
Test script for camera connection with timeout.
"""

import time
import threading
from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera

def test_camera_with_timeout(camera_index, timeout=5):
    print(f"Testing camera at index {camera_index} with {timeout}s timeout")
    
    config = OpenCVCameraConfig(camera_index=camera_index)
    camera = OpenCVCamera(config)
    
    # Try to connect with timeout
    connection_success = [False]
    connection_error = [None]
    
    def connect_with_timeout():
        try:
            camera.connect()
            connection_success[0] = True
        except Exception as e:
            connection_error[0] = e
    
    # Start camera connection in a separate thread
    camera_thread = threading.Thread(target=connect_with_timeout)
    camera_thread.daemon = True
    camera_thread.start()
    
    # Wait for the thread to complete with a timeout
    start_time = time.time()
    while camera_thread.is_alive() and time.time() - start_time < timeout:
        time.sleep(0.1)
        print(".", end="", flush=True)
    
    print()
    
    if camera_thread.is_alive():
        print(f"Camera connection timed out after {timeout} seconds")
        return False
    elif not connection_success[0]:
        print(f"Camera connection failed: {connection_error[0]}")
        return False
    
    print("Camera connected successfully")
    
    # Try to read a frame
    try:
        print("Reading a frame...")
        frame = camera.read()
        print(f"Frame shape: {frame.shape}")
        
        # Disconnect
        print("Disconnecting camera...")
        camera.disconnect()
        print("Camera disconnected")
        return True
    except Exception as e:
        print(f"Error reading frame: {e}")
        return False

if __name__ == "__main__":
    # Test camera index 9
    test_camera_with_timeout(9)