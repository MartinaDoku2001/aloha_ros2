import cv2
import numpy as np

# Test the cluster of indexes around your left wrist camera
test_indexes = [10, 11, 12, 13, 14]

for idx in test_indexes:
    print(f"\n--- Testing /dev/video{idx} ---")
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"Could not open /dev/video{idx}")
        continue
        
    # Try to grab 5 frames to clear the buffer
    for _ in range(5):
        ret, frame = cap.read()
        
    if not ret or frame is None:
        print(f"/dev/video{idx} failed to return a frame.")
    else:
        # Calculate average brightness to see if it is pure black
        avg_pixel_value = np.mean(frame)
        print(f"Success! Frame shape: {frame.shape}")
        print(f"Average frame brightness: {avg_pixel_value:.2f}")
        
        if avg_pixel_value < 1.0:
            print("⚠️ Warning: This stream is completely black (likely a metadata or depth channel).")
        else:
            print("🎉 Found it! This stream contains actual image data.")
            
    cap.release()