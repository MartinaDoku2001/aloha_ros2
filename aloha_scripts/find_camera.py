import cv2
import numpy as np
import time
from pathlib import Path

# Index → role mapping, taken directly from the docker-compose device mounts.
CAMERAS = {
    6:  "cam_right_wrist",
    10: "cam_left_wrist",
    18: "cam_high",
    26: "cam_low",
}

OUT_DIR = Path("assets/camera_snapshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SETTLE_SECONDS = 2.0   # how long to let auto-exposure converge before saving

for idx, name in CAMERAS.items():
    print(f"\n--- Testing /dev/video{idx}  ({name}) ---")
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)

    if not cap.isOpened():
        print(f"Could not open /dev/video{idx}")
        continue

    # Ask for auto-exposure. V4L2 convention is backwards: 3 = auto, 1 = manual.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

    # Settle loop: read for a fixed wall-clock time so auto-exposure can ramp up.
    # Print brightness per frame so you can see whether it climbs or stays flat.
    ret, frame = False, None
    t_end = time.time() + SETTLE_SECONDS
    i = 0
    while time.time() < t_end:
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"  frame {i:3d}  brightness {np.mean(frame):6.2f}")
        i += 1

    if not ret or frame is None:
        print(f"/dev/video{idx} failed to return a frame.")
        cap.release()
        continue

    avg_pixel_value = np.mean(frame)
    print(f"Final frame shape: {frame.shape}")
    print(f"Final frame brightness: {avg_pixel_value:.2f}")

    if avg_pixel_value < 1.0:
        print("⚠️  Warning: stream is completely black (likely metadata/depth channel).")
    else:
        print("🎉 Stream contains actual image data.")

    out_path = OUT_DIR / f"video{idx}_{name}.png"
    cv2.imwrite(str(out_path), frame)
    print(f"Saved → {out_path}")

    cap.release()

print(f"\nDone. Snapshots in: {OUT_DIR.resolve()}")