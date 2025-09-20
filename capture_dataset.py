#!/usr/bin/env python3
"""
Face Dataset Capture Tool

This script captures face images from a webcam to build a training dataset.
It uses OpenCV's DNN face detector to automatically detect faces and saves
cropped face images to a specified directory.

Features:
- Automatic face detection using DNN models
- macOS-optimized camera selection (skips iPhone camera)
- Configurable capture count and confidence threshold
- Real-time preview with face bounding boxes
- Automatic face cropping and resizing

Usage:
    python3 capture_dataset.py --name "PersonName" --count 50
    python3 capture_dataset.py --name "John" --count 100 --show --camera-index 1

Author: Face Recognition System
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def find_model_files():
    """
    Find the required DNN model files for face detection.

    This function searches for the necessary model files (prototxt and caffemodel)
    in the current working directory. These files are required for OpenCV's DNN
    face detection to work.

    Returns:
        tuple: (prototxt_path, caffemodel_path) - Paths to the model files

    Raises:
        FileNotFoundError: If the required model files are not found
    """
    # List of possible prototxt filenames
    prototxt_candidates = [
        "deploy.prototxt",
        "deploy.prototxt.txt",
    ]

    # List of possible caffemodel filenames
    caffemodel_candidates = [
        "res10_300x300_ssd_iter_140000.caffemodel",
        "res10_300x300_ssd_iter_140000_fp16.caffemodel",
    ]

    # Find the first existing prototxt file
    proto = next((p for p in prototxt_candidates if Path(p).is_file()), None)

    # Find the first existing caffemodel file
    model = next((m for m in caffemodel_candidates if Path(m).is_file()), None)

    # Check if both files were found
    if not proto or not model:
        msg = [
            "Face detector model files not found in the current folder.",
            "Expected files:",
            "  - deploy.prototxt",
            "  - res10_300x300_ssd_iter_140000.caffemodel  (or the *_fp16 variant)",
            "",
            "If needed, re-download:",
            "  curl -L -o deploy.prototxt https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "  curl -L -o res10_300x300_ssd_iter_140000_fp16.caffemodel https://raw.githubusercontent.com/opencv/opencv_3rdparty/master/dnn_samples/face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel",
        ]
        raise FileNotFoundError("\n".join(msg))
    return proto, model


def pick_camera(preferred_index=None):
    """
    Select and open the best available camera for macOS.

    This function intelligently selects a camera, preferring built-in cameras
    over iPhone cameras (which are commonly available via Continuity Camera).
    It uses the AVFoundation backend which is optimized for macOS.

    Args:
        preferred_index (int, optional): Specific camera index to use

    Returns:
        cv2.VideoCapture: Opened camera object

    Raises:
        RuntimeError: If no working camera is found
    """
    # Use AVFoundation backend for better macOS compatibility
    backend = cv2.CAP_AVFOUNDATION

    # If a specific camera index is requested, try to use it
    if preferred_index is not None:
        cap = cv2.VideoCapture(preferred_index, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Using camera index {preferred_index}")
                return cap
        cap.release()
        raise RuntimeError(
            f"Could not open camera index {preferred_index}. Try a different index (e.g., 1 or 2).")

    # Try cameras in order of preference (skip iPhone camera at index 0)
    # Built-in cameras are usually at indices 1 and 2
    trial_order = [1, 2, 0, 3, 4]

    for idx in trial_order:
        test = cv2.VideoCapture(idx, backend)
        if test.isOpened():
            ok, _ = test.read()
            if ok:
                print(f"✅ Using camera index {idx}")
                return test
        test.release()

    # If no camera worked, provide helpful error message
    raise RuntimeError(
        "❌ No working camera found. On macOS, make sure Terminal/VS Code has camera access:\n"
        "System Settings → Privacy & Security → Camera → enable for your app.\n"
        "Also try passing an index explicitly, e.g., --camera-index 1"
    )


def detect_faces_dnn(net, frame, conf_thresh=0.6):
    """
    Detect faces in a frame using OpenCV's DNN face detector.

    This function uses a pre-trained deep neural network to detect faces
    in the input frame. It returns bounding boxes for all detected faces
    that meet the confidence threshold.

    Args:
        net: Pre-loaded DNN network for face detection
        frame (numpy.ndarray): Input image frame
        conf_thresh (float): Confidence threshold for face detection (0.0-1.0)

    Returns:
        list: List of bounding boxes as [x1, y1, x2, y2] coordinates
    """
    # Get frame dimensions
    (h, w) = frame.shape[:2]

    # Create a blob from the frame for DNN processing
    # Resize to 300x300 as required by the model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 1.0, (300, 300),
                                 # Mean subtraction values
                                 (104.0, 177.0, 123.0),
                                 swapRB=False, crop=False)

    # Set the blob as input to the network
    net.setInput(blob)

    # Run forward pass to get detections
    detections = net.forward()

    boxes = []

    # Process each detection
    for i in range(detections.shape[2]):
        # Extract confidence score
        conf = float(detections[0, 0, i, 2])

        # Only process detections above confidence threshold
        if conf >= conf_thresh:
            # Extract bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Clamp coordinates to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            # Only add valid boxes (positive width and height)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])

    return boxes


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main function for face dataset capture.

    This function handles command-line arguments, initializes the camera and
    DNN model, then runs the capture loop to collect face images.
    """
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Capture a face dataset using MacBook camera (skip iPhone camera).")

    # Required argument: person's name
    parser.add_argument("--name", required=True,
                        help="Name of the person (folder under dataset/)")

    # Optional arguments with defaults
    parser.add_argument("--count", type=int, default=50,
                        help="Number of face images to capture (default: 50)")
    parser.add_argument("--camera-index", type=int, default=None,
                        help="Force a specific camera index (e.g., 1 for built-in).")
    parser.add_argument("--conf", type=float, default=0.6,
                        help="Face detection confidence threshold (default: 0.6)")
    parser.add_argument("--show", action="store_true",
                        help="Show the capture window with boxes (ESC to quit)")

    # Parse command-line arguments
    args = parser.parse_args()

    # Create output directory for this person's dataset
    out_dir = Path("dataset") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the DNN face detection model
    proto, model = find_model_files()
    print("[INFO] loading face detector...")
    net = cv2.dnn.readNetFromCaffe(proto, model)

    # Open camera
    cap = pick_camera(args.camera_index)

    # Initialize capture variables
    saved = 0  # Counter for saved images
    cooldown = 0  # Delay between saves to avoid near-duplicate images
    print("[INFO] Press ESC/Q to stop early.")

    try:
        # Main capture loop
        while saved < args.count:
            # Read frame from camera
            ok, frame = cap.read()
            if not ok:
                print("⚠️ Failed to read from camera. Retrying...")
                time.sleep(0.1)
                continue

            # Detect faces in the current frame
            boxes = detect_faces_dnn(net, frame, conf_thresh=args.conf)

            # Process detected faces
            if boxes:
                # Calculate area of each detected face
                areas = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes]

                # Select the largest face (most likely the main subject)
                idx = int(np.argmax(areas))
                (x1, y1, x2, y2) = boxes[idx]

                # Extract face region from frame
                face = frame[y1:y2, x1:x2]

                # Save face if cooldown period has passed
                if face.size > 0 and cooldown <= 0:
                    # Resize face to standard 200x200 pixels
                    face_resized = cv2.resize(face, (200, 200))

                    # Generate filename with person name and sequence number
                    filename = out_dir / f"{args.name}_{saved:04d}.jpg"

                    # Save the face image
                    cv2.imwrite(str(filename), face_resized)
                    saved += 1
                    cooldown = 4  # Skip a few frames before next save
                    print(f"[INFO] saved {filename}  ({saved}/{args.count})")

            # Decrease cooldown counter
            if cooldown > 0:
                cooldown -= 1

            # Optional preview window
            if args.show:
                # Create a copy of the frame for preview
                preview = frame.copy()

                # Draw bounding boxes around detected faces
                for (x1, y1, x2, y2) in boxes:
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add progress text to preview
                cv2.putText(preview, f"{args.name}: {saved}/{args.count}",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Display preview window
                cv2.imshow("Capture (ESC/Q to quit)", preview)

                # Check for exit key press
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):  # ESC or Q
                    print("[INFO] Stopping capture...")
                    break

    finally:
        # Cleanup: release camera and close windows
        cap.release()
        cv2.destroyAllWindows()

    # Print completion message
    print(
        f"[INFO] dataset collection complete for {args.name}: {saved} images saved to {out_dir}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {e}\n", file=sys.stderr)
        sys.exit(1)
