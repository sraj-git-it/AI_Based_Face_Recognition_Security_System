# recognize_me.py
"""
Real-time Face Recognition System

This script performs real-time face recognition using a webcam. It compares detected faces
against a database of known faces and identifies strangers. The system can save snapshots
of unknown faces for security purposes.

Features:
- Real-time face detection and recognition
- Known face database management
- Stranger detection and snapshot capture
- Configurable tolerance levels for recognition accuracy
- Live video feed with bounding boxes and labels

Author: Face Recognition System
"""

import os
import sys
import time
import cv2
import numpy as np
import face_recognition

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Directory containing known face images (each image filename becomes the person's name)
KNOWN_DIR = "known_faces"

# Default name to use if no known faces are found
KNOWN_NAME_DEFAULT = "Srajit"

# Face recognition tolerance (lower = more strict, higher = more lenient)
# Range: 0.0 to 1.0, recommended: 0.4-0.6
TOLERANCE = 0.45

# Scale factor for frame resizing to improve performance
# Smaller values = faster processing but lower accuracy
FRAME_RESIZE_SCALE = 0.5

# Camera index (0 = default camera, 1 = second camera, etc.)
CAM_INDEX = 0

# =============================================================================
# STRANGER DETECTION SETTINGS
# =============================================================================

# Enable/disable saving snapshots of unknown faces
SAVE_STRANGER_SNAPSHOTS = True

# Directory to save stranger snapshots
SNAPSHOT_DIR = "stranger_snapshots"

# Create snapshot directory if it doesn't exist
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def load_known():
    """
    Load known face encodings from the known_faces directory.

    This function scans the KNOWN_DIR directory for image files and creates
    face encodings for each person. The filename (without extension) becomes
    the person's name.

    Returns:
        tuple: (encodings_list, names_list) - Lists of face encodings and corresponding names
    """
    encs = []  # List to store face encodings
    names = []  # List to store corresponding names

    # Check if the known faces directory exists
    if not os.path.isdir(KNOWN_DIR):
        print(f"❌ Missing folder: {KNOWN_DIR}")
        return encs, names

    # Iterate through all files in the known faces directory
    for fn in os.listdir(KNOWN_DIR):
        # Only process image files (jpg, jpeg, png)
        if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        # Construct full path to the image file
        path = os.path.join(KNOWN_DIR, fn)

        # Extract person's name from filename (remove extension)
        name = os.path.splitext(fn)[0]

        try:
            # Load the image using face_recognition library
            img = face_recognition.load_image_file(path)

            # Generate face encodings for the image
            e = face_recognition.face_encodings(img)

            # Check if any faces were detected in the image
            if len(e) == 0:
                print(f"⚠️ No face found in {fn}, skipping.")
                continue

            # Add the first (and usually only) face encoding to our list
            encs.append(e[0])
            names.append(name)
            print(f"Loaded known face: {name}")

        except Exception as ex:
            print(f"Error loading {fn}: {ex}")

    return encs, names


def save_snapshot(frame, label):
    """
    Save a snapshot of the current frame with a timestamp.

    This function saves an image file with a timestamp in the filename
    to help track when the snapshot was taken.

    Args:
        frame (numpy.ndarray): The video frame to save
        label (str): Label to include in the filename (e.g., "stranger")
    """
    # Generate timestamp in YYYYMMDD_HHMMSS format
    ts = time.strftime("%Y%m%d_%H%M%S")

    # Create filename with label and timestamp
    fname = f"{label}_{ts}.jpg"

    # Construct full path to save the image
    path = os.path.join(SNAPSHOT_DIR, fname)

    # Save the frame as a JPEG image
    cv2.imwrite(path, frame)
    print(f"Saved snapshot: {path}")


def main():
    """
    Main function that runs the face recognition system.

    This function:
    1. Loads known face encodings
    2. Initializes the camera
    3. Runs the main recognition loop
    4. Handles cleanup on exit
    """
    # Get camera index from command line argument or use default
    cam_index = int(sys.argv[1]) if len(sys.argv) > 1 else CAM_INDEX

    # Load known face encodings and names
    known_encodings, known_names = load_known()

    # Check if we have any known faces loaded
    if len(known_encodings) == 0:
        print("❌ No known faces loaded. Put images into 'known_faces/' and try again.")
        return

    # Initialize camera capture
    cap = cv2.VideoCapture(cam_index)

    # Check if camera opened successfully
    if not cap.isOpened():
        print(f"❌ Cannot open camera index {cam_index}. Try another index.")
        return

    print("Camera opened. Press 'q' to quit.")

    try:
        # Main recognition loop
        while True:
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Resize frame for faster processing
            # Smaller frames = faster face detection but lower accuracy
            small_frame = cv2.resize(
                frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)

            # Convert BGR (OpenCV) to RGB (face_recognition library expects RGB)
            # and ensure contiguous memory layout for better performance
            rgb_small = np.ascontiguousarray(small_frame[:, :, ::-1])

            # Detect faces and compute encodings
            # Wrap in try-catch to handle any face_recognition errors gracefully
            try:
                # Find face locations in the resized frame
                face_locations = face_recognition.face_locations(rgb_small)

                # Generate encodings for detected faces
                face_encodings = face_recognition.face_encodings(
                    rgb_small, face_locations)
            except Exception as e:
                print("Warning: face_recognition error:", e)
                face_locations = []
                face_encodings = []

            # Process each detected face
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Scale coordinates back to original frame size
                # (since we processed a smaller version)
                top = int(top / FRAME_RESIZE_SCALE)
                right = int(right / FRAME_RESIZE_SCALE)
                bottom = int(bottom / FRAME_RESIZE_SCALE)
                left = int(left / FRAME_RESIZE_SCALE)

                # Default to "Stranger" if no match is found
                name = "Stranger"

                # Compare current face with known faces
                matches = face_recognition.compare_faces(
                    known_encodings, face_encoding, tolerance=TOLERANCE)

                # If we found a match, get the name
                if True in matches:
                    name = known_names[matches.index(True)]

                # Choose color based on recognition result
                # Green for known faces, Red for strangers
                color = (0, 255, 0) if name != "Stranger" else (0, 0, 255)

                # Draw bounding box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Draw background rectangle for the name label
                cv2.rectangle(frame, (left, bottom - 25),
                              (right, bottom), color, cv2.FILLED)

                # Draw the name text
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Console feedback for recognition events
                if name == "Stranger":
                    print(
                        f"[{time.strftime('%H:%M:%S')}] Stranger detected at ({left},{top})")

                    # Save snapshot of stranger if enabled
                    if SAVE_STRANGER_SNAPSHOTS:
                        # Save a full-resolution snapshot (original frame)
                        save_snapshot(frame.copy(), "stranger")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] Recognized: {name}")

            # Display the frame with annotations
            cv2.imshow("Recognize Me - press q to quit", frame)

            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup: release camera and close windows
        cap.release()
        cv2.destroyAllWindows()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    main()
