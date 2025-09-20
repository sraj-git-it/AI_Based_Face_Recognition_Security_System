# test_camera.py
"""
Camera Testing Utility

This script helps test and identify available cameras on your system.
It can scan for working camera indices and optionally display live feeds
from specific cameras for testing purposes.

Features:
- Scan for available camera indices
- Test specific camera indices
- Display live camera feed for verification
- Helpful error messages and usage instructions

Usage:
    python3 test_camera.py                    # Scan cameras 0-6
    python3 test_camera.py 0                  # Test camera index 0
    python3 test_camera.py scan 10            # Scan cameras 0-10

Author: Face Recognition System
"""

import cv2
import sys
import time

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def try_open_index(idx, show_preview=False):
    """
    Test if a camera at the given index can be opened.

    This function attempts to open a camera at the specified index and
    optionally displays a live preview window to verify it's working.

    Args:
        idx (int): Camera index to test
        show_preview (bool): Whether to show live camera feed

    Returns:
        bool: True if camera opened successfully, False otherwise
    """
    # Attempt to open camera at the given index
    cap = cv2.VideoCapture(idx)
    opened = cap.isOpened()

    # If camera couldn't be opened, return False
    if not opened:
        return False

    # If preview is requested, show live feed
    if show_preview:
        ret, frame = cap.read()
        if ret:
            window_name = f"Camera {idx} - press q to quit"
            cv2.imshow(window_name, frame)

            # Show live feed until 'q' is pressed
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cv2.destroyWindow(window_name)

    # Release camera resources
    cap.release()
    return opened


def scan_indexes(max_idx=6):
    """
    Scan for working camera indices up to the specified maximum.

    This function tests camera indices from 0 to max_idx and returns
    a list of indices that successfully open.

    Args:
        max_idx (int): Maximum camera index to test (inclusive)

    Returns:
        list: List of working camera indices
    """
    working = []

    # Test each camera index from 0 to max_idx
    for i in range(max_idx+1):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            working.append(i)
        cap.release()

    return working


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Main execution block for camera testing utility.

    Handles command-line arguments and executes the appropriate camera test:
    - No arguments: Scan cameras 0-6 and display results
    - 'scan N': Scan cameras 0-N and display results  
    - Integer: Test specific camera index with live preview

    Usage examples:
        python3 test_camera.py                    # Scan cameras 0-6
        python3 test_camera.py 0                  # Test camera index 0
        python3 test_camera.py scan 10            # Scan cameras 0-10
    """
    # Get command-line arguments
    args = sys.argv[1:]

    # If no arguments provided, scan default range (0-6)
    if not args:
        print("Scanning camera indexes 0..6 ...")
        good = scan_indexes(6)

        if good:
            print("Working camera indexes:", good)
            print("To open one, run: python3 test_camera.py <index>")
        else:
            print("No cameras found in 0..6. Try plugging a camera or increase range: python3 test_camera.py scan 10")
        sys.exit(0)

    # Handle 'scan' command with optional max index
    if args[0].lower() == "scan":
        max_idx = int(args[1]) if len(args) > 1 else 10
        print(f"Scanning 0..{max_idx} ...")
        good = scan_indexes(max_idx)
        print("Working camera indexes:", good)
        sys.exit(0)

    # Handle specific camera index testing
    try:
        idx = int(args[0])
    except ValueError:
        print("Invalid index. Use an integer or 'scan'.")
        sys.exit(1)

    # Test the specified camera index with live preview
    print(f"Trying to open camera index {idx}. Press 'q' in window to quit.")
    opened = try_open_index(idx, show_preview=True)

    if not opened:
        print(f"Camera index {idx} did not open.")
    else:
        print(f"Closed camera index {idx}.")
