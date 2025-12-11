#!/usr/bin/env python3
"""
Camera Permission Setup Helper for macOS
=========================================

This script helps set up camera permissions for the FaceMesh Synth Controller.
Run this first if you're having camera access issues.
"""

import sys
import subprocess
import os

def main():
    print("=== FaceMesh Synth Camera Setup ===\n")

    if sys.platform != "darwin":
        print("This setup script is for macOS only.")
        print("On other platforms, camera permissions are usually handled automatically.")
        return

    print("To use the webcam with Python on macOS, you need to:")
    print("\n1. Grant camera permission to your terminal or IDE:")
    print("   - Open System Settings (or System Preferences)")
    print("   - Go to Privacy & Security > Camera")
    print("   - Enable the checkbox for:")
    print("     • Terminal (if running from command line)")
    print("     • Your IDE (VSCode, PyCharm, etc.)")
    print("     • Python (if it appears)")

    print("\n2. If you just granted permission, you may need to:")
    print("   - Close and reopen your terminal/IDE")
    print("   - Or restart the Python process")

    print("\n3. Test camera access:")
    response = input("\nWould you like to test camera access now? (y/n): ")

    if response.lower() == 'y':
        print("\nTesting camera access...")
        try:
            import cv2
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                print("✅ Camera access successful!")
                ret, frame = cap.read()
                if ret:
                    print("✅ Can read frames from camera")
                cap.release()
            else:
                print("❌ Cannot open camera")
                print("Please check permissions in System Settings")
        except ImportError:
            print("❌ OpenCV not installed. Run: pip install opencv-python")
        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n4. Alternative: Use demo mode without camera:")
    print("   python face_synth.py --demo")

    print("\nFor more help, see: https://support.apple.com/guide/mac-help/")

if __name__ == "__main__":
    main()