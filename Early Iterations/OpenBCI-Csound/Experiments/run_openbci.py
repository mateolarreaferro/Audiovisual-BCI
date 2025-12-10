#!/usr/bin/env python3
"""
Robust OpenBCI Streamer Runner
Handles board reset and connection issues
"""

import time
import sys
import subprocess
import os

def reset_board():
    """Reset the OpenBCI board by power cycling the connection"""
    print("🔄 Resetting OpenBCI board...")
    print("   Please turn OFF the OpenBCI board, wait 3 seconds, then turn it back ON")
    print("   Press Enter when ready...")
    input()
    time.sleep(2)
    
def main():
    port = "/dev/cu.usbserial-DM0258DY"
    
    print("=" * 50)
    print("🧠 OpenBCI Direct Streamer")
    print("=" * 50)
    print(f"Port: {port}")
    print()
    
    # Check if board needs reset
    print("⚠️  Before starting:")
    print("1. Make sure the OpenBCI board is powered ON")
    print("2. If you see connection errors, power cycle the board")
    print()
    
    response = input("Do you need to reset the board? (y/n): ").lower()
    if response == 'y':
        reset_board()
    
    print("\n🚀 Starting OpenBCI streamer...")
    print("   Data will stream to:")
    print("   - Band Powers → Port 5003")
    print("   - Raw EEG → Port 5005")
    print("   - Accelerometer → Port 5004")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        # Run the streamer
        subprocess.run([
            sys.executable,
            "openbci_streamer.py",
            "--port", port,
            "--mode", "all"
        ])
    except KeyboardInterrupt:
        print("\n\n✅ Streamer stopped")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Power cycle the OpenBCI board")
        print("2. Check USB connection")
        print("3. Try running: python test_openbci_connection.py")

if __name__ == "__main__":
    main()