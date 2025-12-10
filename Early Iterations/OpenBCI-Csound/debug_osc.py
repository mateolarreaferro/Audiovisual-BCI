#!/usr/bin/env python3
"""
OpenBCI OSC Debug Script
Listens on port 5003 and shows exactly what OSC messages are received
"""

from pythonosc import dispatcher
from pythonosc import osc_server
import threading
import time

def print_handler(address, *args):
    """Print any OSC message received"""
    print(f"OSC: {address} -> {len(args)} values: {args[:8]}...")  # Show first 8 values
    
    # Show full message for smaller ones
    if len(args) <= 4:
        print(f"     Full values: {args}")

def main():
    print("OpenBCI OSC Debug Listener")
    print("Listening on port 5003...")
    print("Start streaming from OpenBCI GUI now!")
    print("Press Ctrl+C to stop")
    print("-" * 50)
    
    # Create dispatcher to handle all OSC messages
    disp = dispatcher.Dispatcher()
    disp.set_default_handler(print_handler)
    
    # Create and start server
    server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 5003), disp)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping OSC listener...")
        server.shutdown()

if __name__ == "__main__":
    try:
        from pythonosc import dispatcher, osc_server
    except ImportError:
        print("Please install python-osc: pip install python-osc")
        exit(1)
    
    main()