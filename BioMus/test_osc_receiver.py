#!/usr/bin/env python3
"""
OSC Receiver Test Script for BioMus
Listens for OSC messages from the BioMus application and displays them.
"""

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import argparse
import sys


def print_header():
    """Print a nice header for the OSC receiver"""
    print("=" * 80)
    print("BioMus OSC Receiver Test")
    print("=" * 80)
    print()


def handle_raw_channel(address, *args):
    """Handle raw EEG timeseries data for a specific channel"""
    channel_name = address.split('/')[-1]  # Extract channel name from address
    num_samples = len(args)

    # Show first few and last few samples
    if num_samples > 10:
        preview = f"[{args[0]:.2f}, {args[1]:.2f}, {args[2]:.2f}, ..., {args[-3]:.2f}, {args[-2]:.2f}, {args[-1]:.2f}]"
    else:
        preview = f"[{', '.join([f'{x:.2f}' for x in args])}]"

    print(f"📊 RAW {channel_name}: {num_samples} samples")
    print(f"   Data: {preview}")
    print(f"   Range: [{min(args):.2f}, {max(args):.2f}]")
    print()


def handle_raw_chunked(address, *args):
    """Handle chunked raw EEG data (for large datasets)"""
    parts = address.split('/')
    channel_name = parts[-2]
    chunk_id = parts[-1]
    num_samples = len(args)

    print(f"📊 RAW {channel_name} ({chunk_id}): {num_samples} samples")
    print(f"   Range: [{min(args):.2f}, {max(args):.2f}]")
    print()


def handle_bands(address, *args):
    """Handle band power data for a specific channel"""
    channel_name = address.split('/')[-1]

    # Standard EEG bands: delta, theta, alpha, beta, gamma
    band_names = ["delta", "theta", "alpha", "beta", "gamma"]

    print(f"🧠 BANDS {channel_name}:")
    for i, (band_name, value) in enumerate(zip(band_names, args)):
        # Create a simple bar visualization
        bar_length = int(value / 2)  # Scale for display (assuming percentage 0-100)
        bar = "█" * bar_length
        print(f"   {band_name:6s}: {value:6.2f}% {bar}")
    print()


def handle_cv_features(address, *args):
    """Handle computer vision (facial) features"""
    feature_name = address.split('/')[-1]
    value = args[0] if args else 0.0

    # Create a simple bar visualization (0.0 to 1.0 scale)
    bar_length = int(value * 40)
    bar = "█" * bar_length

    print(f"😊 CV {feature_name:20s}: {value:6.3f} {bar}")


def handle_facesynth(address, *args):
    """Handle FaceSynth-format messages"""
    feature_name = address.split('/')[-1]
    value = args[0] if args else 0.0

    # Create a simple bar visualization (0.0 to 1.0 scale)
    bar_length = int(value * 40)
    bar = "█" * bar_length

    print(f"🎭 FaceSynth {feature_name:15s}: {value:6.3f} {bar}")


def handle_default(address, *args):
    """Default handler for any unmatched OSC messages"""
    print(f"❓ Unknown OSC message:")
    print(f"   Address: {address}")
    print(f"   Args: {args}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test OSC receiver for BioMus EEG data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python test_osc_receiver.py                    # Listen on default 127.0.0.1:9000
  python test_osc_receiver.py --port 9001        # Listen on custom port
  python test_osc_receiver.py --ip 0.0.0.0       # Listen on all interfaces
        """
    )
    parser.add_argument(
        "--ip",
        default="127.0.0.1",
        help="IP address to listen on (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="UDP port to listen on (default: 9000)"
    )

    args = parser.parse_args()

    # Create dispatcher and map handlers
    dispatcher = Dispatcher()

    # Raw timeseries data handlers
    dispatcher.map("/eeg/raw/CH*", handle_raw_channel)
    dispatcher.map("/eeg/raw/*/chunk*", handle_raw_chunked)

    # Band power handlers
    dispatcher.map("/eeg/bands/CH*", handle_bands)

    # Computer vision / facial feature handlers
    dispatcher.map("/cv/face/*", handle_cv_features)
    dispatcher.map("/faceSynth/*", handle_facesynth)

    # Fallback for any other messages
    dispatcher.set_default_handler(handle_default)

    # Create and start server
    print_header()
    print(f"🎧 Listening for OSC messages on {args.ip}:{args.port}")
    print(f"📡 Waiting for data from BioMus...")
    print(f"   - Raw EEG: /eeg/raw/CH*")
    print(f"   - Band Powers: /eeg/bands/CH*")
    print(f"   - CV Features: /cv/face/*")
    print(f"   - FaceSynth: /faceSynth/*")
    print()
    print("Press Ctrl+C to stop")
    print("-" * 80)
    print()

    try:
        server = BlockingOSCUDPServer((args.ip, args.port), dispatcher)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n")
        print("=" * 80)
        print("OSC Receiver stopped")
        print("=" * 80)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting OSC server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
