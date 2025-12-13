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


def handle_raw_combined(address, *args):
    """Handle combined raw EEG data (all channels)"""
    num_samples = len(args)

    print(f"📊 RAW (all channels): {num_samples} samples")
    print(f"   Range: [{min(args):.2f}, {max(args):.2f}]")
    print()


def handle_raw_chunked(address, *args):
    """Handle chunked raw EEG data (for large datasets)"""
    parts = address.split('/')
    if len(parts) >= 4:
        channel_name = parts[-2]
        chunk_id = parts[-1]
    else:
        channel_name = "combined"
        chunk_id = parts[-1]
    num_samples = len(args)

    print(f"📊 RAW {channel_name} ({chunk_id}): {num_samples} samples")
    print(f"   Range: [{min(args):.2f}, {max(args):.2f}]")
    print()


def handle_band_individual(address, *args):
    """Handle individual band power for a specific channel"""
    parts = address.split('/')
    channel_name = parts[-2]
    band_info = parts[-1]
    value = args[0] if args else 0.0

    # Check if it's a relative value
    is_relative = band_info.endswith('-relative')
    band_name = band_info.replace('-relative', '')

    if is_relative:
        # 0-1 scale
        bar_length = int(value * 40)
        print(f"🧠 {channel_name} {band_name:6s} (rel): {value:6.3f} {'█' * bar_length}")
    else:
        # Absolute µV²
        bar_length = min(int(value / 2), 40)
        print(f"🧠 {channel_name} {band_name:6s} (abs): {value:7.2f} µV² {'█' * bar_length}")


def handle_band_aggregate(address, *args):
    """Handle cross-channel band aggregates (mean/max/min)"""
    parts = address.split('/')
    band_name = parts[-2] if len(parts) > 3 else parts[-1]
    stat_type = parts[-1] if len(parts) > 3 else "mean"
    value = args[0] if args else 0.0

    bar_length = min(int(value / 2), 40)
    print(f"📈 {band_name:6s} ({stat_type:4s}): {value:7.2f} µV² {'█' * bar_length}")


def handle_cv_features(address, *args):
    """Handle computer vision (facial) features"""
    feature_name = address.split('/')[-1]
    value = args[0] if args else 0.0

    # Create a simple bar visualization (0.0 to 1.0 scale)
    bar_length = int(value * 40)
    bar = "█" * bar_length

    print(f"😊 CV {feature_name:20s}: {value:6.3f} {bar}")


def handle_muse_compat_absolute(address, *args):
    """Handle Muse-compatible combined absolute band power messages"""
    band_name = address.split('/')[-1].replace('_absolute', '')

    print(f"🎵 MUSE-COMPAT {band_name} (absolute):")
    if len(args) >= 4:
        print(f"   CH1: {args[0]:7.2f} µV²  CH2: {args[1]:7.2f} µV²  CH3: {args[2]:7.2f} µV²  CH4: {args[3]:7.2f} µV²")
    else:
        print(f"   Values: {args}")
    print()


def handle_muse_compat_relative(address, *args):
    """Handle Muse-compatible combined relative band power messages"""
    band_name = address.split('/')[-1].replace('_relative', '')

    print(f"🎵 MUSE-COMPAT {band_name} (relative 0-1):")
    if len(args) >= 4:
        bars = [int(val * 20) for val in args[:4]]
        print(f"   CH1: {args[0]:5.3f} {'█' * bars[0]}")
        print(f"   CH2: {args[1]:5.3f} {'█' * bars[1]}")
        print(f"   CH3: {args[2]:5.3f} {'█' * bars[2]}")
        print(f"   CH4: {args[3]:5.3f} {'█' * bars[3]}")
    else:
        print(f"   Values: {args}")
    print()


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

    # Raw timeseries data handlers (new /biomus namespace)
    dispatcher.map("/biomus/raw/CH*", handle_raw_channel)
    dispatcher.map("/biomus/raw", handle_raw_combined)
    dispatcher.map("/biomus/raw/CH*/chunk*", handle_raw_chunked)
    dispatcher.map("/biomus/raw/chunk*", handle_raw_chunked)

    # Band power handlers (new /biomus namespace)
    # Individual channel bands (absolute and relative)
    dispatcher.map("/biomus/bands/CH*/*", handle_band_individual)
    # Cross-channel aggregates
    dispatcher.map("/biomus/bands/*/max", handle_band_aggregate)
    dispatcher.map("/biomus/bands/*/min", handle_band_aggregate)
    dispatcher.map("/biomus/bands/delta", handle_band_aggregate)
    dispatcher.map("/biomus/bands/theta", handle_band_aggregate)
    dispatcher.map("/biomus/bands/alpha", handle_band_aggregate)
    dispatcher.map("/biomus/bands/beta", handle_band_aggregate)
    dispatcher.map("/biomus/bands/gamma", handle_band_aggregate)

    # Computer vision / facial feature handlers (new /cv namespace)
    dispatcher.map("/cv/*", handle_cv_features)

    # Muse-compatible combined messages
    dispatcher.map("/biomus/elements/*_absolute", handle_muse_compat_absolute)
    dispatcher.map("/biomus/elements/*_relative", handle_muse_compat_relative)

    # Fallback for any other messages
    dispatcher.set_default_handler(handle_default)

    # Create and start server
    print_header()
    print(f"🎧 Listening for OSC messages on {args.ip}:{args.port}")
    print(f"📡 Waiting for data from BioMus...")
    print(f"   - Raw EEG: /biomus/raw/CH*, /biomus/raw")
    print(f"   - Band Powers: /biomus/bands/CH*/<band>, /biomus/bands/<band>")
    print(f"   - Muse-compat: /biomus/elements/<band>_absolute, /biomus/elements/<band>_relative")
    print(f"   - CV Features: /cv/*")
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
