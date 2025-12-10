#!/usr/bin/env python3
"""
OpenBCI UDP Sender for Unity BrainWave Visualizer
Sends band power data from OpenBCI to Unity via UDP
"""

import json
import socket
import time
import numpy as np
from scipy import signal
from collections import deque
import threading

# Try importing OpenBCI libraries
try:
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
    BRAINFLOW_AVAILABLE = True
except ImportError:
    print("BrainFlow not installed. Install with: pip install brainflow")
    BRAINFLOW_AVAILABLE = False

class OpenBCIUDPSender:
    def __init__(self, udp_host='127.0.0.1', udp_port=12345, board_id=BoardIds.CYTON_BOARD.value):
        """
        Initialize the OpenBCI UDP sender
        
        Args:
            udp_host: Host IP for UDP transmission
            udp_port: Port for UDP transmission
            board_id: BrainFlow board ID (default is Cyton)
        """
        self.udp_host = udp_host
        self.udp_port = udp_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # EEG bands definition (Hz)
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        
        # Data buffer for band power calculation
        self.buffer_size = 256  # samples
        self.sampling_rate = 250  # Hz (Cyton default)
        self.num_channels = 8
        
        # Initialize BrainFlow if available
        self.board = None
        self.board_id = board_id
        if BRAINFLOW_AVAILABLE:
            self.init_board()
        
        # Buffer for each channel
        self.channel_buffers = [deque(maxlen=self.buffer_size) for _ in range(self.num_channels)]
        
        # Thread control
        self.running = False
        self.thread = None
        
    def init_board(self):
        """Initialize BrainFlow board"""
        params = BrainFlowInputParams()
        # Uncomment and modify based on your connection type
        # params.serial_port = 'COM3'  # Windows
        # params.serial_port = '/dev/ttyUSB0'  # Linux
        # params.serial_port = '/dev/tty.usbserial-*'  # Mac
        
        try:
            BoardShim.enable_dev_board_logger()
            self.board = BoardShim(self.board_id, params)
            self.board.prepare_session()
            self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
            eeg_channels = BoardShim.get_eeg_channels(self.board_id)
            self.num_channels = len(eeg_channels)
            print(f"Board initialized: {self.num_channels} channels at {self.sampling_rate}Hz")
        except Exception as e:
            print(f"Failed to initialize board: {e}")
            self.board = None
    
    def calculate_band_power(self, data, band):
        """
        Calculate band power using FFT
        
        Args:
            data: Signal data
            band: Tuple of (low_freq, high_freq)
        
        Returns:
            Band power value
        """
        if len(data) < self.buffer_size:
            return 0.0
        
        # Apply Hamming window
        windowed = data * np.hamming(len(data))
        
        # Calculate FFT
        fft_vals = np.abs(np.fft.rfft(windowed))
        fft_freq = np.fft.rfftfreq(len(windowed), 1.0/self.sampling_rate)
        
        # Get band power
        idx_band = np.logical_and(fft_freq >= band[0], fft_freq <= band[1])
        band_power = np.mean(fft_vals[idx_band]**2)
        
        return np.sqrt(band_power)  # Return amplitude instead of power
    
    def process_eeg_data(self, eeg_data):
        """
        Process EEG data and calculate band powers
        
        Args:
            eeg_data: Multi-channel EEG data
        
        Returns:
            Dictionary with band powers for each channel
        """
        band_powers = {
            'delta': [],
            'theta': [],
            'alpha': [],
            'beta': [],
            'gamma': []
        }
        
        for ch in range(self.num_channels):
            if ch < len(eeg_data):
                # Update buffer for this channel
                self.channel_buffers[ch].extend(eeg_data[ch])
                
                # Calculate band powers if buffer is full
                if len(self.channel_buffers[ch]) >= self.buffer_size:
                    channel_data = np.array(self.channel_buffers[ch])
                    
                    # Apply filters
                    DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)
                    
                    for band_name, band_range in self.bands.items():
                        power = self.calculate_band_power(channel_data, band_range)
                        band_powers[band_name].append(power)
                else:
                    # Not enough data yet, use zeros
                    for band_name in self.bands.keys():
                        band_powers[band_name].append(0.0)
            else:
                # Channel not available, use zeros
                for band_name in self.bands.keys():
                    band_powers[band_name].append(0.0)
        
        # Normalize band powers (0-1 range)
        for band_name in band_powers:
            max_val = max(band_powers[band_name]) if band_powers[band_name] else 1.0
            if max_val > 0:
                band_powers[band_name] = [p / max_val for p in band_powers[band_name]]
        
        return band_powers
    
    def send_data(self, band_powers, raw_samples=None):
        """
        Send data to Unity via UDP
        
        Args:
            band_powers: Dictionary of band powers
            raw_samples: Optional raw EEG samples
        """
        # Calculate average power across all bands and channels
        avg_power = 0.0
        count = 0
        for band_values in band_powers.values():
            avg_power += sum(band_values)
            count += len(band_values)
        
        if count > 0:
            avg_power /= count
        
        # Prepare data packet
        data_packet = {
            'timestamp': int(time.time() * 1000),
            'bandPowers': band_powers,
            'avgPower': float(avg_power),
            'rawSamples': raw_samples if raw_samples else []
        }
        
        # Send JSON data
        json_data = json.dumps(data_packet)
        self.sock.sendto(json_data.encode('utf-8'), (self.udp_host, self.udp_port))
        
        print(f"Sent data - Avg Power: {avg_power:.3f}")
    
    def generate_test_data(self):
        """Generate test data for debugging without OpenBCI hardware"""
        t = time.time()
        
        band_powers = {}
        for band_name in self.bands.keys():
            band_powers[band_name] = []
            for ch in range(self.num_channels):
                # Generate different patterns for each band
                if band_name == 'alpha':
                    value = (np.sin(t * 2 + ch * 0.5) + 1) * 0.5
                elif band_name == 'beta':
                    value = (np.sin(t * 3 + ch * 0.3) + 1) * 0.3
                elif band_name == 'theta':
                    value = (np.sin(t * 1 + ch * 0.7) + 1) * 0.4
                elif band_name == 'delta':
                    value = (np.sin(t * 0.5 + ch) + 1) * 0.3
                else:  # gamma
                    value = np.random.random() * 0.2
                
                band_powers[band_name].append(float(value))
        
        return band_powers
    
    def stream_loop(self):
        """Main streaming loop"""
        print(f"Starting UDP stream to {self.udp_host}:{self.udp_port}")
        
        while self.running:
            try:
                if self.board and BRAINFLOW_AVAILABLE:
                    # Get real data from OpenBCI
                    data = self.board.get_current_board_data(self.buffer_size)
                    
                    if data.shape[1] > 0:
                        eeg_channels = BoardShim.get_eeg_channels(self.board_id)
                        eeg_data = data[eeg_channels, :]
                        band_powers = self.process_eeg_data(eeg_data)
                        
                        # Send last 64 raw samples
                        raw_samples = eeg_data[0, -64:].tolist() if eeg_data.shape[1] >= 64 else []
                        self.send_data(band_powers, raw_samples)
                else:
                    # Use test data
                    band_powers = self.generate_test_data()
                    self.send_data(band_powers)
                
                time.sleep(0.033)  # ~30 Hz update rate
                
            except Exception as e:
                print(f"Error in stream loop: {e}")
                time.sleep(1)
    
    def start(self):
        """Start streaming"""
        if not self.running:
            self.running = True
            
            if self.board and BRAINFLOW_AVAILABLE:
                self.board.start_stream()
                print("OpenBCI stream started")
            else:
                print("Using test data (no OpenBCI board connected)")
            
            self.thread = threading.Thread(target=self.stream_loop)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self):
        """Stop streaming"""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=2)
        
        if self.board and BRAINFLOW_AVAILABLE:
            try:
                self.board.stop_stream()
                self.board.release_session()
                print("OpenBCI stream stopped")
            except:
                pass
        
        self.sock.close()


def main():
    """Main function"""
    print("OpenBCI to Unity UDP Sender")
    print("-" * 40)
    
    # Configure sender
    sender = OpenBCIUDPSender(
        udp_host='127.0.0.1',  # localhost
        udp_port=12345,         # Unity listening port
        board_id=BoardIds.CYTON_BOARD.value if BRAINFLOW_AVAILABLE else -1
    )
    
    try:
        # Start streaming
        sender.start()
        print("\nStreaming... Press Ctrl+C to stop")
        
        # Keep running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping...")
        sender.stop()
        print("Done!")


if __name__ == "__main__":
    main()