#!/usr/bin/env python3
"""
OpenBCI Data Simulator for Csound Scripts
==========================================

This script simulates various OpenBCI data streams and sends them via OSC
to test the Csound BCI scripts without needing the unreliable OpenBCI GUI.

Data Types Supported:
- Band Power (SimpleBCI.csd, BCI_BandPower.csd) -> Port 5003
- Focus (BCI_Focus.csd) -> Port 5003  
- Accelerometer/Aux (BCI_AccelAux.csd) -> Port 5004
- Raw EEG for FFT (BCI_FFT.csd) -> Port 5005

Usage:
    python openbci_simulator.py --mode [bandpower|focus|accel|fft|all]
    
Requirements:
    pip install python-osc numpy
"""

import argparse
import time
import math
import random
import threading
from typing import Dict, List, Tuple
import numpy as np
from pythonosc import udp_client
from pythonosc.osc_message_builder import OscMessageBuilder


class OpenBCISimulator:
    def __init__(self):
        self.clients = {
            'bandpower': udp_client.SimpleUDPClient("127.0.0.1", 5003),  # SimpleBCI + BandPower
            'focus': udp_client.SimpleUDPClient("127.0.0.1", 5003),      # Focus
            'accel': udp_client.SimpleUDPClient("127.0.0.1", 5004),      # Accelerometer/Aux
            'fft': udp_client.SimpleUDPClient("127.0.0.1", 5005)         # Raw EEG for FFT
        }
        
        self.running = False
        self.threads = []
        
        # Simulation parameters
        self.sample_rate = 250  # OpenBCI sample rate
        self.channels = 16
        
        # Brain wave frequency bands (Hz)
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
    def start_simulation(self, modes: List[str]):
        """Start simulation threads for specified modes"""
        self.running = True
        
        if 'bandpower' in modes or 'all' in modes:
            thread = threading.Thread(target=self.simulate_band_power, daemon=True)
            thread.start()
            self.threads.append(thread)
            print("🧠 Started Band Power simulation (port 5003)")
            
        if 'focus' in modes or 'all' in modes:
            thread = threading.Thread(target=self.simulate_focus, daemon=True)
            thread.start()
            self.threads.append(thread)
            print("🎯 Started Focus simulation (port 5003)")
            
        if 'accel' in modes or 'all' in modes:
            thread = threading.Thread(target=self.simulate_accelerometer, daemon=True)
            thread.start()
            self.threads.append(thread)
            print("📱 Started Accelerometer/Aux simulation (port 5004)")
            
        if 'fft' in modes or 'all' in modes:
            thread = threading.Thread(target=self.simulate_raw_eeg, daemon=True)
            thread.start()
            self.threads.append(thread)
            print("📊 Started Raw EEG simulation (port 5005)")
    
    def stop_simulation(self):
        """Stop all simulation threads"""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=1)
        print("🛑 Simulation stopped")

    def simulate_band_power(self):
        """Simulate brain wave band power data for all 16 channels"""
        print("📡 Sending Band Power data...")
        
        t = 0
        while self.running:
            for channel in range(self.channels):
                # Generate realistic brain wave patterns with some variation
                base_time = time.time() + channel * 0.1  # Slight phase offset per channel
                
                # Simulate different brain states over time
                attention_level = 0.5 + 0.3 * math.sin(base_time * 0.1)  # Slow attention changes
                meditation_level = 0.5 + 0.2 * math.cos(base_time * 0.15)  # Meditation state
                
                # Generate band powers (typical ranges: 10-10000 microvolts^2)
                delta = 1000 + 500 * math.sin(base_time * 0.05) + random.uniform(-100, 100)
                theta = 800 * meditation_level + 200 * math.sin(base_time * 0.08) + random.uniform(-50, 50)
                alpha = 1200 * (1 - attention_level) + 300 * math.cos(base_time * 0.12) + random.uniform(-100, 100)
                beta = 600 * attention_level + 150 * math.sin(base_time * 0.2) + random.uniform(-50, 50) 
                gamma = 200 + 50 * math.sin(base_time * 0.3) + random.uniform(-20, 20)
                
                # Ensure positive values
                delta = max(50, delta)
                theta = max(50, theta) 
                alpha = max(50, alpha)
                beta = max(50, beta)
                gamma = max(50, gamma)
                
                # Send band power data
                self.clients['bandpower'].send_message(
                    f"/openbci/band-power/{channel}",
                    [delta, theta, alpha, beta, gamma]
                )
            
            time.sleep(1.0 / 30)  # 30 Hz update rate
            t += 1

    def simulate_focus(self):
        """Simulate focus/attention metrics"""
        print("📡 Sending Focus data...")
        
        t = 0
        while self.running:
            # Global focus level with realistic patterns
            base_focus = 50  # Base focus level (0-100)
            focus_variation = 25 * math.sin(time.time() * 0.1)  # Slow variations
            noise = random.uniform(-5, 5)  # Random noise
            
            global_focus = max(0, min(100, base_focus + focus_variation + noise))
            
            # Send global focus
            self.clients['focus'].send_message("/openbci/focus", [global_focus])
            
            # Send per-channel focus (with variations)
            for channel in range(8):  # Usually only first 8 channels for focus
                channel_variation = random.uniform(-10, 10)
                channel_focus = max(0, min(100, global_focus + channel_variation))
                
                self.clients['focus'].send_message(
                    f"/openbci/focus/{channel}",
                    [channel_focus]
                )
            
            time.sleep(1.0 / 10)  # 10 Hz update rate
            t += 1

    def simulate_accelerometer(self):
        """Simulate accelerometer and auxiliary sensor data"""
        print("📡 Sending Accelerometer/Aux data...")
        
        t = 0
        while self.running:
            current_time = time.time()
            
            # Simulate natural head movement patterns
            # X-axis: left/right head movement  
            accel_x = 0.1 * math.sin(current_time * 0.5) + 0.05 * random.uniform(-1, 1)
            
            # Y-axis: forward/back head movement
            accel_y = 0.15 * math.cos(current_time * 0.3) + 0.08 * random.uniform(-1, 1)
            
            # Z-axis: up/down head movement (including gravity)
            accel_z = 9.8 + 0.2 * math.sin(current_time * 0.8) + 0.1 * random.uniform(-1, 1)
            
            # Send accelerometer data (in m/s^2)
            self.clients['accel'].send_message(
                "/openbci/accel",
                [accel_x, accel_y, accel_z]
            )
            
            # Simulate auxiliary sensors (could be EMG, temperature, etc.)
            for aux_channel in range(8):
                # Generate auxiliary sensor data (varies by channel type)
                if aux_channel < 2:  # EMG channels
                    aux_val = 100 + 50 * math.sin(current_time * 2 + aux_channel) + random.uniform(-10, 10)
                elif aux_channel < 4:  # Temperature/GSR channels  
                    aux_val = 500 + 20 * math.cos(current_time * 0.1 + aux_channel) + random.uniform(-5, 5)
                else:  # Other sensor types
                    aux_val = 200 + 80 * math.sin(current_time * 0.7 + aux_channel) + random.uniform(-15, 15)
                
                self.clients['accel'].send_message(
                    f"/openbci/aux/{aux_channel}",
                    [aux_val]
                )
            
            time.sleep(1.0 / 50)  # 50 Hz update rate
            t += 1

    def simulate_raw_eeg(self):
        """Simulate raw EEG data for FFT analysis"""
        print("📡 Sending Raw EEG data...")
        
        t = 0
        while self.running:
            current_time = time.time()
            
            for channel in range(8):  # First 8 channels for EEG
                # Generate realistic EEG signal with multiple frequency components
                eeg_signal = 0
                
                # Add different brain wave components
                eeg_signal += 20 * math.sin(2 * math.pi * 2 * current_time + channel * 0.5)   # Delta
                eeg_signal += 15 * math.sin(2 * math.pi * 6 * current_time + channel * 0.3)   # Theta  
                eeg_signal += 25 * math.sin(2 * math.pi * 10 * current_time + channel * 0.8)  # Alpha
                eeg_signal += 10 * math.sin(2 * math.pi * 20 * current_time + channel * 1.2)  # Beta
                eeg_signal += 5 * math.sin(2 * math.pi * 40 * current_time + channel * 0.1)   # Gamma
                
                # Add realistic noise and artifacts
                eeg_signal += random.uniform(-8, 8)  # Random noise
                
                # Occasional "artifacts" (eye blinks, muscle tension)
                if random.random() < 0.01:  # 1% chance
                    eeg_signal += random.uniform(-50, 50)
                
                # Send raw EEG value (in microvolts)
                self.clients['fft'].send_message(
                    f"/openbci/eeg/{channel}",
                    [eeg_signal]
                )
            
            time.sleep(1.0 / self.sample_rate)  # 250 Hz sample rate
            t += 1


def main():
    parser = argparse.ArgumentParser(description='OpenBCI Data Simulator for Csound')
    parser.add_argument(
        '--mode', 
        choices=['bandpower', 'focus', 'accel', 'fft', 'all'],
        default='all',
        help='Data simulation mode (default: all)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        default=0,
        help='Duration in seconds (0 for infinite, default: 0)'
    )
    
    args = parser.parse_args()
    
    simulator = OpenBCISimulator()
    
    print("🎵 OpenBCI Simulator for Csound Scripts")
    print("=" * 50)
    print(f"Mode: {args.mode}")
    print(f"Duration: {'Infinite' if args.duration == 0 else f'{args.duration}s'}")
    print("\nPort Mappings:")
    print("  📊 Band Power & Focus: 127.0.0.1:5003")
    print("  📱 Accelerometer/Aux:  127.0.0.1:5004") 
    print("  🧠 Raw EEG (FFT):      127.0.0.1:5005")
    print("\nPress Ctrl+C to stop...\n")
    
    try:
        modes = [args.mode] if args.mode != 'all' else ['bandpower', 'focus', 'accel', 'fft']
        simulator.start_simulation(modes)
        
        if args.duration > 0:
            time.sleep(args.duration)
            simulator.stop_simulation()
        else:
            # Run until interrupted
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\n⏹️  Stopping simulation...")
        simulator.stop_simulation()
    except Exception as e:
        print(f"❌ Error: {e}")
        simulator.stop_simulation()


if __name__ == "__main__":
    main()