"""
UDP streaming and signal generation for EEG data
Handles demo generation, live relay, and research logging
"""
import socket
import json
import time
import threading
import math
import random
import csv
import os
from typing import Optional, Dict, Any
from datetime import datetime

from .state import app_state, StreamingMode, GeneratorType

class EegPacket:
    """EEG packet structure matching Unity schema"""
    
    def __init__(self, **kwargs):
        self.ts = kwargs.get('ts', time.time())
        self.alpha = kwargs.get('alpha', 0.5)
        self.beta = kwargs.get('beta', 0.5)
        self.theta = kwargs.get('theta', 0.5)
        self.delta = kwargs.get('delta', 0.5)
        self.arousal = kwargs.get('arousal', 0.5)
        self.left = kwargs.get('left', 0.5)
        self.right = kwargs.get('right', 0.5)
        self.front = kwargs.get('front', 0.5)
        self.back = kwargs.get('back', 0.5)
        self.calm = kwargs.get('calm', 0)
        self.engaged = kwargs.get('engaged', 0)
        self.imu = kwargs.get('imu', [0.0, 0.0, 0.0])
    
    def to_json(self) -> str:
        """Convert to JSON string for UDP transmission"""
        data = {
            "ts": self.ts,
            "alpha": self.alpha,
            "beta": self.beta,
            "theta": self.theta,
            "delta": self.delta,
            "arousal": self.arousal,
            "left": self.left,
            "right": self.right,
            "front": self.front,
            "back": self.back,
            "calm": self.calm,
            "engaged": self.engaged,
            "imu": self.imu
        }
        return json.dumps(data, separators=(',', ':'))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'EegPacket':
        """Create packet from JSON string"""
        data = json.loads(json_str)
        return cls(**data)

class SignalGenerator:
    """Generates synthetic EEG signals for demo mode"""
    
    def __init__(self):
        self.start_time = time.time()
        self.last_spike_time = 0
        self.spike_duration = 1.0
        self.spike_active = False
        
    def generate_packet(self) -> EegPacket:
        """Generate a synthetic EEG packet based on current config"""
        config = app_state.get_config()
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # Check for random spikes
        if (current_time - self.last_spike_time) > (1.0 / config.spike_rate) and random.random() < 0.01:
            self.last_spike_time = current_time
            self.spike_active = True
        
        if self.spike_active and (current_time - self.last_spike_time) > self.spike_duration:
            self.spike_active = False
        
        # Generate base values
        if config.generator_type == GeneratorType.SINE:
            packet = self._generate_sine_wave(elapsed, config)
        elif config.generator_type == GeneratorType.STEP:
            packet = self._generate_step_function(elapsed, config)
        elif config.generator_type == GeneratorType.NOISE:
            packet = self._generate_noise(config)
        else:  # MIXED
            packet = self._generate_mixed(elapsed, config)
        
        # Apply spike modulation
        if self.spike_active:
            spike_factor = 1.0 + 0.5 * math.sin((current_time - self.last_spike_time) * math.pi / self.spike_duration)
            packet.alpha *= spike_factor
            packet.beta *= spike_factor
            packet.arousal = min(1.0, packet.arousal * 1.3)
        
        # Generate state flags
        packet.calm = 1 if packet.alpha > 0.7 and packet.arousal < 0.4 else 0
        packet.engaged = 1 if packet.beta > 0.7 and packet.arousal > 0.6 else 0
        
        # Generate synthetic IMU data
        packet.imu = [
            0.1 * math.sin(elapsed * 0.5 + 0),
            0.1 * math.sin(elapsed * 0.3 + 1.57),
            0.98 + 0.02 * math.sin(elapsed * 0.1)
        ]
        
        return packet
    
    def _generate_sine_wave(self, elapsed: float, config) -> EegPacket:
        """Generate sinusoidal EEG values"""
        base_freq = config.frequency
        
        alpha = config.bias + config.amplitude * math.sin(base_freq * elapsed + config.phase_offset['alpha'])
        beta = config.bias + config.amplitude * math.sin(base_freq * elapsed + config.phase_offset['beta'])
        theta = config.bias + config.amplitude * math.sin(base_freq * elapsed + config.phase_offset['theta'])
        delta = config.bias + config.amplitude * math.sin(base_freq * elapsed + config.phase_offset['delta'])
        
        # Add noise
        alpha += random.gauss(0, config.noise_level)
        beta += random.gauss(0, config.noise_level)
        theta += random.gauss(0, config.noise_level)
        delta += random.gauss(0, config.noise_level)
        
        # Calculate derived values
        arousal = (alpha + beta) * 0.5 + random.gauss(0, config.noise_level * 0.5)
        left = config.bias + 0.1 * math.sin(base_freq * elapsed * 0.7)
        right = config.bias - 0.1 * math.sin(base_freq * elapsed * 0.7)
        front = (alpha + theta) * 0.5
        back = (beta + delta) * 0.5
        
        # Clamp to [0, 1]
        return EegPacket(
            alpha=max(0, min(1, alpha)),
            beta=max(0, min(1, beta)),
            theta=max(0, min(1, theta)),
            delta=max(0, min(1, delta)),
            arousal=max(0, min(1, arousal)),
            left=max(0, min(1, left)),
            right=max(0, min(1, right)),
            front=max(0, min(1, front)),
            back=max(0, min(1, back))
        )
    
    def _generate_step_function(self, elapsed: float, config) -> EegPacket:
        """Generate step function EEG values"""
        step_duration = 2.0 / config.frequency
        step_phase = int(elapsed / step_duration) % 4
        
        values = [
            [0.8, 0.2, 0.3, 0.4],  # High alpha, low others
            [0.3, 0.8, 0.2, 0.3],  # High beta
            [0.2, 0.3, 0.8, 0.2],  # High theta
            [0.4, 0.2, 0.3, 0.7]   # High delta
        ]
        
        alpha, beta, theta, delta = values[step_phase]
        
        # Add noise
        alpha += random.gauss(0, config.noise_level)
        beta += random.gauss(0, config.noise_level)
        theta += random.gauss(0, config.noise_level)
        delta += random.gauss(0, config.noise_level)
        
        arousal = (alpha + beta) * 0.6
        
        return EegPacket(
            alpha=max(0, min(1, alpha)),
            beta=max(0, min(1, beta)),
            theta=max(0, min(1, theta)),
            delta=max(0, min(1, delta)),
            arousal=max(0, min(1, arousal)),
            left=config.bias,
            right=config.bias,
            front=(alpha + theta) * 0.5,
            back=(beta + delta) * 0.5
        )
    
    def _generate_noise(self, config) -> EegPacket:
        """Generate random noise EEG values"""
        return EegPacket(
            alpha=max(0, min(1, config.bias + random.gauss(0, config.amplitude))),
            beta=max(0, min(1, config.bias + random.gauss(0, config.amplitude))),
            theta=max(0, min(1, config.bias + random.gauss(0, config.amplitude))),
            delta=max(0, min(1, config.bias + random.gauss(0, config.amplitude))),
            arousal=max(0, min(1, config.bias + random.gauss(0, config.amplitude * 0.5))),
            left=max(0, min(1, config.bias + random.gauss(0, config.amplitude * 0.2))),
            right=max(0, min(1, config.bias + random.gauss(0, config.amplitude * 0.2))),
            front=max(0, min(1, config.bias + random.gauss(0, config.amplitude * 0.3))),
            back=max(0, min(1, config.bias + random.gauss(0, config.amplitude * 0.3)))
        )
    
    def _generate_mixed(self, elapsed: float, config) -> EegPacket:
        """Generate mixed signal combining multiple patterns"""
        # Combine sine and noise
        sine_packet = self._generate_sine_wave(elapsed, config)
        noise_factor = 0.3
        
        return EegPacket(
            alpha=sine_packet.alpha * (1 - noise_factor) + random.random() * noise_factor,
            beta=sine_packet.beta * (1 - noise_factor) + random.random() * noise_factor,
            theta=sine_packet.theta * (1 - noise_factor) + random.random() * noise_factor,
            delta=sine_packet.delta * (1 - noise_factor) + random.random() * noise_factor,
            arousal=sine_packet.arousal,
            left=sine_packet.left,
            right=sine_packet.right,
            front=sine_packet.front,
            back=sine_packet.back
        )

class UdpStreamer:
    """Handles UDP streaming in all modes"""
    
    def __init__(self):
        self.send_socket = None
        self.receive_socket = None
        self.generator = SignalGenerator()
        self.csv_writer = None
        self.csv_file = None
        self.running = False
        self.thread = None
        self.receive_thread = None
        
    def start(self) -> bool:
        """Start streaming in current mode"""
        if self.running:
            return True
            
        config = app_state.get_config()
        
        try:
            # Setup send socket
            self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.send_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Setup receive socket for live relay mode
            if config.mode == StreamingMode.LIVE_RELAY or config.mode == StreamingMode.RESEARCH:
                self.receive_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.receive_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.receive_socket.bind(('0.0.0.0', config.udp_in_port))
                self.receive_socket.settimeout(0.1)  # Non-blocking with timeout
                
                # Start receive thread
                self.receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
                self.receive_thread.start()
            
            # Setup CSV logging for research mode
            if config.mode == StreamingMode.RESEARCH:
                self._setup_csv_logging()
            
            # Start main streaming thread
            self.running = True
            self.thread = threading.Thread(target=self._stream_loop, daemon=True)
            self.thread.start()
            
            app_state.set_running(True)
            print(f"[Streamer] Started in {config.mode.value} mode")
            return True
            
        except Exception as e:
            app_state.add_error_message(f"Failed to start streaming: {e}")
            self.stop()
            return False
    
    def stop(self):
        """Stop streaming and cleanup resources"""
        self.running = False
        app_state.set_running(False)
        
        # Close sockets
        try:
            if self.send_socket:
                self.send_socket.close()
            if self.receive_socket:
                self.receive_socket.close()
        except:
            pass
        
        # Close CSV file
        if self.csv_file:
            try:
                self.csv_file.close()
            except:
                pass
        
        # Wait for threads to finish
        for thread in [self.thread, self.receive_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=1.0)
        
        print("[Streamer] Stopped")
    
    def _stream_loop(self):
        """Main streaming loop"""
        config = app_state.get_config()
        frame_time = 1.0 / config.target_fps
        start_time = time.time()
        
        while self.running:
            loop_start = time.time()
            
            try:
                if config.mode == StreamingMode.DEMO:
                    packet = self.generator.generate_packet()
                    self._send_packet(packet)
                    
                    if config.mode == StreamingMode.RESEARCH:
                        self._log_packet(packet)
                
                # Update stats
                app_state.update_uptime(time.time() - start_time)
                
            except Exception as e:
                app_state.add_error_message(f"Stream loop error: {e}")
                app_state.increment_errors()
            
            # Maintain target frame rate
            elapsed = time.time() - loop_start
            sleep_time = max(0, frame_time - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # Refresh config
            config = app_state.get_config()
            frame_time = 1.0 / config.target_fps
    
    def _receive_loop(self):
        """Receive loop for live relay mode"""
        while self.running and self.receive_socket:
            try:
                data, addr = self.receive_socket.recvfrom(1024)
                packet_str = data.decode('utf-8')
                
                # Parse and relay packet
                packet = EegPacket.from_json(packet_str)
                self._process_received_packet(packet)
                
                app_state.increment_packets_received()
                
            except socket.timeout:
                continue
            except Exception as e:
                app_state.add_error_message(f"Receive error: {e}")
                app_state.increment_errors()
    
    def _process_received_packet(self, packet: EegPacket):
        """Process received packet for relay/research modes"""
        config = app_state.get_config()
        
        # Apply normalization if enabled
        if config.enable_normalization:
            packet = self._normalize_packet(packet)
        
        # Apply threshold flags
        packet.calm = 1 if packet.alpha > config.calm_threshold else 0
        packet.engaged = 1 if packet.beta > config.engaged_threshold else 0
        
        # Forward to Unity
        self._send_packet(packet)
        
        # Log if in research mode
        if config.mode == StreamingMode.RESEARCH:
            self._log_packet(packet)
    
    def _normalize_packet(self, packet: EegPacket) -> EegPacket:
        """Apply normalization to packet values"""
        # Simple min-max normalization (could be enhanced with rolling statistics)
        packet.alpha = max(0, min(1, packet.alpha))
        packet.beta = max(0, min(1, packet.beta))
        packet.theta = max(0, min(1, packet.theta))
        packet.delta = max(0, min(1, packet.delta))
        packet.arousal = max(0, min(1, packet.arousal))
        return packet
    
    def _send_packet(self, packet: EegPacket):
        """Send packet to Unity via UDP"""
        config = app_state.get_config()
        
        try:
            json_data = packet.to_json()
            self.send_socket.sendto(json_data.encode('utf-8'), (config.unity_ip, config.unity_port))
            app_state.increment_packets_sent()
            
        except Exception as e:
            app_state.add_error_message(f"Send error: {e}")
            app_state.increment_errors()
    
    def _setup_csv_logging(self):
        """Setup CSV file for research logging"""
        try:
            # Create data directory
            os.makedirs('data', exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/eeg_session_{timestamp}.csv"
            
            self.csv_file = open(filename, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header
            self.csv_writer.writerow([
                'timestamp', 'ts', 'alpha', 'beta', 'theta', 'delta', 'arousal',
                'left', 'right', 'front', 'back', 'calm', 'engaged',
                'imu_x', 'imu_y', 'imu_z'
            ])
            
            print(f"[CSV] Logging to {filename}")
            
        except Exception as e:
            app_state.add_error_message(f"CSV setup error: {e}")
    
    def _log_packet(self, packet: EegPacket):
        """Log packet to CSV file"""
        if not self.csv_writer:
            return
            
        try:
            self.csv_writer.writerow([
                datetime.now().isoformat(),
                packet.ts,
                packet.alpha,
                packet.beta,
                packet.theta,
                packet.delta,
                packet.arousal,
                packet.left,
                packet.right,
                packet.front,
                packet.back,
                packet.calm,
                packet.engaged,
                packet.imu[0] if len(packet.imu) > 0 else 0,
                packet.imu[1] if len(packet.imu) > 1 else 0,
                packet.imu[2] if len(packet.imu) > 2 else 0
            ])
            
            # Flush periodically
            if app_state.get_stats().packets_sent % 100 == 0:
                self.csv_file.flush()
                
        except Exception as e:
            app_state.add_error_message(f"CSV logging error: {e}")

# Global streamer instance
streamer = UdpStreamer()