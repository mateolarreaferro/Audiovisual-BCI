"""
Thread-safe configuration and state management for EEG streaming
Handles mode switching, parameter updates, and streaming control
"""
import threading
import json
from typing import Dict, Any
from dataclasses import dataclass, asdict
from enum import Enum

class StreamingMode(Enum):
    DEMO = "demo"
    LIVE_RELAY = "live_relay"
    RESEARCH = "research"

class GeneratorType(Enum):
    SINE = "sine"
    STEP = "step"
    NOISE = "noise"
    MIXED = "mixed"

@dataclass
class Config:
    # Network settings
    unity_ip: str = "127.0.0.1"
    unity_port: int = 7777
    udp_in_port: int = 7778
    
    # Streaming control
    mode: StreamingMode = StreamingMode.DEMO
    is_streaming: bool = False
    target_fps: float = 30.0
    
    # Signal processing
    normalization_window_seconds: float = 10.0
    calm_threshold: float = 0.3
    engaged_threshold: float = 0.7
    enable_normalization: bool = True
    
    # Demo generator settings
    generator_type: GeneratorType = GeneratorType.SINE
    frequency: float = 0.5
    amplitude: float = 0.4
    bias: float = 0.5
    noise_level: float = 0.1
    spike_rate: float = 0.02
    phase_offset: Dict[str, float] = None
    
    def __post_init__(self):
        if self.phase_offset is None:
            self.phase_offset = {
                "alpha": 0.0,
                "beta": 1.57,  # π/2
                "theta": 3.14,  # π
                "delta": 4.71,  # 3π/2
            }

@dataclass
class StreamingStats:
    packets_sent: int = 0
    packets_received: int = 0
    send_rate: float = 0.0
    receive_rate: float = 0.0
    last_packet_time: float = 0.0
    error_count: int = 0
    uptime_seconds: float = 0.0

class ThreadSafeState:
    """Thread-safe state management with locks"""
    
    def __init__(self):
        self._lock = threading.RLock()
        self._config = Config()
        self._stats = StreamingStats()
        self._running = False
        
        # Rate calculation
        self._packet_times = []
        self._error_messages = []
        
    def get_config(self) -> Config:
        """Get a copy of current configuration"""
        with self._lock:
            # Create a deep copy to avoid reference issues
            config_dict = asdict(self._config)
            new_config = Config(**config_dict)
            return new_config
    
    def update_config(self, updates: Dict[str, Any]) -> bool:
        """Update configuration with validation"""
        with self._lock:
            try:
                # Validate and convert enum values
                if 'mode' in updates:
                    if isinstance(updates['mode'], str):
                        updates['mode'] = StreamingMode(updates['mode'])
                
                if 'generator_type' in updates:
                    if isinstance(updates['generator_type'], str):
                        updates['generator_type'] = GeneratorType(updates['generator_type'])
                
                # Apply updates
                for key, value in updates.items():
                    if hasattr(self._config, key):
                        setattr(self._config, key, value)
                    else:
                        print(f"Warning: Unknown config key '{key}'")
                        return False
                
                return True
                
            except (ValueError, TypeError) as e:
                print(f"Config update error: {e}")
                return False
    
    def get_stats(self) -> StreamingStats:
        """Get current streaming statistics"""
        with self._lock:
            return StreamingStats(
                packets_sent=self._stats.packets_sent,
                packets_received=self._stats.packets_received,
                send_rate=self._calculate_rate('send'),
                receive_rate=self._calculate_rate('receive'),
                last_packet_time=self._stats.last_packet_time,
                error_count=self._stats.error_count,
                uptime_seconds=self._stats.uptime_seconds
            )
    
    def increment_packets_sent(self, count: int = 1):
        """Increment sent packet counter"""
        with self._lock:
            self._stats.packets_sent += count
            self._packet_times.append(('send', self._get_time()))
            self._trim_packet_times()
    
    def increment_packets_received(self, count: int = 1):
        """Increment received packet counter"""
        with self._lock:
            self._stats.packets_received += count
            self._packet_times.append(('receive', self._get_time()))
            self._trim_packet_times()
    
    def increment_errors(self, count: int = 1):
        """Increment error counter"""
        with self._lock:
            self._stats.error_count += count
    
    def add_error_message(self, message: str):
        """Add error message to log"""
        with self._lock:
            import time
            timestamp = time.strftime("%H:%M:%S")
            self._error_messages.append(f"[{timestamp}] {message}")
            
            # Keep only last 50 errors
            if len(self._error_messages) > 50:
                self._error_messages = self._error_messages[-50:]
    
    def get_recent_errors(self, count: int = 10) -> list:
        """Get recent error messages"""
        with self._lock:
            return self._error_messages[-count:] if self._error_messages else []
    
    def set_running(self, running: bool):
        """Set streaming running state"""
        with self._lock:
            self._running = running
            if running:
                import time
                self._stats.uptime_seconds = 0
                self._stats.last_packet_time = time.time()
    
    def is_running(self) -> bool:
        """Check if streaming is running"""
        with self._lock:
            return self._running
    
    def update_uptime(self, seconds: float):
        """Update uptime counter"""
        with self._lock:
            self._stats.uptime_seconds = seconds
    
    def _calculate_rate(self, packet_type: str) -> float:
        """Calculate packets per second for given type"""
        import time
        current_time = time.time()
        cutoff_time = current_time - 5.0  # 5 second window
        
        recent_packets = [
            t for ptype, t in self._packet_times 
            if ptype == packet_type and t > cutoff_time
        ]
        
        if len(recent_packets) < 2:
            return 0.0
        
        time_span = recent_packets[-1] - recent_packets[0]
        if time_span > 0:
            return (len(recent_packets) - 1) / time_span
        return 0.0
    
    def _trim_packet_times(self):
        """Remove old packet timestamps"""
        import time
        current_time = time.time()
        cutoff_time = current_time - 10.0  # Keep 10 seconds of history
        
        self._packet_times = [
            (ptype, t) for ptype, t in self._packet_times 
            if t > cutoff_time
        ]
    
    def _get_time(self) -> float:
        """Get current timestamp"""
        import time
        return time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert current state to dictionary for JSON serialization"""
        with self._lock:
            config_dict = asdict(self._config)
            
            # Convert enums to strings for JSON
            config_dict['mode'] = self._config.mode.value
            config_dict['generator_type'] = self._config.generator_type.value
            
            stats_dict = asdict(self.get_stats())
            
            return {
                'config': config_dict,
                'stats': stats_dict,
                'running': self._running,
                'recent_errors': self.get_recent_errors()
            }

# Global state instance
app_state = ThreadSafeState()