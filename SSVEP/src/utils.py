"""Utility functions for SSVEP detection system"""

import numpy as np
import time
from typing import List, Optional, Any
from collections import deque
import threading
import logging

logger = logging.getLogger(__name__)


class RingBuffer:
    """Thread-safe ring buffer for real-time data storage"""
    
    def __init__(self, maxlen: int):
        """
        Initialize ring buffer
        
        Args:
            maxlen: Maximum number of elements to store
        """
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()
    
    def append(self, data: Any):
        """
        Add data to the buffer
        
        Args:
            data: Data to add (typically numpy array)
        """
        with self.lock:
            self.buffer.append(data)
    
    def extend(self, data_list: List[Any]):
        """
        Add multiple data points to the buffer
        
        Args:
            data_list: List of data points to add
        """
        with self.lock:
            self.buffer.extend(data_list)
    
    def get_all(self) -> List[Any]:
        """
        Get all data from the buffer
        
        Returns:
            List of all data in buffer (oldest first)
        """
        with self.lock:
            return list(self.buffer)
    
    def get_last_n(self, n: int) -> List[Any]:
        """
        Get the last N elements from the buffer
        
        Args:
            n: Number of elements to retrieve
        
        Returns:
            List of last N elements (oldest first)
        """
        with self.lock:
            if n >= len(self.buffer):
                return list(self.buffer)
            else:
                return list(self.buffer)[-n:]
    
    def clear(self):
        """Clear all data from the buffer"""
        with self.lock:
            self.buffer.clear()
    
    def __len__(self) -> int:
        """Get current buffer length"""
        with self.lock:
            return len(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        with self.lock:
            return len(self.buffer) == self.maxlen


class TimeSeriesBuffer:
    """Ring buffer specifically for time series data (e.g., EEG)"""
    
    def __init__(self, n_channels: int, buffer_duration: float, sampling_rate: float):
        """
        Initialize time series buffer
        
        Args:
            n_channels: Number of data channels
            buffer_duration: Duration of data to keep (seconds)
            sampling_rate: Data sampling rate (Hz)
        """
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.buffer_size = int(buffer_duration * sampling_rate)
        
        # Initialize buffer array
        self.data = np.zeros((n_channels, self.buffer_size))
        self.write_idx = 0
        self.is_full = False
        self.lock = threading.Lock()
        
        logger.info(f"TimeSeriesBuffer initialized: {n_channels} channels, "
                   f"{buffer_duration}s @ {sampling_rate}Hz = {self.buffer_size} samples")
    
    def add_samples(self, samples: np.ndarray):
        """
        Add new samples to the buffer
        
        Args:
            samples: New samples array of shape (n_channels, n_samples)
        """
        if samples.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {samples.shape[0]}")
        
        n_new_samples = samples.shape[1]
        
        with self.lock:
            # Handle wraparound
            if self.write_idx + n_new_samples <= self.buffer_size:
                # No wraparound needed
                self.data[:, self.write_idx:self.write_idx + n_new_samples] = samples
            else:
                # Wraparound needed
                n_before_wrap = self.buffer_size - self.write_idx
                n_after_wrap = n_new_samples - n_before_wrap
                
                self.data[:, self.write_idx:] = samples[:, :n_before_wrap]
                self.data[:, :n_after_wrap] = samples[:, n_before_wrap:]
                
                self.is_full = True
            
            # Update write index
            self.write_idx = (self.write_idx + n_new_samples) % self.buffer_size
    
    def get_latest_samples(self, n_samples: int) -> Optional[np.ndarray]:
        """
        Get the most recent N samples
        
        Args:
            n_samples: Number of samples to retrieve
        
        Returns:
            Array of shape (n_channels, n_samples) or None if not enough data
        """
        with self.lock:
            available = self.buffer_size if self.is_full else self.write_idx
            
            if n_samples > available:
                return None
            
            # Calculate start index
            if self.is_full:
                start_idx = (self.write_idx - n_samples) % self.buffer_size
            else:
                start_idx = max(0, self.write_idx - n_samples)
            
            # Extract data
            if start_idx + n_samples <= self.buffer_size:
                # No wraparound
                return self.data[:, start_idx:start_idx + n_samples].copy()
            else:
                # Wraparound needed
                n_before_wrap = self.buffer_size - start_idx
                n_after_wrap = n_samples - n_before_wrap
                
                result = np.zeros((self.n_channels, n_samples))
                result[:, :n_before_wrap] = self.data[:, start_idx:]
                result[:, n_before_wrap:] = self.data[:, :n_after_wrap]
                
                return result
    
    def get_latest_duration(self, duration: float) -> Optional[np.ndarray]:
        """
        Get the most recent data for a given duration
        
        Args:
            duration: Duration in seconds
        
        Returns:
            Array of shape (n_channels, n_samples) or None if not enough data
        """
        n_samples = int(duration * self.sampling_rate)
        return self.get_latest_samples(n_samples)


class StableVoteFilter:
    """Filter for stable decision making with hold timer"""
    
    def __init__(self, hold_duration_ms: int = 500):
        """
        Initialize stable vote filter
        
        Args:
            hold_duration_ms: How long a decision must be stable before accepting
        """
        self.hold_duration_ms = hold_duration_ms
        self.current_vote = None
        self.vote_start_time = None
        self.stable_decision = None
        self.last_stable_time = None
        self.reselection_cooldown_ms = 1500  # Cooldown before allowing new selection
        
    def update(self, new_vote: Any) -> Optional[Any]:
        """
        Update with a new vote and check for stable decision
        
        Args:
            new_vote: New decision/vote
        
        Returns:
            Stable decision if available, None otherwise
        """
        current_time = time.time() * 1000  # Convert to milliseconds
        
        # Check if we're in cooldown period after a selection
        if self.last_stable_time is not None:
            if current_time - self.last_stable_time < self.reselection_cooldown_ms:
                # During cooldown, return the stable decision only once
                if self.stable_decision is not None:
                    stable_result = self.stable_decision
                    self.stable_decision = None  # Clear to avoid repeated returns
                    return stable_result
                return None
            else:
                # Cooldown expired, fully reset for new selection
                self.stable_decision = None
                self.last_stable_time = None
                self.current_vote = None
                self.vote_start_time = None
        
        if new_vote != self.current_vote:
            # New vote - reset timer
            self.current_vote = new_vote
            self.vote_start_time = current_time
            return None
        
        # Same vote - check if held long enough
        if self.vote_start_time is not None:
            hold_time = current_time - self.vote_start_time
            if hold_time >= self.hold_duration_ms:
                if self.stable_decision != new_vote:
                    # New stable decision
                    self.stable_decision = new_vote
                    self.last_stable_time = current_time
                    logger.info(f"[STABLE] Decision: {new_vote} (held for {hold_time:.0f}ms)")
                    return new_vote
        
        return None
    
    def reset(self):
        """Reset the filter state"""
        self.current_vote = None
        self.vote_start_time = None
        self.stable_decision = None
        self.last_stable_time = None  # Allow immediate reselection after reset


def format_detection_output(frequency: float, snr: float, all_scores: dict, 
                          is_stable: bool = False) -> str:
    """
    Format detection results for console output
    
    Args:
        frequency: Detected frequency
        snr: SNR of detected frequency
        all_scores: Dictionary of all frequency scores
        is_stable: Whether this is a stable detection
    
    Returns:
        Formatted string for console output
    """
    # Format the main prediction
    if is_stable:
        prefix = "[SELECT]"
    else:
        prefix = "Pred="
    
    output = f"{prefix}{frequency:g} SNR={snr:.2f} :: "
    
    # Format all scores
    score_parts = []
    for freq in sorted(all_scores.keys()):
        score = all_scores[freq]
        score_parts.append(f"{freq:g}:{score:.2f}")
    
    output += " | ".join(score_parts)
    
    return output


def calculate_performance_metrics(predictions: List[float], ground_truth: List[float], 
                                target_freqs: List[float]) -> dict:
    """
    Calculate performance metrics for SSVEP detection
    
    Args:
        predictions: List of predicted frequencies
        ground_truth: List of true frequencies
        target_freqs: List of possible target frequencies
    
    Returns:
        Dictionary with performance metrics
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    # Overall accuracy
    correct = sum(1 for p, t in zip(predictions, ground_truth) if p == t)
    accuracy = correct / len(predictions) if len(predictions) > 0 else 0.0
    
    # Per-class metrics
    class_metrics = {}
    for freq in target_freqs:
        # True positives, false positives, false negatives
        tp = sum(1 for p, t in zip(predictions, ground_truth) if p == freq and t == freq)
        fp = sum(1 for p, t in zip(predictions, ground_truth) if p == freq and t != freq)
        fn = sum(1 for p, t in zip(predictions, ground_truth) if p != freq and t == freq)
        
        # Precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[freq] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(1 for t in ground_truth if t == freq)
        }
    
    return {
        'accuracy': accuracy,
        'per_class': class_metrics,
        'num_samples': len(predictions)
    }


def create_synthetic_ssvep(frequency: float, fs: float, duration: float, 
                          n_channels: int = 8, snr: float = 2.0,
                          harmonics: bool = True) -> np.ndarray:
    """
    Create synthetic SSVEP signal for testing
    
    Args:
        frequency: SSVEP frequency in Hz
        fs: Sampling rate in Hz
        duration: Duration in seconds
        n_channels: Number of channels to simulate
        snr: Signal-to-noise ratio (higher = less noise)
        harmonics: Whether to include 2nd harmonic
    
    Returns:
        Synthetic EEG data of shape (n_channels, n_samples)
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    
    # Create SSVEP signal
    ssvep_signal = np.sin(2 * np.pi * frequency * t)
    
    # Add 2nd harmonic if requested
    if harmonics and 2 * frequency < fs / 2:
        ssvep_signal += 0.3 * np.sin(2 * np.pi * 2 * frequency * t)
    
    # Create multi-channel data
    data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Different phase and amplitude for each channel (simulate electrode differences)
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.7, 1.3)
        
        # SSVEP signal
        signal = amplitude * np.sin(2 * np.pi * frequency * t + phase)
        if harmonics and 2 * frequency < fs / 2:
            signal += 0.3 * amplitude * np.sin(2 * np.pi * 2 * frequency * t + phase)
        
        # Add noise
        noise = np.random.randn(n_samples) / snr
        
        # Add some background EEG activity
        for bg_freq in [8.0, 13.0, 20.0]:  # Alpha, beta, low gamma
            if bg_freq < fs / 2:
                bg_amplitude = np.random.uniform(0.1, 0.3)
                signal += bg_amplitude * np.sin(2 * np.pi * bg_freq * t + 
                                              np.random.uniform(0, 2 * np.pi))
        
        data[ch] = signal + noise
    
    return data


def test_utils():
    """Test the utility functions"""
    print("Testing utility functions...")
    
    # Test RingBuffer
    print("\n1. Testing RingBuffer:")
    rb = RingBuffer(5)
    for i in range(8):
        rb.append(f"item_{i}")
        print(f"Added item_{i}, buffer: {rb.get_all()}")
    
    # Test TimeSeriesBuffer
    print("\n2. Testing TimeSeriesBuffer:")
    fs = 125.0
    buffer = TimeSeriesBuffer(n_channels=3, buffer_duration=2.0, sampling_rate=fs)
    
    # Add some data
    for i in range(5):
        samples = np.random.randn(3, 25)  # 3 channels, 25 samples (~0.2s)
        buffer.add_samples(samples)
        
        latest = buffer.get_latest_duration(0.5)  # Get last 0.5 seconds
        if latest is not None:
            print(f"Iteration {i}: Retrieved {latest.shape[1]} samples")
    
    # Test StableVoteFilter
    print("\n3. Testing StableVoteFilter:")
    vote_filter = StableVoteFilter(hold_duration_ms=200)
    
    votes = [10, 10, 12, 12, 12, 12, 12, 10, 10, 10, 10, 10]
    for i, vote in enumerate(votes):
        time.sleep(0.05)  # 50ms delay
        stable = vote_filter.update(vote)
        print(f"Vote {vote} -> Stable: {stable}")
    
    # Test synthetic SSVEP generation
    print("\n4. Testing synthetic SSVEP generation:")
    synthetic_data = create_synthetic_ssvep(frequency=10.0, fs=125.0, 
                                          duration=1.0, n_channels=4)
    print(f"Generated synthetic SSVEP: {synthetic_data.shape}")
    print(f"Data range: [{np.min(synthetic_data):.2f}, {np.max(synthetic_data):.2f}]")
    
    print("\nAll utility tests completed!")


if __name__ == "__main__":
    test_utils()