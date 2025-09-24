#!/usr/bin/env python3
"""Integrated SSVEP system with visual stimulus, calibration, and real-time feedback"""

import numpy as np
import pygame
import time
import threading
import queue
from pylsl import StreamInlet, resolve_streams
from scipy import signal
from scipy.signal import welch
from collections import deque
import sys
import os
import logging

# Add local src directory
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from utils import StableVoteFilter
from ssvep_bci.modules.ssvep_classifier import SSVEPClassifier
from ssvep_ui import SSVEPVisualInterface
from calibration_data_manager import CalibrationDataManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# SSVEP parameters
TARGET_FREQS = [10.0, 15.0]  # Hz
HARMONICS = 2
WINDOW_SEC = 2.0
UPDATE_RATE = 4  # Hz

class IntegratedSSVEP:
    """Combined visual stimulus and SSVEP detector with calibration"""
    
    def __init__(self, fullscreen=False):
        # Visual parameters
        self.frequencies = TARGET_FREQS
        self.labels = ["LEFT", "RIGHT"]
        
        # Initialize UI
        self.ui = SSVEPVisualInterface(self.frequencies, self.labels, fullscreen)
        
        # Initialize calibration data manager
        self.calibration_manager = CalibrationDataManager()
        
        # LSL connection
        self.inlet = None
        self.fs = None
        self.n_channels = None
        self.buffer = deque(maxlen=500)  # Will be resized after connection
        
        # Detection parameters - optimized for better sensitivity
        self.snr_threshold = 0.05  # Very low threshold for initial testing
        self.margin_ratio = 1.05   # Very low margin for easier detection
        self.ema_alpha = 0.3   # More smoothing initially
        self.hold_ms = 800     # Longer hold for stability

        # Calibration and training data
        self.training_data = {freq: [] for freq in self.frequencies}
        self.baseline_noise = None
        self.optimal_channels = None
        self.classifier = None
        
        # State management
        self.running = False
        self.calibrating = False
        self.stimulating = False
        self.current_selection = None
        self.confidence = 0.0
        self.no_detection_count = 0
        
        # Smoothing and filtering
        self.smoothed_powers = np.zeros(len(self.frequencies))
        self.vote_filter = StableVoteFilter(hold_duration_ms=self.hold_ms)
        
        # Communication queue between threads
        self.detection_queue = queue.Queue()
        
        # Calibration state
        self.baseline_data = []
        self.calibration_flicker_start = None
        self.calibration_step = 0
        self.calibration_start_time = None
        self.calibration_message = ""
        self.calibration_duration = 0
        self.calibration_buffer = deque(maxlen=int(self.fs * 70) if hasattr(self, 'fs') and self.fs else 8750)
        self.no_data_count = 0
    
    def connect_lsl(self):
        """Connect to LSL stream from OpenBCI GUI"""
        logger.info("Looking for LSL stream from OpenBCI GUI...")
        
        streams = resolve_streams(wait_time=5.0)
        
        if not streams:
            logger.error("No LSL stream found. Make sure OpenBCI GUI is streaming.")
            return False
        
        self.inlet = StreamInlet(streams[0])
        info = self.inlet.info()
        
        self.fs = info.nominal_srate()
        self.n_channels = info.channel_count()
        
        logger.info(f"Connected to LSL stream: {info.name()}")
        logger.info(f"Sampling rate: {self.fs} Hz, Channels: {self.n_channels}")

        # Update buffer size for calibration (need to hold more data)
        self.calibration_buffer = deque(maxlen=int(self.fs * 70))  # 70 seconds of data for long calibration
        self.buffer = deque(maxlen=int(self.fs * WINDOW_SEC))  # 2 seconds for detection

        # Initialize FBCCA classifier
        config = {
            'HARDWARE': {'sampling_rate': self.fs},
            'STIMULUS': {'frequencies': self.frequencies},
            'CLASSIFIER': {
                'type': 'FBTRCA',
                'n_harmonics': HARMONICS,
                'filter_bank': {'enabled': True, 'n_filters': 5, 'filter_order': 4},
                'window_length': WINDOW_SEC,
                'threshold': self.snr_threshold,
            },
            'FEATURES': {
                'psda': {
                    'nperseg': int(self.fs),
                    'noverlap': int(self.fs * 0.5)
                }
            }
        }
        self.classifier = SSVEPClassifier(config)

        return True
    
    def find_optimal_channels(self, data):
        """Find channels with best SSVEP response"""
        if self.n_channels >= 16:
            # For 16-channel setup: O1, O2, Oz are typically channels 9, 10, 11 (0-indexed: 8, 9, 10)
            # Also include POz (12), P3 (13), P4 (14) for better coverage
            test_channels = [8, 9, 10, 11, 12, 13, 14] if self.n_channels > 14 else list(range(8, self.n_channels))
        elif self.n_channels >= 8:
            # Use last 4 channels
            test_channels = list(range(max(0, self.n_channels-4), self.n_channels))
        else:
            test_channels = list(range(self.n_channels))
        
        # Compute SNR for each channel
        channel_snrs = []
        for ch in test_channels:
            ch_data = data[ch:ch+1, :]
            snr_sum = 0
            for freq in self.frequencies:
                snr = self.compute_ssvep_power(ch_data, freq)
                snr_sum += snr
            channel_snrs.append(snr_sum)
        
        # Select top 4 channels for better coverage
        best_indices = np.argsort(channel_snrs)[-4:]
        self.optimal_channels = [test_channels[i] for i in best_indices]
        
        logger.info(f"Optimal channels selected: {self.optimal_channels}")
        logger.info(f"Channel SNRs: {[f'Ch{test_channels[i]}:{channel_snrs[i]:.2f}' for i in range(len(test_channels))]}")
        return self.optimal_channels
    
    def compute_ssvep_power(self, data, freq):
        """Compute SSVEP power with improved processing"""
        # Bandpass filter
        sos = signal.butter(4, [5, 45], btype='band', fs=self.fs, output='sos')
        filtered = signal.sosfiltfilt(sos, data, axis=1)
        
        # Notch filter for power line noise
        notch_sos = signal.butter(2, [59, 61], btype='bandstop', fs=self.fs, output='sos')
        filtered = signal.sosfiltfilt(notch_sos, filtered, axis=1)
        
        # Compute PSD
        nperseg = min(data.shape[1], int(self.fs * 1.5))
        freqs, psd = welch(filtered, fs=self.fs, nperseg=nperseg, 
                          noverlap=nperseg//2, axis=1)
        
        # Average across channels
        psd_mean = np.mean(psd, axis=0)
        
        # Target frequency power
        target_idx = np.argmin(np.abs(freqs - freq))
        signal_power = psd_mean[target_idx]
        
        # Add harmonic
        if HARMONICS >= 2:
            harmonic_idx = np.argmin(np.abs(freqs - freq * 2))
            if harmonic_idx < len(psd_mean):
                signal_power += 0.3 * psd_mean[harmonic_idx]
        
        # Calculate noise
        noise_band = np.where((freqs >= freq - 2) & (freqs <= freq + 2) & 
                              (np.abs(freqs - freq) > 0.5))[0]
        if len(noise_band) > 0:
            noise_power = np.median(psd_mean[noise_band])
            if self.baseline_noise is not None:
                noise_power = max(noise_power, self.baseline_noise)
            snr = signal_power / (noise_power + 1e-10)
        else:
            snr = signal_power
        
        return snr
    
    def load_previous_calibration(self):
        """Load the best previous calibration session"""
        try:
            best_session = self.calibration_manager.get_best_session(min_segments=20)
            
            if best_session is None:
                logger.warning("No suitable previous calibration found")
                return False
            
            session_data = self.calibration_manager.load_calibration_session(best_session)
            
            # Restore calibration data
            self.training_data = session_data['training_data']
            self.optimal_channels = session_data['optimal_channels']
            self.baseline_noise = session_data['baseline_noise']
            self.snr_threshold = session_data['threshold']
            
            # Train classifier with loaded data
            if self.classifier:
                self.classifier.train(self.training_data)
                
            total_segments = sum(len(segments) for segments in self.training_data.values())
            logger.info(f"Loaded calibration: {best_session}")
            logger.info(f"Total segments: {total_segments}")
            logger.info(f"Optimal channels: {self.optimal_channels}")
            logger.info(f"Threshold: {self.snr_threshold:.3f}")
            logger.info("Ready for detection!")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False
    
    def calibration_phase(self):
        """Run calibration to optimize detection parameters"""
        logger.info("Starting calibration phase...")
        self.calibrating = True
        self.calibration_step = 0
        self.calibration_start_time = None
        self.calibration_message = ""
        self.calibration_flicker_start = None

        # Reset data containers
        self.training_data = {freq: [] for freq in self.frequencies}
        self.calibration_buffer = deque(maxlen=int(self.fs * 70))  # Reset calibration buffer for long sessions
        self.buffer.clear()
        self.baseline_data = []  # Reset baseline data
        self.no_data_count = 0   # Reset data collection counter
        
        # Extended calibration for optimal data collection
        # Need: ~100+ segments per frequency for robust training
        # At 0.5s steps over 2s windows: need ~50s minimum per frequency
        self.calibration_steps = [
            {"message": "Relax and look at the center cross", "duration": 15, "freq_index": None, "target": "baseline"},
            {"message": f"Look at the FLASHING LEFT box ({self.frequencies[0]}Hz)", "duration": 60, "freq_index": 0, "target": "training"},
            {"message": f"Look at the FLASHING RIGHT box ({self.frequencies[1]}Hz)", "duration": 60, "freq_index": 1, "target": "training"},
        ]
        
        self.start_calibration_step()
    
    def start_calibration_step(self):
        """Start the current calibration step"""
        if self.calibration_step >= len(self.calibration_steps):
            self.finish_calibration()
            return
        
        step = self.calibration_steps[self.calibration_step]
        self.calibration_message = step["message"]
        self.calibration_start_time = time.time()
        self.calibration_duration = step["duration"]
        
        # Initialize flicker timing for stimulus steps
        if step["freq_index"] is not None:
            self.calibration_flicker_start = time.time()
        else:
            self.calibration_flicker_start = None
        
        logger.info(f"Calibration step {self.calibration_step + 1}: {self.calibration_message}")
    
    def update_calibration(self):
        """Update calibration progress and collect data"""
        if not self.calibrating or self.calibration_start_time is None:
            return
        
        elapsed = time.time() - self.calibration_start_time
        
        # Collect data more aggressively
        chunk, timestamps = self.inlet.pull_chunk(timeout=0.0, max_samples=64)
        
        if chunk:
            chunk_size = len(chunk)
            if self.calibration_step == 0:  # Baseline
                self.baseline_data.extend(chunk)
                if len(self.baseline_data) % 100 == 0:  # Log every 100 samples
                    logger.info(f"Baseline data: {len(self.baseline_data)} samples collected")
            else:
                # Use calibration buffer to collect all data during the step
                self.calibration_buffer.extend(chunk)
                if len(self.calibration_buffer) % 250 == 0:  # Log every 250 samples (2 seconds)
                    logger.info(f"Training data: {len(self.calibration_buffer)} samples collected")
        else:
            # Log when no data is received
            if hasattr(self, 'no_data_count'):
                self.no_data_count += 1
            else:
                self.no_data_count = 1
                
            if self.no_data_count % 100 == 0:  # Log every 100 empty pulls
                logger.warning(f"No LSL data received for {self.no_data_count} consecutive pulls")
        
        # Check if step is complete
        if elapsed >= self.calibration_duration:
            step = self.calibration_steps[self.calibration_step]
            freq_index = step["freq_index"]

            if self.calibration_step == 0:
                # Process baseline data
                self.process_baseline()
            else:
                # Store training segments for this frequency
                data = np.array(list(self.calibration_buffer)).T  # channels x samples
                logger.info(f"=== CALIBRATION STEP {self.calibration_step} COMPLETE ===")
                logger.info(f"Calibration buffer: {len(self.calibration_buffer)} samples")
                logger.info(f"Expected samples: ~{int(self.fs * self.calibration_duration)} for {self.calibration_duration}s at {self.fs}Hz")
                logger.info(f"Data collection efficiency: {len(self.calibration_buffer)/(self.fs * self.calibration_duration)*100:.1f}%")
                logger.info(f"Data shape: {data.shape}")
                
                # Use optimal channels if available, otherwise use all channels
                if self.optimal_channels and len(self.optimal_channels) > 0:
                    data = data[self.optimal_channels, :]
                    logger.info(f"Using optimal channels {self.optimal_channels}, new shape: {data.shape}")
                else:
                    logger.info(f"Using all channels, shape: {data.shape}")

                window_samples = int(self.fs * WINDOW_SEC)  # 250 samples for 2s at 125Hz
                step_size = int(self.fs * 0.5)  # 62.5 samples for 0.5s step
                
                logger.info(f"Window: {window_samples} samples, Step: {step_size} samples")
                logger.info(f"Data length: {data.shape[1]} samples")
                
                # Collect overlapping segments for better training
                n_segments = 0
                if data.shape[1] >= window_samples:
                    for i in range(0, data.shape[1] - window_samples + 1, step_size):
                        segment = data[:, i:i+window_samples]
                        self.training_data[self.frequencies[freq_index]].append(segment)
                        n_segments += 1
                        if n_segments <= 3:  # Log first few segments
                            logger.info(f"Segment {n_segments}: samples {i} to {i+window_samples-1}, shape: {segment.shape}")
                else:
                    logger.warning(f"Not enough data for segmentation: {data.shape[1]} < {window_samples}")
                
                logger.info(f"Collected {n_segments} segments for {self.frequencies[freq_index]}Hz")

            # Clear the calibration buffer for next step
            self.calibration_buffer.clear()

            self.calibration_step += 1
            if self.calibration_step < len(self.calibration_steps):
                time.sleep(1)  # Short rest between steps
                self.start_calibration_step()
            else:
                self.finish_calibration()
    
    def process_baseline(self):
        """Process baseline data to find optimal channels and noise level"""
        if not hasattr(self, 'baseline_data'):
            self.baseline_data = []
            return
        
        logger.info(f"Processing baseline with {len(self.baseline_data)} samples")
        
        # Need substantial baseline data for good channel selection
        min_baseline_samples = int(self.fs * 10)  # At least 10 seconds
        
        if len(self.baseline_data) >= min_baseline_samples:
            # Use more data for better channel selection
            baseline_array = np.array(self.baseline_data[-int(self.fs*12):]).T  # Use last 12 seconds
            logger.info(f"Baseline array shape: {baseline_array.shape}")
            
            # Find optimal channels
            self.find_optimal_channels(baseline_array)
            
            # Calculate baseline noise
            noise_levels = []
            for freq in range(5, 30):
                if freq not in self.frequencies:
                    noise_levels.append(self.compute_ssvep_power(baseline_array, freq))
            if noise_levels:
                self.baseline_noise = np.median(noise_levels)
                logger.info(f"Baseline noise level: {self.baseline_noise:.2f}")
            else:
                self.baseline_noise = 0.1
                logger.warning("No noise levels calculated, using default")
        else:
            logger.warning(f"Insufficient baseline data: {len(self.baseline_data)} samples")
            # Use default channel selection
            if self.n_channels >= 16:
                self.optimal_channels = [8, 9, 10, 11]  # Default occipital channels
            else:
                self.optimal_channels = list(range(min(4, self.n_channels)))
            logger.info(f"Using default optimal channels: {self.optimal_channels}")
    
    def finish_calibration(self):
        """Complete calibration by training classifier and setting thresholds"""
        if self.classifier and any(self.training_data.values()):
            # Check if we have enough training data
            total_segments = sum(len(segments) for segments in self.training_data.values())
            logger.info(f"Training with {total_segments} total segments")
            
            # Log training data details
            for freq, segments in self.training_data.items():
                logger.info(f"{freq}Hz: {len(segments)} segments")
            
            # Need substantial training data for robust classification
            min_segments_per_freq = 50  # Target ~100 segments per frequency for robust training
            min_total_segments = min_segments_per_freq * len(self.frequencies)
            
            if total_segments >= min_total_segments:
                logger.info("Sufficient training data collected - training classifier")
                self.classifier.train(self.training_data)
                self.calculate_thresholds()
            elif total_segments >= 20:  # Minimum viable training
                logger.warning(f"Limited training data: {total_segments} segments (optimal: {min_total_segments})")
                logger.info("Training with available data - may have reduced accuracy")
                self.classifier.train(self.training_data)
                self.calculate_thresholds()
            else:
                logger.error(f"Critically insufficient training data: {total_segments} segments (minimum: 20)")
                logger.error("This indicates a fundamental LSL data collection problem!")
                logger.info("Possible issues:")
                logger.info("1. OpenBCI GUI not streaming data")
                logger.info("2. LSL connection problems")
                logger.info("3. Data rate too low")
                logger.info("")
                logger.info("Using fallback detection with very low threshold")
                self.snr_threshold = 0.01  # Very low threshold for fallback

        self.calibrating = False
        self.calibration_step = -1
        self.calibration_flicker_start = None
        
        # Save calibration data
        if total_segments >= 10:  # Only save if we have some data
            try:
                metadata = {
                    'sampling_rate': self.fs,
                    'n_channels': self.n_channels,
                    'frequencies': self.frequencies,
                    'window_sec': WINDOW_SEC,
                    'harmonics': HARMONICS
                }
                
                session_name = self.calibration_manager.save_calibration_session(
                    self.training_data,
                    self.optimal_channels,
                    self.baseline_noise,
                    self.snr_threshold,
                    metadata
                )
                
                logger.info(f"Calibration data saved as: {session_name}")
            except Exception as e:
                logger.warning(f"Failed to save calibration data: {e}")
        
        logger.info("Calibration complete!")

        # Clear baseline data to save memory
        if hasattr(self, 'baseline_data'):
            del self.baseline_data

    def calculate_thresholds(self):
        """Calculate detection threshold from calibration scores"""
        all_scores = []
        per_freq_scores = {}
        
        for freq, segments in self.training_data.items():
            freq_scores = []
            for seg in segments:
                features = self.classifier.extract_features(seg)
                score = features.get(freq, 0)
                freq_scores.append(score)
                all_scores.append(score)
            per_freq_scores[freq] = freq_scores
            logger.info(f"{freq}Hz scores: mean={np.mean(freq_scores):.3f}, std={np.std(freq_scores):.3f}, n={len(freq_scores)}")

        if all_scores:
            # Use adaptive threshold based on score distribution
            mean_score = np.mean(all_scores)
            std_score = np.std(all_scores)
            # Set threshold lower than mean to allow detection
            self.snr_threshold = max(0.1, mean_score - 0.5 * std_score)
            logger.info(f"Score statistics: mean={mean_score:.3f}, std={std_score:.3f}")
            logger.info(f"Adaptive score threshold: {self.snr_threshold:.3f}")
        else:
            logger.warning("No scores calculated from training data")
            self.snr_threshold = 0.1
    
    def detection_thread(self):
        """Background thread for SSVEP detection"""
        last_update = time.time()
        
        while self.running:
            try:
                # Pull data from LSL
                chunk, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=32)
                
                if chunk:
                    for sample in chunk:
                        self.buffer.append(sample)
                
                # Process at UPDATE_RATE
                if (time.time() - last_update > 1.0/UPDATE_RATE and
                    len(self.buffer) >= self.fs * WINDOW_SEC and
                    self.stimulating):

                    # Convert buffer to array
                    data = np.array(self.buffer)

                    # Use optimal channels if available
                    if self.optimal_channels:
                        data = data[:, self.optimal_channels].T
                    else:
                        data = data.T

                    # Extract FBCCA features
                    features = self.classifier.extract_features(data)
                    scores = np.array([features.get(freq, 0) for freq in self.frequencies])

                    # Smooth estimates
                    self.smoothed_powers = (
                        self.ema_alpha * scores +
                        (1 - self.ema_alpha) * self.smoothed_powers
                    )
                    scores = self.smoothed_powers

                    # Enhanced detection logic with diagnostic output
                    max_idx = int(np.argmax(scores))
                    max_score = scores[max_idx]
                    second_best = np.partition(scores, -2)[-2] if len(scores) > 1 else 0
                    
                    # Log raw scores every 20 updates for debugging
                    if hasattr(self, 'debug_counter'):
                        self.debug_counter += 1
                    else:
                        self.debug_counter = 1
                    
                    if self.debug_counter % 20 == 0:
                        score_str = ', '.join([f'{self.frequencies[i]}Hz:{scores[i]:.3f}' for i in range(len(scores))])
                        logger.info(f"Raw scores: {score_str}, threshold: {self.snr_threshold:.3f}, max_idx: {max_idx}")

                    # More lenient detection criteria
                    if max_score > self.snr_threshold:
                        # Check if significantly better than second best
                        margin_check = second_best == 0 or max_score > second_best * self.margin_ratio
                        
                        if margin_check:
                            stable = self.vote_filter.update(max_idx)
                            # Better confidence calculation
                            self.confidence = min(1.0, (max_score - self.snr_threshold) / (1.0 - self.snr_threshold))
                        else:
                            # Still detecting but not stable yet
                            self.vote_filter.update(max_idx)
                            stable = None
                            self.confidence = min(1.0, (max_score - self.snr_threshold) / (1.0 - self.snr_threshold) * 0.5)
                    else:
                        # No strong detection - reset
                        stable = None
                        self.vote_filter.reset()
                        self.confidence = 0.0

                    # Send detection result
                    self.detection_queue.put({
                        'selection': stable,
                        'candidate': max_idx if max_score > self.snr_threshold else None,
                        'powers': scores.copy(),
                        'confidence': self.confidence
                    })

                    last_update = time.time()
                    
            except Exception as e:
                logger.error(f"Detection error: {e}")
                time.sleep(0.1)
    
    def draw_main_interface(self):
        """Draw the main interface using UI module"""
        self.ui.clear_screen()
        
        # Title
        title = "SSVEP BCI System"
        self.ui.draw_title(title)
        
        # Instructions
        if not self.stimulating and not self.calibrating:
            instructions = [
                "Press C to calibrate (recommended first)",
                "Press SPACE to start stimulus",
                "Press ESC to exit"
            ]
        elif self.stimulating:
            instructions = [
                f"Look at a box to make selection",
                f"Left: {self.frequencies[0]}Hz | Right: {self.frequencies[1]}Hz",
                "Press SPACE to stop"
            ]
        else:
            instructions = []
        
        self.ui.draw_instructions(instructions)
    
    def draw_calibration(self):
        """Draw calibration interface"""
        if not hasattr(self, 'calibration_message'):
            return
        
        elapsed = time.time() - self.calibration_start_time if self.calibration_start_time else 0
        
        # Draw calibration info
        self.ui.draw_calibration_info(
            self.calibration_step + 1,
            len(self.calibration_steps),
            self.calibration_message,
            elapsed,
            self.calibration_duration
        )
        
        # Draw appropriate visual based on step
        if self.calibration_step == 0:  # Baseline - show center cross
            self.ui.draw_center_cross()
        elif self.calibration_step > 0 and self.calibration_flicker_start:  # Flickering boxes
            elapsed_flicker = time.time() - self.calibration_flicker_start
            target_idx = self.calibration_steps[self.calibration_step]["freq_index"]
            self.ui.draw_calibration_boxes(target_idx, elapsed_flicker)
            
            # Show recording indicator
            self.ui.draw_recording_indicator()
            
            # Show frequency info
            target_freq = self.frequencies[target_idx]
            freq_text = f"Focus on the FLASHING {target_freq}Hz stimulus"
            text = self.ui.font.render(freq_text, True, self.ui.yellow)
            text_rect = text.get_rect(centerx=self.ui.window_size[0]//2,
                                      bottom=self.ui.window_size[1] - 50)
            self.ui.screen.blit(text, text_rect)
    
    def draw_detection_boxes(self, start_time):
        """Draw boxes with detection feedback"""
        # Get current detection result
        detection = None
        candidate = None
        try:
            detection = self.detection_queue.get_nowait()
            # Update selection only if there's a new stable selection
            new_selection = detection.get('selection')
            if new_selection is not None:
                self.current_selection = new_selection
            
            # Get candidate for visual feedback
            candidate = detection.get('candidate')
            
            # If no candidate detected, clear selection after a delay
            if candidate is None:
                self.no_detection_count += 1
                if self.no_detection_count > 10:  # After ~2.5 seconds of no detection
                    self.current_selection = None
            else:
                self.no_detection_count = 0
            
            self.confidence = detection.get('confidence', 0)
        except queue.Empty:
            pass
        
        # Calculate flicker colors
        left_color, right_color = self.ui.update_flicker(start_time)
        
        # Determine borders based on detection
        left_border = None
        right_border = None
        
        if self.current_selection == 0:
            left_border = (self.ui.green, 6)
        elif candidate == 0:
            left_border = (self.ui.yellow, 4)
        
        if self.current_selection == 1:
            right_border = (self.ui.green, 6)
        elif candidate == 1:
            right_border = (self.ui.yellow, 4)
        
        # Draw boxes
        self.ui.draw_boxes(left_color, right_color, left_border, right_border)
        
        # Draw selection feedback
        if self.current_selection is not None:
            selection_text = f"SELECTED: {self.labels[self.current_selection]}"
            color = self.ui.green
        elif candidate is not None:
            selection_text = f"Detecting: {self.labels[candidate]}"
            color = self.ui.yellow
        else:
            selection_text = "Looking for signal..."
            color = self.ui.white
        
        self.ui.draw_selection_feedback(selection_text, color)
        
        # Draw confidence bar
        self.ui.draw_confidence_bar(self.confidence)
        
        # Draw scores if available
        if detection and detection.get('powers') is not None:
            self.ui.draw_scores(detection['powers'], self.snr_threshold)
    
    def run(self):
        """Main application loop"""
        # Setup display
        self.ui.setup_display()
        
        # Connect to LSL
        if not self.connect_lsl():
            logger.error("Failed to connect to LSL stream")
            return
        
        # Start detection thread
        self.running = True
        detection_thread = threading.Thread(target=self.detection_thread)
        detection_thread.daemon = True
        detection_thread.start()
        
        # Main loop
        start_time = None
        
        logger.info("=" * 60)
        logger.info("SSVEP BCI SYSTEM READY")
        logger.info("=" * 60)
        logger.info("IMPORTANT: Calibration will take ~2.5 minutes for optimal results")
        logger.info("- Baseline: 15 seconds (relax, look at center)")
        logger.info("- Left training: 60 seconds (focus on left box)")
        logger.info("- Right training: 60 seconds (focus on right box)")
        logger.info("")
        logger.info("Controls:")
        logger.info("  C = Start new calibration")
        logger.info("  L = Load previous calibration")
        logger.info("  SPACE = Start/stop detection")
        logger.info("  ESC = Exit")
        logger.info("=" * 60)
        
        # Show available calibration sessions
        sessions = self.calibration_manager.list_sessions()
        if sessions:
            logger.info(f"Found {len(sessions)} previous calibration sessions:")
            for i, session in enumerate(sessions[:3]):  # Show top 3
                logger.info(f"  {i+1}. {session['name']}: {session['total_segments']} segments")
        else:
            logger.info("No previous calibration sessions found")
        
        while self.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_SPACE:
                        if not self.calibrating:
                            self.stimulating = not self.stimulating
                            if self.stimulating:
                                start_time = time.time()
                                logger.info("Stimulus started")
                            else:
                                logger.info("Stimulus stopped")
                                self.current_selection = None
                                self.vote_filter.reset()
                    elif event.key == pygame.K_c:
                        if not self.stimulating:
                            self.calibration_phase()
                    elif event.key == pygame.K_l:  # Load previous calibration
                        if not self.stimulating and not self.calibrating:
                            self.load_previous_calibration()
            
            # Update calibration if active
            if self.calibrating:
                self.update_calibration()
            
            # Draw appropriate interface
            if self.calibrating:
                self.draw_calibration()
            else:
                self.draw_main_interface()
                
                if self.stimulating and start_time:
                    self.draw_detection_boxes(start_time)
                else:
                    # Static boxes when not stimulating
                    self.ui.draw_boxes(self.ui.white, self.ui.white)
            
            # Update display
            self.ui.flip()
            self.ui.tick(60)
        
        # Cleanup
        self.ui.quit()
        logger.info("System shutdown")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Integrated SSVEP BCI System')
    parser.add_argument('--fullscreen', action='store_true',
                       help='Run in fullscreen mode')
    
    args = parser.parse_args()
    
    # Create and run system
    system = IntegratedSSVEP(fullscreen=args.fullscreen)
    
    try:
        system.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()