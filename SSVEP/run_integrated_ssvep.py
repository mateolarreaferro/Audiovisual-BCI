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
        self.fullscreen = fullscreen
        self.window_size = (1024, 600)
        
        # Visual elements
        self.box_size = 250
        self.separation = 500  # Increased separation for clearer targets
        
        # Colors
        self.bg_color = (128, 128, 128)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (255, 0, 0)
        self.green = (0, 255, 0)
        self.blue = (0, 100, 255)
        self.yellow = (255, 255, 0)
        
        # LSL connection
        self.inlet = None
        self.fs = None
        self.n_channels = None
        self.buffer = deque(maxlen=500)  # Will be resized after connection
        
        # Detection parameters - optimized for better sensitivity
        self.snr_threshold = 0.25  # Lower threshold for better sensitivity
        self.margin_ratio = 1.10  # Less strict margin for easier detection
        self.ema_alpha = 0.4  # Faster response
        self.hold_ms = 400  # Shorter hold time for quicker selection

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
        
        # Smoothing and filtering
        self.smoothed_powers = np.zeros(len(self.frequencies))
        self.vote_filter = StableVoteFilter(hold_duration_ms=self.hold_ms)
        
        # Communication queue between threads
        self.detection_queue = queue.Queue()
        
        # Calibration state
        self.baseline_data = []
        self.calibration_flicker_start = None  # Track flicker timing during calibration
        
        # Initialize Pygame
        pygame.init()
        
    def setup_display(self):
        """Initialize Pygame display"""
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.window_size = self.screen.get_size()
        else:
            self.screen = pygame.display.set_mode(self.window_size)
        
        pygame.display.set_caption("SSVEP BCI - Integrated System")
        
        # Fonts
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 48)
        
        # Clock
        self.clock = pygame.time.Clock()
        
        # Calculate positions
        center_x = self.window_size[0] // 2
        center_y = self.window_size[1] // 2
        
        self.left_pos = (center_x - self.separation // 2 - self.box_size // 2,
                        center_y - self.box_size // 2)
        self.right_pos = (center_x + self.separation // 2 - self.box_size // 2,
                         center_y - self.box_size // 2)
    
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

        # Update buffer size
        self.buffer = deque(maxlen=int(self.fs * WINDOW_SEC))

        # Initialize FBCCA classifier
        config = {
            'HARDWARE': {'sampling_rate': self.fs},
            'STIMULUS': {'frequencies': self.frequencies},
            'CLASSIFIER': {
                'type': 'FBCCA',
                'n_harmonics': HARMONICS,
                'filter_bank': {'enabled': True, 'n_filters': 5, 'filter_order': 4},
                'window_length': WINDOW_SEC,
                'threshold': self.snr_threshold,
            }
        }
        self.classifier = SSVEPClassifier(config)

        return True
    
    def find_optimal_channels(self, data):
        """Find channels with best SSVEP response"""
        if self.n_channels >= 16:
            # Test occipital channels (typically 7-10 for O1, O2, Oz, POz)
            test_channels = list(range(7, min(11, self.n_channels)))
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
        
        # Select top 3 channels
        best_indices = np.argsort(channel_snrs)[-3:]
        self.optimal_channels = [test_channels[i] for i in best_indices]
        
        logger.info(f"Optimal channels selected: {self.optimal_channels}")
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
    
    def calibration_phase(self):
        """Run calibration to optimize detection parameters"""
        logger.info("Starting calibration phase...")
        self.calibrating = True
        self.calibration_step = 0
        self.calibration_start_time = None
        self.calibration_message = ""
        self.calibration_flicker_start = None  # Initialize flicker timing

        # Reset data containers
        self.training_data = {freq: [] for freq in self.frequencies}
        self.buffer.clear()
        
        # Steps: 0=baseline, 1=left freq, 2=right freq
        self.calibration_steps = [
            {"message": "Relax and look at the center cross", "duration": 8, "freq_index": None},
            {"message": f"Look at the FLASHING LEFT box ({self.frequencies[0]}Hz)", "duration": 20, "freq_index": 0},
            {"message": f"Look at the FLASHING RIGHT box ({self.frequencies[1]}Hz)", "duration": 20, "freq_index": 1},
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
        
        # Collect data
        chunk, _ = self.inlet.pull_chunk(timeout=0.0, max_samples=32)
        if chunk:
            if self.calibration_step == 0:  # Baseline
                for sample in chunk:
                    self.baseline_data.append(sample)
            else:
                for sample in chunk:
                    self.buffer.append(sample)
        
        # Check if step is complete
        if elapsed >= self.calibration_duration:
            step = self.calibration_steps[self.calibration_step]
            freq_index = step["freq_index"]

            if self.calibration_step == 0:
                # Process baseline data
                self.process_baseline()
            else:
                # Store training segments for this frequency
                data = np.array(self.buffer).T  # channels x samples
                if self.optimal_channels:
                    data = data[self.optimal_channels, :]

                window_samples = int(self.fs * WINDOW_SEC)
                for i in range(0, data.shape[1] - window_samples + 1, window_samples):
                    segment = data[:, i:i+window_samples]
                    self.training_data[self.frequencies[freq_index]].append(segment)

            self.buffer.clear()

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
        
        if len(self.baseline_data) > self.fs:
            baseline_array = np.array(self.baseline_data[-int(self.fs*2):]).T
            
            # Find optimal channels
            self.find_optimal_channels(baseline_array)
            
            # Calculate baseline noise
            noise_levels = []
            for freq in range(5, 30):
                if freq not in self.frequencies:
                    noise_levels.append(self.compute_ssvep_power(baseline_array, freq))
            self.baseline_noise = np.median(noise_levels)
            logger.info(f"Baseline noise level: {self.baseline_noise:.2f}")
    
    def finish_calibration(self):
        """Complete calibration by training classifier and setting thresholds"""
        if self.classifier and any(self.training_data.values()):
            # Check if we have enough training data
            total_segments = sum(len(segments) for segments in self.training_data.values())
            logger.info(f"Training with {total_segments} total segments")
            
            if total_segments > 0:
                self.classifier.train(self.training_data)
                self.calculate_thresholds()
            else:
                logger.warning("No training data collected during calibration!")

        self.calibrating = False
        self.calibration_step = -1
        self.calibration_flicker_start = None
        logger.info("Calibration complete!")

        # Clear baseline data to save memory
        if hasattr(self, 'baseline_data'):
            del self.baseline_data

    def calculate_thresholds(self):
        """Calculate detection threshold from calibration scores"""
        scores = []
        for freq, segments in self.training_data.items():
            for seg in segments:
                features = self.classifier.extract_features(seg)
                scores.append(features.get(freq, 0))

        if scores:
            # Use a lower percentile for better sensitivity
            self.snr_threshold = max(0.15, np.percentile(scores, 30))
            logger.info(f"Adaptive score threshold: {self.snr_threshold:.2f}")
        else:
            logger.warning("No scores calculated from training data")
    
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

                    # Enhanced detection logic with better sensitivity
                    max_idx = int(np.argmax(scores))
                    max_score = scores[max_idx]
                    second_best = np.partition(scores, -2)[-2] if len(scores) > 1 else 0

                    # More lenient detection criteria
                    if max_score > self.snr_threshold:
                        # Check if significantly better than second best
                        if second_best == 0 or max_score > second_best * self.margin_ratio:
                            stable = self.vote_filter.update(max_idx)
                            # Normalize confidence to 0-1 range
                            self.confidence = min(1.0, max_score)
                        else:
                            # Still detecting but not stable yet
                            self.vote_filter.update(max_idx)
                            stable = None
                            self.confidence = min(1.0, max_score * 0.7)

                        # Send detection result
                        self.detection_queue.put({
                            'selection': stable,
                            'candidate': max_idx,
                            'powers': scores.copy(),
                            'confidence': self.confidence
                        })
                    else:
                        self.vote_filter.reset()
                        self.detection_queue.put({
                            'selection': None,
                            'candidate': None,
                            'powers': scores.copy(),
                            'confidence': 0.0
                        })

                    last_update = time.time()
                    
            except Exception as e:
                logger.error(f"Detection error: {e}")
                time.sleep(0.1)
    
    def draw_interface(self):
        """Draw the main interface"""
        # Clear screen
        self.screen.fill(self.bg_color)
        
        # Title
        title = "SSVEP BCI System"
        if self.calibrating:
            title += " - CALIBRATION MODE"
        
        text = self.large_font.render(title, True, self.white)
        text_rect = text.get_rect(centerx=self.window_size[0]//2, y=20)
        self.screen.blit(text, text_rect)
        
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
        
        y_offset = 80
        for line in instructions:
            text = self.small_font.render(line, True, self.white)
            text_rect = text.get_rect(centerx=self.window_size[0]//2, y=y_offset)
            self.screen.blit(text, text_rect)
            y_offset += 25
    
    def draw_calibration_interface(self):
        """Draw calibration-specific interface"""
        # Update calibration progress
        self.update_calibration()
        
        if not hasattr(self, 'calibration_message'):
            return
        
        # Don't clear screen here - it's already cleared in the main loop
        
        # Title
        title = f"CALIBRATION - Step {self.calibration_step + 1} of {len(self.calibration_steps)}"
        text = self.large_font.render(title, True, self.yellow)
        text_rect = text.get_rect(centerx=self.window_size[0]//2, y=50)
        self.screen.blit(text, text_rect)
        
        # Current instruction
        instruction = self.calibration_message
        text = self.font.render(instruction, True, self.white)
        text_rect = text.get_rect(centerx=self.window_size[0]//2, y=150)
        self.screen.blit(text, text_rect)
        
        # Progress bar
        if hasattr(self, 'calibration_start_time') and self.calibration_start_time:
            elapsed = time.time() - self.calibration_start_time
            progress = min(1.0, elapsed / self.calibration_duration)
            
            bar_width = 400
            bar_height = 20
            bar_x = (self.window_size[0] - bar_width) // 2
            bar_y = 200
            
            # Background
            pygame.draw.rect(self.screen, self.black,
                           (bar_x, bar_y, bar_width, bar_height))
            
            # Progress fill
            fill_width = int(bar_width * progress)
            pygame.draw.rect(self.screen, self.green,
                           (bar_x, bar_y, fill_width, bar_height))
            
            # Border
            pygame.draw.rect(self.screen, self.white,
                           (bar_x, bar_y, bar_width, bar_height), 2)
            
            # Time remaining
            remaining = max(0, self.calibration_duration - elapsed)
            time_text = f"Time remaining: {remaining:.1f}s"
            text = self.small_font.render(time_text, True, self.white)
            text_rect = text.get_rect(centerx=self.window_size[0]//2, y=bar_y + 30)
            self.screen.blit(text, text_rect)
        
        # Visual cues based on calibration step
        if self.calibration_step == 0:  # Baseline - show center cross
            cross_size = 30
            center_x, center_y = self.window_size[0]//2, self.window_size[1]//2
            
            # Draw cross
            pygame.draw.line(self.screen, self.white,
                           (center_x - cross_size, center_y),
                           (center_x + cross_size, center_y), 4)
            pygame.draw.line(self.screen, self.white,
                           (center_x, center_y - cross_size),
                           (center_x, center_y + cross_size), 4)
        
        elif self.calibration_step > 0:  # Show FLICKERING boxes during calibration
            # Calculate flicker states if we have a flicker start time
            if self.calibration_flicker_start:
                elapsed = time.time() - self.calibration_flicker_start
                
                # Draw both boxes with appropriate flickering
                for i, (pos, label) in enumerate([(self.left_pos, self.labels[0]),
                                                  (self.right_pos, self.labels[1])]):
                    
                    target_idx = self.calibration_steps[self.calibration_step]["freq_index"]
                    
                    if i == target_idx:
                        # This box should flicker at its frequency
                        freq = self.frequencies[i]
                        phase = np.sin(2 * np.pi * freq * elapsed)
                        intensity = int((phase + 1) * 127.5)
                        color = (intensity, intensity, intensity)
                        border_color = self.green if intensity > 127 else self.yellow
                        border_width = 6
                    else:
                        # Non-target box - dim static
                        color = (50, 50, 50)
                        border_color = (100, 100, 100)
                        border_width = 2
                    
                    pygame.draw.rect(self.screen, color,
                                   (pos[0], pos[1], self.box_size, self.box_size))
                    pygame.draw.rect(self.screen, border_color,
                                   (pos[0], pos[1], self.box_size, self.box_size), border_width)
                    
                    # Label
                    text_color = self.white if i == target_idx else (150, 150, 150)
                    text = self.font.render(label, True, text_color)
                    text_rect = text.get_rect(centerx=pos[0] + self.box_size//2,
                                              bottom=pos[1] - 20)
                    self.screen.blit(text, text_rect)
            
            # Show frequency info and instruction
            if self.calibration_flicker_start:
                target_freq = self.frequencies[self.calibration_steps[self.calibration_step]["freq_index"]]
                freq_text = f"Focus on the FLASHING {target_freq}Hz stimulus"
                text = self.font.render(freq_text, True, self.yellow)
                text_rect = text.get_rect(centerx=self.window_size[0]//2,
                                          bottom=self.window_size[1] - 50)
                self.screen.blit(text, text_rect)
                
                # Add a small indicator showing the flashing is active
                indicator_text = "● RECORDING SSVEP RESPONSE"
                text = self.small_font.render(indicator_text, True, self.red)
                text_rect = text.get_rect(centerx=self.window_size[0]//2,
                                          bottom=self.window_size[1] - 20)
                self.screen.blit(text, text_rect)
    
    def draw_boxes(self, left_color, right_color):
        """Draw flickering boxes with selection feedback"""
        # Get current detection result
        detection = None
        try:
            detection = self.detection_queue.get_nowait()
            self.current_selection = detection.get('selection')
            self.confidence = detection.get('confidence', 0)
        except queue.Empty:
            pass
        
        # Draw left box
        left_border_color = self.black
        left_border_width = 3
        
        if self.current_selection == 0:
            left_border_color = self.green
            left_border_width = 6
        elif detection and detection.get('candidate') == 0:
            left_border_color = self.yellow
            left_border_width = 4
        
        pygame.draw.rect(self.screen, left_color,
                        (self.left_pos[0], self.left_pos[1], 
                         self.box_size, self.box_size))
        pygame.draw.rect(self.screen, left_border_color,
                        (self.left_pos[0], self.left_pos[1], 
                         self.box_size, self.box_size), left_border_width)
        
        # Draw right box
        right_border_color = self.black
        right_border_width = 3
        
        if self.current_selection == 1:
            right_border_color = self.green
            right_border_width = 6
        elif detection and detection.get('candidate') == 1:
            right_border_color = self.yellow
            right_border_width = 4
        
        pygame.draw.rect(self.screen, right_color,
                        (self.right_pos[0], self.right_pos[1], 
                         self.box_size, self.box_size))
        pygame.draw.rect(self.screen, right_border_color,
                        (self.right_pos[0], self.right_pos[1], 
                         self.box_size, self.box_size), right_border_width)
        
        # Labels
        for i, (pos, label) in enumerate([(self.left_pos, self.labels[0]), 
                                           (self.right_pos, self.labels[1])]):
            text = self.font.render(label, True, self.white)
            text_rect = text.get_rect(centerx=pos[0] + self.box_size//2,
                                      bottom=pos[1] - 20)
            self.screen.blit(text, text_rect)
        
        # Draw detection feedback
        if self.current_selection is not None:
            selection_text = f"SELECTED: {self.labels[self.current_selection]}"
            color = self.green
        elif detection and detection.get('candidate') is not None:
            selection_text = f"Detecting: {self.labels[detection['candidate']]}"
            color = self.yellow
        else:
            selection_text = "Looking for signal..."
            color = self.white
        
        text = self.font.render(selection_text, True, color)
        text_rect = text.get_rect(centerx=self.window_size[0]//2,
                                  bottom=self.window_size[1] - 80)
        self.screen.blit(text, text_rect)
        
        # Confidence bar
        if self.confidence > 0:
            bar_width = 300
            bar_height = 20
            bar_x = (self.window_size[0] - bar_width) // 2
            bar_y = self.window_size[1] - 50
            
            # Background
            pygame.draw.rect(self.screen, self.black,
                           (bar_x, bar_y, bar_width, bar_height))
            
            # Fill based on confidence
            fill_width = int(bar_width * self.confidence)
            if self.confidence > 0.7:
                bar_color = self.green
            elif self.confidence > 0.4:
                bar_color = self.yellow
            else:
                bar_color = self.red
            
            pygame.draw.rect(self.screen, bar_color,
                           (bar_x, bar_y, fill_width, bar_height))
            
            # Border
            pygame.draw.rect(self.screen, self.white,
                           (bar_x, bar_y, bar_width, bar_height), 2)
        
        # Score display
        if detection and detection.get('powers') is not None:
            powers = detection['powers']
            for i, (freq, power) in enumerate(zip(self.frequencies, powers)):
                text = f"{freq}Hz score: {power:.2f}"
                color = self.green if power > self.snr_threshold else self.white
                snr_text = self.small_font.render(text, True, color)
                snr_rect = snr_text.get_rect(x=20, y=self.window_size[1] - 100 + i*25)
                self.screen.blit(snr_text, snr_rect)
    
    def update_flicker(self, start_time):
        """Calculate flicker states"""
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Calculate phases
        left_phase = np.sin(2 * np.pi * self.frequencies[0] * elapsed)
        right_phase = np.sin(2 * np.pi * self.frequencies[1] * elapsed)
        
        # Convert to colors
        left_intensity = int((left_phase + 1) * 127.5)
        right_intensity = int((right_phase + 1) * 127.5)
        
        left_color = (left_intensity, left_intensity, left_intensity)
        right_color = (right_intensity, right_intensity, right_intensity)
        
        return left_color, right_color
    
    def run(self):
        """Main application loop"""
        # Setup display
        self.setup_display()
        
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
        
        logger.info("System ready. Press C to calibrate, SPACE to start.")
        
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
                    elif event.key == pygame.K_c:
                        if not self.stimulating:
                            self.calibration_phase()
            
            # Draw base interface
            if not self.calibrating:
                self.draw_interface()
            
            # Draw stimulus boxes or calibration
            if self.calibrating:
                # For calibration, we handle everything in one method
                self.draw_calibration_interface()
            elif self.stimulating and start_time:
                # Normal operation - flickering boxes
                left_color, right_color = self.update_flicker(start_time)
                self.draw_boxes(left_color, right_color)
            else:
                # Static boxes when not stimulating
                self.draw_boxes(self.white, self.white)
            
            # Update display
            pygame.display.flip()
            self.clock.tick(60)
        
        # Cleanup
        pygame.quit()
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