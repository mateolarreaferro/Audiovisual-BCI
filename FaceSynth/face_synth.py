#!/usr/bin/env python3
"""
FaceMesh Synth Controller - Real-time facial feature to synthesizer mapping
============================================================================

Installation:
1. Create virtual environment: python3 -m venv venv
2. Activate: source venv/bin/activate (Linux/macOS) or venv\\Scripts\\activate (Windows)
3. Install dependencies: pip install -r requirements.txt
4. Run: python face_synth.py
5. Press 'C' for calibration - move through neutral, smile, mouth open, eyebrow raise, yaw and roll
6. Play with your face to control the synth!

If MediaPipe fails to install, try: pip install mediapipe==0.10.14
"""

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass, field, asdict
import numpy as np

# Core dependencies
import cv2
import mediapipe as mp

# Audio engine
try:
    import pyaudio
    import threading
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: pyaudio not available, audio disabled")

# Optional outputs
try:
    from pythonosc import udp_client, osc_bundle_builder, osc_message_builder
    OSC_AVAILABLE = True
except ImportError:
    OSC_AVAILABLE = False

try:
    import mido
    import rtmidi
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False

# ==================== Constants ====================

# MediaPipe landmark indices for key facial features
UPPER_LIP_CENTER = 13
LOWER_LIP_CENTER = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
LEFT_EYE_CENTER = 159
RIGHT_EYE_CENTER = 386
LEFT_EYEBROW = 70
RIGHT_EYEBROW = 300
NOSE_TIP = 1
LEFT_EAR = 234
RIGHT_EAR = 454
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362

# UI Colors (BGR format)
COLOR_MESH = (100, 100, 100)
COLOR_MOUTH = (0, 255, 0)
COLOR_EYEBROW = (255, 200, 0)
COLOR_YAW = (255, 0, 255)
COLOR_ROLL = (0, 255, 255)
COLOR_SMILE = (0, 200, 255)
COLOR_METER_BG = (50, 50, 50)
COLOR_METER_FG = (0, 255, 100)

# ==================== Data Classes ====================

@dataclass
class CalibrationData:
    """Stores calibration ranges for each facial feature"""
    neutral_mouth: float = 0.05
    max_mouth: float = 0.15
    min_mouth: float = 0.01
    neutral_brow: float = 0.1
    max_brow: float = 0.2
    min_brow: float = 0.05
    neutral_yaw: float = 0.0
    max_yaw: float = 0.5
    min_yaw: float = -0.5
    neutral_roll: float = 0.0
    max_roll: float = 15.0
    min_roll: float = -15.0
    neutral_smile: float = 0.5
    max_smile: float = 1.0
    min_smile: float = 0.0

    def save(self, path: Path):
        """Save calibration to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'CalibrationData':
        """Load calibration from JSON file"""
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                return cls(**data)
        return cls()

# ==================== Utility Functions ====================

def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value between min and max"""
    return max(min_val, min(value, max_val))

def ema_filter(new_val: float, old_val: float, alpha: float = 0.2) -> float:
    """Exponential moving average filter"""
    return alpha * new_val + (1.0 - alpha) * old_val

def exp_scale(value: float, min_freq: float, max_freq: float) -> float:
    """Exponentially scale value for frequency parameters"""
    log_min = np.log(min_freq)
    log_max = np.log(max_freq)
    return np.exp(log_min + value * (log_max - log_min))

def apply_deadband(value: float, threshold: float = 0.03) -> float:
    """Apply deadband near zero to reduce jitter"""
    if abs(value) < threshold:
        return 0.0
    return value

def distance_2d(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate 2D Euclidean distance"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def angle_from_points(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate angle in radians from two points"""
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

# ==================== Face Feature Extractor ====================

class FaceFeatureExtractor:
    """Extract and compute facial features from MediaPipe landmarks"""

    def __init__(self):
        print("  Loading MediaPipe models...")
        self.mp_face_mesh = mp.solutions.face_mesh

        # Initialize with lower confidence for better performance
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        self.last_valid_features = None
        self.lost_face_time = None
        print("  MediaPipe Face Mesh loaded.")

    def extract_features(self, image: np.ndarray) -> Optional[Dict[str, float]]:
        """Extract facial features from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)

        if not results.multi_face_landmarks:
            # Handle face loss with decay
            if self.lost_face_time is None:
                self.lost_face_time = time.time()

            if time.time() - self.lost_face_time > 0.3:  # 300ms hold
                # Decay to neutral
                if self.last_valid_features:
                    for key in self.last_valid_features:
                        self.last_valid_features[key] *= 0.95
                    return self.last_valid_features
            elif self.last_valid_features:
                return self.last_valid_features

            return None

        self.lost_face_time = None
        landmarks = results.multi_face_landmarks[0].landmark
        h, w = image.shape[:2]

        # Convert normalized landmarks to pixel coordinates
        def get_point(idx: int) -> np.ndarray:
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h])

        # Compute mouth openness
        upper_lip = get_point(UPPER_LIP_CENTER)
        lower_lip = get_point(LOWER_LIP_CENTER)
        mouth_left = get_point(MOUTH_LEFT)
        mouth_right = get_point(MOUTH_RIGHT)

        mouth_height = distance_2d(upper_lip, lower_lip)
        mouth_width = distance_2d(mouth_left, mouth_right)
        mouth_openness = mouth_height / max(mouth_width, 1.0)

        # Compute left eyebrow raise
        left_eye = get_point(LEFT_EYE_CENTER)
        left_brow = get_point(LEFT_EYEBROW)
        right_eye = get_point(RIGHT_EYE_CENTER)

        eye_height = abs(left_eye[1] - right_eye[1]) + 20  # Add baseline
        brow_displacement = (left_eye[1] - left_brow[1]) / eye_height

        # Compute head yaw (horizontal rotation)
        left_eye_outer = get_point(LEFT_EYE_OUTER)
        right_eye_outer = get_point(RIGHT_EYE_OUTER)
        left_eye_inner = get_point(LEFT_EYE_INNER)
        right_eye_inner = get_point(RIGHT_EYE_INNER)

        # Simple yaw estimation from eye corner distances
        left_eye_width = distance_2d(left_eye_outer, left_eye_inner)
        right_eye_width = distance_2d(right_eye_outer, right_eye_inner)
        yaw_ratio = (left_eye_width - right_eye_width) / (left_eye_width + right_eye_width)
        head_yaw = yaw_ratio * 0.5  # Scale to radians estimate

        # Compute head roll (tilt)
        eye_angle = angle_from_points(left_eye_outer, right_eye_outer)
        head_roll = np.degrees(eye_angle)

        # Compute smile curvature
        nose_tip = get_point(NOSE_TIP)
        mouth_center = (mouth_left + mouth_right) / 2

        # Proxy: ratio of mouth corner lift vs mouth center
        mouth_curve = (mouth_center[1] - nose_tip[1]) / max(mouth_width, 1.0)
        smile_curvature = 1.0 / (1.0 + np.exp(-5 * (mouth_curve - 0.3)))  # Sigmoid

        features = {
            'mouth_openness': mouth_openness,
            'brow_raise': brow_displacement,
            'head_yaw': head_yaw,
            'head_roll': head_roll,
            'smile_curvature': smile_curvature,
            'landmarks': results.multi_face_landmarks[0]
        }

        self.last_valid_features = features.copy()
        self.last_valid_features.pop('landmarks', None)

        return features

    def close(self):
        """Clean up resources"""
        self.face_mesh.close()

# ==================== Mapping Controller ====================

class MappingController:
    """Handle calibration, normalization, and smoothing of features"""

    def __init__(self, smooth_alpha: float = 0.2, deadband: float = 0.03):
        self.calibration = CalibrationData()
        self.calibration_path = Path("calibration.json")
        self.calibration = CalibrationData.load(self.calibration_path)

        self.smooth_alpha = smooth_alpha
        self.deadband = deadband
        self.smoothed_values = {
            'mouth': 0.0,
            'brow': 0.0,
            'yaw': 0.5,
            'roll': 0.0,
            'smile': 0.0
        }

        # Median filter buffers
        self.median_buffers = {
            key: [0.0, 0.0, 0.0] for key in self.smoothed_values
        }
        self.buffer_idx = 0

        # Calibration mode
        self.calibrating = False
        self.calibration_samples = []

    def normalize_value(self, raw: float, neutral: float, min_val: float, max_val: float) -> float:
        """Normalize raw value to 0-1 range based on calibration"""
        if raw < neutral:
            if min_val >= neutral:
                return 0.0
            return (raw - neutral) / (neutral - min_val) * 0.5
        else:
            if max_val <= neutral:
                return 0.5
            return 0.5 + (raw - neutral) / (max_val - neutral) * 0.5

    def process_features(self, features: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Process raw features into normalized control values"""
        if features is None:
            # Return last smoothed values when no face detected
            return self.smoothed_values

        # Normalize each feature
        norm_mouth = self.normalize_value(
            features['mouth_openness'],
            self.calibration.neutral_mouth,
            self.calibration.min_mouth,
            self.calibration.max_mouth
        )

        norm_brow = self.normalize_value(
            features['brow_raise'],
            self.calibration.neutral_brow,
            self.calibration.min_brow,
            self.calibration.max_brow
        )

        norm_yaw = self.normalize_value(
            features['head_yaw'],
            self.calibration.neutral_yaw,
            self.calibration.min_yaw,
            self.calibration.max_yaw
        )

        norm_roll = features['head_roll']  # Keep in degrees

        norm_smile = self.normalize_value(
            features['smile_curvature'],
            self.calibration.neutral_smile,
            self.calibration.min_smile,
            self.calibration.max_smile
        )

        # Apply median filter
        self.median_buffers['mouth'][self.buffer_idx] = norm_mouth
        self.median_buffers['brow'][self.buffer_idx] = norm_brow
        self.median_buffers['yaw'][self.buffer_idx] = norm_yaw
        self.median_buffers['roll'][self.buffer_idx] = norm_roll
        self.median_buffers['smile'][self.buffer_idx] = norm_smile
        self.buffer_idx = (self.buffer_idx + 1) % 3

        # Get median values
        med_mouth = np.median(self.median_buffers['mouth'])
        med_brow = np.median(self.median_buffers['brow'])
        med_yaw = np.median(self.median_buffers['yaw'])
        med_roll = np.median(self.median_buffers['roll'])
        med_smile = np.median(self.median_buffers['smile'])

        # Apply smoothing
        self.smoothed_values['mouth'] = ema_filter(
            clamp(med_mouth), self.smoothed_values['mouth'], self.smooth_alpha
        )
        self.smoothed_values['brow'] = ema_filter(
            clamp(med_brow), self.smoothed_values['brow'], self.smooth_alpha
        )
        self.smoothed_values['yaw'] = ema_filter(
            clamp(med_yaw), self.smoothed_values['yaw'], self.smooth_alpha
        )
        self.smoothed_values['roll'] = ema_filter(
            clamp(med_roll, -15, 15), self.smoothed_values['roll'], self.smooth_alpha
        )
        self.smoothed_values['smile'] = ema_filter(
            clamp(med_smile), self.smoothed_values['smile'], self.smooth_alpha
        )

        # Apply deadband
        for key in ['mouth', 'brow', 'smile']:
            self.smoothed_values[key] = apply_deadband(
                self.smoothed_values[key], self.deadband
            )

        # Store calibration samples if calibrating
        if self.calibrating and features:
            self.calibration_samples.append({
                'mouth_openness': features['mouth_openness'],
                'brow_raise': features['brow_raise'],
                'head_yaw': features['head_yaw'],
                'head_roll': features['head_roll'],
                'smile_curvature': features['smile_curvature']
            })

        return self.smoothed_values

    def start_calibration(self):
        """Start calibration mode"""
        self.calibrating = True
        self.calibration_samples = []
        print("\n=== CALIBRATION MODE ===")
        print("1. Look straight ahead (neutral) - 3 seconds")
        print("2. Open mouth wide - 2 seconds")
        print("3. Raise eyebrows - 2 seconds")
        print("4. Turn head left/right - 2 seconds")
        print("5. Tilt head left/right - 2 seconds")
        print("6. Big smile - 2 seconds")

    def finish_calibration(self):
        """Process calibration samples and save"""
        if not self.calibration_samples:
            print("No calibration data collected")
            return

        samples = self.calibration_samples

        # Extract features arrays
        mouth_vals = [s['mouth_openness'] for s in samples]
        brow_vals = [s['brow_raise'] for s in samples]
        yaw_vals = [s['head_yaw'] for s in samples]
        roll_vals = [s['head_roll'] for s in samples]
        smile_vals = [s['smile_curvature'] for s in samples]

        # Compute calibration ranges
        self.calibration.neutral_mouth = np.median(mouth_vals[:30])  # First second
        self.calibration.max_mouth = np.percentile(mouth_vals, 95)
        self.calibration.min_mouth = np.percentile(mouth_vals, 5)

        self.calibration.neutral_brow = np.median(brow_vals[:30])
        self.calibration.max_brow = np.percentile(brow_vals, 95)
        self.calibration.min_brow = np.percentile(brow_vals, 5)

        self.calibration.neutral_yaw = np.median(yaw_vals[:30])
        self.calibration.max_yaw = np.percentile(yaw_vals, 95)
        self.calibration.min_yaw = np.percentile(yaw_vals, 5)

        self.calibration.neutral_roll = np.median(roll_vals[:30])
        self.calibration.max_roll = np.percentile(roll_vals, 95)
        self.calibration.min_roll = np.percentile(roll_vals, 5)

        self.calibration.neutral_smile = np.median(smile_vals[:30])
        self.calibration.max_smile = np.percentile(smile_vals, 95)
        self.calibration.min_smile = np.percentile(smile_vals, 5)

        self.calibration.save(self.calibration_path)
        self.calibrating = False
        print("Calibration complete and saved!")

# ==================== Synth Engine ====================

class SynthEngine:
    """PyAudio-based synthesizer with facial control parameters"""

    def __init__(self, no_audio: bool = False):
        self.no_audio = no_audio or not AUDIO_AVAILABLE

        if not self.no_audio:
            # Audio parameters
            self.sample_rate = 44100
            self.buffer_size = 256
            self.channels = 2

            # Initialize PyAudio
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                output=True,
                frames_per_buffer=self.buffer_size,
                stream_callback=self._audio_callback,
                start=False  # Don't start immediately
            )

            # Synthesis parameters
            self.base_freq = 220.0  # A3
            self.current_freq = self.base_freq
            self.target_freq = self.base_freq
            self.phase = 0.0
            self.phase_increment = 2 * np.pi * self.base_freq / self.sample_rate

            # Piano scale quantization (2 octaves, chromatic)
            # From A3 (220 Hz) to A5 (880 Hz)
            self.chromatic_scale = []
            for semitone in range(25):  # 2 octaves + 1 note
                freq = 220.0 * (2 ** (semitone / 12.0))
                self.chromatic_scale.append(freq)

            # Major scale option (C major based on A=440)
            self.major_scale = [
                220.00,   # A3
                246.94,   # B3
                261.63,   # C4
                293.66,   # D4
                329.63,   # E4
                349.23,   # F4
                392.00,   # G4
                440.00,   # A4
                493.88,   # B4
                523.25,   # C5
                587.33,   # D5
                659.25,   # E5
                698.46,   # F5
                783.99,   # G5
                880.00,   # A5
            ]

            self.use_major_scale = False  # Toggle between chromatic and major

            # Oscillator phases for SuperSaw
            self.detune_factors = [0.98, 0.99, 1.0, 1.01, 1.02]
            self.osc_phases = [0.0] * len(self.detune_factors)

            # FM synthesis
            self.fm_phase = 0.0
            self.mod_index = 0.0

            # Filter state (simple one-pole lowpass)
            self.filter_state_l = 0.0
            self.filter_state_r = 0.0
            self.filter_cutoff = 1000.0

            # Vibrato
            self.vib_phase = 0.0
            self.vib_depth = 0.0
            self.vib_rate = 5.5

            # Amplitude and pan
            self.amplitude = 0.5
            self.pan = 0.5

            # Simple reverb (delay lines)
            self.delay_size = int(self.sample_rate * 0.05)  # 50ms
            self.delay_buffer_l = np.zeros(self.delay_size)
            self.delay_buffer_r = np.zeros(self.delay_size)
            self.delay_idx = 0
            self.reverb_mix = 0.2

            # Thread safety
            self.param_lock = threading.Lock()

            # Parameter smoothing
            self.target_params = {
                'vibrato': 0.0,
                'filter': 1000.0,
                'pan': 0.5,
                'pitch': 0.0,
                'mod': 0.0,
                'amp': 0.5,
                'pitch_index': 0  # Index into scale array
            }

            self.current_params = self.target_params.copy()
            self.smooth_factor = 0.95
            self.pitch_smooth_factor = 0.9  # Faster for pitch changes

            # Don't start stream yet - wait until after MediaPipe loads
            self.running = False
            self.stream_started = False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio callback for PyAudio stream"""
        if self.no_audio or not self.running:
            return (np.zeros((frame_count, 2), dtype=np.float32).tobytes(), pyaudio.paContinue)

        try:
            # Smooth parameters
            with self.param_lock:
                for key in self.current_params:
                    self.current_params[key] = (
                        self.smooth_factor * self.current_params[key] +
                        (1 - self.smooth_factor) * self.target_params[key]
                    )

            # Generate audio
            output = np.zeros((frame_count, 2), dtype=np.float32)

            for i in range(frame_count):
                # Vibrato
                vib = np.sin(self.vib_phase) * self.current_params['vibrato'] * 0.1
                self.vib_phase += 2 * np.pi * self.vib_rate / self.sample_rate

                # Smooth frequency transitions for quantized pitch
                self.current_freq = (
                    self.pitch_smooth_factor * self.current_freq +
                    (1 - self.pitch_smooth_factor) * self.target_freq
                )

                # Apply vibrato to the smoothed frequency
                current_freq = self.current_freq * (1 + vib)

                # SuperSaw oscillator
                sample = 0.0
                for j, detune in enumerate(self.detune_factors):
                    freq = current_freq * detune
                    self.osc_phases[j] += 2 * np.pi * freq / self.sample_rate
                    if self.osc_phases[j] > 2 * np.pi:
                        self.osc_phases[j] -= 2 * np.pi
                    sample += np.sin(self.osc_phases[j]) * 0.2

                # FM synthesis
                mod_freq = current_freq * 2.0
                self.fm_phase += 2 * np.pi * mod_freq / self.sample_rate
                if self.fm_phase > 2 * np.pi:
                    self.fm_phase -= 2 * np.pi

                modulation = np.sin(self.fm_phase) * self.current_params['mod'] * current_freq
                fm_carrier = np.sin(self.phase + modulation) * 0.3
                self.phase += 2 * np.pi * current_freq / self.sample_rate
                if self.phase > 2 * np.pi:
                    self.phase -= 2 * np.pi

                sample += fm_carrier

                # Apply amplitude
                sample *= self.current_params['amp']

                # Simple lowpass filter
                alpha = min(1.0, self.current_params['filter'])
                filtered_l = alpha * sample + (1 - alpha) * self.filter_state_l
                self.filter_state_l = filtered_l
                filtered_r = alpha * sample + (1 - alpha) * self.filter_state_r
                self.filter_state_r = filtered_r

                # Pan
                left_gain = np.sqrt(1.0 - self.current_params['pan'])
                right_gain = np.sqrt(self.current_params['pan'])

                # Simple reverb (echo)
                delayed_l = self.delay_buffer_l[self.delay_idx]
                delayed_r = self.delay_buffer_r[self.delay_idx]

                output[i, 0] = filtered_l * left_gain + delayed_l * self.reverb_mix
                output[i, 1] = filtered_r * right_gain + delayed_r * self.reverb_mix

                # Update delay buffer
                self.delay_buffer_l[self.delay_idx] = filtered_l * 0.5
                self.delay_buffer_r[self.delay_idx] = filtered_r * 0.5
                self.delay_idx = (self.delay_idx + 1) % self.delay_size

            return (output.tobytes(), pyaudio.paContinue)
        except Exception as e:
            print(f"Audio callback error: {e}")
            return (np.zeros((frame_count, 2), dtype=np.float32).tobytes(), pyaudio.paContinue)

    def start_audio(self):
        """Start the audio stream after MediaPipe is initialized"""
        if not self.no_audio and not self.stream_started:
            print("Starting audio...")
            self.running = True
            self.stream.start_stream()
            self.stream_started = True

    def update_parameters(self, controls: Dict[str, float]):
        """Update synth parameters from control values"""
        if self.no_audio:
            return

        with self.param_lock:
            # Vibrato depth and amplitude from mouth openness
            self.target_params['vibrato'] = controls['mouth'] * 0.5
            self.target_params['amp'] = 0.3 + controls['mouth'] * 0.6

            # Filter cutoff from eyebrow raise
            self.target_params['filter'] = exp_scale(controls['brow'], 300, 9000) / self.sample_rate

            # Pan from head yaw
            self.target_params['pan'] = controls['yaw']

            # Quantized pitch from head roll (2 octaves)
            # Map roll from -15 to +15 degrees to scale indices
            roll_normalized = (controls['roll'] + 15.0) / 30.0  # 0 to 1

            # Select scale
            scale = self.major_scale if self.use_major_scale else self.chromatic_scale

            # Map to scale index
            scale_index = int(roll_normalized * (len(scale) - 1))
            scale_index = clamp(scale_index, 0, len(scale) - 1)

            # Set target frequency from scale
            self.target_freq = scale[scale_index]
            self.target_params['pitch_index'] = scale_index

            # Modulation index from smile
            self.target_params['mod'] = controls['smile'] * 5.0

    def get_latency(self) -> float:
        """Get estimated audio latency in ms"""
        if self.no_audio:
            return 0.0
        return (self.buffer_size / self.sample_rate) * 1000

    def close(self):
        """Clean shutdown"""
        if not self.no_audio:
            self.running = False
            time.sleep(0.1)  # Allow callback to finish
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()

# ==================== OSC Client ====================

class OSCClient:
    """Send control values via OSC"""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        if not OSC_AVAILABLE:
            raise ImportError("python-osc not installed")

        self.client = udp_client.SimpleUDPClient(host, port)
        self.enabled = False
        print(f"OSC client ready: {host}:{port}")

    def send_controls(self, controls: Dict[str, float]):
        """Send control bundle"""
        if not self.enabled:
            return

        # Send as bundle for atomicity
        bundle = osc_bundle_builder.OscBundleBuilder(
            osc_bundle_builder.IMMEDIATELY
        )

        for key, value in controls.items():
            msg = osc_message_builder.OscMessageBuilder(address=f"/faceSynth/{key}")
            msg.add_arg(float(value))
            bundle.add_content(msg.build())

        self.client.send(bundle.build())

# ==================== MIDI Client ====================

class MIDIClient:
    """Send control values via MIDI CC"""

    def __init__(self, port_name: Optional[str] = None):
        if not MIDI_AVAILABLE:
            raise ImportError("mido/rtmidi not installed")

        self.port = None
        self.enabled = False

        # Find or create port
        available_ports = mido.get_output_names()

        if port_name and port_name in available_ports:
            self.port = mido.open_output(port_name)
        elif available_ports:
            self.port = mido.open_output(available_ports[0])
        else:
            self.port = mido.open_output('FaceSynth Out', virtual=True)

        print(f"MIDI output ready: {self.port.name}")

    def send_controls(self, controls: Dict[str, float]):
        """Send MIDI CC messages"""
        if not self.enabled or not self.port:
            return

        # CC1 Mod Wheel - vibrato/mouth
        self.port.send(mido.Message(
            'control_change',
            control=1,
            value=int(controls['mouth'] * 127)
        ))

        # CC74 Brightness - filter/brow
        self.port.send(mido.Message(
            'control_change',
            control=74,
            value=int(controls['brow'] * 127)
        ))

        # CC10 Pan - yaw
        self.port.send(mido.Message(
            'control_change',
            control=10,
            value=int(controls['yaw'] * 127)
        ))

        # Pitch Bend - roll
        pitch_val = int(8192 + (controls['roll'] / 15.0) * 8191)
        pitch_val = clamp(pitch_val, 0, 16383)
        self.port.send(mido.Message(
            'pitchwheel',
            pitch=pitch_val
        ))

        # CC11 Expression - amplitude
        amp_val = 0.3 + controls['mouth'] * 0.6
        self.port.send(mido.Message(
            'control_change',
            control=11,
            value=int(amp_val * 127)
        ))

    def close(self):
        """Clean up"""
        if self.port:
            self.port.close()

# ==================== UI Renderer ====================

class UIRenderer:
    """Handle OpenCV visualization"""

    def __init__(self):
        self.show_graphics = True
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def draw_frame(self, image: np.ndarray, features: Optional[Dict],
                   controls: Dict[str, float], fps: float, latency: float):
        """Draw full UI overlay"""
        h, w = image.shape[:2]

        if self.show_graphics and features and 'landmarks' in features:
            # Draw face mesh
            self.mp_drawing.draw_landmarks(
                image,
                features['landmarks'],
                mp.solutions.face_mesh.FACEMESH_TESSELATION,
                None,
                self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # Highlight key landmarks
            landmarks = features['landmarks'].landmark

            # Draw feature points
            points = [
                (UPPER_LIP_CENTER, COLOR_MOUTH),
                (LOWER_LIP_CENTER, COLOR_MOUTH),
                (LEFT_EYEBROW, COLOR_EYEBROW),
                (LEFT_EYE_CENTER, COLOR_EYEBROW),
                (NOSE_TIP, COLOR_YAW),
                (LEFT_EYE_OUTER, COLOR_ROLL),
                (RIGHT_EYE_OUTER, COLOR_ROLL),
                (MOUTH_LEFT, COLOR_SMILE),
                (MOUTH_RIGHT, COLOR_SMILE)
            ]

            for idx, color in points:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(image, (x, y), 3, color, -1)

        # Draw meters
        meter_y = h - 140  # Moved up to make room for pitch display
        meter_height = 15
        meter_width = 200
        meter_spacing = 20

        meters = [
            ("Mouth", controls['mouth'], COLOR_MOUTH),
            ("Brow", controls['brow'], COLOR_EYEBROW),
            ("Yaw", controls['yaw'], COLOR_YAW),
            ("Roll", (controls['roll'] + 15) / 30.0, COLOR_ROLL),
            ("Smile", controls['smile'], COLOR_SMILE)
        ]

        # Add pitch indicator if available
        note_names = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        pitch_idx = int(((controls['roll'] + 15) / 30.0) * 24)  # 0-24 for 2 octaves
        pitch_idx = max(0, min(24, pitch_idx))
        note_name = note_names[pitch_idx % 12]
        octave = 3 + pitch_idx // 12
        cv2.putText(image, f"Pitch: {note_name}{octave}", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for i, (label, value, color) in enumerate(meters):
            y = meter_y + i * meter_spacing

            # Background
            cv2.rectangle(image, (10, y), (10 + meter_width, y + meter_height),
                         COLOR_METER_BG, -1)

            # Foreground
            fill_width = int(value * meter_width)
            cv2.rectangle(image, (10, y), (10 + fill_width, y + meter_height),
                         color, -1)

            # Label
            cv2.putText(image, f"{label}: {value:.2f}", (220, y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw stats
        cv2.putText(image, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(image, f"Audio Latency: {latency:.1f}ms", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Controls legend
        legend = [
            "C: Calibrate",
            "S: Toggle OSC",
            "M: Toggle MIDI",
            "G: Toggle Graphics",
            "P: Toggle Scale",
            "Q: Quit"
        ]
        for i, text in enumerate(legend):
            cv2.putText(image, text, (w - 150, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return image

# ==================== Demo Mode ====================

def run_demo_mode(synth: SynthEngine, osc_client: Optional[OSCClient],
                  midi_client: Optional[MIDIClient]):
    """Run synthetic demo mode without camera"""
    print("\n=== DEMO MODE ===")
    print("Sweeping control values over 10 seconds...")

    start_time = time.time()
    duration = 10.0

    while True:
        elapsed = time.time() - start_time
        if elapsed > duration:
            break

        t = elapsed / duration

        # Generate sine sweeps at different rates
        controls = {
            'mouth': (np.sin(t * np.pi * 4) + 1) / 2,
            'brow': (np.sin(t * np.pi * 3) + 1) / 2,
            'yaw': (np.sin(t * np.pi * 2) + 1) / 2,
            'roll': np.sin(t * np.pi * 2) * 15,
            'smile': (np.sin(t * np.pi * 5) + 1) / 2
        }

        # Update outputs
        synth.update_parameters(controls)

        if osc_client and osc_client.enabled:
            osc_client.send_controls(controls)

        if midi_client and midi_client.enabled:
            midi_client.send_controls(controls)

        # Print values
        print(f"\rT:{t:.2f} " + " ".join(f"{k[:3]}:{v:.2f}" for k, v in controls.items()), end="")

        time.sleep(1/30)

    print("\nDemo complete!")

# ==================== Main Application ====================

def check_camera_permissions():
    """Check if we have camera permissions on macOS"""
    if sys.platform == "darwin":  # macOS
        import subprocess
        try:
            # Try to check camera authorization status
            result = subprocess.run(
                ["tccutil", "check", "Camera"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except:
            # tccutil might not be available or might need admin rights
            return None
    return True

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="FaceMesh Synth Controller")
    parser.add_argument("--camera-index", type=int, default=0,
                       help="Camera index (default: 0)")
    parser.add_argument("--osc-host", type=str, default="127.0.0.1",
                       help="OSC host (default: 127.0.0.1)")
    parser.add_argument("--osc-port", type=int, default=9000,
                       help="OSC port (default: 9000)")
    parser.add_argument("--midi-port", type=str, default=None,
                       help="MIDI port name")
    parser.add_argument("--no-audio", action="store_true",
                       help="Disable audio output")
    parser.add_argument("--smooth-alpha", type=float, default=0.2,
                       help="Smoothing factor (default: 0.2)")
    parser.add_argument("--deadband", type=float, default=0.03,
                       help="Deadband threshold (default: 0.03)")
    parser.add_argument("--demo", action="store_true",
                       help="Run demo mode without camera")

    args = parser.parse_args()

    # Initialize components first
    print("Initializing audio engine...")
    synth = SynthEngine(no_audio=args.no_audio)

    print("Initializing OSC/MIDI...")
    osc_client = None
    if OSC_AVAILABLE:
        try:
            osc_client = OSCClient(args.osc_host, args.osc_port)
        except Exception as e:
            print(f"OSC client failed: {e}")

    midi_client = None
    if MIDI_AVAILABLE:
        try:
            midi_client = MIDIClient(args.midi_port)
        except Exception as e:
            print(f"MIDI client failed: {e}")

    # Run demo mode if requested
    if args.demo:
        synth.start_audio()  # Start audio for demo
        run_demo_mode(synth, osc_client, midi_client)
        synth.close()
        return

    print("Initializing face tracking (this may take a moment)...")
    extractor = FaceFeatureExtractor()
    mapper = MappingController(args.smooth_alpha, args.deadband)
    ui = UIRenderer()

    print("Face tracking initialized.")

    # Open camera with permission handling
    print("\n=== Camera Access Required ===")
    print("If prompted, please grant camera access in System Preferences")
    print("macOS: System Preferences > Security & Privacy > Camera")
    print("Add Terminal or your Python IDE to allowed apps\n")

    # Try different camera backends
    cap = None
    backends = [
        (cv2.CAP_AVFOUNDATION, "AVFoundation"),
        (cv2.CAP_ANY, "Default"),
    ]

    for backend, name in backends:
        print(f"Trying camera with {name} backend...")
        cap = cv2.VideoCapture(args.camera_index, backend)
        if cap.isOpened():
            print(f"Successfully opened camera with {name}")
            break
        cap.release()

    if not cap or not cap.isOpened():
        print(f"\nError: Cannot open camera {args.camera_index}")
        print("\nTroubleshooting:")
        print("1. Grant camera permission to Terminal/Python in System Preferences")
        print("2. Try running with: python face_synth.py --demo")
        print("3. Check if another app is using the camera")
        return

    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Now that MediaPipe is fully initialized, start audio
    synth.start_audio()

    # Main loop
    print("\n=== FaceMesh Synth Controller Started ===")
    print("Controls:")
    print("  C: Calibrate facial features")
    print("  P: Toggle between Chromatic/Major scale")
    print("  S: Toggle OSC output")
    print("  M: Toggle MIDI output")
    print("  G: Toggle graphics overlay")
    print("  Q: Quit")
    print("\nCamera resolution:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("\nHead Roll → Pitch (2 octaves, quantized to scale)")
    print("Current scale: Chromatic")

    cv2.namedWindow("FaceMesh Synth Controller")

    frame_count = 0
    fps = 0
    fps_time = time.time()
    calibration_start = None

    try:
        while True:
            # Timing
            loop_start = time.time()

            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break

            # Process frame
            frame = cv2.flip(frame, 1)  # Mirror for natural interaction
            features = extractor.extract_features(frame)
            controls = mapper.process_features(features)

            # Update synth
            synth.update_parameters(controls)

            # Send OSC/MIDI
            if osc_client and osc_client.enabled:
                osc_client.send_controls(controls)

            if midi_client and midi_client.enabled:
                midi_client.send_controls(controls)

            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_time)
                fps_time = time.time()

            # Draw UI
            frame = ui.draw_frame(frame, features, controls, fps, synth.get_latency())

            # Auto-finish calibration after 12 seconds
            if mapper.calibrating:
                if calibration_start is None:
                    calibration_start = time.time()
                elif time.time() - calibration_start > 12:
                    mapper.finish_calibration()
                    calibration_start = None

            cv2.imshow("FaceMesh Synth Controller", frame)

            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                mapper.start_calibration()
            elif key == ord('s'):
                if osc_client:
                    osc_client.enabled = not osc_client.enabled
                    print(f"OSC: {'ON' if osc_client.enabled else 'OFF'}")
            elif key == ord('m'):
                if midi_client:
                    midi_client.enabled = not midi_client.enabled
                    print(f"MIDI: {'ON' if midi_client.enabled else 'OFF'}")
            elif key == ord('g'):
                ui.show_graphics = not ui.show_graphics
            elif key == ord('p'):  # Toggle between chromatic and major scale
                if not synth.no_audio:
                    synth.use_major_scale = not synth.use_major_scale
                    scale_name = "Major" if synth.use_major_scale else "Chromatic"
                    print(f"Pitch scale: {scale_name}")

            # Frame rate limiting
            elapsed = time.time() - loop_start
            if elapsed < 1/60:
                time.sleep(1/60 - elapsed)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        print("\nShutting down...")
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        synth.close()

        if midi_client:
            midi_client.close()

        print("Goodbye!")

if __name__ == "__main__":
    main()

"""
Next Steps:
- Multiple voices with chord quantization for harmonic layers
- Gesture-based triggers for rhythmic events and sequences
- HRTF-based 3D panning for immersive spatial audio
- Pose-stabilized face cropping for improved tracking robustness
"""