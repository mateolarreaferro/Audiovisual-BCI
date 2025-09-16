import time
import math
import threading
import logging
import glob
import platform
from collections import deque

import numpy as np
import pyaudio
from scipy import signal

# Lab Streaming Layer
from pylsl import StreamInlet, resolve_streams

# BrainFlow (for signal processing only)
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowOperations, DetrendOperations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bci_csound.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# -----------------------------
# CONFIG
# -----------------------------

def connect_lsl():
    """Connect to LSL streams from OpenBCI GUI"""
    logger.info("Looking for LSL streams from OpenBCI GUI...")

    streams = resolve_streams(wait_time=5.0)

    if not streams:
        logger.error("No LSL streams found. Make sure OpenBCI GUI is running and streaming.")
        logger.info("Instructions:")
        logger.info("1. Open OpenBCI GUI")
        logger.info("2. Connect to your board")
        logger.info("3. Start streaming")
        logger.info("4. Enable LSL output: Stream 1=TimeSeriesRaw, Stream 2=Accel/Aux")
        return None, None, None, None

    # Find EEG and Accelerometer streams
    eeg_inlet = None
    accel_inlet = None
    sample_rate = None
    n_eeg_channels = None

    for stream in streams:
        stream_info = stream
        stream_name = stream_info.name()
        stream_type = stream_info.type()
        channel_count = stream_info.channel_count()

        logger.info(f"Found stream: {stream_name} (type: {stream_type}, channels: {channel_count})")

        # Identify streams based on name pattern and channel count
        # obci_eeg1 = TimeSeriesRaw (16 channels), obci_eeg2 = Accel/Aux (3 channels)
        if stream_name == "obci_eeg1" or (stream_type == "EEG" and channel_count > 3):
            eeg_inlet = StreamInlet(stream_info)
            sample_rate = stream_info.nominal_srate()
            n_eeg_channels = channel_count
            logger.info(f"Using as EEG stream: {stream_name}, {sample_rate}Hz, {n_eeg_channels} channels")
        elif stream_name == "obci_eeg2" or channel_count == 3:
            accel_inlet = StreamInlet(stream_info)
            logger.info(f"Using as Accelerometer stream: {stream_name}, {channel_count} channels")

    if eeg_inlet is None:
        logger.error("No EEG stream found. Make sure Stream 1 is set to TimeSeriesRaw with type EEG")
        return None, None, None, None

    if accel_inlet is None:
        logger.warning("No accelerometer stream found. Reverb control will be disabled.")

    return eeg_inlet, accel_inlet, sample_rate, n_eeg_channels

# Connect to LSL streams
EEG_INLET, ACCEL_INLET, SAMPLE_RATE, N_EEG_CHANNELS = connect_lsl()

if EEG_INLET is None:
    logger.error("Failed to connect to EEG LSL stream. Please configure OpenBCI GUI LSL properly.")
    exit(1)

# EEG channels from the EEG stream
EEG_CHANNELS = list(range(N_EEG_CHANNELS))

# Accelerometer channels (if available from separate stream)
if ACCEL_INLET:
    ACCEL_CHANNELS = [0, 1, 2]  # X, Y, Z from accelerometer stream
else:
    ACCEL_CHANNELS = []

# Alpha band (Hz)
ALPHA_LO, ALPHA_HI = 8.0, 12.0

# Optional: choose posterior-ish channels if you know your montage (example indices within EEG_CHANNELS list)
# Otherwise we'll average all EEG channels.
ALPHA_PREFERRED = None  # e.g., [6,7,8,9] to point into EEG_CHANNELS

# Pitch mapping (Hz)
FREQ_MIN, FREQ_MAX = 220.0, 880.0

# Reverb wet mapping from accelerometer X absolute value (g’s)
ACCEL_WET_MAX_G = 1.2   # cap at ~1.2 g for head/cap movement (tune)

# Smoothing
ALPHA_SMOOTH = 0.2      # 0..1 EMA for alpha ratio
ACCEL_SMOOTH = 0.3      # EMA for accel X

# PSD window
PSD_WINDOW = WindowOperations.HAMMING.value
PSD_SEGMENT_LEN = int(2.0 * SAMPLE_RATE)     # 2s windows
PSD_OVERLAP = int(0.5 * PSD_SEGMENT_LEN)     # 50% overlap

# Control rate (how often we push values into Csound, seconds)
CONTROL_PERIOD = 0.05   # 20 Hz


# -----------------------------
# AUDIO SYNTHESIS
# -----------------------------

class AudioSynthesizer:
    """Simple real-time audio synthesizer with saw wave and reverb"""

    def __init__(self, sample_rate=44100, buffer_size=512):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size

        # Synthesis parameters (controlled by BCI)
        self.frequency = 440.0
        self.amplitude = 0.2
        self.reverb_wet = 0.0

        # Oscillator phase
        self.phase = 0.0

        # Simple reverb delay lines (comb filters)
        self.delay_times = [0.029, 0.031, 0.037, 0.041]  # in seconds
        self.delay_buffers = []
        self.delay_indices = []
        for delay_time in self.delay_times:
            buffer_len = int(delay_time * sample_rate)
            self.delay_buffers.append(np.zeros(buffer_len))
            self.delay_indices.append(0)

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()
        self.stream = None

    def generate_saw_wave(self, num_samples):
        """Generate antialiased saw wave"""
        samples = np.zeros(num_samples)
        phase_increment = self.frequency / self.sample_rate

        for i in range(num_samples):
            # Simple saw wave (-1 to 1)
            samples[i] = 2.0 * (self.phase - 0.5)

            # Update phase
            self.phase += phase_increment
            if self.phase >= 1.0:
                self.phase -= 1.0

        return samples * self.amplitude

    def apply_reverb(self, dry_signal):
        """Apply simple reverb using comb filters"""
        wet_signal = np.zeros_like(dry_signal)

        # Process through each comb filter
        for idx, (buffer, buffer_idx) in enumerate(zip(self.delay_buffers, self.delay_indices)):
            buffer_len = len(buffer)

            for i in range(len(dry_signal)):
                # Read from delay buffer
                delayed = buffer[buffer_idx]

                # Mix with input and feedback
                buffer[buffer_idx] = dry_signal[i] + delayed * 0.5

                # Add to wet signal
                wet_signal[i] += delayed

                # Update buffer index
                buffer_idx = (buffer_idx + 1) % buffer_len

            self.delay_indices[idx] = buffer_idx

        # Normalize and mix dry/wet
        wet_signal = wet_signal / len(self.delay_buffers)
        return dry_signal * (1 - self.reverb_wet) + wet_signal * self.reverb_wet

    def audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback for real-time audio generation"""
        # Generate saw wave
        dry_signal = self.generate_saw_wave(frame_count)

        # Apply reverb
        output = self.apply_reverb(dry_signal)

        # Convert to stereo and format for PyAudio
        stereo = np.column_stack((output, output))
        return (stereo.astype(np.float32).tobytes(), pyaudio.paContinue)

    def start(self):
        """Start audio stream"""
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.buffer_size,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()
        logger.info("Audio synthesis started")

    def stop(self):
        """Stop audio stream"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()
        logger.info("Audio synthesis stopped")

    def set_parameters(self, frequency=None, amplitude=None, reverb_wet=None):
        """Update synthesis parameters"""
        if frequency is not None:
            self.frequency = np.clip(frequency, 20, 20000)
        if amplitude is not None:
            self.amplitude = np.clip(amplitude, 0, 1)
        if reverb_wet is not None:
            self.reverb_wet = np.clip(reverb_wet, 0, 1)


# -----------------------------
# Helper functions
# -----------------------------

def ema(prev, x, alpha):
    """Exponential moving average."""
    return alpha * x + (1 - alpha) * prev


def safe_norm(val, lo, hi):
    """Clamp then normalize to 0..1."""
    v = max(lo, min(hi, val))
    return (v - lo) / (hi - lo)


def alpha_ratio_from_block(eeg_block, fs):
    """
    Compute an alpha-relative power metric across channels.
    Uses Welch PSD via BrainFlow; returns ratio in 0..1-ish.
    """
    n_ch, n_samp = eeg_block.shape
    logger.debug(f"Processing EEG block: {n_ch} channels, {n_samp} samples")
    ratios = []

    for ch in range(n_ch):
        sig = np.ascontiguousarray(eeg_block[ch].astype(np.float64))

        try:
            # Ensure array is contiguous for BrainFlow functions
            sig = np.ascontiguousarray(sig)
            # Detrend, notch, and bandlimit a bit to reduce drift/line noise (optional)
            DataFilter.detrend(sig, DetrendOperations.CONSTANT.value)
            # Notch filter for 60Hz noise (US power line frequency)
            # Using direct bandstop filter instead of environmental noise filter
            notch_sos = signal.butter(2, [59, 61], btype='bandstop', fs=fs, output='sos')
            sig = signal.sosfiltfilt(notch_sos, sig)

            # Ensure array is contiguous after scipy filtering
            sig = np.ascontiguousarray(sig)

            # PSD (Welch) - ensure all parameters are integers
            psd, freqs = DataFilter.get_psd_welch(
                sig,
                PSD_WINDOW,
                PSD_SEGMENT_LEN,
                PSD_OVERLAP,
                int(fs)
            )

            # Integrate alpha band
            alpha = DataFilter.get_band_power(psd, freqs, ALPHA_LO, ALPHA_HI)

            # Integrate a broad total (delta..gamma) to ratio out amplitude differences
            total = DataFilter.get_band_power(psd, freqs, 1.0, 45.0)

            ratio = (alpha / total) if total > 1e-12 else 0.0
            ratios.append(ratio)
            logger.debug(f"Channel {ch}: alpha={alpha:.6f}, total={total:.6f}, ratio={ratio:.6f}")

        except Exception as e:
            logger.error(f"Error processing channel {ch}: {e}")
            ratios.append(0.0)

    mean_ratio = float(np.mean(ratios))
    logger.debug(f"Mean alpha ratio across {n_ch} channels: {mean_ratio:.6f}")
    return mean_ratio


def hz_from_alpha_ratio(ratio):
    """Map alpha ratio (approx 0..0.6) to frequency range."""
    # Stretch and clamp a bit
    x = safe_norm(ratio, 0.05, 0.55)
    return FREQ_MIN + x * (FREQ_MAX - FREQ_MIN)


def wet_from_accel_x_g(ax_g):
    """Map |accel X| in g to reverb wet (0..1)."""
    x = abs(ax_g)
    x = min(x, ACCEL_WET_MAX_G)
    return x / ACCEL_WET_MAX_G


# -----------------------------
# Main
# -----------------------------

def main():
    global EEG_INLET, ACCEL_INLET  # Access global LSL inlets

    logger.info("Starting BCI-Audio mapper...")
    logger.info(f"Sample rate: {SAMPLE_RATE} Hz")
    logger.info(f"EEG channels: {EEG_CHANNELS}")
    logger.info(f"Accel channels: {ACCEL_CHANNELS}")

    # Fixed volume level (you can adjust this or make it a command-line argument)
    VOLUME = 0.3  # Range: 0.0 to 1.0
    logger.info(f"Volume level: {VOLUME}")

    # Initialize audio synthesizer
    synth = None
    try:
        logger.info("Initializing audio synthesizer...")
        synth = AudioSynthesizer(sample_rate=44100, buffer_size=512)
        synth.amplitude = VOLUME  # Set initial volume
        synth.start()
        logger.info("Audio synthesizer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize audio synthesizer: {e}")
        return

    logger.info("Starting main processing loop... Press Ctrl+C to stop.")
    logger.info("Tip: Adjust VOLUME constant in the code (0.0 to 1.0) to change loudness")
    try:
        # EMA states
        alpha_ema = 0.2
        accel_ema = 0.0

        # Data buffers for 2s window
        block_secs = 2.0
        block_len = int(block_secs * SAMPLE_RATE)
        eeg_buffer = deque(maxlen=block_len)
        accel_buffer = deque(maxlen=100)  # Smaller buffer for accelerometer
        logger.info(f"Using {block_secs}s blocks ({block_len} samples)")

        loop_count = 0
        while True:
            time.sleep(CONTROL_PERIOD)

            # Pull EEG data
            eeg_chunk, _ = EEG_INLET.pull_chunk(timeout=0.0, max_samples=32)
            if eeg_chunk:
                for sample in eeg_chunk:
                    eeg_buffer.append(sample)

            # Pull accelerometer data (if available)
            current_accel = None
            if ACCEL_INLET:
                accel_chunk, _ = ACCEL_INLET.pull_chunk(timeout=0.0, max_samples=10)
                if accel_chunk:
                    for sample in accel_chunk:
                        accel_buffer.append(sample)
                    # Use most recent accelerometer sample
                    if len(accel_buffer) > 0:
                        current_accel = accel_buffer[-1]

            # Check if we have enough EEG data
            if len(eeg_buffer) < block_len:
                if loop_count % 100 == 0:  # Log every 5 seconds at 20Hz
                    logger.debug(f"Waiting for EEG data: {len(eeg_buffer)}/{block_len} samples available")
                loop_count += 1
                continue

            # Convert EEG buffer to numpy array (samples x channels)
            eeg_data = np.array(eeg_buffer).T  # Transpose to (channels x samples)
            logger.debug(f"Retrieved {eeg_data.shape[1]} EEG samples")

            # EEG block shape: (len(EEG_CHANNELS), block_len)
            eeg_block = eeg_data[EEG_CHANNELS, :]

            # If you want "posterior only", pick subset
            if ALPHA_PREFERRED is not None:
                pick = np.array(ALPHA_PREFERRED, dtype=int)
                eeg_block = eeg_block[pick, :]
                logger.debug(f"Using preferred alpha channels: {ALPHA_PREFERRED}")

            # Accel (ax, ay, az) - from separate accelerometer stream
            if current_accel is not None and len(current_accel) >= 3:
                ax = float(current_accel[0]) / 1000.0  # Convert mG to g
                logger.debug(f"Accelerometer X (raw): {current_accel[0]:.3f}, converted: {ax:.6f} g")
            else:
                # No accelerometer data available, use default
                ax = 0.0

            # Alpha ratio
            alpha_r = alpha_ratio_from_block(eeg_block, SAMPLE_RATE)
            alpha_ema = ema(alpha_ema, alpha_r, ALPHA_SMOOTH)

            # Map → freq
            freq = hz_from_alpha_ratio(alpha_ema)

            # Accel X → wet
            accel_ema = ema(accel_ema, ax, ACCEL_SMOOTH)
            wet = wet_from_accel_x_g(accel_ema)
            wet = max(0.0, min(1.0, wet))

            # Update synthesizer parameters (no amplitude change - using fixed volume)
            try:
                synth.set_parameters(
                    frequency=float(freq),
                    reverb_wet=float(wet)
                    # amplitude stays constant at VOLUME level
                )
            except Exception as e:
                logger.error(f"Error updating synthesizer parameters: {e}")

            # Log current values every second (20 loops at 20Hz)
            if loop_count % 20 == 0:
                logger.info(f"Alpha ratio: {alpha_ema:.3f} | Freq: {freq:6.1f} Hz | "
                           f"Accel X: {accel_ema:+.3f} g | Reverb: {wet:.2f}")

            loop_count += 1

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, stopping...")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")
    finally:
        logger.info("Cleaning up...")
        try:
            # Close LSL inlets
            del EEG_INLET
            if ACCEL_INLET:
                del ACCEL_INLET
            logger.info("LSL connections closed")
        except Exception as e:
            logger.error(f"Error closing LSL connections: {e}")

        try:
            if synth:
                synth.stop()
            logger.info("Audio synthesizer stopped")
        except Exception as e:
            logger.error(f"Error stopping audio synthesizer: {e}")

        logger.info("BCI-Audio mapper stopped successfully")


if __name__ == "__main__":
    main()
