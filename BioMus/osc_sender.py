# osc_sender.py
from typing import List, Dict, Optional
from pythonosc.udp_client import SimpleUDPClient
import logging
import numpy as np

logger = logging.getLogger(__name__)


class OSCSender:
    def __init__(self, ip: str = "127.0.0.1", port: int = 9000):
        self.ip = ip
        self.port = port
        self.client: Optional[SimpleUDPClient] = None
        self.enabled: bool = False
        self.send_raw: bool = True
        self.send_bands_enabled: bool = False
        # Max floats per OSC message to avoid UDP size limits
        # Conservative limit: ~1000 floats (4000 bytes) to stay well under UDP limits
        self.max_floats_per_message = 1000

        # Fixed normalization ranges based on neuroscience research (in µV²)
        # These are typical ranges for adult EEG during resting/active states
        self.band_ranges = {
            'delta': (0.5, 100.0),    # 0.5-4 Hz
            'theta': (0.5, 50.0),     # 4-8 Hz
            'alpha': (1.0, 100.0),    # 8-13 Hz
            'beta': (0.5, 30.0),      # 13-30 Hz
            'gamma': (0.5, 20.0)      # 30-50 Hz
        }

    def configure(self, ip: str, port: int, enabled: bool,
                  send_raw: bool, send_bands: bool):
        self.ip = ip
        self.port = port
        self.enabled = enabled
        self.send_raw = send_raw
        self.send_bands_enabled = send_bands
        if enabled:
            self.client = SimpleUDPClient(self.ip, self.port)
        else:
            self.client = None

    def _ensure_client(self):
        if not self.enabled or self.client is None:
            return False
        return True

    def send_timeseries(self, channel_names: List[str], data: List[List[float]]):
        """
        data: list of channels, each is a list of samples (same length).
        Sends both per-channel messages and combined message:
        - /biomus/raw/CH1, /biomus/raw/CH2, etc. (per-channel)
        - /biomus/raw (all channels combined)
        If even a single channel is too large, we chunk it.
        """
        if not self._ensure_client() or not self.send_raw:
            return

        try:
            # Send per-channel messages
            for ch_idx, ch_data in enumerate(data):
                ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"CH{ch_idx+1}"

                # If channel data is small enough, send as single message
                if len(ch_data) <= self.max_floats_per_message:
                    self.client.send_message(f"/biomus/raw/{ch_name}", ch_data)
                else:
                    # Chunk the data
                    for chunk_idx, i in enumerate(range(0, len(ch_data), self.max_floats_per_message)):
                        chunk = ch_data[i:i + self.max_floats_per_message]
                        self.client.send_message(f"/biomus/raw/{ch_name}/chunk{chunk_idx}", chunk)

            # Send combined message with all channels
            # Flatten all channel data into a single array
            # Format: [CH1_samples..., CH2_samples..., CH3_samples..., CH4_samples...]
            all_data = []
            for ch_data in data:
                all_data.extend(ch_data)

            if len(all_data) <= self.max_floats_per_message:
                self.client.send_message("/biomus/raw", all_data)
            else:
                # Chunk the combined data
                for chunk_idx, i in enumerate(range(0, len(all_data), self.max_floats_per_message)):
                    chunk = all_data[i:i + self.max_floats_per_message]
                    self.client.send_message(f"/biomus/raw/chunk{chunk_idx}", chunk)

        except Exception as e:
            logger.error(f"Failed to send timeseries OSC: {e}")

    def _normalize_band_value(self, band_name: str, value: float) -> float:
        """
        Normalize a band power value to 0-1 range using fixed research-based ranges.
        """
        min_val, max_val = self.band_ranges.get(band_name.lower(), (0.0, 100.0))
        normalized = (value - min_val) / (max_val - min_val)
        # Clamp to 0-1 range
        return max(0.0, min(1.0, normalized))

    def send_bands(
        self,
        channel_names: List[str],
        bands: List[str],
        values: List[List[float]],
    ):
        """
        values: shape [n_channels x n_bands]
        Sends comprehensive band power messages:
        - Per-channel individual bands: /biomus/bands/CH1/delta, /biomus/bands/CH1/theta, etc.
        - Per-channel relative: /biomus/bands/CH1/delta-relative, etc. (0-1 normalized)
        - Cross-channel averages: /biomus/bands/delta, /biomus/bands/theta, etc.
        - Cross-channel stats: /biomus/bands/delta/max, /biomus/bands/delta/min, etc.
        - Muse-compatible combined: /biomus/elements/delta_absolute, /biomus/elements/delta_relative, etc.
        """
        if not self._ensure_client() or not self.send_bands_enabled:
            return

        try:
            n_channels = len(values)
            n_bands = len(bands)

            # Convert to numpy for easier calculations
            values_array = np.array(values)  # shape: [n_channels, n_bands]

            # 1. Send per-channel individual band powers (absolute and relative)
            for ch_idx, ch_vals in enumerate(values):
                ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"CH{ch_idx+1}"

                for band_idx, band_name in enumerate(bands):
                    abs_value = ch_vals[band_idx]

                    # Send absolute value
                    self.client.send_message(f"/biomus/bands/{ch_name}/{band_name}", abs_value)

                    # Send relative (normalized) value
                    rel_value = self._normalize_band_value(band_name, abs_value)
                    self.client.send_message(f"/biomus/bands/{ch_name}/{band_name}-relative", rel_value)

            # 2. Send cross-channel aggregates (mean, max, min)
            for band_idx, band_name in enumerate(bands):
                band_values = values_array[:, band_idx]  # All channels for this band

                # Mean across channels
                mean_val = float(np.mean(band_values))
                self.client.send_message(f"/biomus/bands/{band_name}", mean_val)

                # Max and min across channels
                max_val = float(np.max(band_values))
                min_val = float(np.min(band_values))
                self.client.send_message(f"/biomus/bands/{band_name}/max", max_val)
                self.client.send_message(f"/biomus/bands/{band_name}/min", min_val)

            # 3. Send Muse-compatible combined messages (all 4 channels in one message)
            # Format: /biomus/elements/<band>_absolute sends [CH1, CH2, CH3, CH4]
            # Format: /biomus/elements/<band>_relative sends [CH1_rel, CH2_rel, CH3_rel, CH4_rel]
            for band_idx, band_name in enumerate(bands):
                # Absolute values: all channels for this band
                abs_values = [float(values[ch_idx][band_idx]) for ch_idx in range(n_channels)]
                self.client.send_message(f"/biomus/elements/{band_name}_absolute", abs_values)

                # Relative values: normalized for all channels
                rel_values = [self._normalize_band_value(band_name, values[ch_idx][band_idx])
                             for ch_idx in range(n_channels)]
                self.client.send_message(f"/biomus/elements/{band_name}_relative", rel_values)

        except Exception as e:
            logger.error(f"Failed to send bands OSC: {e}")
