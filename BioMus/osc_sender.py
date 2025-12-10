# osc_sender.py
from typing import List, Dict, Optional
from pythonosc.udp_client import SimpleUDPClient
import logging

logger = logging.getLogger(__name__)


class OSCSender:
    def __init__(self, ip: str = "127.0.0.1", port: int = 9000):
        self.ip = ip
        self.port = port
        self.client: Optional[SimpleUDPClient] = None
        self.enabled: bool = False
        self.send_raw: bool = True
        self.send_bands: bool = False
        # Max floats per OSC message to avoid UDP size limits
        # Conservative limit: ~1000 floats (4000 bytes) to stay well under UDP limits
        self.max_floats_per_message = 1000

    def configure(self, ip: str, port: int, enabled: bool,
                  send_raw: bool, send_bands: bool):
        self.ip = ip
        self.port = port
        self.enabled = enabled
        self.send_raw = send_raw
        self.send_bands = send_bands
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
        We send per-channel messages to avoid UDP size limits: /eeg/raw/CH1, /eeg/raw/CH2, etc.
        If even a single channel is too large, we chunk it.
        """
        if not self._ensure_client() or not self.send_raw:
            return

        try:
            for ch_idx, ch_data in enumerate(data):
                ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"CH{ch_idx+1}"

                # If channel data is small enough, send as single message
                if len(ch_data) <= self.max_floats_per_message:
                    self.client.send_message(f"/eeg/raw/{ch_name}", ch_data)
                else:
                    # Chunk the data
                    for chunk_idx, i in enumerate(range(0, len(ch_data), self.max_floats_per_message)):
                        chunk = ch_data[i:i + self.max_floats_per_message]
                        self.client.send_message(f"/eeg/raw/{ch_name}/chunk{chunk_idx}", chunk)
        except Exception as e:
            logger.error(f"Failed to send timeseries OSC: {e}")

    def send_bands(
        self,
        channel_names: List[str],
        bands: List[str],
        values: List[List[float]],
    ):
        """
        values: shape [n_channels x n_bands]
        We send per-channel messages: /eeg/bands/CH1, /eeg/bands/CH2, etc.
        Each message contains [delta, theta, alpha, beta, gamma] for that channel.
        """
        if not self._ensure_client() or not self.send_bands:
            return

        try:
            for ch_idx, ch_vals in enumerate(values):
                ch_name = channel_names[ch_idx] if ch_idx < len(channel_names) else f"CH{ch_idx+1}"
                # Band powers are typically just 5 values per channel, safe to send
                self.client.send_message(f"/eeg/bands/{ch_name}", ch_vals)
        except Exception as e:
            logger.error(f"Failed to send bands OSC: {e}")
