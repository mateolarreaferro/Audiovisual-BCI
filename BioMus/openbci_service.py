# openbci_service.py
import threading
from typing import Optional, List, Tuple

import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations

from osc_sender import OSCSender


class GanglionService:
    def __init__(self):
        self.board: Optional[BoardShim] = None
        self.board_id = None  # Will be set based on connection type
        self.params = BrainFlowInputParams()
        self.connected = False
        self.streaming = False
        self.lock = threading.Lock()
        self.osc = OSCSender()

    # ---------- Connection control ----------

    def connect(self, serial_port: str = "", mac_address: str = "", timeout: int = 15):
        """
        Connect to Ganglion using either:
        - Native Bluetooth (GANGLION_NATIVE_BOARD): Leave serial_port empty, optionally set mac_address
        - BLED112 Dongle (GANGLION_BOARD): Set serial_port
        """
        if self.connected:
            return

        # Determine board type based on connection method
        if serial_port:
            # Using BLED112 dongle - serial connection
            self.board_id = BoardIds.GANGLION_BOARD
            self.params.serial_port = serial_port
            self.params.mac_address = ""
        else:
            # Using native Bluetooth
            self.board_id = BoardIds.GANGLION_NATIVE_BOARD
            self.params.serial_port = ""
            self.params.mac_address = mac_address  # Optional - BrainFlow will autodiscover if empty

        self.params.timeout = timeout

        BoardShim.enable_dev_board_logger()
        self.board = BoardShim(self.board_id, self.params)
        self.board.prepare_session()
        self.connected = True

    def disconnect(self):
        if not self.connected or self.board is None:
            return
        if self.streaming:
            self.stop_stream()
        self.board.release_session()
        self.board = None
        self.connected = False

    # ---------- Streaming control ----------

    def start_stream(self, buffer_size: int = 45000):
        """
        buffer_size in number of data points. 45000 is a typical default.
        """
        if not self.connected or self.board is None:
            raise RuntimeError("Board not connected")
        if self.streaming:
            return
        self.board.start_stream(buffer_size)
        self.streaming = True

    def stop_stream(self):
        if not self.streaming or self.board is None:
            return
        self.board.stop_stream()
        self.streaming = False

    # ---------- Test / config ----------

    def send_test_signal_on(self):
        """
        Use Ganglion ASCII command '[' to enable synthetic square wave.
        """
        if self.board is not None:
            self.board.config_board("[")

    def send_test_signal_off(self):
        if self.board is not None:
            self.board.config_board("]")

    # ---------- Data access helpers ----------

    def _get_exg_channels(self) -> List[int]:
        if self.board_id is None:
            return []
        return BoardShim.get_exg_channels(self.board_id)

    def get_timeseries_window(
        self,
        window_sec: float = 4.0,
        max_points: int = 512,
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Returns (channel_names, data[channels][samples])
        """
        if not (self.connected and self.streaming and self.board):
            return [], []

        sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        n_samples = int(window_sec * sampling_rate)
        with self.lock:
            data = self.board.get_current_board_data(n_samples)
        ch_indices = self._get_exg_channels()
        if data.shape[1] == 0:
            return [], []

        ts_data = []
        for ch in ch_indices:
            channel_series = data[ch, :]
            # downsample if too many points
            if channel_series.size > max_points:
                step = int(np.floor(channel_series.size / max_points))
                channel_series = channel_series[::step]
            ts_data.append(channel_series.tolist())

        channel_names = [f"CH{idx+1}" for idx in range(len(ch_indices))]
        return channel_names, ts_data

    def get_fft_spectrum(
        self,
        window_sec: float = 4.0,
        max_freq: float = 50.0,
    ) -> Tuple[List[str], List[float], List[List[float]]]:
        """
        Returns (channel_names, freqs, psd[channels][freqs])
        """
        if not (self.connected and self.streaming and self.board):
            return [], [], []

        sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        n_samples = int(window_sec * sampling_rate)

        with self.lock:
            data = self.board.get_current_board_data(n_samples)

        ch_indices = self._get_exg_channels()
        if data.shape[1] == 0:
            return [], [], []

        channel_names = [f"CH{idx+1}" for idx in range(len(ch_indices))]
        all_psd: List[List[float]] = []
        freq_list: List[float] = []

        for ch_idx, ch in enumerate(ch_indices):
            sig = data[ch, :].astype(np.float64)
            if sig.size < 32:
                all_psd.append([])
                continue

            # detrend + PSD with Welch via BrainFlow helper
            DataFilter.detrend(sig, DetrendOperations.LINEAR.value)
            fft_len = DataFilter.get_nearest_power_of_two(sig.size)
            psd, freqs = DataFilter.get_psd_welch(
                sig, fft_len, fft_len // 2, sampling_rate,
                WindowOperations.HANNING.value
            )

            # Filter to max_freq
            mask = freqs <= max_freq
            filtered_freqs = freqs[mask]
            filtered_psd = psd[mask]

            if ch_idx == 0:
                freq_list = filtered_freqs.tolist()

            all_psd.append(filtered_psd.tolist())

        return channel_names, freq_list, all_psd

    def get_band_powers(
        self,
        window_sec: float = 4.0,
        bands: Optional[List[Tuple[str, float, float]]] = None,
    ) -> Tuple[List[str], List[str], List[List[float]]]:
        """
        Returns (channel_names, band_names, band_values[channels][bands])
        """
        if bands is None:
            bands = [
                ("delta", 1.0, 4.0),
                ("theta", 4.0, 8.0),
                ("alpha", 8.0, 13.0),
                ("beta", 13.0, 30.0),
                ("gamma", 30.0, 45.0),
            ]

        if not (self.connected and self.streaming and self.board):
            return [], [b[0] for b in bands], []

        sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        n_samples = int(window_sec * sampling_rate)

        with self.lock:
            data = self.board.get_current_board_data(n_samples)

        ch_indices = self._get_exg_channels()
        if data.shape[1] == 0:
            return [], [b[0] for b in bands], []

        band_names = [b[0] for b in bands]
        channel_names = [f"CH{idx+1}" for idx in range(len(ch_indices))]
        all_band_vals: List[List[float]] = []

        for ch in ch_indices:
            sig = data[ch, :].astype(np.float64)
            if sig.size < 32:
                all_band_vals.append([0.0] * len(bands))
                continue

            # detrend + PSD with Welch via BrainFlow helper
            DataFilter.detrend(
                sig, DetrendOperations.LINEAR.value
            )
            # choose fft_len as nearest power of 2 <= len(sig)
            fft_len = DataFilter.get_nearest_power_of_two(sig.size)
            psd, freqs = DataFilter.get_psd_welch(
                sig, fft_len, fft_len // 2, sampling_rate,
                WindowOperations.HANNING.value
            )

            # freqs, psd are numpy arrays
            # integrate band power
            ch_band_vals: List[float] = []
            for _, fmin, fmax in bands:
                bp = DataFilter.get_band_power(psd, freqs, fmin, fmax)
                ch_band_vals.append(float(bp))
            all_band_vals.append(ch_band_vals)

        return channel_names, band_names, all_band_vals

    # ---------- OSC glue ----------

    def configure_osc(self, ip: str, port: int, enabled: bool,
                      send_raw: bool, send_bands: bool):
        self.osc.configure(ip, port, enabled, send_raw, send_bands)

    def osc_push_timeseries(self, channel_names: List[str], data: List[List[float]]):
        self.osc.send_timeseries(channel_names, data)

    def osc_push_bands(self, channel_names: List[str],
                       band_names: List[str],
                       values: List[List[float]]):
        self.osc.send_bands(channel_names, band_names, values)
