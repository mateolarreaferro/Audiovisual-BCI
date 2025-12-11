# openbci_service.py
import threading
from typing import Optional, List, Tuple

import numpy as np

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations, FilterTypes

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

        # Configure board: disable test signal and enable all channels
        try:
            import time
            time.sleep(0.5)  # Wait for board to be fully ready

            # Disable test signal
            self.board.config_board("]")
            print("[BioMus] Sent test signal OFF command (])")
            time.sleep(0.1)

            # Enable all 4 channels (! @ # $ for channels 1-4)
            self.board.config_board("!")
            self.board.config_board("@")
            self.board.config_board("#")
            self.board.config_board("$")
            print("[BioMus] Enabled all 4 channels (! @ # $)")
            time.sleep(0.1)

        except Exception as e:
            print(f"[BioMus] Failed to configure board: {e}")
            # If command fails, continue anyway
            pass

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
        # Ensure test signal is OFF before starting stream
        import time
        try:
            self.board.config_board("]")
            print("[BioMus] Before stream: Sent test signal OFF command (])")
            time.sleep(0.2)
        except Exception as e:
            print(f"[BioMus] Failed to send OFF command before stream: {e}")
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

    def _preprocess_signal(self, signal: np.ndarray, sampling_rate: int, apply_filters: bool = False) -> np.ndarray:
        """
        Apply light EEG preprocessing:
        - Detrending (always applied)
        - Optional: Notch filter (60 Hz) to remove power line interference
        """
        if signal.size < 8:
            return signal

        # Create a copy to avoid modifying original data
        filtered = signal.copy()

        # Always apply detrending to remove DC offset
        DataFilter.detrend(filtered, DetrendOperations.LINEAR.value)

        # Only apply notch filter if requested (can be aggressive on small datasets)
        if apply_filters and signal.size >= 64:
            try:
                # Apply notch filter at 60 Hz (US power line frequency)
                DataFilter.perform_bandstop(
                    filtered,
                    sampling_rate,
                    centerfreq=60.0,
                    bandwidth_hz=2.0,  # Narrow notch width
                    order=2,  # Lower order for stability
                    filter_type=FilterTypes.BUTTERWORTH.value,
                    ripple=0
                )
            except Exception:
                # If filter fails, just return detrended signal
                pass

        return filtered

    def get_timeseries_window(
        self,
        window_sec: float = 4.0,
        max_points: int = 512,
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Returns (channel_names, data[channels][samples])
        Data is preprocessed with bandpass and notch filters, then smoothed
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
            channel_series = data[ch, :].astype(np.float64)

            # Apply light smoothing using moving average (window size = 3)
            if channel_series.size >= 3:
                smoothed = channel_series.copy()
                DataFilter.perform_rolling_filter(smoothed, 3, operation=0)  # 0 = mean
                channel_series = smoothed

            # Downsample if too many points
            if channel_series.size > max_points:
                step = int(np.floor(channel_series.size / max_points))
                channel_series = channel_series[::step]

            ts_data.append(channel_series.tolist())

        channel_names = [f"CH{idx+1}" for idx in range(len(ch_indices))]
        return channel_names, ts_data

    def get_fft_spectrum(
        self,
        window_sec: float = 4.0,
        min_freq: float = 0.5,
        max_freq: float = 40.0,
    ) -> Tuple[List[str], List[float], List[List[float]]]:
        """
        Returns (channel_names, freqs, psd[channels][freqs])
        PSD values are in log scale (dB) for better visualization
        Signal is preprocessed with bandpass and notch filters
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

            # Detrend to remove linear trends
            DataFilter.detrend(sig, DetrendOperations.LINEAR.value)

            # Choose fft_len as nearest power of 2, but MUST be < data length
            # Use 50% overlap for good PSD estimation
            fft_len = DataFilter.get_nearest_power_of_two(sig.size)
            while fft_len >= sig.size and fft_len > 2:
                fft_len = fft_len // 2

            overlap = fft_len // 2  # 50% overlap

            psd, freqs = DataFilter.get_psd_welch(
                sig, fft_len, overlap, sampling_rate,
                WindowOperations.HANNING.value
            )

            # Filter to frequency range
            mask = (freqs >= min_freq) & (freqs <= max_freq)
            filtered_freqs = freqs[mask]
            filtered_psd = psd[mask]

            # Convert to dB scale for better visualization: 10 * log10(PSD)
            # Add small epsilon to avoid log(0)
            psd_db = 10 * np.log10(filtered_psd + 1e-10)

            if ch_idx == 0:
                freq_list = filtered_freqs.tolist()

            all_psd.append(psd_db.tolist())

        return channel_names, freq_list, all_psd

    def get_band_powers(
        self,
        window_sec: float = 4.0,
        bands: Optional[List[Tuple[str, float, float]]] = None,
        use_relative: bool = True,
    ) -> Tuple[List[str], List[str], List[List[float]]]:
        """
        Returns (channel_names, band_names, band_values[channels][bands])
        Signal is preprocessed with bandpass and notch filters
        If use_relative=True, returns relative band power (percentage of total power)
        """
        if bands is None:
            bands = [
                ("delta", 0.5, 4.0),
                ("theta", 4.0, 8.0),
                ("alpha", 8.0, 13.0),
                ("beta", 13.0, 30.0),
                ("gamma", 30.0, 40.0),
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

            # Detrend to remove linear trends
            DataFilter.detrend(sig, DetrendOperations.LINEAR.value)

            # Choose fft_len as nearest power of 2, but MUST be < data length
            fft_len = DataFilter.get_nearest_power_of_two(sig.size)
            while fft_len >= sig.size and fft_len > 2:
                fft_len = fft_len // 2

            overlap = fft_len // 2  # 50% overlap

            psd_tuple = DataFilter.get_psd_welch(
                sig, fft_len, overlap, sampling_rate,
                WindowOperations.HANNING.value
            )

            # Calculate band powers
            ch_band_vals: List[float] = []
            for _, fmin, fmax in bands:
                bp = DataFilter.get_band_power(psd_tuple, fmin, fmax)
                ch_band_vals.append(float(bp))

            # Convert to relative power (percentage) if requested
            if use_relative:
                total_power = sum(ch_band_vals)
                if total_power > 0:
                    ch_band_vals = [(bp / total_power) * 100 for bp in ch_band_vals]
                else:
                    ch_band_vals = [0.0] * len(bands)

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
