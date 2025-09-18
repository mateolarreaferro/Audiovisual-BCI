"""Dataset helpers for offline SSVEP training"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.io import loadmat


class BenchmarkSSVEPDataset:
    """Utility class to work with the public benchmark SSVEP dataset.

    The dataset is described in:
    Nakanishi et al., A comparison study of canonical correlation analysis
    based methods for detecting steady-state visual evoked potentials,
    PLoS ONE, 2015.
    """

    FILENAME_PATTERN = "S*.mat"

    def __init__(self, root_dir: Path | str):
        self.root = Path(root_dir)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")

    def available_subjects(self) -> List[str]:
        """Return sorted list of available subject file stems."""
        subjects = sorted(p.stem for p in self.root.glob(self.FILENAME_PATTERN))
        if not subjects:
            raise FileNotFoundError(
                f"No subject files matching {self.FILENAME_PATTERN} found in {self.root}"
            )
        return subjects

    def _standardize_array(self, data: np.ndarray, n_freqs: int) -> np.ndarray:
        """Convert raw array to shape (n_freqs, n_trials, n_channels, n_samples)."""
        if data.ndim != 4:
            raise ValueError(
                f"Expected 4D array for dataset (found shape {data.shape}). "
                "Please verify the downloaded benchmark files."
            )

        # Move frequency axis to position 0
        try:
            freq_axis = next(idx for idx, dim in enumerate(data.shape) if dim == n_freqs)
        except StopIteration as exc:
            raise ValueError(
                "Could not infer frequency axis from dataset. "
                "Ensure the correct benchmark dataset is provided."
            ) from exc

        arr = np.moveaxis(data, freq_axis, 0)

        # Move sample axis (largest dimension) to the end
        remaining_axes = [1, 2, 3]
        sample_axis = max(remaining_axes, key=lambda idx: arr.shape[idx])
        arr = np.moveaxis(arr, sample_axis, -1)

        # Determine channel axis (typical channel counts)
        channel_candidates = {4, 6, 8, 16, 32, 64}
        remaining_axes = [1, 2]
        channel_axis = None
        for axis in remaining_axes:
            if arr.shape[axis] in channel_candidates:
                channel_axis = axis
                break
        if channel_axis is None:
            # Fallback: choose smaller dimension as channel axis
            channel_axis = min(remaining_axes, key=lambda idx: arr.shape[idx])

        trial_axis = 1 if channel_axis == 2 else 2

        if channel_axis != 2:
            arr = np.swapaxes(arr, channel_axis, 2)
        if trial_axis != 1:
            arr = np.swapaxes(arr, trial_axis, 1)

        # Final shape should be (n_freqs, n_trials, n_channels, n_samples)
        return np.asarray(arr, dtype=np.float32)

    def load_subject(self, subject: str) -> Dict[str, np.ndarray]:
        """Load a single subject file and standardize its structure."""
        subject_path = Path(subject)
        if not subject_path.is_file():
            subject_path = self.root / f"{subject}.mat"
        if not subject_path.exists():
            raise FileNotFoundError(f"Subject file not found: {subject_path}")

        mat = loadmat(subject_path)

        if 'data' not in mat:
            raise KeyError("Benchmark dataset file is missing the 'data' field")

        data = mat['data']

        freq_keys = ['freqs', 'frequencies', 'stimulus', 'stimulus_frequencies']
        freqs = None
        for key in freq_keys:
            if key in mat:
                freqs = np.squeeze(mat[key])
                break
        if freqs is None:
            raise KeyError("Frequency list not found in benchmark dataset file")

        fs = None
        for key in ['fs', 'Fs', 'sampling_rate', 'srate']:
            if key in mat:
                fs = float(np.squeeze(mat[key]))
                break
        if fs is None:
            fs = 250.0  # Default sampling rate for the benchmark dataset

        standardized = self._standardize_array(np.asarray(data), len(freqs))

        return {
            'subject': subject_path.stem,
            'fs': fs,
            'frequencies': freqs.astype(float),
            'data': standardized,
        }

    def extract_segments(
        self,
        subject_data: Dict[str, np.ndarray],
        target_freqs: Sequence[float],
        window_sec: float,
        start_offset: float = 0.5,
        max_trials: Optional[int] = None,
    ) -> Dict[float, List[np.ndarray]]:
        """Return calibration-ready segments for selected frequencies."""
        fs = subject_data['fs']
        data = subject_data['data']  # shape: (n_freqs, n_trials, n_channels, n_samples)
        freqs = subject_data['frequencies']

        window_samples = int(round(window_sec * fs))
        start_sample = int(round(start_offset * fs))
        end_sample = start_sample + window_samples

        if end_sample > data.shape[-1]:
            raise ValueError(
                f"Requested window ({window_sec}s from {start_offset}s) exceeds trial length"
            )

        segments: Dict[float, List[np.ndarray]] = {float(freq): [] for freq in target_freqs}

        for target in target_freqs:
            freq_idx = int(np.argmin(np.abs(freqs - target)))
            actual_freq = float(freqs[freq_idx])

            trials = data[freq_idx]  # shape: (n_trials, n_channels, n_samples)
            if max_trials is not None:
                trials = trials[:max_trials]

            for trial in trials:
                segment = trial[:, start_sample:end_sample]
                if segment.shape[-1] == window_samples:
                    segments.setdefault(actual_freq, []).append(segment.astype(np.float32))

        return segments

    @staticmethod
    def concatenate_segments(
        segment_sets: Sequence[Dict[float, List[np.ndarray]]]
    ) -> Dict[float, List[np.ndarray]]:
        """Merge multiple segment dictionaries by frequency."""
        merged: Dict[float, List[np.ndarray]] = {}
        for segment_dict in segment_sets:
            for freq, segments in segment_dict.items():
                merged.setdefault(freq, []).extend(segments)
        return merged
