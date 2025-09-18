"""
SSVEP Classification Module
Implements CCA, FBCCA, and PSDA methods for SSVEP detection
"""

import numpy as np
from scipy import signal, linalg
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class SSVEPClassifier:
    """
    SSVEP classifier using multiple detection methods
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SSVEP classifier
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.fs = config['HARDWARE']['sampling_rate']
        self.frequencies = config['STIMULUS']['frequencies']
        self.n_harmonics = config['CLASSIFIER']['n_harmonics']
        self.window_length = config['CLASSIFIER']['window_length']
        self.threshold = config['CLASSIFIER']['threshold']
        
        # Filter bank configuration
        self.use_filter_bank = config['CLASSIFIER']['filter_bank']['enabled']
        self.n_filters = config['CLASSIFIER']['filter_bank']['n_filters']
        self.filter_order = config['CLASSIFIER']['filter_bank']['filter_order']
        
        # Classification method
        self.method = config['CLASSIFIER']['type']
        
        # Reference signals for CCA
        self.reference_signals = {}
        self._generate_reference_signals()

        # Filter bank filters
        self.filter_bank = None
        self.filter_bank_weights = None
        if self.use_filter_bank:
            self._create_filter_bank()

        # Calibration data
        self.templates = {}
        self.trained = False
        self.trca_filters = {}
        self.trca_templates = {}

    def _generate_reference_signals(self):
        """Generate reference signals for each stimulus frequency"""
        n_samples = int(self.window_length * self.fs)
        t = np.arange(n_samples) / self.fs
        
        for freq in self.frequencies:
            refs = []
            
            # Generate sine and cosine references for fundamental and harmonics
            for harmonic in range(1, self.n_harmonics + 1):
                refs.append(np.sin(2 * np.pi * harmonic * freq * t))
                refs.append(np.cos(2 * np.pi * harmonic * freq * t))
            
            self.reference_signals[freq] = np.array(refs).T
    
    def _create_filter_bank(self):
        """Create filter bank for FBCCA"""
        self.filter_bank = []

        # Define sub-band ranges
        for i in range(self.n_filters):
            # Sub-bands: 6-14, 14-22, 22-30, 30-38, 38-46 Hz (example)
            low_freq = 6 + i * 8
            high_freq = min(14 + i * 8, 45)

            # Create bandpass filter
            sos = signal.butter(self.filter_order,
                              [low_freq, high_freq],
                              btype='band',
                              fs=self.fs,
                              output='sos')
            self.filter_bank.append(sos)

        # Pre-compute weights for filter bank combination
        self.filter_bank_weights = self._compute_filter_bank_weights(len(self.filter_bank))

    @staticmethod
    def _compute_filter_bank_weights(n_filters: int) -> np.ndarray:
        """Return weighting coefficients for filter-bank based methods"""
        if n_filters <= 0:
            return np.array([1.0])

        # Standard FBCCA weighting heuristic (m^-1.25 + 0.25)
        fb_indices = np.arange(1, n_filters + 1)
        weights = (fb_indices ** -1.25) + 0.25
        return weights / np.sum(weights)

    def _prepare_eeg_data(self, eeg_data: np.ndarray) -> np.ndarray:
        """Ensure EEG data has shape (channels, samples)"""

        if eeg_data.ndim != 2:
            raise ValueError("EEG data must be 2D (channels x samples)")

        # Internally we work with shape (channels, samples)
        if eeg_data.shape[0] < eeg_data.shape[1]:
            return eeg_data

        return eeg_data.T

    def extract_features(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        Extract SSVEP features from EEG data

        Args:
            eeg_data: EEG data (channels x samples)
            
        Returns:
            Dictionary of features for each frequency
        """
        features = {}
        eeg_data = self._prepare_eeg_data(eeg_data)

        if self.method == 'CCA':
            features = self._cca_features(eeg_data)
        elif self.method == 'FBCCA':
            features = self._fbcca_features(eeg_data)
        elif self.method == 'PSDA':
            features = self._psda_features(eeg_data)
        elif self.method in ['TRCA', 'FBTRCA']:
            features = self._trca_features(eeg_data)
        elif self.method == 'ensemble':
            # Combine multiple methods
            cca_features = self._cca_features(eeg_data)
            psda_features = self._psda_features(eeg_data)

            # Weighted combination
            for freq in self.frequencies:
                features[freq] = 0.7 * cca_features.get(freq, 0) + 0.3 * psda_features.get(freq, 0)
        
        return features
    
    def _cca_features(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        Extract features using Canonical Correlation Analysis

        Args:
            eeg_data: EEG data (channels x samples)

        Returns:
            CCA coefficients for each frequency
        """
        features = {}

        # Convert to shape (samples, channels) for CCA
        data = eeg_data.T if eeg_data.shape[0] <= eeg_data.shape[1] else eeg_data

        # Truncate or pad to desired window length
        n_samples = int(self.window_length * self.fs)
        if data.shape[0] > n_samples:
            data = data[:n_samples, :]
        elif data.shape[0] < n_samples:
            padding = n_samples - data.shape[0]
            data = np.vstack([data, np.zeros((padding, data.shape[1]))])

        # Compute CCA for each frequency
        for freq in self.frequencies:
            try:
                ref_signals = self.reference_signals[freq][:data.shape[0], :]

                cca = CCA(n_components=1, max_iter=500)
                cca.fit(data, ref_signals)

                X_c, Y_c = cca.transform(data, ref_signals)
                corr = np.corrcoef(X_c.T, Y_c.T)[0, 1]

                features[freq] = abs(corr)

            except Exception as e:
                print(f"CCA error for {freq}Hz: {e}")
                features[freq] = 0.0

        return features
    
    def _fbcca_features(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        Extract features using Filter Bank CCA
        
        Args:
            eeg_data: EEG data (channels x samples)
            
        Returns:
            FBCCA coefficients for each frequency
        """
        if not self.use_filter_bank:
            return self._cca_features(eeg_data)
        
        features = {}
        
        # Initialize features
        for freq in self.frequencies:
            features[freq] = 0.0
        
        # Apply each filter and compute CCA
        for fb_idx, sos_filter in enumerate(self.filter_bank):
            # Filter the data
            filtered_data = signal.sosfiltfilt(sos_filter, eeg_data, axis=1)
            
            # Get CCA features for filtered data
            fb_features = self._cca_features(filtered_data)
            
            # Weight by filter bank index (higher weight for lower frequencies)
            weight = (self.n_filters - fb_idx) / self.n_filters
            
            # Accumulate weighted features
            for freq in self.frequencies:
                features[freq] += weight * fb_features.get(freq, 0)
        
        # Normalize
        max_val = max(features.values()) if features else 1.0
        if max_val > 0:
            for freq in features:
                features[freq] /= max_val
        
        return features
    
    def _psda_features(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """
        Extract features using Power Spectral Density Analysis
        
        Args:
            eeg_data: EEG data (channels x samples)
            
        Returns:
            PSDA features for each frequency
        """
        features = {}
        
        # Compute PSD using Welch's method
        nperseg = self.config['FEATURES']['psda']['nperseg']
        noverlap = self.config['FEATURES']['psda']['noverlap']
        
        # Average PSD across channels
        psd_list = []
        for ch in range(eeg_data.shape[0]):
            f, psd = signal.welch(eeg_data[ch, :], 
                                 fs=self.fs,
                                 nperseg=nperseg,
                                 noverlap=noverlap)
            psd_list.append(psd)
        
        # Average across channels
        psd_avg = np.mean(psd_list, axis=0)
        
        # Extract power at target frequencies and harmonics
        for freq in self.frequencies:
            power = 0.0
            
            for harmonic in range(1, self.n_harmonics + 1):
                target_freq = freq * harmonic
                
                # Find closest frequency bin
                freq_idx = np.argmin(np.abs(f - target_freq))
                
                # Sum power in a small window around target frequency
                window_size = 0.5  # Hz
                freq_range = np.where((f >= target_freq - window_size) & 
                                     (f <= target_freq + window_size))[0]
                
                if len(freq_range) > 0:
                    power += np.sum(psd_avg[freq_range])
            
            features[freq] = power
        
        # Normalize features
        max_power = max(features.values()) if features else 1.0
        if max_power > 0:
            for freq in features:
                features[freq] /= max_power
        
        return features
    
    def _train_trca(self, training_data: Dict[float, List[np.ndarray]]):
        """Train TRCA or FBTRCA spatial filters"""
        self.trca_filters = {}
        self.trca_templates = {}

        if not training_data:
            return

        if self.filter_bank is None or not self.use_filter_bank:
            # Single broad bandpass (fallback)
            self.filter_bank = [signal.butter(4, [6, 45], btype='band', fs=self.fs, output='sos')]
            self.filter_bank_weights = np.array([1.0])

        # Iterate over frequencies and compute TRCA filters
        for freq, segments in training_data.items():
            if not segments:
                continue

            try:
                trials = np.stack([self._prepare_eeg_data(seg) for seg in segments], axis=0)
            except ValueError:
                # Skip frequencies with inconsistent segment shapes
                continue
            n_trials, n_channels, _ = trials.shape

            if n_trials < 2:
                # Need at least two trials for TRCA covariance estimation
                continue

            freq_filters = []
            freq_templates = []

            for sos in self.filter_bank:
                filtered_trials = signal.sosfiltfilt(sos, trials, axis=2)
                spatial_filter, template = self._compute_trca_filter(filtered_trials)
                freq_filters.append(spatial_filter)
                freq_templates.append(template)

            self.trca_filters[freq] = np.vstack(freq_filters)
            self.trca_templates[freq] = np.vstack(freq_templates)

    def _compute_trca_filter(self, trials: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute TRCA spatial filter and template for a set of trials"""

        # trials shape: (n_trials, n_channels, n_samples)
        n_trials, n_channels, n_samples = trials.shape

        # Remove DC component per trial
        trials = trials - trials.mean(axis=2, keepdims=True)

        # Within-trial covariance (Q) and cross-trial covariance (S)
        Q = np.zeros((n_channels, n_channels))
        S = np.zeros_like(Q)

        for trial in trials:
            Q += np.cov(trial)

        for i in range(n_trials):
            Xi = trials[i]
            for j in range(i + 1, n_trials):
                Xj = trials[j]
                S += Xi @ Xj.T + Xj @ Xi.T

        # Symmetrize matrices and add regularization
        Q = (Q + Q.T) / 2.0 + 1e-6 * np.eye(n_channels)
        S = (S + S.T) / 2.0

        # Solve generalized eigenvalue problem S w = lambda Q w
        eigvals, eigvecs = linalg.eigh(S, Q)

        # Select eigenvector with largest eigenvalue
        max_idx = np.argmax(eigvals.real)
        spatial_filter = eigvecs[:, max_idx].real
        spatial_filter /= np.linalg.norm(spatial_filter) + 1e-12

        # Create template by averaging spatially filtered trials
        average_trial = trials.mean(axis=0)
        template = spatial_filter @ average_trial

        return spatial_filter, template

    def _trca_features(self, eeg_data: np.ndarray) -> Dict[str, float]:
        """Calculate TRCA-based features"""
        if not self.trca_filters or not self.trca_templates:
            return {freq: 0.0 for freq in self.frequencies}

        features = {}

        for freq in self.frequencies:
            if freq not in self.trca_filters:
                features[freq] = 0.0
                continue

            filters = self.trca_filters[freq]
            templates = self.trca_templates[freq]

            responses = []
            for fb_idx in range(filters.shape[0]):
                sos = self.filter_bank[fb_idx] if self.filter_bank else None
                data_fb = eeg_data
                if sos is not None:
                    data_fb = signal.sosfiltfilt(sos, data_fb, axis=1)

                projected = filters[fb_idx] @ data_fb
                template = templates[fb_idx]

                if projected.std() == 0 or template.std() == 0:
                    corr = 0.0
                else:
                    corr = np.corrcoef(projected, template)[0, 1]

                if np.isnan(corr):
                    corr = 0.0

                responses.append(corr)

            responses = np.array(responses)
            weights = self.filter_bank_weights[:len(responses)] if self.filter_bank_weights is not None else 1.0
            score = float(np.sum(weights * responses))
            features[freq] = score

        return features

    def train(self, training_data: Dict[float, np.ndarray]):
        """
        Train classifier with calibration data

        Args:
            training_data: Dictionary mapping frequencies to EEG data arrays
        """
        self.templates = {}

        if self.method in ['TRCA', 'FBTRCA']:
            self._train_trca(training_data)
            print(f"Trained TRCA classifier for {len(self.trca_filters)} frequencies")
            self.trained = True
            return

        for freq, data_list in training_data.items():
            if freq not in self.frequencies:
                continue

            # Compute average template for each frequency
            templates = []

            for eeg_data in data_list:
                prepared = self._prepare_eeg_data(eeg_data)

                # Extract features or store raw template
                if self.method in ['CCA', 'FBCCA']:
                    templates.append(prepared)
                else:
                    features = self.extract_features(prepared)
                    templates.append(features[freq])

            # Average templates
            if templates:
                if isinstance(templates[0], np.ndarray):
                    self.templates[freq] = np.mean(templates, axis=0)
                else:
                    self.templates[freq] = np.mean(templates)

        self.trained = True
        print(f"Trained SSVEP classifier with {len(self.templates)} frequency templates")

    def get_trca_channel_importance(self) -> Optional[np.ndarray]:
        """Return average absolute spatial weights per channel for TRCA-based models"""
        if not self.trca_filters:
            return None

        all_weights = []
        for filters in self.trca_filters.values():
            # filters shape: (n_filters, n_channels)
            all_weights.append(np.mean(np.abs(filters), axis=0))

        if not all_weights:
            return None

        return np.mean(all_weights, axis=0)
    
    def predict(self, eeg_data: np.ndarray) -> Tuple[int, float]:
        """
        Predict SSVEP target from EEG data
        
        Args:
            eeg_data: EEG data (channels x samples)
            
        Returns:
            Tuple of (predicted target index, confidence)
        """
        # Extract features
        eeg_data = self._prepare_eeg_data(eeg_data)
        features = self.extract_features(eeg_data)
        
        # Find frequency with highest score
        max_score = 0.0
        predicted_freq = None
        
        for freq, score in features.items():
            if score > max_score:
                max_score = score
                predicted_freq = freq
        
        # Convert frequency to target index
        if predicted_freq is not None and predicted_freq in self.frequencies:
            target_idx = self.frequencies.index(predicted_freq)
        else:
            target_idx = 0
        
        # Calculate confidence (normalized score)
        confidence = max_score
        
        return target_idx, confidence
    
    def predict_proba(self, eeg_data: np.ndarray) -> np.ndarray:
        """
        Predict probability distribution over targets
        
        Args:
            eeg_data: EEG data (channels x samples)
            
        Returns:
            Probability array for each target
        """
        # Extract features
        eeg_data = self._prepare_eeg_data(eeg_data)
        features = self.extract_features(eeg_data)
        
        # Convert to probabilities using softmax
        scores = np.array([features.get(freq, 0) for freq in self.frequencies])
        
        # Apply softmax
        exp_scores = np.exp(scores - np.max(scores))
        probabilities = exp_scores / np.sum(exp_scores)
        
        return probabilities
    
    def calculate_snr(self, eeg_data: np.ndarray, target_freq: float) -> float:
        """
        Calculate Signal-to-Noise Ratio for a target frequency
        
        Args:
            eeg_data: EEG data (channels x samples)
            target_freq: Target frequency to analyze
            
        Returns:
            SNR in dB
        """
        # Compute PSD
        f, psd = signal.welch(np.mean(eeg_data, axis=0), 
                            fs=self.fs,
                            nperseg=self.fs*2)
        
        # Signal power (at target frequency and harmonics)
        signal_power = 0.0
        noise_indices = np.ones(len(f), dtype=bool)
        
        for harmonic in range(1, self.n_harmonics + 1):
            freq = target_freq * harmonic
            freq_idx = np.argmin(np.abs(f - freq))
            
            # Signal band (±0.5 Hz)
            signal_band = np.where((f >= freq - 0.5) & (f <= freq + 0.5))[0]
            signal_power += np.sum(psd[signal_band])
            noise_indices[signal_band] = False
        
        # Noise power (everything else in 3-45 Hz band)
        noise_band = np.where((f >= 3) & (f <= 45) & noise_indices)[0]
        noise_power = np.mean(psd[noise_band]) if len(noise_band) > 0 else 1e-10
        
        # Calculate SNR in dB
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        
        return snr_db
    
    def calculate_itr(self, accuracy: float, n_targets: int, 
                     selection_time: float) -> float:
        """
        Calculate Information Transfer Rate
        
        Args:
            accuracy: Classification accuracy (0-1)
            n_targets: Number of possible targets
            selection_time: Time for one selection in seconds
            
        Returns:
            ITR in bits per minute
        """
        if accuracy <= 0 or accuracy >= 1 or selection_time <= 0:
            return 0.0
        
        # ITR formula for BCI
        p = accuracy
        
        if p == 1:
            bit_rate = np.log2(n_targets)
        else:
            bit_rate = np.log2(n_targets) + p * np.log2(p) + (1 - p) * np.log2((1 - p) / (n_targets - 1))
        
        # Convert to bits per minute
        itr = (60.0 / selection_time) * bit_rate
        
        return max(0, itr)