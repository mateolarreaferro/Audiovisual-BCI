#!/usr/bin/env python3
"""Offline training pipeline for TRCA-based SSVEP classification."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import signal
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from calibration_data_manager import CalibrationDataManager
from ssvep_bci.modules.datasets import BenchmarkSSVEPDataset
from ssvep_bci.modules.ssvep_classifier import SSVEPClassifier


logger = logging.getLogger("offline_trca")


def build_classifier_config(fs: float, freqs: List[float], window: float, method: str) -> Dict:
    """Create a configuration dictionary for :class:`SSVEPClassifier`."""
    use_filter_bank = method.upper() == "FBTRCA"
    return {
        'HARDWARE': {'sampling_rate': fs},
        'STIMULUS': {'frequencies': freqs},
        'CLASSIFIER': {
            'type': method,
            'n_harmonics': 2,
            'filter_bank': {'enabled': use_filter_bank, 'n_filters': 5, 'filter_order': 4},
            'window_length': window,
            'threshold': 0.0,
        },
        'FEATURES': {
            'psda': {
                'nperseg': int(fs),
                'noverlap': int(fs * 0.5)
            }
        }
    }


def estimate_baseline_noise(training_data: Dict[float, List[np.ndarray]],
                             fs: float,
                             freqs: List[float]) -> float:
    """Estimate baseline noise level from training segments."""
    noise_values = []
    freq_set = set(freqs)

    for freq, segments in training_data.items():
        for segment in segments:
            avg_signal = np.mean(segment, axis=0)
            nperseg = min(len(avg_signal), int(fs))
            f_axis, psd = signal.welch(avg_signal, fs=fs, nperseg=nperseg)
            mask = (f_axis >= 5) & (f_axis <= 45)
            for target in freq_set:
                mask &= (np.abs(f_axis - target) > 0.5)
            if np.any(mask):
                noise_values.append(float(np.median(psd[mask])))

    if not noise_values:
        return 0.1

    return float(np.median(noise_values))


def compute_threshold(classifier: SSVEPClassifier,
                      training_data: Dict[float, List[np.ndarray]]) -> float:
    """Derive a score threshold from training segments."""
    scores = []

    for freq, segments in training_data.items():
        for segment in segments:
            features = classifier.extract_features(segment)
            scores.append(features.get(freq, 0.0))

    if not scores:
        return 0.1

    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores))
    return max(0.05, mean_score - 0.5 * std_score)


def cross_validate(classifier_config: Dict,
                   training_data: Dict[float, List[np.ndarray]],
                   folds: int) -> Dict:
    """Evaluate classifier using stratified cross validation."""
    freq_list = sorted(training_data.keys())
    segments = []
    labels = []

    for idx, freq in enumerate(freq_list):
        freq_segments = training_data[freq]
        segments.extend(freq_segments)
        labels.extend([idx] * len(freq_segments))

    if not segments:
        raise RuntimeError("No segments available for cross validation")

    segments_arr = np.stack(segments, axis=0)
    labels_arr = np.array(labels)

    min_class_segments = min(len(training_data[freq]) for freq in freq_list)
    if min_class_segments < 2:
        raise RuntimeError("Each frequency requires at least two segments for validation")

    effective_folds = min(folds, min_class_segments)
    if effective_folds < 2:
        effective_folds = 2

    skf = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=42)

    accuracies = []
    confusion_counts = np.zeros((len(freq_list), len(freq_list)), dtype=float)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(segments_arr, labels_arr), start=1):
        fold_training = {freq: [] for freq in freq_list}
        for idx in train_idx:
            freq = freq_list[labels_arr[idx]]
            fold_training[freq].append(segments_arr[idx])

        classifier = SSVEPClassifier(classifier_config)
        classifier.train(fold_training)

        preds = []
        for idx in test_idx:
            pred_idx, _ = classifier.predict(segments_arr[idx])
            preds.append(pred_idx)

        preds_arr = np.array(preds)
        truth = labels_arr[test_idx]
        fold_accuracy = float(np.mean(preds_arr == truth))
        accuracies.append(fold_accuracy)
        confusion_counts += confusion_matrix(truth, preds_arr, labels=range(len(freq_list)))
        logger.info("Fold %d accuracy: %.3f", fold_idx, fold_accuracy)

    row_sums = confusion_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        confusion = np.divide(
            confusion_counts,
            row_sums,
            out=np.zeros_like(confusion_counts, dtype=float),
            where=row_sums != 0,
        )
    return {
        'folds': effective_folds,
        'accuracies': accuracies,
        'mean_accuracy': float(np.mean(accuracies)),
        'confusion_counts': confusion_counts.tolist(),
        'confusion_matrix': confusion.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Offline TRCA training using the benchmark SSVEP dataset")
    parser.add_argument('--dataset-root', required=True, help='Path to extracted benchmark dataset directory')
    parser.add_argument('--subjects', nargs='+', default=['S1'],
                        help="Subject IDs to use (e.g., S1 S2) or 'all'")
    parser.add_argument('--frequencies', nargs='+', type=float, default=[10.0, 15.0],
                        help='Target stimulation frequencies to extract (Hz)')
    parser.add_argument('--window', type=float, default=2.0, help='Window length for segments (seconds)')
    parser.add_argument('--start-offset', type=float, default=0.5,
                        help='Offset from stimulus onset for window start (seconds)')
    parser.add_argument('--max-trials', type=int, default=None,
                        help='Limit number of trials per frequency (optional)')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--method', choices=['TRCA', 'FBTRCA'], default='FBTRCA',
                        help='Classifier variant to train')
    parser.add_argument('--calibration-dir', type=Path, default=Path('calibration_data'),
                        help='Directory where calibration sessions are stored')
    parser.add_argument('--no-save', action='store_true', help='Skip saving calibration session')
    parser.add_argument('--log-level', default='INFO', help='Logging level')

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format='[%(levelname)s] %(message)s')

    dataset = BenchmarkSSVEPDataset(args.dataset_root)

    if len(args.subjects) == 1 and args.subjects[0].lower() == 'all':
        subjects = dataset.available_subjects()
    else:
        subjects = args.subjects

    logger.info("Using subjects: %s", ', '.join(subjects))

    segment_sets = []
    fs_values = []

    for subject in subjects:
        subject_data = dataset.load_subject(subject)
        fs_values.append(subject_data['fs'])
        segments = dataset.extract_segments(
            subject_data,
            target_freqs=args.frequencies,
            window_sec=args.window,
            start_offset=args.start_offset,
            max_trials=args.max_trials,
        )
        segment_sets.append(segments)
        logger.info("Subject %s segments: %s", subject_data['subject'],
                    {float(freq): len(seg_list) for freq, seg_list in segments.items()})

    training_data = BenchmarkSSVEPDataset.concatenate_segments(segment_sets)
    freq_list = sorted(training_data.keys())

    for freq in freq_list:
        logger.info("Frequency %.2f Hz -> %d segments", freq, len(training_data[freq]))

    if not training_data:
        raise RuntimeError("No training data extracted from dataset")

    if len(set(fs_values)) != 1:
        raise RuntimeError("Mixed sampling rates detected across subjects")

    fs = fs_values[0]
    classifier_config = build_classifier_config(fs, freq_list, args.window, args.method)

    cv_results = cross_validate(classifier_config, training_data, args.folds)
    logger.info("Mean CV accuracy: %.3f", cv_results['mean_accuracy'])

    classifier = SSVEPClassifier(classifier_config)
    classifier.train(training_data)

    channel_importance = classifier.get_trca_channel_importance()
    if channel_importance is not None:
        sorted_channels = np.argsort(channel_importance)[::-1]
        top_channels = sorted_channels[: min(6, len(sorted_channels))]
        logger.info("Top TRCA channels: %s", ', '.join(map(str, top_channels)))
        optimal_channels = top_channels.tolist()
    else:
        optimal_channels = list(range(training_data[freq_list[0]][0].shape[0]))

    baseline_noise = estimate_baseline_noise(training_data, fs, freq_list)
    threshold = compute_threshold(classifier, training_data)

    logger.info("Estimated baseline noise: %.4f", baseline_noise)
    logger.info("Adaptive score threshold: %.4f", threshold)

    metadata = {
        'source': 'benchmark_ssvep_dataset',
        'subjects': subjects,
        'frequencies': freq_list,
        'window_sec': args.window,
        'start_offset_sec': args.start_offset,
        'sampling_rate': fs,
        'method': args.method,
        'cv': cv_results,
    }

    if not args.no_save:
        manager = CalibrationDataManager(args.calibration_dir)
        session_name = manager.save_calibration_session(
            training_data=training_data,
            optimal_channels=optimal_channels,
            baseline_noise=baseline_noise,
            threshold=threshold,
            metadata=metadata,
        )
        logger.info("Saved calibration session: %s", session_name)
    else:
        logger.info("Calibration data not saved (no-save flag)")

    # Print JSON summary for convenience
    summary = {
        'mean_accuracy': cv_results['mean_accuracy'],
        'folds': cv_results['folds'],
        'frequencies': freq_list,
        'baseline_noise': baseline_noise,
        'threshold': threshold,
        'optimal_channels': optimal_channels,
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
