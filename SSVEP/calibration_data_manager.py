#!/usr/bin/env python3
"""Calibration data management for SSVEP system"""

import numpy as np
import pickle
import json
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CalibrationDataManager:
    """Manages saving and loading of calibration data"""
    
    def __init__(self, data_dir="calibration_data"):
        """
        Initialize calibration data manager
        
        Args:
            data_dir: Directory to store calibration data
        """
        self.data_dir = data_dir
        self.ensure_data_dir()
    
    def ensure_data_dir(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created calibration data directory: {self.data_dir}")
    
    def save_calibration_session(self, training_data, optimal_channels, 
                                baseline_noise, threshold, metadata=None):
        """
        Save a complete calibration session
        
        Args:
            training_data: Dictionary of frequency -> list of data segments
            optimal_channels: List of optimal channel indices
            baseline_noise: Baseline noise level
            threshold: Detection threshold
            metadata: Additional metadata (sampling rate, etc.)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"calibration_{timestamp}"
        
        # Prepare data for saving
        session_data = {
            'timestamp': timestamp,
            'training_data': {},
            'optimal_channels': optimal_channels,
            'baseline_noise': baseline_noise,
            'threshold': threshold,
            'metadata': metadata or {}
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for freq, segments in training_data.items():
            session_data['training_data'][str(freq)] = [seg.tolist() for seg in segments]
        
        # Save as both pickle (full data) and JSON (metadata)
        pickle_path = os.path.join(self.data_dir, f"{session_name}.pkl")
        json_path = os.path.join(self.data_dir, f"{session_name}_meta.json")
        
        # Save full data
        with open(pickle_path, 'wb') as f:
            pickle.dump(session_data, f)
        
        # Save metadata only
        metadata_only = {
            'timestamp': timestamp,
            'optimal_channels': optimal_channels,
            'baseline_noise': baseline_noise,
            'threshold': threshold,
            'metadata': session_data['metadata'],
            'training_segments': {str(freq): len(segments) for freq, segments in training_data.items()},
            'total_segments': sum(len(segments) for segments in training_data.values())
        }
        
        with open(json_path, 'w') as f:
            json.dump(metadata_only, f, indent=2)
        
        logger.info(f"Calibration session saved: {session_name}")
        logger.info(f"Total segments: {metadata_only['total_segments']}")
        
        return session_name
    
    def load_calibration_session(self, session_name):
        """
        Load a calibration session
        
        Args:
            session_name: Name of the session to load
            
        Returns:
            Dictionary with training_data, optimal_channels, etc.
        """
        pickle_path = os.path.join(self.data_dir, f"{session_name}.pkl")
        
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Calibration session not found: {session_name}")
        
        with open(pickle_path, 'rb') as f:
            session_data = pickle.load(f)
        
        # Convert lists back to numpy arrays
        training_data = {}
        for freq_str, segments_list in session_data['training_data'].items():
            freq = float(freq_str)
            training_data[freq] = [np.array(seg) for seg in segments_list]
        
        session_data['training_data'] = training_data
        
        logger.info(f"Loaded calibration session: {session_name}")
        return session_data
    
    def list_sessions(self):
        """List all available calibration sessions"""
        sessions = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('_meta.json'):
                session_name = filename.replace('_meta.json', '')
                json_path = os.path.join(self.data_dir, filename)
                
                try:
                    with open(json_path, 'r') as f:
                        metadata = json.load(f)
                    
                    sessions.append({
                        'name': session_name,
                        'timestamp': metadata.get('timestamp'),
                        'total_segments': metadata.get('total_segments', 0),
                        'optimal_channels': metadata.get('optimal_channels', []),
                        'threshold': metadata.get('threshold', 0)
                    })
                except Exception as e:
                    logger.warning(f"Error reading session metadata {filename}: {e}")
        
        # Sort by timestamp (newest first)
        sessions.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return sessions
    
    def get_best_session(self, min_segments=50):
        """
        Get the best calibration session based on data quality
        
        Args:
            min_segments: Minimum number of segments required
            
        Returns:
            Session name of the best session, or None if none found
        """
        sessions = self.list_sessions()
        
        # Filter sessions with enough data
        good_sessions = [s for s in sessions if s['total_segments'] >= min_segments]
        
        if not good_sessions:
            logger.warning(f"No sessions found with >= {min_segments} segments")
            return None
        
        # Return the most recent good session
        best_session = good_sessions[0]
        logger.info(f"Best session: {best_session['name']} ({best_session['total_segments']} segments)")
        
        return best_session['name']
    
    def cleanup_old_sessions(self, keep_recent=5):
        """
        Clean up old calibration sessions, keeping only the most recent ones
        
        Args:
            keep_recent: Number of recent sessions to keep
        """
        sessions = self.list_sessions()
        
        if len(sessions) <= keep_recent:
            return
        
        sessions_to_delete = sessions[keep_recent:]
        
        for session in sessions_to_delete:
            session_name = session['name']
            
            # Delete both pickle and JSON files
            pickle_path = os.path.join(self.data_dir, f"{session_name}.pkl")
            json_path = os.path.join(self.data_dir, f"{session_name}_meta.json")
            
            try:
                if os.path.exists(pickle_path):
                    os.remove(pickle_path)
                if os.path.exists(json_path):
                    os.remove(json_path)
                
                logger.info(f"Deleted old session: {session_name}")
            except Exception as e:
                logger.warning(f"Error deleting session {session_name}: {e}")