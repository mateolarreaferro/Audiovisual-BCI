#!/usr/bin/env python3
"""
Simple Brainwave Server
Sends selected brainwave band data to Unity via UDP
"""

import socket
import time
import numpy as np
from flask import Flask, render_template, request, jsonify
import threading
import json
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations

app = Flask(__name__)

# Global state
current_band = "alpha"
current_power = 0.0
is_streaming = False
board = None
udp_socket = None

# Network settings
UNITY_HOST = "127.0.0.1"
UNITY_PORT = 12345
ACCEL_PORT = 12346

# Band frequency ranges
BAND_RANGES = {
    "delta": (1, 4),
    "theta": (4, 8), 
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50)
}

def setup_udp():
    """Setup UDP socket for Unity communication"""
    global udp_socket
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"UDP socket ready for {UNITY_HOST}:{UNITY_PORT}")

def send_to_unity(band, power):
    """Send band power to Unity"""
    if udp_socket:
        try:
            message = f"{band}:{power:.3f}"
            udp_socket.sendto(message.encode(), (UNITY_HOST, UNITY_PORT))
            print(f"Sent to {UNITY_HOST}:{UNITY_PORT} -> {message}")
        except Exception as e:
            print(f"UDP send error: {e}")
    else:
        print("UDP socket not initialized!")

def send_accelerometer_data(x, y, z):
    """Send accelerometer data to Unity"""
    if udp_socket:
        try:
            message = f"{x:.3f},{y:.3f},{z:.3f}"
            udp_socket.sendto(message.encode(), (UNITY_HOST, ACCEL_PORT))
        except Exception as e:
            print(f"Accelerometer UDP send error: {e}")

def calculate_band_power(data, sampling_rate, band_range):
    """Calculate power in specific frequency band"""
    try:
        # Get EEG channels - BrainFlow synthetic board has multiple channels
        eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
        if len(eeg_channels) == 0:
            print("No EEG channels found")
            return 0.0
            
        # Use first EEG channel
        eeg_channel = eeg_channels[0]
        eeg_data = data[eeg_channel].copy()
        
        print(f"Data shape: {data.shape}, EEG channel: {eeg_channel}, samples: {len(eeg_data)}")
        
        if len(eeg_data) < 256:  # Need enough samples for good FFT
            print(f"Not enough samples: {len(eeg_data)}")
            return 0.0
        
        # Take last 256 samples for analysis
        if len(eeg_data) > 256:
            eeg_data = eeg_data[-256:]
            
        # Apply bandpass filter
        DataFilter.perform_bandpass(eeg_data, sampling_rate, 
                                   band_range[0], band_range[1], 
                                   2, FilterTypes.BUTTERWORTH.value, 0)
        
        # Calculate power spectral density
        nfft = len(eeg_data)
        noverlap = nfft // 2
        psd = DataFilter.get_psd_welch(eeg_data, nfft, noverlap, sampling_rate, 
                                      WindowOperations.HANNING.value)
        
        # Get frequency bands
        freqs = psd[1]
        powers = psd[0]
        
        # Find indices for our frequency band
        band_mask = (freqs >= band_range[0]) & (freqs <= band_range[1])
        band_power = np.mean(powers[band_mask]) if np.any(band_mask) else 0.0
        
        # Normalize to 0-1 range (synthetic data has known ranges)
        normalized_power = min(max(band_power / 100.0, 0.0), 1.0)
        
        print(f"Band {band_range}: raw_power={band_power:.3f}, normalized={normalized_power:.3f}")
        return normalized_power
        
    except Exception as e:
        print(f"Band power calculation error: {e}")
        import traceback
        traceback.print_exc()
        return 0.0

def brainflow_thread():
    """Background thread for BrainFlow data acquisition"""
    global current_power, is_streaming, board
    
    try:
        # Setup BrainFlow (using synthetic board for demo)
        params = BrainFlowInputParams()
        board_id = BoardIds.SYNTHETIC_BOARD.value
        board = BoardShim(board_id, params)
        
        board.prepare_session()
        board.start_stream()
        is_streaming = True
        print("BrainFlow streaming started")
        
        while is_streaming:
            time.sleep(0.1)  # 10Hz update rate
            
            # Get recent data
            data = board.get_board_data()
            if data.shape[1] > 0:
                sampling_rate = BoardShim.get_sampling_rate(board_id)
                band_range = BAND_RANGES[current_band]
                
                # Calculate power for selected band
                power = calculate_band_power(data, sampling_rate, band_range)
                current_power = power
                
                # Send to Unity
                send_to_unity(current_band, current_power)
                
                # Simulate accelerometer data (replace with real sensor)
                fake_accel_x = np.sin(time.time() * 0.5) * 0.3
                fake_accel_y = np.cos(time.time() * 0.3) * 0.2  
                fake_accel_z = 0.9 + np.sin(time.time() * 0.1) * 0.1
                send_accelerometer_data(fake_accel_x, fake_accel_y, fake_accel_z)
                
    except Exception as e:
        print(f"BrainFlow thread error: {e}")
        is_streaming = False
    finally:
        if board and board.is_prepared():
            board.stop_stream()
            board.release_session()

@app.route('/')
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simple Brainwave Controller</title>
        <style>
            body { font-family: Arial; margin: 50px; background: #1a1a1a; color: white; }
            .container { max-width: 600px; }
            .band-selector { margin: 20px 0; }
            .band-button { 
                padding: 15px 30px; 
                margin: 10px; 
                font-size: 18px; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer;
                color: white;
                font-weight: bold;
            }
            .delta { background: #8a2be2; }
            .theta { background: #4169e1; }
            .alpha { background: #32cd32; }
            .beta { background: #ff8c00; }
            .gamma { background: #ff1493; }
            .status { margin: 30px 0; padding: 20px; background: #333; border-radius: 8px; }
            .power-bar { 
                width: 100%; 
                height: 30px; 
                background: #555; 
                border-radius: 15px; 
                overflow: hidden; 
                margin: 10px 0;
            }
            .power-fill { height: 100%; background: #32cd32; transition: width 0.3s; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧠 Simple Brainwave Controller</h1>
            
            <div class="band-selector">
                <h3>Select Band:</h3>
                <button class="band-button delta" onclick="selectBand('delta')">DELTA (1-4 Hz)</button>
                <button class="band-button theta" onclick="selectBand('theta')">THETA (4-8 Hz)</button>
                <button class="band-button alpha" onclick="selectBand('alpha')">ALPHA (8-13 Hz)</button>
                <button class="band-button beta" onclick="selectBand('beta')">BETA (13-30 Hz)</button>
                <button class="band-button gamma" onclick="selectBand('gamma')">GAMMA (30-50 Hz)</button>
            </div>
            
            <div class="status">
                <h3>Status:</h3>
                <p>Selected Band: <span id="current-band">alpha</span></p>
                <p>Streaming: <span id="streaming-status">false</span></p>
                <p>Current Power:</p>
                <div class="power-bar">
                    <div class="power-fill" id="power-fill" style="width: 0%"></div>
                </div>
                <p id="power-value">0.000</p>
            </div>
            
            <div>
                <button onclick="toggleStreaming()" id="stream-button" 
                        style="padding: 15px 30px; font-size: 16px; background: #32cd32; 
                               color: white; border: none; border-radius: 8px; cursor: pointer;">
                    START STREAMING
                </button>
            </div>
        </div>
        
        <script>
            let isStreaming = false;
            
            function selectBand(band) {
                fetch('/set_band', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({band: band})
                });
                document.getElementById('current-band').textContent = band.toUpperCase();
            }
            
            function toggleStreaming() {
                const action = isStreaming ? 'stop' : 'start';
                fetch(`/${action}_streaming`, {method: 'POST'})
                    .then(response => response.json())
                    .then(data => {
                        isStreaming = data.streaming;
                        document.getElementById('streaming-status').textContent = isStreaming;
                        document.getElementById('stream-button').textContent = 
                            isStreaming ? 'STOP STREAMING' : 'START STREAMING';
                        document.getElementById('stream-button').style.background = 
                            isStreaming ? '#ff4444' : '#32cd32';
                    });
            }
            
            // Update status every 100ms
            setInterval(() => {
                fetch('/status')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('current-band').textContent = data.band.toUpperCase();
                        document.getElementById('streaming-status').textContent = data.streaming;
                        document.getElementById('power-value').textContent = data.power.toFixed(3);
                        document.getElementById('power-fill').style.width = (data.power * 100) + '%';
                    });
            }, 100);
        </script>
    </body>
    </html>
    """

@app.route('/set_band', methods=['POST'])
def set_band():
    global current_band
    data = request.get_json()
    current_band = data.get('band', 'alpha')
    print(f"Band changed to: {current_band}")
    
    # Send test data when band changes
    send_to_unity(current_band, 0.3)
    
    return jsonify({"success": True, "band": current_band})

@app.route('/start_streaming', methods=['POST'])
def start_streaming():
    global is_streaming
    if not is_streaming:
        threading.Thread(target=brainflow_thread, daemon=True).start()
    
    # Send test message immediately
    send_to_unity(current_band, 0.5)
    
    return jsonify({"streaming": is_streaming})

@app.route('/stop_streaming', methods=['POST']) 
def stop_streaming():
    global is_streaming
    is_streaming = False
    return jsonify({"streaming": is_streaming})

@app.route('/status')
def status():
    return jsonify({
        "band": current_band,
        "power": current_power,
        "streaming": is_streaming
    })

if __name__ == '__main__':
    setup_udp()
    print("🧠 Simple Brainwave Server")
    print("=" * 40)
    print(f"Web UI: http://localhost:5001")
    print(f"Unity UDP: {UNITY_HOST}:{UNITY_PORT}")
    print(f"Accelerometer UDP: {UNITY_HOST}:{ACCEL_PORT}")
    print("=" * 40)
    
    app.run(host='0.0.0.0', port=5001, debug=False)