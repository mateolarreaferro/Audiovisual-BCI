"""
Flask web server for EEG streaming control panel
Provides web UI for mode selection, parameter adjustment, and monitoring
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
import threading
import signal
import sys
import os
from typing import Dict, Any

from .state import app_state, StreamingMode, GeneratorType
from .streaming import streamer

app = Flask(__name__)

@app.route('/')
def index():
    """Main control panel page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get current streaming status and statistics"""
    try:
        state_dict = app_state.to_dict()
        return jsonify({
            'success': True,
            'data': state_dict
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/config', methods=['POST'])
def update_config():
    """Update streaming configuration"""
    try:
        updates = request.get_json()
        
        if not updates:
            return jsonify({
                'success': False,
                'error': 'No configuration data provided'
            }), 400
        
        success = app_state.update_config(updates)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Configuration updated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid configuration parameters'
            }), 400
            
    except Exception as e:
        app_state.add_error_message(f"Config update error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/start', methods=['POST'])
def start_streaming():
    """Start EEG streaming"""
    try:
        if app_state.is_running():
            return jsonify({
                'success': False,
                'error': 'Streaming is already running'
            }), 400
        
        success = streamer.start()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Streaming started successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to start streaming'
            }), 500
            
    except Exception as e:
        app_state.add_error_message(f"Start streaming error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stop', methods=['POST'])
def stop_streaming():
    """Stop EEG streaming"""
    try:
        if not app_state.is_running():
            return jsonify({
                'success': False,
                'error': 'Streaming is not running'
            }), 400
        
        streamer.stop()
        
        return jsonify({
            'success': True,
            'message': 'Streaming stopped successfully'
        })
        
    except Exception as e:
        app_state.add_error_message(f"Stop streaming error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/trigger/<trigger_type>', methods=['POST'])
def manual_trigger(trigger_type):
    """Manually trigger calm or engaged state"""
    try:
        if not app_state.is_running():
            return jsonify({
                'success': False,
                'error': 'Streaming is not running'
            }), 400
        
        config = app_state.get_config()
        
        # Create a manual trigger packet
        from .streaming import EegPacket
        packet = EegPacket()
        
        if trigger_type == 'calm':
            packet.calm = 1
            packet.alpha = 0.8
            packet.arousal = 0.3
        elif trigger_type == 'engaged':
            packet.engaged = 1
            packet.beta = 0.8
            packet.arousal = 0.8
        else:
            return jsonify({
                'success': False,
                'error': f'Unknown trigger type: {trigger_type}'
            }), 400
        
        # Send the trigger packet
        streamer._send_packet(packet)
        
        return jsonify({
            'success': True,
            'message': f'{trigger_type.title()} trigger sent'
        })
        
    except Exception as e:
        app_state.add_error_message(f"Manual trigger error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/test_signal', methods=['POST'])
def send_test_signal():
    """Send a test signal with specified parameters"""
    try:
        if not app_state.is_running():
            return jsonify({
                'success': False,
                'error': 'Streaming is not running'
            }), 400
        
        data = request.get_json() or {}
        
        from .streaming import EegPacket
        packet = EegPacket(
            alpha=float(data.get('alpha', 0.5)),
            beta=float(data.get('beta', 0.5)),
            theta=float(data.get('theta', 0.5)),
            delta=float(data.get('delta', 0.5)),
            arousal=float(data.get('arousal', 0.5)),
            calm=int(data.get('calm', 0)),
            engaged=int(data.get('engaged', 0))
        )
        
        streamer._send_packet(packet)
        
        return jsonify({
            'success': True,
            'message': 'Test signal sent'
        })
        
    except Exception as e:
        app_state.add_error_message(f"Test signal error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/data')
def list_data_files():
    """List available CSV data files"""
    try:
        data_dir = 'data'
        if not os.path.exists(data_dir):
            return jsonify([])
        
        files = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(data_dir, filename)
                stat = os.stat(filepath)
                files.append({
                    'name': filename,
                    'size': stat.st_size,
                    'modified': stat.st_mtime
                })
        
        # Sort by modification time, newest first
        files.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify(files)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/data/<filename>')
def download_data_file(filename):
    """Download a CSV data file"""
    try:
        data_dir = os.path.abspath('data')
        return send_from_directory(data_dir, filename, as_attachment=True)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 404

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\n[Server] Received signal {signum}, shutting down...")
    streamer.stop()
    sys.exit(0)

def create_app():
    """Application factory"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    return app

if __name__ == '__main__':
    app = create_app()
    
    print("=" * 60)
    print("🧠 OpenBCI EEG Terrain Visualizer - Control Panel")
    print("=" * 60)
    print("Starting Flask server...")
    print("Web interface: http://localhost:5000")
    print("UDP Output: Unity (127.0.0.1:7777)")
    print("UDP Input: Live Relay (0.0.0.0:7778)")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    # Start Flask app
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n[Server] Keyboard interrupt received")
        streamer.stop()
    except Exception as e:
        print(f"[Server] Error: {e}")
        streamer.stop()
    finally:
        print("[Server] Shutdown complete")