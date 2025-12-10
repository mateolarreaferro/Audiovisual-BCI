#!/usr/bin/env python3
"""
EEG Terrain Visualizer - Main Entry Point
Starts the Flask web server and streaming system
"""
import sys
import os

# Add app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

if __name__ == '__main__':
    from app.server import create_app
    
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
    
    try:
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n[Server] Shutdown complete")
    except Exception as e:
        print(f"[Server] Error: {e}")
        sys.exit(1)