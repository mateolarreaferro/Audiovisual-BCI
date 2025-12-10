# Simple Brainwave Terrain Visualizer

A minimal brainwave visualization system with Unity terrain and Python/Flask control interface.

## Features

- **Unity Side**: Dynamic terrain that responds to brainwave band power
- **Python Side**: Simple web interface to select which band to visualize
- **Camera Tilt**: Camera responds to accelerometer data (simulated)
- **Real-time**: UDP communication between Python and Unity

## Setup

### Python Requirements
```bash
pip install flask brainflow numpy
```

### Unity Setup
1. Open Unity project
2. Create empty GameObject
3. Attach `SimpleTerrainVisualizer` script
4. Attach `AccelerometerCamera` to Main Camera
5. Press Play

### Usage
1. Run Python server: `python simple_brainwave_server.py`
2. Open web interface: http://localhost:5000
3. Select brainwave band (Delta, Theta, Alpha, Beta, Gamma)
4. Click "START STREAMING"
5. Watch Unity terrain respond to selected band

## Network Ports
- **Port 12345**: Brainwave data (format: "band:power")
- **Port 12346**: Accelerometer data (format: "x,y,z")
- **Port 5000**: Web interface

## Files
- `SimpleTerrainVisualizer.cs` - Unity terrain visualization
- `AccelerometerCamera.cs` - Camera tilt from accelerometer
- `simple_brainwave_server.py` - Python server with web UI

That's it. Simple and clean.