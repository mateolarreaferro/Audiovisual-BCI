# BioMus Project Structure

## 📁 File Organization

```
BioMus/
├── 📄 main.py                    # FastAPI application entry point
│   └── Routes: /, /api/*, /ws/stream
│
├── 🧠 openbci_service.py         # BrainFlow integration
│   ├── GanglionService class
│   ├── Connection management
│   ├── Stream control
│   ├── Time series processing
│   └── Band power computation (Welch PSD)
│
├── 📡 osc_sender.py              # OSC communication
│   ├── OSCSender class
│   ├── Raw data transmission (/eeg/raw)
│   └── Band power transmission (/eeg/bands)
│
├── 🌐 templates/
│   └── index.html                # Web UI
│       ├── ElevenLabs-inspired design
│       ├── Chart.js visualizations
│       ├── WebSocket client
│       └── Real-time updates
│
├── 📋 requirements.txt           # Python dependencies
│
├── 🛠️ setup.sh                   # Automated setup script
│
├── 📖 README.md                  # Full documentation
│
├── 🚀 QUICKSTART.md              # Quick start guide
│
├── 🗂️ .gitignore                # Git ignore patterns
│
└── 📝 instructions.rtf           # Original design spec

```

## 🔄 Data Flow

```
OpenBCI Ganglion (Hardware)
         ↓
   [BrainFlow SDK]
         ↓
  openbci_service.py
    ├─→ Time Series Mode
    │   ├─→ WebSocket → Browser
    │   └─→ OSC → External Apps
    │
    └─→ Bands Mode
        ├─→ Welch PSD
        ├─→ WebSocket → Browser
        └─→ OSC → External Apps
```

## 🎯 Key Components

### Backend (Python/FastAPI)

**main.py**
- REST API endpoints for device control
- WebSocket endpoint for real-time streaming
- Template rendering for web UI

**openbci_service.py**
- Board connection/disconnection
- Stream start/stop
- Data acquisition from BrainFlow buffer
- Signal processing (downsampling, PSD, band integration)
- Test signal control

**osc_sender.py**
- UDP OSC client management
- Message formatting and transmission
- Configurable endpoints and data types

### Frontend (HTML/CSS/JS)

**index.html**
- Responsive dark theme UI
- Real-time Chart.js visualizations
- WebSocket client for live data
- Interactive controls for all features

## 🔌 API Endpoints

### REST API
- `GET /` - Web interface
- `POST /api/connect` - Connect to Ganglion
- `POST /api/disconnect` - Disconnect from board
- `GET /api/status` - Get connection/streaming status
- `POST /api/start` - Start data stream
- `POST /api/stop` - Stop data stream
- `POST /api/test_signal` - Toggle test signal
- `POST /api/osc_config` - Configure OSC output

### WebSocket
- `WS /ws/stream` - Real-time data stream
  - Accepts: `{mode, window_sec, interval_ms}`
  - Sends: `{type: "timeseries"|"bands", ...}`

## 📊 Signal Processing Pipeline

### Time Series Mode
1. Pull N samples from BrainFlow buffer
2. Downsample if needed (max 512 points)
3. Format as [channels][samples]
4. Send to UI and OSC

### Bands Mode
1. Pull N samples (based on window_sec)
2. For each channel:
   - Detrend (linear)
   - Compute Welch PSD
   - Integrate power in frequency bands:
     - Delta: 1-4 Hz
     - Theta: 4-8 Hz
     - Alpha: 8-13 Hz
     - Beta: 13-30 Hz
     - Gamma: 30-45 Hz
3. Format as [channels][bands]
4. Send to UI and OSC

## 🎨 UI Features

- **Connection Panel**: Serial/MAC input, connect/disconnect buttons
- **Stream Control**: Start/stop, test signal toggle
- **Mode Selector**: Time series vs. bands view
- **Main Visualization**: Chart.js real-time plots
- **Configuration Panel**: Window length, update interval
- **OSC Panel**: IP/port, enable/disable, content selection
- **Status Indicators**: Connection state, streaming state, OSC state

## 🧪 Testing

### Without Hardware
1. Click "Connect" (may fail but sets up state)
2. Click "Test signal" to enable synthetic data
3. Click "Start stream"
4. Observe synthetic square wave in UI

### With Ganglion
1. Power on Ganglion
2. Enter correct serial port/MAC
3. Click "Connect" and "Start stream"
4. Attach electrodes or use test signal

## 🔧 Configuration Options

### Board Parameters (openbci_service.py)
- `board_id`: BoardIds.GANGLION_BOARD
- `buffer_size`: 45000 samples default
- `timeout`: 15 seconds default

### Signal Processing
- `window_sec`: 1-10 seconds (default: 4)
- `max_points`: 512 for time series display
- `fft_len`: Nearest power of 2 to window size

### OSC Settings
- `ip`: Default 127.0.0.1
- `port`: Default 9000
- `send_raw`: Boolean
- `send_bands`: Boolean

## 📦 Dependencies

### Python Packages
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `brainflow` - OpenBCI SDK
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `python-osc` - OSC protocol
- `jinja2` - Template engine

### External Libraries
- BrainFlow native libraries (platform-specific)

### Frontend Libraries (CDN)
- Chart.js - Visualization library

## 🚀 Deployment Notes

### Local Development
```bash
uvicorn main:app --reload
```

### Production Considerations
- Use production ASGI server (gunicorn + uvicorn workers)
- Add HTTPS support
- Implement authentication if needed
- Add data validation and error handling
- Consider rate limiting for API endpoints
- Add logging and monitoring

## 🔮 Future Enhancements

- [ ] Data recording to CSV/HDF5
- [ ] Playback mode for recorded data
- [ ] Advanced filtering (notch, bandpass)
- [ ] ICA artifact rejection
- [ ] Multiple board support
- [ ] Custom OSC message schemas
- [ ] Per-channel controls (gain, visibility)
- [ ] Real-time impedance checking
- [ ] Spectrogram visualization
- [ ] Custom band definitions
