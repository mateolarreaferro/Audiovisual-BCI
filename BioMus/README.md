# BioMus

Real-time EEG visualization for OpenBCI Ganglion with OSC output.

## Setup

```bash
./setup.sh
```

## Run

```bash
./run.sh
```

## Usage

### Bluetooth (Recommended)
1. Turn on your Ganglion board (blue LED should blink)
2. Open http://localhost:8000
3. **Leave the input field empty**
4. Click **Connect** (BrainFlow will auto-discover via Bluetooth)
5. Click **Start Stream**
6. Switch between **Time series** and **Bands** modes

### BLED112 Dongle
1. Plug in BLED112 dongle
2. Find serial port: `ls /dev/tty.usbmodem*` or `ls /dev/cu.usbmodem*`
3. Enter the port in the input field (e.g., `/dev/tty.usbmodem14101`)
4. Click **Connect** → **Start Stream**

## OSC Output

- Set IP:port (default: 127.0.0.1:9000)
- Toggle raw/bands output
- Click **Enable OSC**
- Messages: `/eeg/raw` and `/eeg/bands`

## Features

- Native Bluetooth LE support (auto-discovery)
- BLED112 dongle support
- Real-time visualization with smooth animations
- Frequency band analysis (delta, theta, alpha, beta, gamma)
- OSC output for integration with other tools
- Configurable window length and refresh rate
