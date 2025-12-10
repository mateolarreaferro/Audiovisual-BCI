# BrainFlow Unity Installation Guide

## Quick Start (Without Hardware)

The project is now set up to work immediately with simulated brainwave data, so you can test the visualization system right away:

1. Open Unity
2. Open the scene (Assets/Scenes/SampleScene)
3. Go to **Tools > Brainwave > Setup Wizard**
4. Leave "Use Real OpenBCI Hardware" unchecked
5. Click **Setup Scene**
6. Press Play - You should see a sphere responding to simulated brainwaves!

## Installing BrainFlow for Real OpenBCI Hardware

### Step 1: Install NuGetForUnity

1. Go to: https://github.com/GlitchEnzo/NuGetForUnity/releases
2. Download the latest `.unitypackage` file (e.g., `NuGetForUnity.3.1.4.unitypackage`)
3. In Unity, go to **Assets > Import Package > Custom Package**
4. Select the downloaded file and click **Import**

### Step 2: Install BrainFlow Package

1. After NuGetForUnity is installed, you'll see a new **NuGet** menu in Unity
2. Go to **NuGet > Manage NuGet Packages**
3. In the search box, type: `brainflow`
4. Click **Install** on the brainflow package
5. Wait for installation to complete

### Step 3: Enable BrainFlow in Your Project

1. Go to **Edit > Project Settings > Player**
2. Expand **Other Settings**
3. Find **Scripting Define Symbols**
4. Add: `BRAINFLOW_INSTALLED`
5. Press Enter and wait for Unity to recompile

### Step 4: Configure for Your OpenBCI Board

1. Use the Setup Wizard: **Tools > Brainwave > Setup Wizard**
2. Check "Use Real OpenBCI Hardware"
3. Enter your serial port:
   - **Windows**: `COM3` or `COM4` (check Device Manager)
   - **Mac**: `/dev/tty.usbserial-XXXXXXX` (run `ls /dev/tty.*` in Terminal)
   - **Linux**: `/dev/ttyUSB0` (run `ls /dev/ttyUSB*` in Terminal)
4. Click **Setup Scene**

## Troubleshooting

### "BrainFlow is not installed" Warning
This is normal before installing BrainFlow. The system will use simulated data until you complete the installation steps above.

### Can't Find NuGetForUnity Menu
- Make sure you imported the package correctly
- Try restarting Unity
- Check Console for any import errors

### BrainFlow Package Not Found in NuGet
- Make sure you have internet connection
- Try refreshing the package list
- Check if NuGet source is configured correctly in NuGet settings

### Serial Port Not Working
1. Make sure OpenBCI board is powered on
2. Close OpenBCI GUI if it's running (can't share serial port)
3. Verify serial port name:
   - Windows: Device Manager > Ports (COM & LPT)
   - Mac: `ls /dev/tty.usbserial*` in Terminal
   - Linux: `ls /dev/ttyUSB*` or `ls /dev/ttyACM*` in Terminal

### DLL Errors on Windows
Install Visual C++ Redistributables:
- Download from Microsoft: https://aka.ms/vs/17/release/vc_redist.x64.exe

### Unity Crashes on Play
- Reduce buffer size in BrainFlowManager
- Try SYNTHETIC_BOARD first to test the setup
- Check Console for error messages before crash

## Testing Your Setup

### With Simulated Data:
1. Keep boardType as SYNTHETIC_BOARD
2. Press Play
3. You should see the sphere changing size/color based on simulated brainwaves

### With Real Hardware:
1. Connect OpenBCI board via USB
2. Attach electrodes properly
3. Set boardType to your board model (CYTON_BOARD, etc.)
4. Enter correct serial port
5. Press Play
6. Check Console for "Board initialized successfully"

## Visual Modes

The BrainwaveVisualizer component offers several modes:
- **Scale**: Object size changes with brain activity
- **Color**: Color changes based on intensity
- **Rotation**: Rotation speed varies with signals
- **Position**: Object moves based on activity
- **Combined**: All effects together

## Band Powers Explained

- **Delta (1-4 Hz)**: Deep sleep, unconscious
- **Theta (4-8 Hz)**: Drowsiness, meditation
- **Alpha (8-13 Hz)**: Relaxed, eyes closed
- **Beta (13-30 Hz)**: Active thinking, focus
- **Gamma (30-50 Hz)**: High-level cognitive processing

## Next Steps

1. Experiment with different visualization modes
2. Adjust amplification factors for better response
3. Try closing your eyes to see alpha waves increase
4. Create custom visualizations using the OnBandPowerCalculated event
5. Record sessions for later playback and analysis

## Support

- BrainFlow Documentation: https://brainflow.readthedocs.io/
- OpenBCI Documentation: https://docs.openbci.com/
- Unity Forums: https://forum.unity.com/