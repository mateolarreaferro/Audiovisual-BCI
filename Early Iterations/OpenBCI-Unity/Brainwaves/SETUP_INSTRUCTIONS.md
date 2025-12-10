# BrainFlow Unity Integration Setup Instructions

## Prerequisites
- Unity 2021.3 or newer
- OpenBCI hardware (Cyton board or compatible)
- OpenBCI GUI installed and configured

## Setup Steps

### 1. Install NuGetForUnity
1. Open your Unity project
2. Download NuGetForUnity from: https://github.com/GlitchEnzo/NuGetForUnity/releases
3. Download the `.unitypackage` file from the latest release
4. In Unity, go to `Assets > Import Package > Custom Package`
5. Select the downloaded NuGetForUnity package and import it

### 2. Install BrainFlow Package
1. After NuGetForUnity is installed, go to `NuGet > Manage NuGet Packages` in Unity
2. Search for "brainflow"
3. Click Install on the brainflow package
4. Wait for the installation to complete

### 3. Scene Setup
1. Open your scene in Unity (or create a new one)
2. Create an empty GameObject: `GameObject > Create Empty`
3. Name it "BrainFlowManager"
4. Add the `BrainFlowManager.cs` script to this GameObject
5. Configure the script parameters in the Inspector:
   - **Board Type**: Select your OpenBCI board type (e.g., CYTON_BOARD for Cyton)
   - **Serial Port**: Enter your device's serial port (e.g., "COM3" on Windows, "/dev/tty.usbserial-xxx" on Mac)
   - **Start Stream On Awake**: Check this to automatically start streaming

### 4. Create Visualization Object
1. Create another empty GameObject: `GameObject > Create Empty`
2. Name it "BrainwaveVisualizer"
3. Add the `BrainwaveVisualizer.cs` script to this GameObject
4. Configure visualization settings:
   - **Visual Mode**: Choose how to visualize (Scale, Color, Rotation, Position, or Combined)
   - **Smoothing Speed**: Adjust for smoother transitions
   - **Amplification Factor**: Increase to make effects more pronounced
   - **Band Power Mapping**: Select which frequency bands to use

### 5. Add UI Display (Optional)
1. Create a Canvas if you don't have one: `GameObject > UI > Canvas`
2. Create an empty GameObject as a child of the Canvas
3. Add the `BrainwaveUIDisplay.cs` script to it
4. The script will automatically create basic UI elements if they're missing

### 6. Configure OpenBCI Connection

#### For Direct Serial Connection:
1. Connect your OpenBCI board via USB
2. Find the serial port:
   - Windows: Check Device Manager for COM port
   - Mac: Run `ls /dev/tty.*` in Terminal
   - Linux: Run `ls /dev/ttyUSB*` in Terminal
3. Enter the serial port in the BrainFlowManager component

#### For OpenBCI GUI Streaming:
If you want to use OpenBCI GUI's LSL streaming:
1. In OpenBCI GUI, set up LSL streaming
2. Modify the BrainFlowManager script to use `BoardIds.STREAMING_BOARD`
3. Set the streaming parameters according to your LSL configuration

### 7. Testing the Setup

1. **Start OpenBCI Hardware**:
   - Power on your OpenBCI board
   - Ensure electrodes are properly connected

2. **Run Unity Scene**:
   - Press Play in Unity Editor
   - Check the Console for connection status
   - You should see "Board initialized successfully" if connected

3. **Verify Data Reception**:
   - Look for the UI display showing band powers
   - The visualization object should respond to brain activity
   - Try closing your eyes to see alpha wave increases

## Troubleshooting

### Common Issues:

1. **"Failed to initialize board" error**:
   - Check serial port is correct
   - Ensure OpenBCI board is powered on
   - Close OpenBCI GUI if using direct serial connection
   - Check board isn't being used by another application

2. **No data visualization**:
   - Verify streaming is started (check IsStreaming in Inspector)
   - Ensure electrodes have good contact
   - Check Console for any error messages
   - Verify the correct board type is selected

3. **DLL/Library errors**:
   - Ensure BrainFlow package is properly installed via NuGet
   - On Windows, you may need Visual C++ Redistributables
   - Check that Unity's platform settings match your system architecture

4. **Performance issues**:
   - Reduce buffer size in BrainFlowManager
   - Decrease sampling rate if possible
   - Use simpler visualization modes

## Testing with Synthetic Data

For testing without hardware:
1. Change Board Type to `SYNTHETIC_BOARD` in BrainFlowManager
2. This will generate synthetic EEG data for testing
3. Useful for verifying the visualization pipeline works

## Next Steps

- Experiment with different visualization modes
- Adjust band power weights for your specific use case
- Create custom visualizations by extending the scripts
- Add neurofeedback training features
- Implement data recording and playback

## Additional Resources

- BrainFlow Documentation: https://brainflow.readthedocs.io/
- OpenBCI Documentation: https://docs.openbci.com/
- Unity Input System: https://docs.unity3d.com/Manual/com.unity.inputsystem.html