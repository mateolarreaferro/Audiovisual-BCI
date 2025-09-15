# Basic SSVEP Prototype — OpenBCI Cyton + Daisy (16-channel)

A small, reliable SSVEP demo that runs in real time with OpenBCI Cyton + Daisy (16 channels, ~125 Hz). No novelty. Just what works.

## Goals

- 3–4 on-screen flicker targets
- Real-time frequency selection using PSD/SNR
- Stable selection after ~0.5–1.0 s fixation

## Hardware and Setup

**Board**: OpenBCI Cyton + Daisy (16 channels, ~125 Hz)

**Reference/Bias**: SRB2 reference, BIAS to BIAS

**Electrodes (recommended)**: O1, Oz, O2, PO3, POz, PO4, P3, Pz, P4 (use extra parietal/occipital sites as available)

**Display**: 60 Hz monitor, windowed or fullscreen with reliable vsync

**Flicker frequencies (60 Hz friendly)**: 7.5, 10, 12, 15 Hz

Seat the participant ~60–80 cm from the screen. Keep cables still, minimize blinks and jaw movement during trials.

## Project Structure

```
ssvep/
  README.md
  requirements.txt
  config.py
  run_stimulus.py          # Visual flicker UI (frame-locked)
  run_ssvep.py             # Acquisition + real-time detection

  src/
    acquisition.py         # BrainFlow reader for Cyton+Daisy (16 ch)
    filters.py             # Bandpass + notch
    detector_psd.py        # Welch PSD + SNR scoring
    utils.py               # Ring buffer and stable-vote helper
    synthetic.py           # Simple synthetic generator (optional)
```

## Environment and Install

**Python**: 3.10 or newer

### Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

If PsychoPy is heavy on the machine, Pygame can substitute, but PsychoPy gives better frame timing.

## Configuration

Edit `config.py` for your setup:

```python
BOARD_ID = "CYTON_DAISY"     # or "CYTON" if using 8-ch
SERIAL_PORT = "COM5"         # e.g. COM5 on Windows, /dev/tty.usbserial-DMxxxx on macOS
WINDOW_SEC = 2.0             # analysis window length
STEP_SEC = 0.2               # hop size
BANDPASS = (6.0, 45.0)       # Hz
NOTCH = 60.0                 # set to 50.0 for 50 Hz regions
FREQS = [7.5, 10.0, 12.0, 15.0]
HARMONICS = 2                # 1 = fundamental only, 2 = add 2nd harmonic
USE_CHANNELS = None          # None = use all EEG channels
SNR_NEIGHBOR_BW = 1.0        # sidebands for noise floor (excludes ±0.3 Hz around peak)
VOTE_HOLD_MS = 500           # must hold the top class this long before select
LOGGING = True
```

**Notes**:
- On macOS, SERIAL_PORT looks like `/dev/tty.usbserial-XXXX`
- Sampling rate is read from BrainFlow at runtime. Expect ~125 Hz on Cyton + Daisy.

## Run

Start the visual stimulus in one terminal:

```bash
python run_stimulus.py
```

Start acquisition and detection in another terminal:

```bash
python run_ssvep.py
```

You should see logs like:

```
[INFO] Sampling rate: 125 Hz, EEG chans: 16
Pred=10.0 SNR=2.45 :: 7.5:1.22 | 10.0:2.45 | 12.0:1.18 | 15.0:0.97
[SELECT] 10.0 Hz
```

### Command Line Options

**Stimulus Options**:
```bash
python run_stimulus.py --freqs 7.5 10.0 12.0 15.0 --fullscreen
python run_stimulus.py --test  # Test mode with simulated feedback
```

**Detection Options**:
```bash
# Use synthetic data for testing
python run_ssvep.py --synthetic 10.0

# Real OpenBCI with custom settings  
python run_ssvep.py --board CYTON_DAISY --port COM5 --window 2.5

# Occipital channels only
python run_ssvep.py --occipital-only

# Custom channel selection (0-based indices)
python run_ssvep.py --channels 0 1 2 7 8 9
```

## Usage Tips

- Place the flicker boxes far apart
- Ask the participant to fixate on one box at a time
- Keep the PsychoPy window in focus and vsync on

## How It Works

1. **Acquisition**: BrainFlow pulls 16-ch EEG from Cyton + Daisy
2. **Preprocess**: Bandpass 6–45 Hz and 50/60 Hz notch
3. **Detection**: Welch PSD averaged across selected channels. Compute SNR around fundamental and optional 2nd harmonic. Pick the highest SNR frequency.
4. **Stability**: A small "hold" timer confirms the selection only if the same frequency stays on top for VOTE_HOLD_MS.

## Electrode Montage (suggested)

Back-of-head emphasis: O1, Oz, O2, PO3, POz, PO4, P3, Pz, P4

Keep impedances low and stable. Use conductive gel or saline as appropriate.

## Troubleshooting

### No detection or unstable output

- Confirm printed sampling rate ~125 Hz
- Verify SERIAL_PORT and that the dongle is paired
- Increase WINDOW_SEC to 2.5–3.0 s and keep STEP_SEC at 0.2 s
- Keep HARMONICS = 2, and ensure NOTCH matches mains power
- Reduce channels to occipital/parietal if noisy

### Stimulus timing issues

- Ensure monitor refresh is truly 60 Hz
- Keep the PsychoPy window active. Avoid screen recording overlays

## Offline Training with Public Datasets

If you are just getting started or want a stronger initialization before
recording your own calibration data, you can pre-train the classifier using the
"Benchmark SSVEP" dataset released by Nakanishi et al. (PLoS ONE, 2015). This
dataset contains multi-subject recordings at many stimulation frequencies,
including 10 Hz and 15 Hz.

1. Download the dataset archive from
   [https://figshare.com/articles/dataset/Benchmark_Dataset_for_SSVEP-based_BCI/2003670](https://figshare.com/articles/dataset/Benchmark_Dataset_for_SSVEP-based_BCI/2003670)
   and extract the `.mat` files to a local folder.
2. Run the offline trainer to build TRCA/FBTRCA templates and save a
   calibration session:

   ```bash
   cd SSVEP
   python offline_trca_training.py \
       --dataset-root /path/to/extracted/dataset \
       --subjects S1 S2 \
       --frequencies 10 15 \
       --window 2.0 --start-offset 0.5
   ```

   The script prints cross-validation accuracy, selects the most informative
   channels, estimates an adaptive detection threshold, and stores the resulting
   calibration in `calibration_data/` so it can be loaded via the main UI (`L`
   key).
3. Launch `run_integrated_ssvep.py`, press `L`, and choose the newly created
   calibration session to bootstrap real-time detection before collecting your
   own data.

The offline training workflow relies on the improved TRCA/FBTRCA classifier in
`ssvep_bci/modules/ssvep_classifier.py`, which tends to offer sharper class
separation than the original PSD-only approach when enough template trials are
available.

## Definition of Done

- Real-time classification among 3–4 targets with visible logs of predicted frequency
- Selection triggers after about 0.5–1.0 s of steady gaze
- Runs for at least 5 minutes without crashing. Handles brief data dropouts
- All main parameters are in config.py

## Testing with Synthetic Data

For quick validation without hardware:

```bash
# Test detection pipeline with synthetic 12 Hz SSVEP
python run_ssvep.py --synthetic 12.0 --duration 30

# Test individual components
python src/detector_psd.py    # Test PSD detector
python src/filters.py         # Test filters
python src/synthetic.py       # Test synthetic data generation
```

## Stretch Upgrades (optional, nice to have)

### Occipital-only switch
Add a CLI flag `--occipital-only` that restricts channels to O1, Oz, O2, PO3, POz, PO4, Pz. ✅ Implemented

### Synthetic mode
Add `python run_ssvep.py --synthetic 10.0` to bypass BrainFlow and feed synthetic.synth_ssvep(10.0, fs=125, dur=4, n_channels=8) through the pipeline for quick validation. ✅ Implemented

### CCA detector
Implement a second detector that uses CCA with sin/cos reference signals over 1–2 s windows and a simple majority vote. Keep PSD as default. (Future work)

## Safety

This is a research prototype, not a medical device. Provide breaks, avoid excessive flicker exposure, and follow standard lab safety practices.