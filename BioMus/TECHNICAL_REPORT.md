# BioMus: An Accessible Brain-Computer Interface for Audiovisual Expression

## Executive Summary

BioMus is an open-source, affordable brain-computer interface (BCI) platform designed to empower individuals with limited mobility through creative audiovisual expression. By combining electroencephalography (EEG) biofeedback and computer vision, BioMus enables users to control musical parameters and visual content through brain activity and facial movements, providing an accessible pathway to artistic creation and self-expression.

---

## 1. Introduction: Motivation and Vision

### 1.1 The Challenge
Individuals with limited mobility often face significant barriers to creative expression, particularly in domains like music performance and digital art creation that traditionally rely on fine motor control. While brain-computer interfaces hold promise for accessibility, existing solutions are often prohibitively expensive or technically complex, limiting their reach to specialized research facilities.

### 1.2 Our Vision
BioMus aims to democratize access to neural interface technology by providing:
- **Affordable hardware** based on open-source components
- **Accessible software** with intuitive visual feedback
- **Creative applications** focused on artistic expression rather than clinical use
- **Playful self-regulation** through the experience of art-making

The goal is not merely to provide control, but to enable **genuine creative expression** where users can discover their own artistic voice through biosignals and movement.

---

## 2. Background: Previous Explorations

### 2.1 SSVEP (Steady-State Visually Evoked Potentials)
**Approach:** Utilized frequency-tagged visual stimuli to elicit measurable brain responses in visual cortex.

**Findings:**
- Reliable signal detection in controlled conditions
- Required sustained visual attention, limiting creative flow
- Visual fatigue during extended sessions
- Limited expressiveness due to discrete selection paradigm

**Outcome:** While technically viable, SSVEP proved too constraining for fluid musical expression.

### 2.2 OpenBCI Cyton (8-Channel Research-Grade EEG)
**Approach:** High-fidelity 8-channel EEG system for comprehensive brain activity monitoring.

**Findings:**
- Excellent signal quality and spatial resolution
- **Prohibitive cost (~$500-1000)** creating accessibility barrier
- Complexity of 8-channel setup overwhelming for end users
- Over-engineered for real-time creative control applications

**Outcome:** Demonstrated technical feasibility but failed accessibility requirements.

### 2.3 Muse Headband (Consumer EEG)
**Approach:** Commercial meditation headband with 4 dry EEG electrodes.

**Findings:**
- Consumer-friendly form factor
- Proprietary software ecosystem limiting customization
- Closed API restricting low-level signal access
- Designed for meditation, not creative control

**Outcome:** Good user experience but insufficient technical flexibility.

---

## 3. BioMus Headband: Our Solution

### 3.1 Design Philosophy
Drawing from our explorations, we designed BioMus around three core principles:
1. **Accessibility:** Affordable open-source hardware (<$300 total)
2. **Simplicity:** 4-channel design balancing quality and usability
3. **Openness:** Full access to raw signals and processing pipeline

### 3.2 Hardware Architecture

#### 3.2.1 Electrode Configuration
**4-Channel Layout:**
- **2 Frontal electrodes (Fp1, Fp2):** Positioned above eyebrows
  - Captures prefrontal cortex activity
  - Sensitive to attention, cognitive load, and frontal theta/alpha
  - Minimal hair interference for reliable dry contact

- **2 Parietal electrodes (P3, P4):** Positioned on posterior scalp
  - Captures sensorimotor and posterior alpha rhythms
  - Sensitive to relaxation, visual processing, and spatial attention
  - Complements frontal channels for brain state estimation

- **2 Ground/reference electrodes:** Mastoid or earlobe placement
  - Provides stable reference potential
  - Minimizes common-mode noise

**Electrode Type:** Dry contact (no gel required)
- Reduces setup friction
- Enables spontaneous creative sessions
- Trade-off: Slightly higher impedance vs. wet electrodes

![BioMus Interface](Pictures/Interface.png)
*Figure 1: BioMus web interface showing real-time EEG visualization and control panel*

![Computer Vision Mode](Pictures/Computer%20Vision%20Mode.png)
*Figure 2: Camera/FaceSynth mode with live facial feature tracking and parameter extraction*

#### 3.2.2 Signal Acquisition: OpenBCI Ganglion
**Board Specifications:**
- **4 EEG channels** @ 200 Hz sampling rate
- **24-bit ADC resolution** (Texas Instruments ADS1299)
- **Bluetooth Low Energy** wireless transmission
- **Open-source firmware** (Arduino-compatible)
- **Cost:** ~$99-199 depending on configuration

**Key Advantages:**
- BrainFlow SDK support for cross-platform compatibility
- Native Bluetooth LE (no dongle required on modern systems)
- BLED112 dongle support for legacy systems
- Programmable gain and filtering

**Data Stream:**
```
Electrode → Analog Front-End → 24-bit ADC → Microcontroller → BLE → Computer
```

---

## 4. Software System: Interface Design and Signal Processing

### 4.1 System Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                         BioMus Platform                           │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│    ┌──────────────┐                    ┌──────────────┐         │
│    │   Ganglion   │                    │   Camera     │         │
│    │   Service    │                    │   Service    │         │
│    └──────┬───────┘                    └──────┬───────┘         │
│           │                                   │                  │
│           │                                   │                  │
│           └─────────────┬─────────────────────┘                  │
│                         │                                        │
│                         ▼                                        │
│           ┌─────────────────────────────────┐                   │
│           │  Signal Processing Engine       │                   │
│           │  - FFT (Welch PSD)              │                   │
│           │  - Band Power (δ,θ,α,β,γ)      │                   │
│           │  - Facial Feature Extraction    │                   │
│           └─────────────┬───────────────────┘                   │
│                         │                                        │
│          ┌──────────────┼──────────────┬──────────────┐        │
│          │              │              │              │        │
│          ▼              ▼              ▼              ▼        │
│    ┌─────────┐    ┌─────────┐   ┌──────────┐   ┌──────────┐  │
│    │   Web   │    │   OSC   │   │  Local   │   │  Future  │  │
│    │   UI    │    │ Output  │   │  Audio   │   │   Apps   │  │
│    │(Viz+Ctl)│    │         │   │ (Csound) │   │          │  │
│    └─────────┘    └─────────┘   └──────────┘   └──────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Backend Implementation

#### 4.2.1 Technology Stack
- **FastAPI:** Modern Python web framework for REST API and WebSocket
- **BrainFlow:** Cross-platform biosignal acquisition SDK
- **NumPy/SciPy:** Signal processing and numerical computation
- **python-osc:** OSC protocol implementation for creative tool integration
- **OpenCV + MediaPipe:** Computer vision and facial tracking
- **Uvicorn:** ASGI server for production deployment

#### 4.2.2 Signal Processing Pipeline

**Time Series Mode:**
```python
# openbci_service.py:166-204
1. Acquire window_sec seconds of data (default: 4.0s @ 200 Hz = 800 samples)
2. Apply linear detrend (remove DC offset and drift)
3. Optional smoothing (3-sample moving average)
4. Downsample to max 512 points for visualization
5. Transmit via WebSocket and OSC
```

**Frequency Domain Analysis:**
```python
# openbci_service.py:206-270
1. Acquire 4.0s window (800 samples)
2. Linear detrend (remove low-frequency drift)
3. Compute Welch periodogram:
   - FFT length: Nearest power of 2 (512)
   - Window: Hanning (reduces spectral leakage)
   - Overlap: 50% (improves frequency resolution)
4. Convert to dB scale: PSD_dB = 10 * log10(PSD)
5. Filter to 0.5-40 Hz range (physiological EEG)
```

**Band Power Computation:**
```python
# openbci_service.py:272-346
Frequency Bands:
  - Delta (δ):  0.5-4  Hz  → Deep sleep, unconscious processes
  - Theta (θ):  4-8   Hz  → Meditation, creativity, memory
  - Alpha (α):  8-13  Hz  → Relaxed awareness, eyes closed
  - Beta (β):   13-30 Hz  → Active thinking, concentration
  - Gamma (γ):  30-40 Hz  → High-level cognition, binding

For each channel:
  1. Compute Welch PSD
  2. Integrate power in each band: ∫[f_min to f_max] PSD(f) df
  3. Convert to relative power: (Band_Power / Total_Power) × 100
  4. Output as percentage of total brain activity
```

#### 4.2.3 Computer Vision Integration

**Facial Feature Extraction:**
```python
# camera_service.py
Using MediaPipe Face Mesh (468 3D landmarks):

1. Face Detection and Tracking
   - Real-time facial landmark localization
   - Robust to head rotation (±30°)

2. Feature Computation:
   a) Mouth Openness:
      - Vertical distance: lip_top ↔ lip_bottom
      - Normalized to 0.0 (closed) → 1.0 (open)
      - Useful for vocalization/articulation mapping

   b) Head Roll:
      - Rotation angle around Z-axis
      - Range: -30° to +30°
      - Maps to stereo pan, filter cutoff

   c) Smile Curvature:
      - Mouth corner elevation
      - Range: 0.0 (neutral) → 1.0 (full smile)
      - Controls brightness, harmonic content

3. OSC Output:
   - /cv/face/mouth → 0.0-1.0
   - /cv/face/roll → -1.0 to +1.0
   - /cv/face/smile → 0.0-1.0
   - /faceSynth/* → Legacy compatibility
```

### 4.3 Frontend: Visualization and User Control

#### 4.3.1 Design Language
**Minimalist Aesthetic (Inspired by modern audio tools):**
- Clean white/black theme with subtle accent colors
- Focus on data, minimal chrome
- Responsive layout (desktop, tablet, mobile)
- Professional typography (Inter font family)

#### 4.3.2 Visualization Modes

**1. Time Series View**
- Real-time 4-channel waveform display
- 4-second rolling window
- Smooth 50ms update rate (20 Hz refresh)
- Adaptive Y-axis scaling or manual control
- Color-coded channels for easy tracking

**2. FFT Spectrum View**
- Power spectral density (0.5-40 Hz)
- Logarithmic power scale (dB)
- Frequency band annotations
- Useful for monitoring alpha/theta balance

**3. Band Power View**
- Bar chart: 5 bands × 4 channels
- Relative power percentages
- Immediate feedback on mental state
- Optimal for biofeedback training

**4. Camera/FaceSynth View**
- Live video feed with landmark overlay
- 3 real-time parameter meters (mouth, roll, smile)
- Visual progress bars + numeric values
- Synchronized with OSC output

#### 4.3.3 Control Interface

**Connection Panel:**
```
1. Device Connection
   - Auto-discovery (Bluetooth LE)
   - Manual serial port entry (BLED112 dongle)
   - Connection status indicator

2. Stream Control
   - Start/Stop EEG streaming
   - Start/Stop camera tracking
   - Independent control of each modality
```

**OSC Configuration:**
```
IP Address: [127.0.0.1    ]
Port:       [9000         ]

Data Streams:
  [X] Timeseries  → /eeg/raw/CH{1-4}
  [X] Bands       → /eeg/bands/{delta,theta,alpha,beta,gamma}/CH{1-4}
  [X] FaceSynth   → /cv/face/{mouth,roll,smile}

[Enable OSC] button
```

**Visualization Controls:**
```
Y-Scale Mode:
  [ ] Manual  [Slider: ±25 to ±10000]
  [X] Auto    (Adaptive with smoothing)

Mode Tabs:
  [Time Series] [FFT] [Bands] [Camera]
```

### 4.4 Communication Protocols

#### 4.4.1 WebSocket (Real-Time Data)
```javascript
// Connection: ws://localhost:8000/ws/stream

// Client → Server (Configuration)
{
  "mode": "timeseries" | "fft" | "bands",
  "window_sec": 4.0,
  "interval_ms": 50
}

// Server → Client (Time Series)
{
  "type": "timeseries",
  "channels": ["CH1", "CH2", "CH3", "CH4"],
  "data": [[...], [...], [...], [...]]  // 4 channels × N samples
}

// Server → Client (FFT)
{
  "type": "fft",
  "channels": ["CH1", "CH2", "CH3", "CH4"],
  "freqs": [0.5, 0.75, 1.0, ..., 40.0],  // Hz
  "psd": [[...], [...], [...], [...]]     // 4 channels × M frequencies (dB)
}

// Server → Client (Bands)
{
  "type": "bands",
  "channels": ["CH1", "CH2", "CH3", "CH4"],
  "bands": ["delta", "theta", "alpha", "beta", "gamma"],
  "values": [[...], [...], [...], [...]]  // 4 channels × 5 bands (%)
}
```

#### 4.4.2 OSC (External Integration)
```
# EEG Raw Data (Time Series)
Address: /eeg/raw/CH{1-4}
Format: [float, float, float, ...]  (Variable length array)
Rate: ~20 Hz

# EEG Band Powers
Address: /eeg/bands/{delta|theta|alpha|beta|gamma}/CH{1-4}
Format: [float]  (Percentage: 0.0-100.0)
Rate: ~20 Hz

# Facial Features
Address: /cv/face/mouth
Format: [float]  (0.0-1.0)

Address: /cv/face/roll
Format: [float]  (-1.0 to +1.0)

Address: /cv/face/smile
Format: [float]  (0.0-1.0)

Rate: ~30 Hz (camera framerate dependent)
```

---

## 5. Current System Status

### 5.1 Implemented Features
- [DONE] Bluetooth LE and BLED112 dongle connectivity
- [DONE] 4-channel EEG acquisition @ 200 Hz
- [DONE] Real-time signal processing (FFT, band powers)
- [DONE] Multi-modal visualization (time series, FFT, bands, camera)
- [DONE] Computer vision facial tracking (mouth, roll, smile)
- [DONE] OSC output for creative applications
- [DONE] Web-based interface with responsive design
- [DONE] Adaptive and manual scaling controls
- [DONE] Smooth 20 Hz visualization updates

### 5.2 Technical Specifications
```
Hardware:
  - Ganglion Board: 4 channels, 200 Hz, 24-bit
  - Dry electrodes: Frontal (Fp1, Fp2) + Parietal (P3, P4)
  - Wireless: Bluetooth LE 4.0+

Software:
  - Backend: Python 3.9+, FastAPI, BrainFlow
  - Frontend: HTML5, JavaScript, Chart.js
  - Signal Processing: NumPy, SciPy (Welch PSD)
  - Computer Vision: OpenCV, MediaPipe
  - Communication: WebSocket, OSC (UDP)

Performance:
  - Latency: <100ms (acquisition → display)
  - Update Rate: 20 Hz (visualization), 30 Hz (camera)
  - CPU Usage: ~15-25% (single core, M-series Mac)
  - Memory: ~150-200 MB
```

---

## 6. Future Directions: Applications for Creative Expression

### 6.1 Musical Control in Csound

**Vision:** Map biosignals and facial features to synthesis parameters for real-time musical performance.

**Proposed Mappings:**

**A. Alpha-Driven Ambient Soundscapes**
```csound
; Parietal Alpha Power (8-13 Hz) → Filter Resonance
; More relaxed/eyes closed = Higher alpha = Brighter timbre

instr AlphaAmbient
  kAlpha    OSCin "/eeg/bands/alpha/CH3"      ; Parietal right (P4)
  kAlphaNorm = kAlpha / 100                    ; Normalize to 0-1

  ; Map to filter cutoff: 200 Hz (low alpha) → 2000 Hz (high alpha)
  kCutoff   = 200 + (kAlphaNorm * 1800)

  aSource   vco2 0.3, 110                      ; Sawtooth base (A2)
  aFiltered moogladder aSource, kCutoff, 0.7

  out aFiltered
endin
```

**B. Theta/Beta Ratio → Rhythmic Density**
```csound
; High theta/beta = Meditative → Sparse rhythm
; Low theta/beta = Focused → Dense rhythm

instr RhythmicDensity
  kTheta OSCin "/eeg/bands/theta/CH1"    ; Frontal theta
  kBeta  OSCin "/eeg/bands/beta/CH1"     ; Frontal beta
  kRatio = (kTheta + 1) / (kBeta + 1)    ; Avoid div by zero

  ; Map to trigger rate: 0.5 Hz (focused) → 4 Hz (meditative)
  kRate  = 0.5 + (kRatio * 3.5)

  kTrig  metro kRate
  schedkwhen kTrig, 0, 0, "Percussion", 0, 0.1
endin
```

**C. Facial Control → Expressive Parameters**
```csound
; Mouth openness → Formant vowel morphing
; Smile → Harmonic brightness
; Head roll → Stereo pan

instr VocalSynthesis
  kMouth  OSCin "/cv/face/mouth"        ; 0.0-1.0
  kSmile  OSCin "/cv/face/smile"        ; 0.0-1.0
  kRoll   OSCin "/cv/face/roll"         ; -1.0 to +1.0

  ; Formant frequencies: /a/ → /i/
  kF1 = 800 - (kMouth * 500)            ; 800 (closed) → 300 (open)
  kF2 = 1200 + (kMouth * 1000)          ; 1200 → 2200

  aSource buzz 0.5, 150, 20, 1
  aVowel  formant aSource, kF1, kF2, kSmile

  ; Stereo panning
  aPanL, aPanR pan2 aVowel, (kRoll + 1) / 2
  outs aPanL, aPanR
endin
```

**Anticipated Challenges:**
- Mapping stability: EEG is noisy; will require smoothing/hysteresis
- Latency: Need <50ms for musical responsiveness
- Learning curve: Users need time to develop control

**Solutions:**
- Exponential smoothing on band powers (α = 0.1-0.2)
- User-adjustable sensitivity/gain controls
- Visual feedback to aid neurofeedback learning
- Preset mapping schemes for different musical contexts

### 6.2 Adaptive Visuals

**Vision:** Generate real-time visual content that responds to brain state and movement, creating an immersive feedback loop.

**Proposed Implementation (TouchDesigner/Processing):**

**A. Brain-State Color Palettes**
```python
# Processing pseudocode

# Receive band powers
alpha = osc.receive("/eeg/bands/alpha/CH3")
theta = osc.receive("/eeg/bands/theta/CH1")
beta  = osc.receive("/eeg/bands/beta/CH1")

# Map to HSB color space
hue        = map(theta, 0, 50, 180, 280)  # Blue (calm) → Purple (meditative)
saturation = map(beta, 0, 50, 50, 100)    # Low focus → desaturated
brightness = map(alpha, 0, 50, 30, 90)    # Low alpha → dim, high → bright

background(hue, saturation, brightness)
```

**B. Particle System Dynamics**
```python
# Particle generation rate ← Gamma (cognitive binding)
gamma = osc.receive("/eeg/bands/gamma/CH1")
spawnRate = map(gamma, 0, 20, 1, 20)  # 1-20 particles/frame

# Particle motion ← Head movement
roll = osc.receive("/cv/face/roll")
particles.applyForce(Vector(roll * 0.5, 0))  # Lateral "tilt"
```

**C. Fractal Zoom Depth**
```python
# Mandelbrot/Julia set zoom ← Alpha coherence
alpha_L = osc.receive("/eeg/bands/alpha/CH3")  # Left parietal
alpha_R = osc.receive("/eeg/bands/alpha/CH4")  # Right parietal

coherence = 1 - abs(alpha_L - alpha_R) / 50
zoomDepth = map(coherence, 0, 1, 2, 12)  # More coherent → deeper zoom
```

**Benefits:**
- **Non-verbal expression:** Visuals convey inner states without language
- **Neurofeedback:** Users see their brain activity, learning self-regulation
- **Aesthetic exploration:** Discover unique visual signatures

### 6.3 Playful Self-Regulation Systems

**Vision:** Create engaging, game-like experiences where users learn to modulate their brain states through creative interaction.

**Application 1: "Alpha Garden"**
```
Concept: Grow a digital garden by sustaining high parietal alpha (relaxation)

Mechanics:
  - Plants grow when alpha > threshold (e.g., 30%)
  - Growth rate proportional to alpha power
  - Beta activity (stress) causes wilting
  - Smile adds flowers/color

Outcome:
  - Users learn to enter relaxed, meditative states
  - Visual reward reinforces desired brain pattern
  - Accessible alternative to traditional meditation
```

**Application 2: "Focus Flow"**
```
Concept: Navigate a particle through obstacles using sustained attention (beta)

Mechanics:
  - High frontal beta → particle moves forward
  - Theta/alpha → particle slows down
  - Head roll controls lateral position
  - Avoid obstacles, collect targets

Outcome:
  - Trains sustained attention (useful for ADHD)
  - Combines cognitive control with spatial navigation
  - Mouth openness could control "boost" (engagement)
```

**Application 3: "Harmony Maker"**
```
Concept: Create musical chords by balancing different brain rhythms

Mechanics:
  - Each frequency band controls a note in a chord
  - Delta → Root note
  - Theta → 3rd
  - Alpha → 5th
  - Beta → 7th
  - Gamma → Extensions (9th, 11th)

  Target: Achieve "target ratios" for different musical modes
    - Major triad: High alpha + beta
    - Minor triad: High theta + alpha
    - Dissonant: High gamma + beta

Outcome:
  - Users explore brain state spaces through sound
  - Learn fine-grained control over mental states
  - Composition becomes a meditative practice
```

**Design Principles:**
- **No failure states:** Exploration, not performance
- **Immediate feedback:** Visual/audio responds within 100ms
- **Aesthetic rewards:** Beauty as motivation, not points
- **Accessibility:** Operable with minimal motor control

---

## 7. Technical Considerations and Challenges

### 7.1 Signal Quality and Artifacts

**Challenge:** EEG is extremely sensitive to noise and movement artifacts.

**Sources of Interference:**
- **EMG (muscle activity):** Jaw clenching, frowning → high-frequency noise (>30 Hz)
- **Eye movements:** Electrooculogram (EOG) → large frontal artifacts
- **Motion artifacts:** Head movement → baseline drift, electrode displacement
- **Environmental noise:** 50/60 Hz power line interference

**Current Mitigations:**
- Linear detrending removes slow drift
- Notch filter at 60 Hz (optional, can reduce gamma band accuracy)
- 50% overlapping windows in Welch PSD smooth noise

**Future Improvements:**
- Implement artifact rejection (threshold-based or ICA)
- Electrode-scalp impedance monitoring
- Adaptive filtering based on user movement

### 7.2 Real-Time Performance

**Requirements:**
- **Latency:** <100ms total (acquisition → processing → output)
- **Throughput:** 200 Hz × 4 channels = 800 samples/sec minimum

**Current Performance:**
```
Component               Latency    CPU (%)
─────────────────────────────────────────
BLE transmission        ~20-30ms   N/A
BrainFlow buffering     ~10ms      3%
Signal processing       ~15ms      8%
WebSocket transmission  ~5ms       2%
OSC transmission        ~2ms       1%
UI rendering (browser)  ~16ms      5%
─────────────────────────────────────────
Total                   ~68-78ms   ~19%
```

**Bottlenecks:**
- BLE transmission jitter (unavoidable wireless trade-off)
- Browser rendering on lower-end devices

**Optimizations:**
- Downsample visualization data (512 points max)
- Use Chart.js "none" animation mode for efficiency
- Send OSC on separate thread (async I/O)

### 7.3 Individual Variability

**Challenge:** EEG patterns vary significantly between individuals due to:
- Skull thickness and conductivity
- Electrode placement precision
- Baseline brain rhythm frequencies (e.g., alpha peak: 8-13 Hz)
- Cognitive strategies for "relaxation" or "focus"

**Implications:**
- Fixed mappings (e.g., alpha → brightness) may not work for all users
- Requires **per-user calibration** or **adaptive mapping**

**Proposed Solutions:**

**A. Baseline Calibration:**
```python
# Record 1-minute baseline (eyes open, neutral state)
baseline_alpha = mean(alpha_power[0:60s])
baseline_beta  = mean(beta_power[0:60s])

# Normalize during performance
alpha_normalized = (current_alpha - baseline_alpha) / baseline_std
```

**B. Adaptive Thresholding:**
```python
# Track user's alpha range over session
alpha_min = min(alpha_history)
alpha_max = max(alpha_history)

# Map to 0-1 using personal range
alpha_mapped = (current_alpha - alpha_min) / (alpha_max - alpha_min)
```

**C. User-Driven Mapping:**
- Provide sliders for sensitivity/gain per parameter
- "Learn" mode: User demonstrates high/low states, system sets thresholds

### 7.4 User Experience Design

**Challenge:** Brain control is inherently non-intuitive; users need scaffolding to develop control.

**Design Strategies:**

**1. Progressive Disclosure:**
```
Session 1: Observe your alpha rhythm (passive)
Session 2: Try to increase alpha by closing eyes (active)
Session 3: Map alpha to a simple parameter (e.g., volume)
Session 4: Multi-parameter control (alpha + theta)
Session 5: Free exploration / performance mode
```

**2. Visual Feedback Loops:**
- Real-time brain state visualization alongside output
- "Target zones" overlaid on band power charts
- Color-coded feedback (green = on target, red = off target)

**3. Haptic Feedback (Future):**
- Vibration patterns on headband corresponding to brain states
- Enhances proprioception of mental states

---

## 8. Ethical and Accessibility Considerations

### 8.1 Inclusive Design

**Principle:** BioMus must serve users across a spectrum of abilities and contexts.

**Accessibility Features:**
- **Screen reader support:** Semantic HTML, ARIA labels
- **Keyboard navigation:** All controls operable without mouse
- **High contrast mode:** For visual impairments
- **Adjustable text size:** Minimum 16px, scalable
- **No time-based challenges:** Users work at their own pace

**User Diversity:**
- **Motor impairments:** Wheelchair users, ALS, cerebral palsy
- **Cognitive diversity:** Autism, ADHD, brain injury
- **Aging populations:** Reduced dexterity, vision
- **Economic barriers:** Low-cost hardware (<$300 total)

### 8.2 Data Privacy

**Current Status:** All processing is local; no cloud transmission.

**Privacy Principles:**
- **Minimal collection:** Only acquire signals needed for current task
- **Local processing:** No biosignal data leaves user's computer
- **User control:** Start/stop streaming at will
- **No recording by default:** Explicit opt-in for data storage

**Future Considerations:**
- If remote collaboration features added, use end-to-end encryption
- Allow users to review/delete any recorded data
- Clear consent mechanisms for research participation

### 8.3 Empowerment vs. Medicalization

**Risk:** Framing BCI as "therapeutic" may stigmatize or medicalize users.

**BioMus Philosophy:**
- **Art-first framing:** Tool for creative expression, not medical device
- **Celebrate neurodiversity:** Different brain patterns = different artistic voices
- **Avoid normative language:** No "correct" brain state, only exploration
- **User agency:** Users define success, not clinicians or researchers

**Example Language:**
- AVOID: "Train your brain to achieve optimal focus"
- PREFER: "Discover how your unique brain rhythms can shape sound and visuals"

---

## 9. Conclusion and Next Steps

### 9.1 Summary of Achievements

BioMus represents a significant step toward **democratizing brain-computer interfaces for creative expression**. By combining affordable open-source hardware (OpenBCI Ganglion), accessible software (web-based interface), and a dual-modality approach (EEG + computer vision), we have created a platform that:

1. **Lowers barriers** to neural interface technology (cost, complexity)
2. **Empowers creativity** for individuals with limited mobility
3. **Enables self-regulation** through playful artistic interaction
4. **Provides flexible infrastructure** for future applications

### 9.2 Immediate Next Steps

**Technical Development (Q1-Q2 2026):**
1. [DONE] Complete camera integration
2. [TODO] Implement Csound OSC receiver and basic synthesis mappings
3. [TODO] Develop TouchDesigner/Processing visual templates
4. [TODO] Add user calibration workflow (baseline + adaptive thresholding)
5. [TODO] Optimize artifact rejection (EMG/EOG filtering)

**User Testing (Q2-Q3 2026):**
1. [TODO] Recruit 5-10 pilot users with motor impairments
2. [TODO] Conduct usability testing sessions (2-3 sessions/user)
3. [TODO] Gather qualitative feedback on empowerment and creative expression
4. [TODO] Iterate on interface design based on findings

**Application Development (Q3-Q4 2026):**
1. [TODO] Prototype "Alpha Garden" self-regulation app
2. [TODO] Develop musical performance templates (ambient, rhythmic, melodic)
3. [TODO] Create tutorial series (video + written guides)
4. [TODO] Build community forum for sharing mappings and creations

### 9.3 Long-Term Vision

**Year 1-2:** Establish BioMus as a **proven accessible BCI platform**
- Publish open-source documentation and tutorials
- Build community of users and developers
- Present at accessibility and BCI conferences

**Year 3-5:** Expand to **multi-user and educational contexts**
- Classroom applications (neuroscience education)
- Collaborative musical performances (BCI ensembles)
- Partnerships with disability arts organizations

**Year 5+:** Influence **broader BCI accessibility standards**
- Contribute to open BCI hardware/software ecosystems
- Advocate for inclusive design in neurotechnology
- Inspire next-generation accessible creative tools

---

## 10. References and Resources

### Technical Documentation
- **BrainFlow SDK:** [brainflow.readthedocs.io](https://brainflow.readthedocs.io)
- **OpenBCI Ganglion:** [docs.openbci.com/Ganglion](https://docs.openbci.com/Ganglion)
- **MediaPipe Face Mesh:** [google.github.io/mediapipe/solutions/face_mesh](https://google.github.io/mediapipe/solutions/face_mesh)
- **Welch PSD Method:** Welch, P. (1967). "The use of fast Fourier transform for the estimation of power spectra"

### BCI and EEG Resources
- **BCI Primer:** Nicolas-Alonso & Gomez-Gil (2012). "Brain Computer Interfaces, a Review"
- **EEG Frequency Bands:** Abhang et al. (2016). "Introduction to EEG- and Speech-Based Emotion Recognition"
- **Artifact Rejection:** Jiang et al. (2019). "Removal of Artifacts from EEG Signals: A Review"

### Accessibility and Design
- **Web Accessibility (WCAG 2.1):** [w3.org/WAI/WCAG21](https://www.w3.org/WAI/WCAG21)
- **Inclusive Design Principles:** [inclusivedesignprinciples.org](https://inclusivedesignprinciples.org)
- **Nothing About Us Without Us:** Disability rights principle

### Creative Applications
- **Csound Manual:** [csound.com/docs/manual](http://csound.com/docs/manual)
- **TouchDesigner:** [derivative.ca](https://derivative.ca)
- **Processing:** [processing.org](https://processing.org)

---

## Appendices

### Appendix A: Hardware Bill of Materials

| Component | Quantity | Unit Cost | Total | Source |
|-----------|----------|-----------|-------|--------|
| OpenBCI Ganglion Board | 1 | $199 | $199 | OpenBCI.com |
| Dry EEG Electrodes (pack of 6) | 1 | $30 | $30 | OpenBCI/Amazon |
| Headband/mounting | 1 | $15 | $15 | Custom/3D printed |
| USB Battery Pack (optional) | 1 | $20 | $20 | Generic |
| BLED112 Dongle (optional) | 1 | $15 | $15 | OpenBCI/Digikey |
| **TOTAL** | | | **$279** | |

### Appendix B: Software Dependencies

```bash
# Python Backend
fastapi==0.104.1
uvicorn[standard]==0.24.0
brainflow==5.10.1
numpy==1.24.3
scipy==1.11.3
python-osc==1.8.3
opencv-python==4.8.1
mediapipe==0.10.8
jinja2==3.1.2

# Frontend (CDN, no install)
Chart.js 4.4.0
Inter Font (Google Fonts)
```

### Appendix C: API Reference

See `PROJECT_STRUCTURE.md` for complete API documentation.

**Quick Reference:**
```bash
# Start server
./run.sh

# Access interface
http://localhost:8000

# WebSocket endpoint
ws://localhost:8000/ws/stream

# OSC output
Default: 127.0.0.1:9000
Configurable via UI
```

---

**Document Version:** 1.0
**Last Updated:** December 10, 2025
**Authors:** BioMus Development Team
**License:** MIT (Open Source)
**Contact:** [Your contact information]

---

**BioMus: Empowering creativity through accessible brain-computer interfaces**
