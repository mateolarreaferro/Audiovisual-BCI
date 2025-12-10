<CsoundSynthesizer>
; ================================================================================
; OPENBCI FFT LISTENER
; ================================================================================
; Receives raw EEG data and performs real-time FFT analysis
; OSC Address: /openbci/eeg/{channel}
; Data Format: Single float per channel representing raw EEG voltage
; ================================================================================
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 100
nchnls = 2
0dbfs = 1.0

; Common waveform tables
gisine   ftgen 1, 0, 16384, 10, 1
gisquare ftgen 2, 0, 16384, 10, 1, 0, .33, 0, .2, 0, .14, 0, .11, 0, .09
gisaw    ftgen 3, 0, 16384, 10, 1, .2, 0, .4, 0, .6, 0, .8, 0, 1, 0, .8, 0, .6, 0, .4, 0, .2

; Scale tables
giscale1 ftgen 111, 0, 8, -2, 72, 74, 76, 79, 81, 83, 72, 84
giscale2 ftgen 112, 0, 8, -2, 52, 50, 48, 65, 60, 52, 48, 60
giscale3 ftgen 113, 0, 8, -2, 60, 62, 69, 67, 71, 74, 72, 48
giscale4 ftgen 114, 0, 8, -2, 48, 43, 50, 48, 43, 55, 60, 43

; Reverb globals
garvbL init 0
garvbR init 0

; Delay globals
gadelL init 0
gadelR init 0

; BCI control values
gkf1 init 0.5
gkf2 init 0.5
gkf3 init 0.5
gkf4 init 0.5
gkf5 init 0.5
gkf6 init 0.5
gkf7 init 0.5
gkf8 init 0.5
gkf9 init 0.5
gkf10 init 0.5
gkf11 init 0.5
gkf12 init 0.5
gkf13 init 0.5
gkf14 init 0.5
gkf15 init 0.5
gkf16 init 0.5

; FFT-specific globals
gkfft_ready init 0

; Schedule instruments
schedule 1, 0, -1        ; FFT Listener
schedule 3, 0, -1        ; Reverb
schedule 4, 0, -1        ; Delay

; MIDI setup
massign 1, 2

; ==============================================================================
; Instrument 1: Raw EEG FFT Data Listener
; ==============================================================================
instr 1
    ; Initialize OSC handle
    gihandle OSCinit 5005
    
    ; Local variables for raw EEG data
    keeg_val init 0
    kdata_received init 0
    
    ; Buffer for FFT analysis (using audio rate variables)
    abuf1, abuf2, abuf3, abuf4 init 0
    abuf5, abuf6, abuf7, abuf8 init 0
    
    ; Listen to raw EEG data for first 8 channels
    kk1 OSClisten gihandle, "/openbci/eeg/0", "f", keeg_val
    if kk1 > 0 then
        abuf1 = keeg_val * 0.001  ; Scale raw EEG voltage
        kdata_received = 1
    endif
    
    kk2 OSClisten gihandle, "/openbci/eeg/1", "f", keeg_val
    if kk2 > 0 then
        abuf2 = keeg_val * 0.001
        kdata_received = 1
    endif
    
    kk3 OSClisten gihandle, "/openbci/eeg/2", "f", keeg_val
    if kk3 > 0 then
        abuf3 = keeg_val * 0.001
        kdata_received = 1
    endif
    
    kk4 OSClisten gihandle, "/openbci/eeg/3", "f", keeg_val
    if kk4 > 0 then
        abuf4 = keeg_val * 0.001
        kdata_received = 1
    endif
    
    kk5 OSClisten gihandle, "/openbci/eeg/4", "f", keeg_val
    if kk5 > 0 then
        abuf5 = keeg_val * 0.001
        kdata_received = 1
    endif
    
    kk6 OSClisten gihandle, "/openbci/eeg/5", "f", keeg_val
    if kk6 > 0 then
        abuf6 = keeg_val * 0.001
        kdata_received = 1
    endif
    
    kk7 OSClisten gihandle, "/openbci/eeg/6", "f", keeg_val
    if kk7 > 0 then
        abuf7 = keeg_val * 0.001
        kdata_received = 1
    endif
    
    kk8 OSClisten gihandle, "/openbci/eeg/7", "f", keeg_val
    if kk8 > 0 then
        abuf8 = keeg_val * 0.001
        kdata_received = 1
    endif
    
    ; Perform simple spectral analysis using resonant filters
    ; Delta band (0.5-4 Hz) - using very low frequency filters
    adelta1 reson abuf1, 2, 2, 1
    adelta2 reson abuf2, 2, 2, 1
    adelta3 reson abuf3, 2, 2, 1
    adelta4 reson abuf4, 2, 2, 1
    
    ; Theta band (4-8 Hz)
    atheta1 reson abuf1, 6, 2, 1
    atheta2 reson abuf2, 6, 2, 1
    atheta3 reson abuf3, 6, 2, 1
    atheta4 reson abuf4, 6, 2, 1
    
    ; Alpha band (8-13 Hz)
    aalpha1 reson abuf1, 10, 3, 1
    aalpha2 reson abuf2, 10, 3, 1
    aalpha3 reson abuf3, 10, 3, 1
    aalpha4 reson abuf4, 10, 3, 1
    
    ; Beta band (13-30 Hz)
    abeta1 reson abuf5, 20, 8, 1
    abeta2 reson abuf6, 20, 8, 1
    abeta3 reson abuf7, 20, 8, 1
    abeta4 reson abuf8, 20, 8, 1
    
    ; Gamma band (30-100 Hz) - using higher frequency filters
    agamma1 reson abuf5, 50, 20, 1
    agamma2 reson abuf6, 50, 20, 1
    agamma3 reson abuf7, 50, 20, 1
    agamma4 reson abuf8, 50, 20, 1
    
    ; Convert filtered signals to control values using RMS
    krms1 rms adelta1
    krms2 rms adelta2
    krms3 rms atheta1
    krms4 rms atheta2
    krms5 rms aalpha1
    krms6 rms aalpha2
    krms7 rms aalpha3
    krms8 rms aalpha4
    krms9 rms abeta1
    krms10 rms abeta2
    krms11 rms abeta3
    krms12 rms abeta4
    krms13 rms agamma1
    krms14 rms agamma2
    krms15 rms agamma3
    krms16 rms agamma4
    
    ; Map FFT analysis results to synthesis control values (scaled like alpha band)
    if kdata_received == 1 then
        gkf1 = krms1 * 0.0001   ; Delta channels 1-2
        gkf2 = krms2 * 0.0001
        
        gkf3 = krms3 * 0.0001   ; Theta channels 3-4
        gkf4 = krms4 * 0.0001
        
        gkf5 = krms5 * 0.0001   ; Alpha channels 5-8
        gkf6 = krms6 * 0.0001
        gkf7 = krms7 * 0.0001
        gkf8 = krms8 * 0.0001
        
        gkf9 = krms9 * 0.0001   ; Beta channels 9-12
        gkf10 = krms10 * 0.0001
        gkf11 = krms11 * 0.0001
        gkf12 = krms12 * 0.0001
        
        gkf13 = krms13 * 0.0001 ; Gamma channels 13-16
        gkf14 = krms14 * 0.0001
        gkf15 = krms15 * 0.0001
        gkf16 = krms16 * 0.0001
        
        gkfft_ready = 1
    endif
    
    ; Fallback smoothing when no data received - same as SimpleBCI.csd
    if kdata_received == 0 then
        gkf1 = gkf1 * 0.995 + 0.5 * 0.005
        gkf2 = gkf2 * 0.995 + 0.5 * 0.005
        gkf3 = gkf3 * 0.995 + 0.5 * 0.005
        gkf4 = gkf4 * 0.995 + 0.5 * 0.005
        gkf5 = gkf5 * 0.995 + 0.5 * 0.005
        gkf6 = gkf6 * 0.995 + 0.5 * 0.005
        gkf7 = gkf7 * 0.995 + 0.5 * 0.005
        gkf8 = gkf8 * 0.995 + 0.5 * 0.005
        gkf9 = gkf9 * 0.995 + 0.5 * 0.005
        gkf10 = gkf10 * 0.995 + 0.5 * 0.005
        gkf11 = gkf11 * 0.995 + 0.5 * 0.005
        gkf12 = gkf12 * 0.995 + 0.5 * 0.005
        gkf13 = gkf13 * 0.995 + 0.5 * 0.005
        gkf14 = gkf14 * 0.995 + 0.5 * 0.005
        gkf15 = gkf15 * 0.995 + 0.5 * 0.005
        gkf16 = gkf16 * 0.995 + 0.5 * 0.005
    endif
    
    ; Connection monitoring and debug output
    ktimer timeinsts
    if kdata_received == 1 then
        printks "FFT Analysis: Delta: %.3f %.3f | Theta: %.3f %.3f | Alpha: %.3f %.3f %.3f %.3f\n", 1, gkf1, gkf2, gkf3, gkf4, gkf5, gkf6, gkf7, gkf8
        printks "FFT Analysis: Beta: %.3f %.3f %.3f %.3f | Gamma: %.3f %.3f %.3f %.3f\n", 1, gkf9, gkf10, gkf11, gkf12, gkf13, gkf14, gkf15, gkf16
        kdata_received = 0
    elseif ktimer > 5 && int(ktimer) % 5 == 0 then
        printks "Waiting for Raw EEG data for FFT analysis...\n", 5
    endif
endin

; ==============================================================================
; Instrument 2: MIDI-controlled Synthesis with FFT modulation
; ==============================================================================
instr 2
    ; MIDI CC controls with fallback defaults
    kspeed1 midic7 21, .01, 40
    kspeed2 midic7 22, .01, 50
    kspeed3 midic7 23, .01, 100
    kspeed4 midic7 24, .01, 10
    
    ; Use defaults if no MIDI controller
    if kspeed1 == 0 then
        kspeed1 = 10
    endif
    if kspeed2 == 0 then
        kspeed2 = 15
    endif
    if kspeed3 == 0 then
        kspeed3 = 25
    endif
    if kspeed4 == 0 then
        kspeed4 = 5
    endif
    
    ; Create metro triggers
    ktrig1 metro kspeed1
    ktrig2 metro kspeed2
    ktrig3 metro kspeed3
    ktrig4 metro kspeed4
    
    ; Sample and hold BCI values - use first 8 channels for synthesis
    kf1 samphold gkf1, ktrig1
    kf2 samphold gkf2, ktrig2
    kf3 samphold gkf3, ktrig3
    kf4 samphold gkf4, ktrig4
    kf5 samphold gkf5, ktrig1  ; Reuse metro triggers in a pattern
    kf6 samphold gkf6, ktrig2
    kf7 samphold gkf7, ktrig3
    kf8 samphold gkf8, ktrig4
    
    ; Scale FFT values for pitch modulation (ensure they're in good range)
    kf1_scaled = abs(kf1) * 3 + 0.2  ; Scale and offset for musical range
    kf2_scaled = abs(kf2) * 3 + 0.2
    kf3_scaled = abs(kf3) * 3 + 0.2
    kf4_scaled = abs(kf4) * 3 + 0.2
    kf5_scaled = abs(kf5) * 2 + 0.3  ; Vary scaling for different channels
    kf6_scaled = abs(kf6) * 2 + 0.3
    kf7_scaled = abs(kf7) * 1.5 + 0.4
    kf8_scaled = abs(kf8) * 1.5 + 0.4
    
    ; Get MIDI note frequency
    icps cpsmidi
    
    ; Generate oscillators modulated by FFT data - 8 channels
    aout1 oscili 0.3, icps + cpspch(kf1_scaled + 2), 1
    aout2 oscili 0.3, icps + cpspch(kf2_scaled + 2), 1
    aout3 oscili 0.3, icps + cpspch(kf3_scaled + 2), 1
    aout4 oscili 0.3, icps + cpspch(kf4_scaled + 2), 1
    aout5 oscili 0.2, icps * 0.5 + cpspch(kf5_scaled + 1), 2  ; Lower octave with square wave
    aout6 oscili 0.2, icps * 0.5 + cpspch(kf6_scaled + 1), 2
    aout7 oscili 0.1, icps * 2 + cpspch(kf7_scaled + 3), 3    ; Higher octave with saw wave
    aout8 oscili 0.1, icps * 2 + cpspch(kf8_scaled + 3), 3
    
    ; Apply ADSR envelope
    aadsr madsr 1, 0.5, 0.8, 0.9
    
    ; Mix all oscillators with different weighting
    aout = ((aout1 + aout2 + aout3 + aout4 + aout5 + aout6 + aout7 + aout8) / 12) * aadsr
    
    ; Send to effects
    garvbL += aout * 0.8
    garvbR += aout * 0.8
    gadelL += aout * 0.8
    gadelR += aout * 0.8
    
    ; Direct output
    outs aout, aout
endin

; ==============================================================================
; Instrument 3: Reverb
; ==============================================================================
instr 3
    denorm garvbL
    denorm garvbR
    aout1, aout2 reverbsc garvbL, garvbR, 0.8, 8000
    outs aout1, aout2
    clear garvbL
    clear garvbR
endin

; ==============================================================================
; Instrument 4: Delay
; ==============================================================================
instr 4
    adelL init 0
    adelR init 0
    denorm gadelL
    denorm gadelR
    adelL delay gadelL + (adelL * 0.72), 2
    adelR delay gadelR + (adelR * 0.7), 2
    adelOutL moogvcf adelL, 2000, 0.4
    adelOutR moogvcf adelR, 2000, 0.6
    outs adelOutL, adelOutR
    clear gadelL
    clear gadelR
endin

</CsInstruments>
<CsScore>
; Empty score - all instruments scheduled in orchestra
</CsScore>

</CsoundSynthesizer>









<bsbPanel>
 <label>Widgets</label>
 <objectName/>
 <x>100</x>
 <y>100</y>
 <width>320</width>
 <height>240</height>
 <visible>true</visible>
 <uuid/>
 <bgcolor mode="background">
  <r>240</r>
  <g>240</g>
  <b>240</b>
 </bgcolor>
</bsbPanel>
<bsbPresets>
</bsbPresets>
