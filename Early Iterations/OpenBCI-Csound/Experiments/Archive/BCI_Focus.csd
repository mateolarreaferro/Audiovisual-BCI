<CsoundSynthesizer>
; ================================================================================
; OPENBCI FOCUS LISTENER
; ================================================================================
; Receives focus metrics calculated from band powers
; OSC Address: /openbci/focus/{channel} or /openbci/focus (global)
; Data Format: Single float representing focus level (0-1 typically)
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

; Focus-specific globals
gkfocus_global init 0.5
gkfocus_trend init 0.5

; Schedule instruments
schedule 1, 0, -1        ; Focus Listener
schedule 3, 0, -1        ; Reverb
schedule 4, 0, -1        ; Delay

; MIDI setup
massign 1, 2

; ==============================================================================
; Instrument 1: Focus Data Listener
; ==============================================================================
instr 1
    ; Initialize OSC handle
    gihandle OSCinit 5003
    
    ; Local variables for focus data
    kfocus_val init 0
    kdata_received init 0
    
    ; Listen for global focus message first (most common)
    kk_global OSClisten gihandle, "/openbci/focus", "f", kfocus_val
    if kk_global > 0 then
        gkfocus_global = kfocus_val
        
        ; Map focus value using SimpleBCI.csd scaling approach 
        ; Focus typically ranges 0-100, so scale similar to alpha band
        gkf1 = kfocus_val * 0.0001  
        gkf2 = kfocus_val * 0.0001 * 1.1  ; Slight variations 
        gkf3 = kfocus_val * 0.0001 * 0.9
        gkf4 = kfocus_val * 0.0001 * 1.2
        gkf5 = kfocus_val * 0.0001 * 0.8
        gkf6 = kfocus_val * 0.0001 * 1.3
        gkf7 = kfocus_val * 0.0001 * 0.7
        gkf8 = kfocus_val * 0.0001 * 1.4
        gkf9 = kfocus_val * 0.0001 * 0.6
        gkf10 = kfocus_val * 0.0001 * 1.5
        gkf11 = kfocus_val * 0.0001 * 0.5
        gkf12 = kfocus_val * 0.0001 * 1.6
        gkf13 = kfocus_val * 0.0001 * 0.4
        gkf14 = kfocus_val * 0.0001 * 1.7
        gkf15 = kfocus_val * 0.0001 * 0.3
        gkf16 = kfocus_val * 0.0001 * 1.8
        
        kdata_received = 1
    endif
    
    ; Also listen for per-channel focus data for all 16 channels
    kk1 OSClisten gihandle, "/openbci/focus/0", "f", kfocus_val
    if kk1 > 0 then
        gkf1 = kfocus_val * 0.0001
        kdata_received = 1
    endif
    
    kk2 OSClisten gihandle, "/openbci/focus/1", "f", kfocus_val
    if kk2 > 0 then
        gkf2 = kfocus_val * 0.0001
        kdata_received = 1
    endif
    
    kk3 OSClisten gihandle, "/openbci/focus/2", "f", kfocus_val
    if kk3 > 0 then
        gkf3 = kfocus_val * 0.0001
        kdata_received = 1
    endif
    
    kk4 OSClisten gihandle, "/openbci/focus/3", "f", kfocus_val
    if kk4 > 0 then
        gkf4 = kfocus_val * 0.0001
        kdata_received = 1
    endif
    
    kk5 OSClisten gihandle, "/openbci/focus/4", "f", kfocus_val
    if kk5 > 0 then
        gkf5 = kfocus_val * 0.0001
        kdata_received = 1
    endif
    
    kk6 OSClisten gihandle, "/openbci/focus/5", "f", kfocus_val
    if kk6 > 0 then
        gkf6 = kfocus_val * 0.0001
        kdata_received = 1
    endif
    
    kk7 OSClisten gihandle, "/openbci/focus/6", "f", kfocus_val
    if kk7 > 0 then
        gkf7 = kfocus_val * 0.0001
        kdata_received = 1
    endif
    
    kk8 OSClisten gihandle, "/openbci/focus/7", "f", kfocus_val
    if kk8 > 0 then
        gkf8 = kfocus_val * 0.0001
        kdata_received = 1
    endif
    
    kk9 OSClisten gihandle, "/openbci/focus/8", "f", kfocus_val
    if kk9 > 0 then
        gkf9 = kfocus_val * 0.0001
        kdata_received = 1
    endif
    
    kk10 OSClisten gihandle, "/openbci/focus/9", "f", kfocus_val
    if kk10 > 0 then
        gkf10 = kfocus_val * 0.0001
        kdata_received = 1
    endif
    
    kk11 OSClisten gihandle, "/openbci/focus/10", "f", kfocus_val
    if kk11 > 0 then
        gkf11 = kfocus_val * 0.0001
        kdata_received = 1
    endif
    
    kk12 OSClisten gihandle, "/openbci/focus/11", "f", kfocus_val
    if kk12 > 0 then
        gkf12 = kfocus_val * 0.0001
        kdata_received = 1
    endif
    
    kk13 OSClisten gihandle, "/openbci/focus/12", "f", kfocus_val
    if kk13 > 0 then
        gkf13 = kfocus_val * 0.0001
        kdata_received = 1
    endif
    
    kk14 OSClisten gihandle, "/openbci/focus/13", "f", kfocus_val
    if kk14 > 0 then
        gkf14 = kfocus_val * 0.0001
        kdata_received = 1
    endif
    
    kk15 OSClisten gihandle, "/openbci/focus/14", "f", kfocus_val
    if kk15 > 0 then
        gkf15 = kfocus_val * 0.0001
        kdata_received = 1
    endif
    
    kk16 OSClisten gihandle, "/openbci/focus/15", "f", kfocus_val
    if kk16 > 0 then
        gkf16 = kfocus_val * 0.0001
        kdata_received = 1
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
        printks "Focus Data: Global=%.3f Channels(1-8): %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n", 1, gkfocus_global, gkf1, gkf2, gkf3, gkf4, gkf5, gkf6, gkf7, gkf8
        kdata_received = 0
    elseif ktimer > 5 && int(ktimer) % 5 == 0 then
        printks "Waiting for Focus data...\n", 5
    endif
endin

; ==============================================================================
; Instrument 2: MIDI-controlled Synthesis with Focus modulation
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
    
    ; Scale focus values for pitch modulation (ensure they're in good range)
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
    
    ; Generate oscillators modulated by focus data - 8 channels
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
