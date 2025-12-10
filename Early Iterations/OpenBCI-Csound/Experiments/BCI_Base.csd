<CsoundSynthesizer>
; ================================================================================
; OPENBCI BASE TEMPLATE - Shared functionality for all data types
; ================================================================================
; This contains all common synthesis, effects, and utility code
; Include this template and override the BCI listener instrument for each data type
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

; =============================================================================
; SHARED BCI CONTROL VALUES - All data types map to these
; =============================================================================
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

; OSC Configuration
giosc_port init 5004
gSosc_base_addr = "/openbci"

; Schedule base instruments
schedule 2, 0, -1        ; Synthesis (when triggered by MIDI)
schedule 3, 0, -1        ; Reverb
schedule 4, 0, -1        ; Delay

; MIDI setup
massign 1, 2

; ==============================================================================
; Instrument 1: BCI Data Listener (OVERRIDE THIS IN EACH SPECIFIC SCRIPT)
; ==============================================================================
instr 1
    ; This instrument should be overridden in each specific data type script
    printks "Base BCI listener - should be overridden\n", 5
endin

; ==============================================================================
; Instrument 2: MIDI-controlled Synthesis with BCI modulation
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
    
    ; Scale BCI values for pitch modulation (ensure they're in good range)
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
    
    ; Generate oscillators modulated by BCI data - 8 channels
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