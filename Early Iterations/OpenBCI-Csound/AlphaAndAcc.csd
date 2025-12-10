<CsoundSynthesizer>
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

; Scale tables (currently unused)
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

; BCI globals - using alpha band power from all 16 channels
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

; >>> NEW: Global wet control (0=dry, 1=wet), driven by accelerometer X in [-1,1]
gkWet init 0

; Schedule instruments
schedule 1, 0, -1        ; BCI + ACCEL OSC Listener
schedule 3, 0, -1        ; Reverb
schedule 4, 0, -1        ; Delay

; MIDI setup
massign 1, 2

; ==============================================================================
; Instrument 1: OpenBCI Band Power + Accelerometer Listener
; ==============================================================================
instr 1
    ; Initialize OSC handle
    gihandle OSCinit 5003
    
    ; Local variables for band power data [delta, theta, alpha, beta, gamma] - 16 channels
    kdelta1, ktheta1, kalpha1, kbeta1, kgamma1 init 0
    kdelta2, ktheta2, kalpha2, kbeta2, kgamma2 init 0  
    kdelta3, ktheta3, kalpha3, kbeta3, kgamma3 init 0
    kdelta4, ktheta4, kalpha4, kbeta4, kgamma4 init 0
    kdelta5, ktheta5, kalpha5, kbeta5, kgamma5 init 0
    kdelta6, ktheta6, kalpha6, kbeta6, kgamma6 init 0
    kdelta7, ktheta7, kalpha7, kbeta7, kgamma7 init 0
    kdelta8, ktheta8, kalpha8, kbeta8, kgamma8 init 0
    kdelta9, ktheta9, kalpha9, kbeta9, kgamma9 init 0
    kdelta10, ktheta10, kalpha10, kbeta10, kgamma10 init 0
    kdelta11, ktheta11, kalpha11, kbeta11, kgamma11 init 0
    kdelta12, ktheta12, kalpha12, kbeta12, kgamma12 init 0
    kdelta13, ktheta13, kalpha13, kbeta13, kgamma13 init 0
    kdelta14, ktheta14, kalpha14, kbeta14, kgamma14 init 0
    kdelta15, ktheta15, kalpha15, kbeta15, kgamma15 init 0
    kdelta16, ktheta16, kalpha16, kbeta16, kgamma16 init 0
    
    ; Listen to OpenBCI band power data (all 16 channels)
    kk1 OSClisten gihandle, "/openbci/band-power/0", "fffff", kdelta1, ktheta1, kalpha1, kbeta1, kgamma1
    kk2 OSClisten gihandle, "/openbci/band-power/1", "fffff", kdelta2, ktheta2, kalpha2, kbeta2, kgamma2
    kk3 OSClisten gihandle, "/openbci/band-power/2", "fffff", kdelta3, ktheta3, kalpha3, kbeta3, kgamma3
    kk4 OSClisten gihandle, "/openbci/band-power/3", "fffff", kdelta4, ktheta4, kalpha4, kbeta4, kgamma4
    kk5 OSClisten gihandle, "/openbci/band-power/4", "fffff", kdelta5, ktheta5, kalpha5, kbeta5, kgamma5
    kk6 OSClisten gihandle, "/openbci/band-power/5", "fffff", kdelta6, ktheta6, kalpha6, kbeta6, kgamma6
    kk7 OSClisten gihandle, "/openbci/band-power/6", "fffff", kdelta7, ktheta7, kalpha7, kbeta7, kgamma7
    kk8 OSClisten gihandle, "/openbci/band-power/7", "fffff", kdelta8, ktheta8, kalpha8, kbeta8, kgamma8
    kk9 OSClisten gihandle, "/openbci/band-power/8", "fffff", kdelta9, ktheta9, kalpha9, kbeta9, kgamma9
    kk10 OSClisten gihandle, "/openbci/band-power/9", "fffff", kdelta10, ktheta10, kalpha10, kbeta10, kgamma10
    kk11 OSClisten gihandle, "/openbci/band-power/10", "fffff", kdelta11, ktheta11, kalpha11, kbeta11, kgamma11
    kk12 OSClisten gihandle, "/openbci/band-power/11", "fffff", kdelta12, ktheta12, kalpha12, kbeta12, kgamma12
    kk13 OSClisten gihandle, "/openbci/band-power/12", "fffff", kdelta13, ktheta13, kalpha13, kbeta13, kgamma13
    kk14 OSClisten gihandle, "/openbci/band-power/13", "fffff", kdelta14, ktheta14, kalpha14, kbeta14, kgamma14
    kk15 OSClisten gihandle, "/openbci/band-power/14", "fffff", kdelta15, ktheta15, kalpha15, kbeta15, kgamma15
    kk16 OSClisten gihandle, "/openbci/band-power/15", "fffff", kdelta16, ktheta16, kalpha16, kbeta16, kgamma16

    ; >>> NEW: Accelerometer listeners (X in [-1,1])
    kx      init 0
    ky      init 0
    kz      init 0
    kx1     init 0
    kx2     init 0
    kacc1   OSClisten gihandle, "/openbci/accel", "fff", kx, ky, kz   ; common OpenBCI schema
    kacc2   OSClisten gihandle, "/accel",        "f",   kx1           ; generic single-float
    kacc3   OSClisten gihandle, "/accel/x",      "f",   kx2           ; explicit x

    ; Choose any that arrived this control period
    if      (kacc1 > 0) then
        kxsel = kx
    elseif  (kacc2 > 0) then
        kxsel = kx1
    elseif  (kacc3 > 0) then
        kxsel = kx2
    else
        kxsel = (gkWet * 2) - 1      ; keep previous mapped state
    endif

    ; Map X in [-1,1] -> wet in [0,1], clamp and smooth
    kWetRaw  = limit((kxsel + 1) * 0.5, 0, 1)
    kWetS    port  kWetRaw, 0.1       ; ~100 ms smoothing
    gkWet    = kWetS

    ; Update alpha globals (scale to ~0-1)
    if kk1 > 0 then
        gkf1 = kalpha1 * 0.0001
    endif
    if kk2 > 0 then
        gkf2 = kalpha2 * 0.0001
    endif
    if kk3 > 0 then
        gkf3 = kalpha3 * 0.0001
    endif
    if kk4 > 0 then
        gkf4 = kalpha4 * 0.0001
    endif
    if kk5 > 0 then
        gkf5 = kalpha5 * 0.0001
    endif
    if kk6 > 0 then
        gkf6 = kalpha6 * 0.0001
    endif
    if kk7 > 0 then
        gkf7 = kalpha7 * 0.0001
    endif
    if kk8 > 0 then
        gkf8 = kalpha8 * 0.0001
    endif
    if kk9 > 0 then
        gkf9 = kalpha9 * 0.0001
    endif
    if kk10 > 0 then
        gkf10 = kalpha10 * 0.0001
    endif
    if kk11 > 0 then
        gkf11 = kalpha11 * 0.0001
    endif
    if kk12 > 0 then
        gkf12 = kalpha12 * 0.0001
    endif
    if kk13 > 0 then
        gkf13 = kalpha13 * 0.0001
    endif
    if kk14 > 0 then
        gkf14 = kalpha14 * 0.0001
    endif
    if kk15 > 0 then
        gkf15 = kalpha15 * 0.0001
    endif
    if kk16 > 0 then
        gkf16 = kalpha16 * 0.0001
    endif

    ; Fallback smoothing when no band-power data received
    if kk1 == 0 then
        gkf1 = gkf1 * 0.995 + 0.5 * 0.005
    endif
    if kk2 == 0 then
        gkf2 = gkf2 * 0.995 + 0.5 * 0.005
    endif
    if kk3 == 0 then
        gkf3 = gkf3 * 0.995 + 0.5 * 0.005
    endif
    if kk4 == 0 then
        gkf4 = gkf4 * 0.995 + 0.5 * 0.005
    endif
    if kk5 == 0 then
        gkf5 = gkf5 * 0.995 + 0.5 * 0.005
    endif
    if kk6 == 0 then
        gkf6 = gkf6 * 0.995 + 0.5 * 0.005
    endif
    if kk7 == 0 then
        gkf7 = gkf7 * 0.995 + 0.5 * 0.005
    endif
    if kk8 == 0 then
        gkf8 = gkf8 * 0.995 + 0.5 * 0.005
    endif
    if kk9 == 0 then
        gkf9 = gkf9 * 0.995 + 0.5 * 0.005
    endif
    if kk10 == 0 then
        gkf10 = gkf10 * 0.995 + 0.5 * 0.005
    endif
    if kk11 == 0 then
        gkf11 = gkf11 * 0.995 + 0.5 * 0.005
    endif
    if kk12 == 0 then
        gkf12 = gkf12 * 0.995 + 0.5 * 0.005
    endif
    if kk13 == 0 then
        gkf13 = gkf13 * 0.995 + 0.5 * 0.005
    endif
    if kk14 == 0 then
        gkf14 = gkf14 * 0.995 + 0.5 * 0.005
    endif
    if kk15 == 0 then
        gkf15 = gkf15 * 0.995 + 0.5 * 0.005
    endif
    if kk16 == 0 then
        gkf16 = gkf16 * 0.995 + 0.5 * 0.005
    endif
    
    ; Connection monitoring and debug output
    ktimer timeinsts
    if kk1 > 0 || kk2 > 0 || kk3 > 0 || kk4 > 0 || kk5 > 0 || kk6 > 0 || kk7 > 0 || kk8 > 0 || kk9 > 0 || kk10 > 0 || kk11 > 0 || kk12 > 0 || kk13 > 0 || kk14 > 0 || kk15 > 0 || kk16 > 0 then
        printks "Alpha power (1-8): %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n", 1, gkf1, gkf2, gkf3, gkf4, gkf5, gkf6, gkf7, gkf8
        printks "Alpha power (9-16): %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f\n", 1, gkf9, gkf10, gkf11, gkf12, gkf13, gkf14, gkf15, gkf16
    elseif ktimer > 5 && int(ktimer) % 5 == 0 then
        printks "Waiting for OpenBCI band power data (16 channels)...\n", 5
    endif

    ; >>> NEW: Wet debug
    if int(ktimer) % 1 == 0 then
        printks "Wet (from accel X): %.2f\n", 1, gkWet
    endif
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
    kf5 samphold gkf5, ktrig1
    kf6 samphold gkf6, ktrig2
    kf7 samphold gkf7, ktrig3
    kf8 samphold gkf8, ktrig4
    
    ; Scale BCI values for pitch modulation
    kf1_scaled = abs(kf1) * 3 + 0.2
    kf2_scaled = abs(kf2) * 3 + 0.2
    kf3_scaled = abs(kf3) * 3 + 0.2
    kf4_scaled = abs(kf4) * 3 + 0.2
    kf5_scaled = abs(kf5) * 2 + 0.3
    kf6_scaled = abs(kf6) * 2 + 0.3
    kf7_scaled = abs(kf7) * 1.5 + 0.4
    kf8_scaled = abs(kf8) * 1.5 + 0.4
    
    ; Get MIDI note frequency
    icps cpsmidi
    
    ; Oscillators modulated by BCI alpha waves - 8 channels
    aout1 oscili 0.3, icps + cpspch(kf1_scaled + 2), 1
    aout2 oscili 0.3, icps + cpspch(kf2_scaled + 2), 1
    aout3 oscili 0.3, icps + cpspch(kf3_scaled + 2), 1
    aout4 oscili 0.3, icps + cpspch(kf4_scaled + 2), 1
    aout5 oscili 0.2, icps * 0.5 + cpspch(kf5_scaled + 1), 2
    aout6 oscili 0.2, icps * 0.5 + cpspch(kf6_scaled + 1), 2
    aout7 oscili 0.1, icps * 2 + cpspch(kf7_scaled + 3), 3
    aout8 oscili 0.1, icps * 2 + cpspch(kf8_scaled + 3), 3
    
    ; ADSR envelope
    aadsr madsr 1, 0.5, 0.8, 0.9
    
    ; Mix
    aout = ((aout1 + aout2 + aout3 + aout4 + aout5 + aout6 + aout7 + aout8) / 12) * aadsr
    
    ; Sends to FX (unchanged)
    garvbL += aout * 0.8
    garvbR += aout * 0.8
    gadelL += aout * 0.8
    gadelR += aout * 0.8
    
    ; >>> CHANGED: Dry output scaled by (1 - wet)
    kd = 1 - gkWet
    outs aout * kd, aout * kd
endin

; ==============================================================================
; Instrument 3: Reverb (wet scaled by gkWet)
; ==============================================================================
instr 3
    denorm garvbL
    denorm garvbR
    aout1, aout2 reverbsc garvbL, garvbR, 0.8, 8000
    ; >>> CHANGED: scale wet return by gkWet
    outs aout1 * gkWet, aout2 * gkWet
    clear garvbL
    clear garvbR
endin

; ==============================================================================
; Instrument 4: Delay (wet scaled by gkWet)
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
    ; >>> CHANGED: scale wet return by gkWet
    outs adelOutL * gkWet, adelOutR * gkWet
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
