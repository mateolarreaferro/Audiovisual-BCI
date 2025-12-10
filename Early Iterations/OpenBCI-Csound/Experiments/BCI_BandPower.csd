<CsoundSynthesizer>
; ================================================================================
; OPENBCI BAND POWER LISTENER
; ================================================================================
; Receives individual frequency band powers per channel
; OSC Address: /openbci/band-power/{channel}
; Data Format: delta, theta, alpha, beta, gamma (5 floats per channel)
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

; Schedule instruments
schedule 1, 0, -1        ; Band Power Listener
schedule 3, 0, -1        ; Reverb
schedule 4, 0, -1        ; Delay

; MIDI setup
massign 1, 2

; ==============================================================================
; Instrument 1: Intelligent Brain Wave Band Power Listener  
; ==============================================================================
instr 1
    ; Initialize OSC handle
    gihandle OSCinit 5001
    
    ; Brain wave band storage for first 8 channels (most important)
    kdelta1, kdelta2, kdelta3, kdelta4, kdelta5, kdelta6, kdelta7, kdelta8 init 0
    ktheta1, ktheta2, ktheta3, ktheta4, ktheta5, ktheta6, ktheta7, ktheta8 init 0
    kalpha1, kalpha2, kalpha3, kalpha4, kalpha5, kalpha6, kalpha7, kalpha8 init 0
    kbeta1, kbeta2, kbeta3, kbeta4, kbeta5, kbeta6, kbeta7, kbeta8 init 0
    kgamma1, kgamma2, kgamma3, kgamma4, kgamma5, kgamma6, kgamma7, kgamma8 init 0
    kdata_received init 0
    kchan init 0
    
    ; Listen to band power data for first 8 channels
    kk1 OSClisten gihandle, "/openbci/band-power/0", "fffff", kdelta1, ktheta1, kalpha1, kbeta1, kgamma1
    if kk1 > 0 then
        kdata_received = 1
    endif
    
    kk2 OSClisten gihandle, "/openbci/band-power/1", "fffff", kdelta2, ktheta2, kalpha2, kbeta2, kgamma2
    if kk2 > 0 then
        kdata_received = 1
    endif
    
    kk3 OSClisten gihandle, "/openbci/band-power/2", "fffff", kdelta3, ktheta3, kalpha3, kbeta3, kgamma3
    if kk3 > 0 then
        kdata_received = 1
    endif
    
    kk4 OSClisten gihandle, "/openbci/band-power/3", "fffff", kdelta4, ktheta4, kalpha4, kbeta4, kgamma4
    if kk4 > 0 then
        kdata_received = 1
    endif
    
    kk5 OSClisten gihandle, "/openbci/band-power/4", "fffff", kdelta5, ktheta5, kalpha5, kbeta5, kgamma5
    if kk5 > 0 then
        kdata_received = 1
    endif
    
    kk6 OSClisten gihandle, "/openbci/band-power/5", "fffff", kdelta6, ktheta6, kalpha6, kbeta6, kgamma6
    if kk6 > 0 then
        kdata_received = 1
    endif
    
    kk7 OSClisten gihandle, "/openbci/band-power/6", "fffff", kdelta7, ktheta7, kalpha7, kbeta7, kgamma7
    if kk7 > 0 then
        kdata_received = 1
    endif
    
    kk8 OSClisten gihandle, "/openbci/band-power/7", "fffff", kdelta8, ktheta8, kalpha8, kbeta8, kgamma8
    if kk8 > 0 then
        kdata_received = 1
    endif
    
    ; Map brain waves to musical elements intelligently
    if kdata_received == 1 then
        ; DELTA (0.5-4Hz) - Deep unconscious, bass foundation
        gkf1 = (kdelta1 + kdelta2) * 0.00005  ; Slow bass pulse
        gkf2 = (kdelta3 + kdelta4) * 0.00005  ; Sub-bass harmonics
        
        ; THETA (4-8Hz) - Creativity, meditation, flowing patterns  
        gkf3 = (ktheta1 + ktheta2 + ktheta3) * 0.00003  ; Creative flow
        gkf4 = (ktheta4 + ktheta5 + ktheta6) * 0.00003  ; Meditative patterns
        
        ; ALPHA (8-13Hz) - Relaxed awareness, main arpeggios
        gkf5 = (kalpha1 + kalpha2) * 0.00008  ; Primary arpeggio
        gkf6 = (kalpha3 + kalpha4) * 0.00008  ; Secondary arpeggio  
        gkf7 = (kalpha5 + kalpha6) * 0.00008  ; Tertiary arpeggio
        gkf8 = (kalpha7 + kalpha8) * 0.00008  ; Quaternary arpeggio
        
        ; BETA (13-30Hz) - Active thinking, rhythmic patterns
        gkf9 = (kbeta1 + kbeta3 + kbeta5) * 0.00006   ; Fast rhythmic
        gkf10 = (kbeta2 + kbeta4 + kbeta6) * 0.00006  ; Syncopated rhythm
        
        ; GAMMA (30-100Hz) - High cognition, frequency sparkles
        gkf11 = kgamma1 * 0.0001  ; High freq sparkle 1
        gkf12 = kgamma3 * 0.0001  ; High freq sparkle 2  
        gkf13 = kgamma5 * 0.0001  ; High freq sparkle 3
        gkf14 = kgamma7 * 0.0001  ; High freq sparkle 4
        
        ; MIXED BRAIN STATE INDICATORS
        ktotal_alpha = kalpha1 + kalpha2 + kalpha3 + kalpha4 + kalpha5 + kalpha6 + kalpha7 + kalpha8
        ktotal_beta = kbeta1 + kbeta2 + kbeta3 + kbeta4 + kbeta5 + kbeta6 + kbeta7 + kbeta8
        
        ; Brain state blend controls
        gkf15 = (ktotal_alpha - ktotal_beta) * 0.000005  ; Alpha dominance vs Beta
        gkf16 = (ktotal_alpha + ktotal_beta) * 0.000008  ; Overall brain activity
    endif
    
    ; Fallback smoothing when no data received
    if kdata_received == 0 then
        gkf1 = gkf1 * 0.995 + 0.01 * 0.005     ; Slow decay for bass
        gkf2 = gkf2 * 0.995 + 0.01 * 0.005
        gkf3 = gkf3 * 0.995 + 0.02 * 0.005     ; Medium decay for creativity
        gkf4 = gkf4 * 0.995 + 0.02 * 0.005
        gkf5 = gkf5 * 0.99 + 0.05 * 0.01       ; Faster decay for arpeggios
        gkf6 = gkf6 * 0.99 + 0.05 * 0.01
        gkf7 = gkf7 * 0.99 + 0.05 * 0.01
        gkf8 = gkf8 * 0.99 + 0.05 * 0.01
        gkf9 = gkf9 * 0.985 + 0.03 * 0.015     ; Medium-fast decay for rhythm
        gkf10 = gkf10 * 0.985 + 0.03 * 0.015
        gkf11 = gkf11 * 0.98 + 0.001 * 0.02    ; Fast decay for sparkles
        gkf12 = gkf12 * 0.98 + 0.001 * 0.02
        gkf13 = gkf13 * 0.98 + 0.001 * 0.02
        gkf14 = gkf14 * 0.98 + 0.001 * 0.02
        gkf15 = gkf15 * 0.99                   ; State blends fade naturally
        gkf16 = gkf16 * 0.99
    endif
    
    ; Enhanced monitoring with brain wave analysis
    ktimer timeinsts
    if kdata_received == 1 then
        kdominant_wave init 0
        kmax_power = kalpha1 + kalpha2 + kalpha3 + kalpha4
        kdominant_wave = 2  ; Alpha default
        
        ; Determine dominant brain wave
        if kbeta1 + kbeta2 + kbeta3 + kbeta4 > kmax_power then
            kdominant_wave = 3  ; Beta dominant
            kmax_power = kbeta1 + kbeta2 + kbeta3 + kbeta4
        endif
        
        if ktheta1 + ktheta2 + ktheta3 + ktheta4 > kmax_power then
            kdominant_wave = 1  ; Theta dominant
        endif
        
        if kdelta1 + kdelta2 + kdelta3 + kdelta4 > kmax_power then
            kdominant_wave = 0  ; Delta dominant
        endif
        
        if kgamma1 + kgamma2 + kgamma3 + kgamma4 > kmax_power then
            kdominant_wave = 4  ; Gamma dominant
        endif
        
        if kdominant_wave == 0 then
            Swave = "DELTA"
        elseif kdominant_wave == 1 then  
            Swave = "THETA"
        elseif kdominant_wave == 2 then
            Swave = "ALPHA"
        elseif kdominant_wave == 3 then
            Swave = "BETA"
        else
            Swave = "GAMMA"
        endif
        
        printks "Brain State: %s dominant | Activity: %.3f | Balance: %.3f\n", 1, Swave, gkf16, gkf15
        kdata_received = 0
    elseif ktimer > 5 && int(ktimer) % 5 == 0 then
        printks "Waiting for Brain Wave Band Power data...\n", 5
    endif
endin

; ==============================================================================
; Instrument 2: MIDI-controlled Synthesis with Band Power modulation
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
    
    ; Scale band power values for pitch modulation (ensure they're in good range)
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
    
    ; Generate oscillators modulated by band power data - 8 channels
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
