<CsoundSynthesizer>
; ================================================================================
; SIMPLE ACCELEROMETER PITCH CONTROL
; ================================================================================
; Only X accelerometer value controls pitch - clean and simple
; OSC Address: /openbci/accel
; ================================================================================
<CsOptions>
-odac -d
</CsOptions>
<CsInstruments>

sr = 44100
ksmps = 100
nchnls = 2
0dbfs = 1.0

; Simple waveforms
gisine ftgen 1, 0, 16384, 10, 1

; Scale for mapping accelerometer to pitch
giscale ftgen 100, 0, 8, -2, 60, 62, 64, 67, 69, 71, 72, 74  ; C major scale

; Global accelerometer X value
gkaccel_x init 0

; MIDI setup - route MIDI notes to instrument 3
massign 0, 3

; ==============================================================================
; Instrument 1: OSC Data Receiver (always on)
; ==============================================================================
instr 1
    ; Start OSC listener for GUI (port 12345)
    gihandle OSCinit 12345
    
    ; Declare local variables
    kaxis, kvalue init 0
    
    ; Listen to GUI accelerometer data (separate messages per axis)
    kk OSClisten gihandle, "/openbci/accel", "if", kaxis, kvalue
    if kk > 0 then
        ; Only store X axis (axis 0), ignore Y (1) and Z (2)
        if kaxis == 0 then
            gkaccel_x = kvalue
            printks "Accel X: %.3f -> Pitch control\n", 0.2, gkaccel_x
        endif
    endif
    
    ; Smooth out the value when no data
    gkaccel_x = gkaccel_x * 0.99
endin

; ==============================================================================
; Instrument 2: Effect processor for keyboard notes
; ==============================================================================
instr 2
    ; This instrument does nothing - just placeholder
endin

; ==============================================================================
; Instrument 3: MIDI-triggered note player with head-controlled effects
; ==============================================================================
instr 3
    ; Get MIDI note frequency
    icps cpsmidi
    
    ; Simple envelope for MIDI notes
    aenv madsr 0.1, 0.1, 0.8, 0.5
    
    ; Basic oscillator
    aosc poscil aenv, icps, 1
    
    ; Smooth accelerometer for effects
    ksmooth_accel portk gkaccel_x, 0.1
    
    ; Map head position to dry/wet mix (data is in m/s², ±4G = ±39.24 m/s²)
    ; Left tilt (negative) = dry, Right tilt (positive) = wet
    kwet = ((ksmooth_accel + 40) / 80)  ; Convert -40 to +40 m/s² range to 0-1
    kwet limit kwet, 0, 1
    kdry = 1 - kwet
    
    ; Add dramatic reverb effect
    awet reverb aosc, 5.0
    
    ; Mix dry and wet signals based on head position
    amix = (aosc * kdry) + (awet * kwet * 3)  ; Boost wet signal
    
    ; Output with more volume
    aout = amix * 0.5
    outs aout, aout
    
    ; Debug print
    printks "Head: %.2f -> Dry: %.2f, Wet: %.2f\n", 0.2, gkaccel_x, kdry, kwet
endin

</CsInstruments>
<CsScore>
; Start instruments
i1 0 3600  ; OSC receiver only

; Keep running for 1 hour
e
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
