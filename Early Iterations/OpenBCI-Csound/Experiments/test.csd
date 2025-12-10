<CsoundSynthesizer>
; ================================================================================
; OPENBCI ACCEL → REALTIME PAN / BRIGHTNESS / TEMPO
; ================================================================================
; Listens to /openbci/accel on UDP 5004
; - Head tilt (X accel) → pan (L/R), filter brightness, arpeggio speed
; - Continuous updates (not only on note ticks)
; ================================================================================
<CsOptions>
-odac -d
</CsOptions>

<CsInstruments>
sr     = 44100
ksmps  = 100
nchnls = 2
0dbfs  = 1.0

gisine  ftgen 1, 0, 16384, 10, 1
giscale ftgen 100, 0, 8, -2, 60, 62, 64, 67, 69, 71, 72, 74  ; C major

; Globals updated by OSC
gkAx init 0            ; raw accel X in m/s^2
gkAy init 0
gkAz init 0

; Smoothed + normalized (to ±1 g)
gkXg init 0
gkYg init 0
gkZg init 0

; ==============================================================================
; Instrument 1: OSC receiver (always on)
; ==============================================================================
instr 1
  gihandle OSCinit 5004

  kx  init 0
  ky  init 0
  kz  init 0

  ; Receive accel in m/s^2 (Python multiplies by 9.81)
  kk OSClisten gihandle, "/openbci/accel", "fff", kx, ky, kz
  if kk > 0 then
      gkAx = kx
      gkAy = ky
      gkAz = kz
  endif

  ; Smooth & normalize to g units (-1..1)
  gkXg portk (gkAx / 9.81), 0.12
  gkYg portk (gkAy / 9.81), 0.12
  gkZg portk (gkAz / 9.81), 0.12

  ; Occasionally print for debug
  printks "Accel X(g): %6.2f | Y(g): %6.2f | Z(g): %6.2f\n", 0.5, gkXg, gkYg, gkZg
endin

; ==============================================================================
; Instrument 2: Arpeggiator with continuous motion control
; ==============================================================================
instr 2
  ; Use X tilt as main control (roll-ish). Range clamp to [-1,1].
  kx limit gkXg, -1, 1

  ; Map X to 0..1 for panner
  kpan = (kx * 0.5) + 0.5

  ; Map X to filter brightness (Hz)
  ; (-1) => darker (~300 Hz), (+1) => brighter (~3000 Hz)
  kf   = 300 + ( (kx + 1) * 0.5 ) * 2700
  kf   limit kf, 200, 5000

  ; Map X to arpeggio speed (Hz) : 0.5..5 Hz
  kspeed = 0.5 + ( (kx + 1) * 0.5 ) * 4.5
  kspeed limit kspeed, 0.5, 5.0

  ; Trigger next note by tempo
  ktrig metro kspeed

  ; State: current scale index and freq
  kidx init 0
  kcurcps init 261.63

  if ktrig == 1 then
      kidx = (kidx + 1) % 8
      kmidi table kidx, 100
      kcurcps = cpsmidinn(kmidi)
  endif

  ; === Audio path (ALWAYS RUNS) ===
  ; Gentle sustain so motion is audible continuously
  aenv linsegr 0.0, 0.02, 0.5, 0.2, 0.5
  aosc poscil aenv, kcurcps, 1

  ; Motion-controlled brightness
  af  moogvcf aosc, kf, 0.3

  ; Motion-controlled panning
  aL, aR pan2 af, kpan
  outs aL*0.8, aR*0.8

  ; Debug every ~0.5s
  printks "X: %+.2f | pan: %.2f | filt: %4.0f Hz | tempo: %.2f Hz | note: %.1f Hz\n", 0.5, kx, kpan, kf, kspeed, kcurcps
endin

</CsInstruments>

<CsScore>
i1 0 3600   ; OSC receiver
i2 0 3600   ; Motion-controlled arpeggio
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
