<CsoundSynthesizer>
<CsInstruments>

sr = 44100
ksmps = 100
nchnls = 2
0dbfs = 1.0

schedule 1, 0, -1
schedule 3, 0, -1
schedule 4, 0, -1

garvbL init 0
garvbR init 0

gadelL init 0
gadelR init 0

ctrlinit 1, 21,0, 22,0, 23,0, 24,0

massign 1, 2

gkf1 init 0
gkf2 init 0
gkf3 init 0
gkf4 init 0
gkf5 init 0
gkf6 init 0

instr 1

	gihandle		OSCinit 9000  ; BioMus default port (was 5003 for Muse)


; BIOMUS API - Muse-Compatible Messages

; BioMus now sends Muse-compatible combined messages!
; Format: /biomus/elements/<band>_absolute sends [CH1, CH2, CH3, CH4]
; Format: /biomus/elements/<band>_relative sends [CH1, CH2, CH3, CH4]
; NOTE: BioMus (Ganglion) has 4 channels, Muse has 6. We use 4 channels.



; DELTA ABSOLUTE
;kk  OSClisten gihandle, "/biomus/elements/delta_absolute", "ffff", gkf1, gkf2, gkf3, gkf4

; THETA ABSOLUTE - ACTIVE (this is what the original script uses)
kk  OSClisten gihandle, "/biomus/elements/theta_absolute", "ffff", gkf1, gkf2, gkf3, gkf4

; ALPHA ABSOLUTE
;kk  OSClisten gihandle, "/biomus/elements/alpha_absolute", "ffff", gkf1, gkf2, gkf3, gkf4

; BETA ABSOLUTE
;kk  OSClisten gihandle, "/biomus/elements/beta_absolute", "ffff", gkf1, gkf2, gkf3, gkf4

; GAMMA ABSOLUTE
;kk  OSClisten gihandle, "/biomus/elements/gamma_absolute", "ffff", gkf1, gkf2, gkf3, gkf4


; DELTA RELATIVE (0-1 normalized)
;kk  OSClisten gihandle, "/biomus/elements/delta_relative", "ffff", gkf1, gkf2, gkf3, gkf4

; THETA RELATIVE
;kk  OSClisten gihandle, "/biomus/elements/theta_relative", "ffff", gkf1, gkf2, gkf3, gkf4

; ALPHA RELATIVE
;kk  OSClisten gihandle, "/biomus/elements/alpha_relative", "ffff", gkf1, gkf2, gkf3, gkf4

; BETA RELATIVE
;kk  OSClisten gihandle, "/biomus/elements/beta_relative", "ffff", gkf1, gkf2, gkf3, gkf4

; GAMMA RELATIVE
;kk  OSClisten gihandle, "/biomus/elements/gamma_relative", "ffff", gkf1, gkf2, gkf3, gkf4



endin

instr 2

 ; Novation LaunchKeyMini:  CC 21,22,23,24

 kspeed1 midic7 21, .01, 40
 printk2 kspeed1
  kspeed2 midic7 22, .01, 50
   printk2 kspeed2
    kspeed3 midic7 23, .01, 100
     printk2 kspeed3
        kspeed4 midic7 24, .01, 10
         printk2 kspeed4
 ktrig1 metro kspeed1
  ktrig2  metro kspeed2
   ktrig3 metro kspeed3
    ktrig4   metro kspeed4
kf1 samphold gkf1, ktrig1
kf2 samphold gkf2, ktrig2
kf3 samphold gkf3, ktrig3
kf4 samphold gkf4, ktrig4
icps cpsmidi
 aout1 = oscili(0.5, icps+cpspch( (kf1+2) ))
 aout2 = oscili(0.5, icps+cpspch( (kf2+2) ))
 aout3 = oscili(0.5, icps+cpspch( (kf3+2) ))
 aout4 = oscili(0.5, icps+cpspch( (kf4+2) ))
 aadsr madsr 1, 0.5, 0.8, .9
 aout = ((aout1 + aout2 + aout3 + aout4)/8) * aadsr
 garvbL += aout * 0.8
 garvbR += aout * 0.8
 gadelL += aout * 0.8
 gadelR += aout * 0.8
 outs aout, aout
 endin

instr revsc
    denorm garvbL
    denorm garvbR
    aout1, aout2 reverbsc garvbL, garvbR, 0.8, 8000
    outs aout1, aout2
    clear garvbL
    clear garvbR
endin

instr del
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
 <bgcolor mode="nobackground">
  <r>255</r>
  <g>255</g>
  <b>255</b>
 </bgcolor>
</bsbPanel>
<bsbPresets>
</bsbPresets>
