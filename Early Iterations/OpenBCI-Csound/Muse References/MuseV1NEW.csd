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

massign 1, 2

gkf1 init 0
gkf2 init 0
gkf3 init 0
gkf4 init 0
gkf5 init 0
gkf6 init 0


instr 1

	gihandle	OSCinit 5003

/*

Frequency Ranges
      
+ delta_absolute  1-4Hz

DELTA WAVES (.5 TO 3 HZ)
Delta Waves, the slowest but loudest brainwaves

Delta brainwaves are slow, loud brainwaves (low frequency and deeply penetrating, like a drum beat). They are generated in deepest meditation and dreamless sleep. Delta waves suspend external awareness and are the source of empathy. Healing and regeneration are stimulated in this state, and that is why deep restorative sleep is so essential to the healing process.

  
+ theta_absolute  4-8Hz  

THETA WAVES (3 TO 8 HZ)
Theta brainwaves, occur in sleep and are also dominant in deep meditation.

Theta brainwaves occur most often in sleep but are also dominant in deep meditation. Theta is our gateway to learning, memory, and intuition. In theta, our senses are withdrawn from the external world and focused on signals originating from within. It is that twilight state which we normally only experience fleetingly as we wake or drift off to sleep. In theta we are in a dream; vivid imagery, intuition and information beyond our normal conscious awareness. It’s where we hold our ‘stuff’, our fears, troubled history, and nightmares.

   
+ alpha_absolute  7.5-13Hz   

ALPHA WAVES (8 TO 12 HZ)
Alpha brainwaves occur during quietly flowing thoughts, but not quite meditation.

Alpha brainwaves are dominant during quietly flowing thoughts, and in some meditative states. Alpha is ‘the power of now’, being here, in the present. Alpha is the resting state for the brain. Alpha waves aid overall mental coordination, calmness, alertness, mind/body integration and learning.
  

+ beta_absolute 13-30Hz  

BETA WAVES (12 TO 38 HZ)
Beta brainwaves are present in our normal waking state of consciousness.

Beta brainwaves dominate our normal waking state of consciousness when attention is directed towards cognitive tasks and the outside world. Beta is a ‘fast’ activity, present when we are alert, attentive, engaged in problem solving, judgment, decision making, or focused mental activity.

Beta brainwaves are further divided into three bands; Lo-Beta (Beta1, 12-15Hz) can be thought of as a 'fast idle', or musing. Beta (Beta2, 15-22Hz) is high engagement or actively figuring something out. Hi-Beta (Beta3, 22-38Hz) is highly complex thought, integrating new experiences, high anxiety, or excitement. Continual high frequency processing is not a very efficient way to run the brain, as it takes a tremendous amount of energy. 

   
+ gamma_absolute  30-44Hz

GAMMA WAVES (38 TO 42 HZ)
Gamma brainwaves are the fastest of brain waves and relate to simultaneous processing of information from different brain areas

Gamma brainwaves are the fastest of brain waves (high frequency, like a flute), and relate to simultaneous processing of information from different brain areas. Gamma brainwaves pass information rapidly and quietly. The most subtle of the brainwave frequencies, the mind has to be quiet to access gamma. 

Gamma was dismissed as 'spare brain noise' until researchers discovered it was highly active when in states of universal love, altruism, and the ‘higher virtues’. Gamma is also above the frequency of neuronal firing, so how it is generated remains a mystery. It is speculated that gamma rhythms modulate perception and consciousness, and that a greater presence of gamma relates to expanded consciousness and spiritual emergence.

*/

;kk  OSClisten gihandle, "/muse/elements/delta_absolute", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/delta_absolute", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/theta_absolute", "ffff", gkf1, gkf2, gkf3, gkf4

kk  OSClisten gihandle, "/muse/elements/alpha_absolute", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/beta_absolute", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/gamma_absolute", "ffff", gkf1, gkf2, gkf3, gkf4  


/*

Relative Band Powers

The relative band powers are calculated by dividing the absolute linear-scale power in one band over the sum of the absolute linear-scale powers in all bands. The linear-scale band power can be calculated from the log-scale band power thusly: linear-scale band power = 10^ (log-scale band power).

Therefore, the relative band powers can be calculated as percentages of linear-scale band powers in each band. For example:

alpha_relative = (10^alpha_absolute / (10^alpha_absolute + 10^beta_absolute + 10^delta_absolute + 10^gamma_absolute + 10^theta_absolute))

The resulting value is between 0 and 1. However, the value will never be 0 or 1.
These values are emitted at 10Hz.

*/

;kk  OSClisten gihandle, "/muse/elements/delta_relative", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/theta_relative", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/alpha_relative", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/beta_relative", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/gamma_relative", "ffff", gkf1, gkf2, gkf3, gkf4



; EEG

;kk OSClisten gihandle, "/muse/eeg", "ffffff", gkf1, gkf2, gkf3, gkf4, gkf5, gkf6

;kk  OSClisten gihandle, "/muse/variance_eeg", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/notch_filtered_eeg", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/variance_notch_filtered_eeg", "ffff", gkf1, gkf2, gkf3, gkf4



; Accelerometer & Gyro

;kk  OSClisten gihandle, "/muse/acc", "fff", gkf1, gkf2, gkf3

;kk  OSClisten gihandle, "/muse/gyro", "fff", gkf1, gkf2, gkf3



; Session Scores

;kk  OSClisten gihandle, "/muse/elements/delta_session_score", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/delta_absolute/elements/theta_session_score", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/delta_absolute/elements/alpha_session_score", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/delta_absolute/elements/beta_session_score", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/delta_absolute/elements/gamma_session_score", "ffff", gkf1, gkf2, gkf3, gkf4


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
