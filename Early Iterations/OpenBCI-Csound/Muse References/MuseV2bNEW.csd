<CsoundSynthesizer>

<CsInstruments>

sr = 44100
ksmps = 100
nchnls = 2
0dbfs = 1.0

schedule 1, 0, -1
schedule "revsc", 0, -1
schedule "del", 0, -1

gisine   ftgen 1, 0, 16384, 10, 1	;sine wave
gisquare ftgen 2, 0, 16384, 10, 1, 0 , .33, 0, .2 , 0, .14, 0 , .11, 0, .09 ;odd harmonics
gisaw    ftgen 3, 0, 16384, 10, 0, .2, 0, .4, 0, .6, 0, .8, 0, 1, 0, .8, 0, .6, 0, .4, 0,.2 ;even harmonics

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

ctrlinit 1, 21,33, 26,111, 22,00, 23,00
ctrlinit 1, 24,00, 3,127, 4,127, 25,60
ctrlinit 1, 6,100, 27,110, 28,100, 31,59

giscale1 ftgen 111, 0, 8, -2, 72, 74, 76, 79, 81, 83, 72, 84
giscale2 ftgen 112, 0, 8, -2, 60, 55, 60, 53, 48, 52, 53, 55
giscale3 ftgen 113, 0, 8, -2, 48, 57, 84, 59, 48, 72, 41, 36
giscale4 ftgen 114, 0, 8, -2, 36, 43, 45, 36, 41, 36, 41, 45

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

;kk  OSClisten gihandle, "/muse/elements/theta_absolute", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/alpha_absolute", "ffff", gkf1, gkf2, gkf3, gkf4

kk  OSClisten gihandle, "/muse/elements/beta_absolute", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/gamma_absolute", "ffff", gkf1, gkf2, gkf3, gkf4  



; EEG

;kk OSClisten gihandle, "/muse/eeg", "ffffff", gkf1, gkf2, gkf3, gkf4, gkf5, gkf6;

;kk  OSClisten gihandle, "/muse/variance_eeg", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/notch_filtered_eeg", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/variance_notch_filtered_eeg", "ffff", gkf1, gkf2, gkf3, gkf4



; Accelerometer & Gyro

;kk  OSClisten gihandle, "/muse/acc", "fff", gkf1, gkf2, gkf3

;kk  OSClisten gihandle, "/muse/gyro", "fff", gkf1, gkf2, gkf3



/*

Relative Band Powers (NOTE: Not in the range of MIDI instrument2 )

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


; Session Scores (NOTE: Not in range of MIDI instrument 2)

;kk  OSClisten gihandle, "/muse/elements/delta_session_score", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/delta_absolute/elements/theta_session_score", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/delta_absolute/elements/alpha_session_score", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/delta_absolute/elements/beta_session_score", "ffff", gkf1, gkf2, gkf3, gkf4

;kk  OSClisten gihandle, "/muse/elements/delta_absolute/elements/gamma_session_score", "ffff", gkf1, gkf2, gkf3, gkf4

endin





instr 2
 
 ; Novation LaunchKeyMini:  CC 21,22,23,24
  ; nanoKEY Studio:  CC 20,21,22,23,24,25,26,27
   ; nanoKONTROL2: CC 0,1,2,3,4,5,6,7 (sliders)
    ; nanoKONTROL2: CC 8,9,10,11,12,13,14,15 (knobs
  
  kgain1 midic7 21, 0, 1
   kspeed1 midic7 25, .1, 8
 		printk2 kspeed1
    ktrig1 metro kspeed1
     ksize1 midic7 3, 1, 8
  		printk2 ksize1
  	  ksend1 midic7 4, 0, 1	
  	  		
   kgain2 midic7 22, 0, 1
    kspeed2 midic7 26, .1, 8
     ktrig2  metro kspeed2
       ksize2 midic7 5, 1, 8     
  	     ksend2 midic7 6, 0, 1	
   	      
    kgain3 midic7 23, 0, 1
      kspeed3 midic7 27, .1, 4
       ktrig3  metro kspeed3
         ksize3 midic7 7, 1, 8 
  	       ksend3 midic7 8, 0, 1	     
              
      kgain4 midic7 24, 0, 1 
        kspeed4 midic7 28, .1, 4
          ktrig4   metro kspeed4
            ksize4 midic7 9, 1, 8         
             ksend4 midic7 10, 0, 1	 
 
    agate1 triglinseg ktrig1,.7,.5,0
    agate2 triglinseg ktrig2,0,.1,.7,.2,0
    agate3 triglinseg ktrig3,.4,.5,0
    agate4 triglinseg ktrig4,.4,.6,0
    
kf1 samphold gkf1, ktrig1
 kf2 samphold gkf2, ktrig2
  kf3 samphold gkf3, ktrig3
   kf4 samphold gkf4, ktrig4
   
   kndx1 randomh 0,(ksize1)+kf1, kspeed1
 kpitch1 table int(abs(kndx1)), giscale1
 
    kndx2 randomh 0,(ksize2)+kf2, kspeed2
 kpitch2 table int(abs(kndx2)), giscale2
 
    kndx3 randomh 0,(ksize3)+kf3, kspeed3
 kpitch3 table int(abs(kndx3)), giscale3
 
    kndx4 randomh 0,(ksize4)+kf4, kspeed4
 kpitch4 table int(abs(kndx4)), giscale4

iNum notnum
print iNum 
iTrans = (exp(log(2.0)*((iNum)-69.0)/12.0))
print iTrans

 
aout1 = oscili(0.5*kgain1, (iTrans)*cpsmidinn(kpitch1),2,-1);+cpsmidinn(iNum))
 aout2 = oscili(0.5*kgain2, (iTrans)*cpsmidinn(kpitch2),3,-1);+cpsmidinn(iNum))
  aout3 = oscili(0.5*kgain3, (iTrans)*cpsmidinn(kpitch3),2,-1);+cpsmidinn(iNum))
   aout4 = oscili(0.8*kgain4, (iTrans)*cpsmidinn(kpitch4),3,-1);+cpsmidinn(iNum))
   
 aadsr1 madsr .01, 0.5, 0.8, 1.89
  aadsr2 madsr .02, 0.4, 0.9, 1.99
   aadsr3 madsr .001, 0.5, 0.8, 1.79
    aadsr4 madsr .0001, 0.4, 0.9, 3.09

aoutL = ((aout1*agate1*aadsr1) + (aout3*agate2*aadsr3))
aoutR = ((aout2*agate3*aadsr2) + (aout4*agate4*aadsr4))

 garvbL += aoutL * 0.5
 garvbR += aoutR * 0.5
 
; gadelL += aoutL * 0.4
; gadelR += aoutR * 0.4
 
 outs aoutL, aoutR
 endin


instr revsc
    denorm garvbL
    denorm garvbR
    aout1, aout2 reverbsc garvbL, garvbR, 0.8, 8000
    outs aout1, aout2
    clear garvbL
    clear garvbR
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
 <bgcolor mode="background">
  <r>240</r>
  <g>240</g>
  <b>240</b>
 </bgcolor>
</bsbPanel>
<bsbPresets>
</bsbPresets>
