; ==============================================================================
; BCI Utilities Include File
; Common functions and instruments for BCI processing
; ==============================================================================

; ==============================================================================
; OSC Listener with Fallback Macros for different channel counts
; ==============================================================================
#define OSC_LISTEN_8CH(handle'path'v1'v2'v3'v4'v5'v6'v7'v8) #
    kk OSClisten $handle, $path, "ffffffff", $v1, $v2, $v3, $v4, $v5, $v6, $v7, $v8
    if kk == 0 then
        $v1 = $v1 * 0.9 + 0.5 * 0.1
        $v2 = $v2 * 0.9 + 0.5 * 0.1
        $v3 = $v3 * 0.9 + 0.5 * 0.1
        $v4 = $v4 * 0.9 + 0.5 * 0.1
        $v5 = $v5 * 0.9 + 0.5 * 0.1
        $v6 = $v6 * 0.9 + 0.5 * 0.1
        $v7 = $v7 * 0.9 + 0.5 * 0.1
        $v8 = $v8 * 0.9 + 0.5 * 0.1
    endif
#

; 16-channel OSC listener macro for band power data (5 values per channel)
#define OSC_LISTEN_5CH_BANDPOWER(handle'path'v1'v2'v3'v4'v5) #
    kk OSClisten $handle, $path, "fffff", $v1, $v2, $v3, $v4, $v5
    if kk == 0 then
        $v1 = $v1 * 0.95 + 0.5 * 0.05
        $v2 = $v2 * 0.95 + 0.5 * 0.05
        $v3 = $v3 * 0.95 + 0.5 * 0.05
        $v4 = $v4 * 0.95 + 0.5 * 0.05
        $v5 = $v5 * 0.95 + 0.5 * 0.05
    endif
#

; ==============================================================================
; Value Scaling Macro - scales BCI values to useful ranges
; ==============================================================================
#define SCALE_BCI_VALUE(input'min'max) #
    ($input * ($max - $min) + $min)
#

; ==============================================================================
; 16-Channel BCI Global Variable Initialization
; ==============================================================================
#define INIT_16CH_BCI_GLOBALS(prefix) #
    g$prefix.1 init 0.5
    g$prefix.2 init 0.5
    g$prefix.3 init 0.5
    g$prefix.4 init 0.5
    g$prefix.5 init 0.5
    g$prefix.6 init 0.5
    g$prefix.7 init 0.5
    g$prefix.8 init 0.5
    g$prefix.9 init 0.5
    g$prefix.10 init 0.5
    g$prefix.11 init 0.5
    g$prefix.12 init 0.5
    g$prefix.13 init 0.5
    g$prefix.14 init 0.5
    g$prefix.15 init 0.5
    g$prefix.16 init 0.5
#

; ==============================================================================
; Channel Averaging Macro - average multiple channels into fewer outputs
; ==============================================================================
#define AVERAGE_CHANNELS_4TO2(in1'in2'in3'in4'out1'out2) #
    $out1 = ($in1 + $in2) * 0.5
    $out2 = ($in3 + $in4) * 0.5
#

#define AVERAGE_CHANNELS_16TO4(in1'in2'in3'in4'in5'in6'in7'in8'in9'in10'in11'in12'in13'in14'in15'in16'out1'out2'out3'out4) #
    $out1 = ($in1 + $in2 + $in3 + $in4) * 0.25
    $out2 = ($in5 + $in6 + $in7 + $in8) * 0.25
    $out3 = ($in9 + $in10 + $in11 + $in12) * 0.25
    $out4 = ($in13 + $in14 + $in15 + $in16) * 0.25
#

; ==============================================================================
; Common Waveform Tables
; ==============================================================================
opcode InitWaveforms, 0, 0
    gisine   ftgen 1, 0, 16384, 10, 1
    gisquare ftgen 2, 0, 16384, 10, 1, 0, .33, 0, .2, 0, .14, 0, .11, 0, .09
    gisaw    ftgen 3, 0, 16384, 10, 1, .2, 0, .4, 0, .6, 0, .8, 0, 1, 0, .8, 0, .6, 0, .4, 0, .2
    gifour   ftgen 4, 0, 16384, 10, 1, .5, .3, .2, .1, .06, .04, .028, .010
    gifive   ftgen 5, 0, 16384, 10, 1, 0, 0, .6, 0, 0, .5, 0, 0, .4, 0, 0, .3, 0, 0, .2, 0, 0, .1
endop

; ==============================================================================
; Common Scale Tables
; ==============================================================================
opcode InitScales, 0, 0
    giscale1 ftgen 111, 0, 8, -2, 72, 74, 76, 79, 81, 83, 72, 84
    giscale2 ftgen 112, 0, 8, -2, 52, 50, 48, 65, 60, 52, 48, 60
    giscale3 ftgen 113, 0, 8, -2, 60, 62, 69, 67, 71, 74, 72, 48
    giscale4 ftgen 114, 0, 8, -2, 48, 43, 50, 48, 43, 55, 60, 43
endop

; ==============================================================================
; Reverb Send Globals
; ==============================================================================
opcode InitReverbGlobals, 0, 0
    garvbL init 0
    garvbR init 0
endop

; ==============================================================================
; Delay Send Globals
; ==============================================================================
opcode InitDelayGlobals, 0, 0
    gadelL init 0
    gadelR init 0
endop

; ==============================================================================
; Standard Reverb Instrument
; ==============================================================================
instr Reverb
    denorm garvbL
    denorm garvbR
    kfeedback chnget "reverb_feedback"
    kcutoff chnget "reverb_cutoff"
    
    ; Default values if channels not set
    if kfeedback == 0 then
        kfeedback = 0.8
    endif
    if kcutoff == 0 then
        kcutoff = 8000
    endif
    
    aout1, aout2 reverbsc garvbL, garvbR, kfeedback, kcutoff
    outs aout1, aout2
    clear garvbL
    clear garvbR
endin

; ==============================================================================
; Standard Delay Instrument
; ==============================================================================
instr Delay
    adelL init 0
    adelR init 0
    denorm gadelL
    denorm gadelR
    
    kfeedbackL chnget "delay_feedbackL"
    kfeedbackR chnget "delay_feedbackR"
    ktime chnget "delay_time"
    kcutoffL chnget "delay_cutoffL"
    kcutoffR chnget "delay_cutoffR"
    kresonance chnget "delay_resonance"
    
    ; Default values
    if kfeedbackL == 0 then
        kfeedbackL = 0.72
    endif
    if kfeedbackR == 0 then
        kfeedbackR = 0.7
    endif
    if ktime == 0 then
        ktime = 2
    endif
    if kcutoffL == 0 then
        kcutoffL = 2000
    endif
    if kcutoffR == 0 then
        kcutoffR = 2000
    endif
    if kresonance == 0 then
        kresonance = 0.5
    endif
    
    adelL delay gadelL + (adelL * kfeedbackL), ktime
    adelR delay gadelR + (adelR * kfeedbackR), ktime
    adelOutL moogvcf adelL, kcutoffL, kresonance
    adelOutR moogvcf adelR, kcutoffR, kresonance * 1.2
    
    outs adelOutL, adelOutR
    clear gadelL
    clear gadelR
endin

; ==============================================================================
; BCI Connection Monitor
; ==============================================================================
instr BCIMonitor
    ; Check for timeout
    ktimer timeinsts
    kconnected init 0
    
    ; Simple connection check (you can expand this)
    if ktimer > $TIMEOUT_SECONDS && kconnected == 0 then
        prints "Warning: No BCI data received for %d seconds\n", $TIMEOUT_SECONDS
    endif
endin

; ==============================================================================
; MIDI Controller Mapping Macro
; ==============================================================================
#define MIDI_CONTROL(cc'min'max'default) #
    (midic7:k($cc, $min, $max) == 0 ? $default : midic7:k($cc, $min, $max))
#