"""
=====================================================================================
 jaudiopy full library walkthrough + a 1-minute heavy, dark, bass-driven gangsta beat
=====================================================================================

This file does two things:
  1) Gives a compact tour of the jaudiopy library and the parameters of the
     functions it uses (plus a reference appendix for the parts it does NOT use).
  2) Builds a real, complete beat (hard kick/clap/hihat + a heavy distorted 808
     sub-bass + a dark minor drone + a sparse minor-key stab), mixes and masters
     it loud and heavy, and saves it as exactly 60 seconds of audio.

Install (per the library's own README):
    pip install jaudiopy
    pip install jaudiopy[numpy]     # optional faster numpy processing engine
    pip install jaudiopy[playback]  # optional, more reliable terminal playback
    pip install jaudiopy[formats]   # optional mp3/ogg/... support (needs ffmpeg too)

The core library has zero required dependencies (it only uses array/math/random/wave
from the Python standard library), so it works even with no pip install at all.

---------------------------------------------------------------------------------
Section 0: the library's two processing engines (and how to switch between them)
---------------------------------------------------------------------------------
Every piece of DSP in jaudiopy is implemented twice:
  - pure-python engine (default) -> jaudiopy/*.py
  - numpy engine (faster)        -> jaudiopy/np_backend/*.py
Both produce identical output; the numpy one is just faster for long renders.
If numpy/scipy happen to be installed, even the pure-python engine will silently
use them under the hood for speed - without changing behavior or output. So you
never *need* the numpy engine, it's purely a performance option.

There are two different ways this "engine choice" shows up, depending on what
part of the library you're using:

  A) High-level classes that take a `change_to_numpy` flag - AudioFile and
     BeatBuilder. Just pass the flag; the class internally imports the right
     backend for you:
         from jaudiopy import AudioFile, BeatBuilder
         song = AudioFile("song.wav", change_to_numpy=True)     # numpy engine
         bb   = BeatBuilder(style="trap_808", bpm=140, key="E",
                             change_to_numpy=True)               # numpy engine
     (change_to_numpy=False, the default, uses the pure-python engine.)

  B) Everything else - the plain function/class imports this script uses
     (drums, StepSequencer, MelodySequencer, AdvancedMixer, pad_synth,
     piano_synth, bass_808_line, effects, ...). These have NO flag; the engine
     is decided purely by WHICH MODULE PATH you import from:
         # pure-python engine (what this script uses):
         from jaudiopy import drums, StepSequencer, MelodySequencer, AdvancedMixer
         from jaudiopy import pad_synth, piano_synth, bass_808_line, effects

         # numpy engine - same names, different import path:
         from jaudiopy.np_backend import drums
         from jaudiopy.np_backend.sequencer import StepSequencer, MelodySequencer
         from jaudiopy.np_backend.mixer import AdvancedMixer
         from jaudiopy.np_backend.instruments import pad_synth, piano_synth, bass_808_line
         from jaudiopy.np_backend import effects
     `jaudiopy.audio_io` (save_wav_stereo/save_wav_mono/load_audio/...) is shared
     by both engines - you import it the same way no matter which engine you used
     to build the audio, and it auto-optimizes the write step when the buffer
     came from the numpy engine.
     Note: you cannot mix buffers from the two engines together (e.g. AdvancedMixer
     from the pure engine cannot mix a numpy-engine buffer) - pick one engine and
     import everything for that render from the matching module path.

This script sticks to the pure-python engine (plain `from jaudiopy import ...`),
same as the library's own example scripts in tests/.
"""

from jaudiopy import (
    drums,                # synthesized drum sound bank (kick/snare/hihat/perc...)
    StepSequencer,         # step-pattern (X/x/.) sequencer for drums
    MelodySequencer,       # note-list sequencer for melody/chords
    AdvancedMixer,          # multi-track mixer with bus (group) support
    pad_synth,               # pad synth (sustained tone - used here as a dark low drone)
    piano_synth,              # piano synth (used here for a sparse, moody minor stab)
    bass_808_line,             # 808 bass with pitch glide (signature trap/gangsta sound)
    effects,                    # effects collection (filters, distortion, sidechain, ...)
)
from jaudiopy.audio_io import save_wav_stereo    # write a stereo AudioBuffer pair to .wav
from jaudiopy.mastering import master_with_preset  # apply a mastering chain from a named preset

# =====================================================================================
# Section 1: overall track settings
# =====================================================================================
# This is the opposite of a soft/cute beat on purpose: a fast, hard trap tempo,
# a minor key, a heavily driven sub-808, and dense hihat rolls - the classic
# "dark gangsta trap" formula. Almost everything melodic is kept low, sparse and
# in the background; the drums and the 808 bass carry the whole track.
BPM = 140                # hard trap tempo - fast and aggressive
STEPS_PER_BEAT = 8       # each beat splits into 8 micro-steps -> room for dense hihat rolls
BARS = 4                 # loop length = 4 bars (we repeat this loop later to reach 1 minute)
SR = 44100               # standard sample rate (also the default in every function below)
LOOP_DURATION = (BARS * 4) * (60 / BPM)  # exact intended length of one loop, in seconds (4 beats
                                          # per bar x BARS bars). Every track gets trimmed to
                                          # exactly this length before mixing - see the note after
                                          # drum_bus is built, in Section 2, for why that matters.


# =====================================================================================
# Section 2: drums  ->  jaudiopy.drums  +  jaudiopy.StepSequencer
# =====================================================================================
# drums.list_sounds() lists every family and variant available, e.g.:
#   kick : default, 808, sub, acoustic, punchy
#   snare: default, tight, fat, electro, rimshot
#   hihat: closed, open, trap, pedal
#   clap : default, tight
#   tom  : low, mid, high
#   cymbal: ride, crash
#   perc : cowbell, shaker, tambourine, conga, clave, woodblock, triangle
#
# drums.build_kit(mapping) takes a dict {short_letter: (family, variant)} and
# returns a dict {short_letter: sound-making function}. Each function takes no
# arguments (build_kit already bakes in each sound's default parameters).
kit = drums.build_kit({
    "K": ("kick", "808"),        # 808 kick: deep and heavy - the backbone of the beat
    "C": ("clap", "tight"),      # sharp, hard-hitting clap for the backbeat
    "H": ("hihat", "trap"),      # trap-style hihat (short, ticky - built for dense rolls)
    "O": ("hihat", "open"),      # open hihat for accents/tension
    "R": ("snare", "rimshot"),   # rimshot: a sharp, woody accent used for extra grit
})

# Pattern language: each character is one micro-step.
#   "X"  = hit at high velocity (default 1.0)
#   "x"  = softer hit (default 0.6, but can be overridden - see drum_sounds below)
#   "."  = silence
# Pattern length must equal STEPS_PER_BEAT x number of beats; here 8 x 4 = 32
# characters for one bar. StepSequencer.render_kit repeats the one-bar pattern
# `bars` times for you (the bars= argument).
drum_patterns = {
    # Kick: a clear hit on 1, one syncopated ghost, a hit before 3, a hit on 4.
    # Deliberately NOT placed on the same steps as the hihat roll below - stacking
    # a kick onset directly under a fast hihat roll is exactly what produces an
    # ugly, crackly pile-up of transients (more on this below).
    "K": "X......." "....x..." "..X....." "X.......",
    # Clap: a single hard hit on beat 3 (the backbeat) - simple and clean
    "C": "........" "........" "X......." "........",
    # Hihat: steady 8th notes for 3 beats, with just ONE small 2-hit roll at the
    # very end of the bar (not a long 4-5 hit machine-gun roll). A short roll
    # like this still reads as "trap" without creating a wall of overlapping
    # noise-based hits.
    "H": "x.x.x.x." "x.x.x.x." "x.x.x.x." "x.x.xx.x",
    # Open hihat: a single accent near the end, clear of the small roll above
    "O": "........" "........" "........" ".......X",
    # Rimshot: one sparse accent, away from the kick/clap downbeats
    "R": "....x..." "........" "........" "........",
}

# Maps each pattern character to (sound function, velocity). If you only give a
# function, the velocity defaults based on the letter's case (X=1.0, x=0.6).
drum_sounds = {
    "K": {"X": kit["K"], "x": (kit["K"], 0.7)},   # ghost kick stays fairly firm, not soft
    "C": {"X": kit["C"]},
    "H": {"X": kit["H"], "x": (kit["H"], 0.35)},   # reined in further - see the drum_bus
                                                     # high_shelf note above and the mastering
                                                     # note in Section 8 for why hihats in
                                                     # particular need care on this engine
    "O": {"X": (kit["O"], 0.8)},
    "R": {"x": (kit["R"], 0.6)},
}

# StepSequencer(bpm, steps_per_beat, sr):
#   bpm            = tempo (beats per minute)
#   steps_per_beat = how many micro-steps per beat (higher = finer rolls possible)
#   sr             = output sample rate
seq = StepSequencer(bpm=BPM, steps_per_beat=STEPS_PER_BEAT, sr=SR)

# render_kit(patterns, sounds, bars, volumes=None, swing=0.0):
#   patterns = dict of one-bar patterns (defined above)
#   sounds   = letter -> sound/velocity mapping (defined above)
#   bars     = how many times to repeat the one-bar pattern (here BARS=4)
#   volumes  = optional per-channel volume applied before the final mixdown
#   swing    = swing amount (0 to 1); kept low here - a tight, driving, mechanical
#              feel suits a hard beat better than a loose, human swing
drum_bus = seq.render_kit(drum_patterns, drum_sounds, bars=BARS, swing=0.03)

# IMPORTANT - loop length: render_kit doesn't cut sounds off at the pattern
# boundary - a sound plays out its own natural length wherever it starts. The
# open hihat ("O") fires on the very last step of the pattern, and open hihats
# ring out for a while (~0.35-0.5s) - so drum_bus ends up noticeably LONGER
# than the "musical" 4-bar length (measured: ~7.33s here, versus the intended
# 6.857s). The melodic tracks below (drone/stab/bass) don't have that trailing
# hit, so they come out at the correct length - meaning drum_bus alone was
# quietly the longest track in the whole mix. Section 9 repeats this loop to
# build the full 60 seconds, and a length mismatch like that means the loop
# doesn't actually repeat where you think it does: everything drifts out of
# sync with the drums by that same ~0.47s on every single repeat, and the
# repeat() seam lands in the middle of what's now a lopsided, drums-only tail
# - which is exactly what produced the periodic click/crackle at a fixed,
# unexpected interval (~7.33s apart, not the intended 6.857s) in an earlier
# version of this script. The fix: trim every track to the SAME exact loop
# length before mixing, with a very short fade so the trim itself is inaudible.
drum_bus = drum_bus.slice(0.0, LOOP_DURATION)
drum_bus.fade_out(0.01)

# IMPORTANT - taming the hihat's inherent brightness: every hihat sound in this
# library (drums.hihat / hihat_trap / etc.) is built from "differentiated noise"
# (see drums._diff_noise: essentially y[n] = x[n] - x[n-1], a simple high-pass
# filter applied to white noise). That's a deliberate, bright, "ticky" noise
# source - but it also means the hihat is bright/sizzly by construction, before
# any mastering EQ even touches it. Combined with a fast trap tempo and a lot of
# hihat hits, that reads as an audible "sizzle/spark" character riding under the
# beat. effects.high_shelf(buf, freq, gain_db) with a NEGATIVE gain_db cuts
# (rather than boosts) everything above `freq` - used here to pull back some of
# that inherent brightness at the source, on the drum bus specifically, rather
# than trying to fix it later in the master (where it would affect the bass and
# pad too).
effects.high_shelf(drum_bus, freq=6000, gain_db=-4.0)

# IMPORTANT - gain staging: render_kit/mix_buffers just SUMS the drum channels
# with no normalization, so when several hits land on the same step the peak
# can go well above 1.0 (often 2x-3x). Always normalize a summed drum bus
# before doing anything else with it (compression, distortion, etc.) - this is
# standard "gain staging" practice in audio production.
drum_bus.normalize(0.9)

# NOTE ON distortion() and dense/noisy sources: an earlier version of this
# script ran effects.distortion() on the whole drum bus for extra grit. In
# practice that made things worse, not better: hihats and claps are already
# synthesized from filtered noise (broadband, not a clean single pitch), and
# jaudiopy's distortion/saturation are simple tanh waveshapers with no
# oversampling/anti-aliasing. Driving noise-heavy material through a waveshaper
# like that just piles on more uncontrolled high-frequency energy - it reads
# as "static/crackle", not "heavy". Lesson: save distortion/saturation for
# tonal, harmonic sources (like the 808 bass below), not for noise-based
# percussion. So this drum bus is left clean after normalizing - the weight
# comes from the kick/808 choice, the arrangement, and the mastering stage.


# =====================================================================================
# Section 3: dark low drone (background weight, not a "pretty" pad)  ->  MelodySequencer + pad_synth
# =====================================================================================
# MelodySequencer(bpm, sr) is a melody/chord sequencer that, instead of an X/x/.
# pattern, takes a list of events. Each event is a 4-tuple:
#   (note_or_list_of_notes_or_None, octave, length_in_beats, velocity)
# Passing a list of notes instead of a single note (i.e. a chord) plays them all
# at once. Passing None instead of a note is a rest of that same beat length.
mel = MelodySequencer(bpm=BPM, sr=SR)

# E minor -> C -> G -> D: a dark, minimal natural-minor progression. Kept in a
# LOW octave and played as a sustained drone rather than a bright chord stab -
# this is meant to sit under the beat as weight/tension, not as a melody you
# hum along to. Each chord lasts a full bar (4 beats), so 4 chords x 4 beats =
# 16 beats = exactly matches BARS=4 bars of drums above.
chords = [
    (["E", "G", "B"], 2, 4, 0.35),    # E minor
    (["C", "E", "G"], 2, 4, 0.32),    # C major
    (["G", "B", "D"], 2, 4, 0.32),    # G major
    (["D", "F#", "A"], 2, 4, 0.30),   # D major
]

# render(events, instrument_fn):
#   events        = the event list above
#   instrument_fn = a function with signature (note, octave, duration) returning
#                   an AudioBuffer; here we use pad_synth purely as a low drone -
#                   two slightly detuned sine waves + a 2000Hz lowpass + a slow
#                   attack envelope, kept quiet and heavily sidechained below.
drone_line = mel.render(chords, pad_synth)

# Trimmed to the same exact LOOP_DURATION as drum_bus above, for the same reason -
# MelodySequencer's own envelope release can leave this a fraction of a
# millisecond longer than the nominal length, and everything needs to line up
# exactly for the loop repeat in Section 9 to be seamless.
drone_line = drone_line.slice(0.0, LOOP_DURATION)
drone_line.fade_out(0.01)


# =====================================================================================
# Section 4: sparse minor stab (a moody accent, not a cheerful hook)  ->  MelodySequencer + piano_synth
# =====================================================================================
# Instead of a bright, busy arpeggio, this beat only plays the root note of each
# chord once per bar, as a single low, decaying piano hit - a common "dark trap"
# touch that adds a bit of melodic identity without making the beat sound playful.
# Built with a small loop so you can see MelodySequencer accepts any generated
# list of events, not just hand-written ones.
stab_events = []
for chord_notes, octave, chord_beats, base_vel in chords:
    root_note = chord_notes[0]                      # just the root, not the full chord
    stab_events.append((root_note, octave + 1, chord_beats, base_vel * 0.8))

stab_line = mel.render(stab_events, piano_synth)
stab_line = stab_line.slice(0.0, LOOP_DURATION)  # same loop-length trim as drone_line above
stab_line.fade_out(0.01)


# =====================================================================================
# Section 5: heavy distorted 808 sub-bass  ->  jaudiopy.bass_808_line
# =====================================================================================
# bass_808_line(notes, bpm, sr, glide_time, drive):
#   notes      = list of (note, octave, length_in_beats) - no velocity, since it
#                has its own attack/decay shaping built in
#   glide_time = time (seconds) for the pitch to slide between two consecutive
#                notes (instead of re-triggering the envelope) - the classic
#                trap 808 "glide/slide" effect. Slowed down here (0.18s) for a
#                heavier, more ominous slide.
#   drive      = amount of saturation/distortion on the bass; this one IS a
#                tonal/harmonic source (a single low sine tone), so it takes
#                distortion far better than the noise-based drums do - pushing
#                it (3.0, above the usual ~2.0-2.5) still gives a dirty, heavy
#                low end without the aliasing/noise problems distortion causes
#                on broadband sources. bass_808_line also normalizes its own
#                output internally, so it comes out already safely gain-staged.
bass_notes = [
    ("E", 0, 4),  # deep sub under the Em chord (octave 0 = as low as it gets)
    ("C", 0, 4),  # under the C chord
    ("G", 0, 4),  # under the G chord
    ("D", 0, 4),  # under the D chord
]
bass_line = bass_808_line(bass_notes, bpm=BPM, sr=SR, glide_time=0.18, drive=3.0)
bass_line = bass_line.slice(0.0, LOOP_DURATION)  # same loop-length trim as drone_line/stab_line
bass_line.fade_out(0.01)


# =====================================================================================
# Section 6: sidechain  ->  jaudiopy.effects.sidechain
# =====================================================================================
# sidechain(buf, pattern, bpm, steps_per_beat, depth, release):
#   buf            = the buffer we want to "duck" under something else (here: drone, stab, bass)
#   pattern        = an X/x/. pattern deciding when the trigger (e.g. the kick) fires
#   depth          = how much gain drops on each trigger (0 to 1; higher = more noticeable pump)
#   release        = how long gain takes to recover (seconds; smaller = snappier, harder pump)
# This mutates buf in place and returns it. Depths are pushed higher here than a
# soft beat would use - a hard, obvious pump under the kick is part of the
# "heavy gangsta bass" sound.
full_kick_pattern = drum_patterns["K"] * BARS  # kick pattern for the full 4 bars (not just one)

effects.sidechain(drone_line, full_kick_pattern, bpm=BPM, steps_per_beat=STEPS_PER_BEAT,
                   depth=0.5, release=0.18)
effects.sidechain(stab_line, full_kick_pattern, bpm=BPM, steps_per_beat=STEPS_PER_BEAT,
                   depth=0.4, release=0.16)
effects.sidechain(bass_line, full_kick_pattern, bpm=BPM, steps_per_beat=STEPS_PER_BEAT,
                   depth=0.5, release=0.14)   # a firm but not razor-fast pump - the trap/gangsta bounce


# =====================================================================================
# Section 7: mixing with buses  ->  jaudiopy.AdvancedMixer
# =====================================================================================
# AdvancedMixer inherits from the plain Mixer and adds the ability to group
# tracks onto a "bus"; a bus's volume/pan multiplies onto every track in it.
#
# add_track(name, buf, volume=1.0, pan=0.0):
#   volume = gain multiplier for this track (1.0 = unchanged)
#   pan    = stereo position from -1 (full left) to +1 (full right); 0 = center
mixer = AdvancedMixer(sr=SR)
mixer.add_track("drums", drum_bus, volume=0.9, pan=0.0)      # see the mastering note below on why
                                                                # this (and the hihat velocity above)
                                                                # got reined in slightly
mixer.add_track("drone", drone_line, volume=0.55, pan=0.0)   # kept low - background weight, not a lead
mixer.add_track("stab", stab_line, volume=0.4, pan=0.0)      # kept low - a moody accent, not a hook
mixer.add_track("bass808", bass_line, volume=1.1, pan=0.0)   # boosted - the bass should dominate

# add_bus(name, track_names, volume=1.0, pan=0.0):
#   track_names = the names of the tracks that belong to this bus (its volume/pan affects all of them)
mixer.add_bus("atmosphere", ["drone", "stab"], volume=1.0, pan=0.0)
mixer.add_bus("low_end", ["bass808"], volume=1.25, pan=0.0)  # extra push on the whole low-end bus

# render() just sums every track/bus with NO normalization (same as drum_bus earlier) -
# a mix this dense typically comes out with a peak around 1.2-1.3, i.e. already "too hot"
# before mastering EQ/compression even starts. Gain-staging this properly - normalizing to
# a safe peak BEFORE handing it to the mastering chain - matters just as much here as it
# did for the drum bus back in Section 2; skipping it means every EQ boost below compounds
# on top of an already-hot signal instead of a properly-staged one.
raw_left, raw_right = mixer.render()
raw_left.normalize(0.9)
raw_right.normalize(0.9)

# master_with_preset(left, right, preset, **overrides) - see Section 8 for the full
# parameter reference. Used directly here (instead of the render_and_master shortcut)
# specifically so the normalize() step above can happen in between rendering and
# mastering.
left, right = master_with_preset(
    raw_left, raw_right,
    preset="warm",           # NOT "loud_edm" - see the explanation in Section 8. "warm" applies
                               # zero boost to the high/"air" band, which matters a lot here: this
                               # engine's hihats/claps are synthesized from filtered noise, and
                               # boosting the air band (as "loud_edm" does by design) boosts
                               # exactly the frequency range that noise lives in, turning ordinary
                               # hihats into an audible hiss/crackle. "warm" also compresses far
                               # more gently (ratio 2.5 vs 6.0), which avoids the "zipper" artifact
                               # discussed in Section 8.
    bass_gain=5.0,            # override: push well past the preset's own 3.0 - this is where the
                               # "heavy" character actually comes from now, not from squashing the
                               # whole mix or brightening the highs.
    presence_gain=1.0,        # a little more than the preset default (0.5), for some clarity/edge
                               # without touching the air band at all.
    comp_threshold=-17,       # override: slightly GENTLER than the preset's own -18 (not more
                               # aggressive, as an earlier version of this script had it). A tighter
                               # threshold means the compressor's gain-reduction envelope reacts to
                               # more of the signal, more often - including ducking hard on every
                               # kick hit and then recovering over the release time (0.15s inside
                               # master_chain_advanced). Since kicks land every fraction of a
                               # second in a pattern this busy, that gain envelope is almost always
                               # riding up and down - which amplitude-modulates whatever's still
                               # ringing at the same time (the hihats) right along with it. That's
                               # heard as an audible "sizzle" pumping in time with the kick, not
                               # just steady brightness. Backing the threshold off a little reduces
                               # how often/hard that ducking happens, without giving up the loudness
                               # the compressor still provides on the parts of the mix that do cross
                               # the (slightly higher) threshold.
    target_crest_db=10,       # override: a bit tighter than the preset's gentle default of 12, for
                               # more loudness, while staying well clear of the aggressive 8.0 that
                               # caused problems - see Section 8.
    add_saturation=0.08,      # a light touch of saturation across the master - enough for glue,
                               # not enough to add grit on top of the already-driven 808.
)


# =====================================================================================
# Section 8: mastering  ->  jaudiopy.mastering
# =====================================================================================
# master_with_preset(left, right, preset, **overrides) starts from one of the
# ready-made MASTER_PRESETS and lets you override any of its parameters:
#   balanced / warm / bright / loud_edm / vocal_pop
# Each preset sets these parameters (which you can also call directly via
# master_chain_advanced):
#   bass_gain      = low-shelf boost amount (100Hz)
#   presence_gain  = boost/cut of the presence band (2500Hz)
#   air_gain       = high-shelf "air" boost (10kHz)
#   comp_threshold, comp_ratio = master compressor threshold and ratio
#   limiter_ceiling = final ceiling in dB before clipping
#   width          = stereo width (stereo_widener)
#   target_crest_db = target "crest factor" (peak vs. average difference); a
#                     smaller number means a louder/more compressed sound (less
#                     dynamic - "heavier"), a bigger one means more dynamic/natural.
#                     Passing None disables the whole loudness_maximizer stage.
#   haas_mix, haas_delay_ms = amount/delay of the Haas effect for more natural stereo width
#
# Extra parameters from master_chain_advanced (also reachable through master_with_preset):
#   add_exciter     = turn the harmonic exciter on/off
#   add_saturation  = amount of gentle saturation before compression
#
# WHY THIS SCRIPT USES "warm" INSTEAD OF THE OBVIOUS-LOOKING "loud_edm":
# earlier versions of this beat used "loud_edm" (the loudest/most compressed built-in
# preset) and came out sounding crackly/hissy/broken. Two real, measured causes, both
# tied to that specific preset:
#   1) loud_edm applies air_gain=+2.0 (a boost around 10kHz). jaudiopy's hihats/claps are
#      synthesized from filtered noise, not samples - boosting the air band boosts exactly
#      the frequency range that noise lives in, which is audible as hiss/crackle rather than
#      "brightness". "warm" uses air_gain=0.0, so this never happens.
#   2) loud_edm's loudness_maximizer chases a tight target_crest_db=8.0 by repeatedly pushing
#      gain up and hard-limiting (up to 8 passes). Because the drum/hihat sounds are built
#      from randomized noise (a fresh, unseeded draw every run - see drums._diff_noise), how
#      peaky the mix gets varies run to run, and chasing a tight target sometimes reacts hard
#      to a random unlucky peak, producing an audible "zipper" burst. Measured across many
#      renders, this was worse and far less consistent than gentler presets/targets.
# Combined with properly gain-staging the full mix before mastering (Section 7) and reining
# in the hihat level slightly (Section 2), switching to "warm" + a few targeted overrides
# (mainly bass_gain, pushed well above the preset default) gave a beat that measured clean
# AND came out louder (higher RMS) than the "loud_edm" attempts ever did - a good reminder
# that on a lightweight, non-oversampled synthesis engine like this one, "loudest preset" and
# "loudest-sounding result" are not the same thing.


# =====================================================================================
# Section 9: stretching the loop to exactly 1 minute  ->  AudioBuffer methods
# =====================================================================================
# What's in left/right right now is just one LOOP_DURATION-long loop (~6.86 seconds
# at 140 BPM). To reach 60 seconds we need to repeat it several times.
#
# AudioBuffer.repeat(times) does this the simple way - it multiplies the buffer's
# length by `times` by hard-concatenating copies of itself, with NO crossfade at the
# seams. Even with every track trimmed to an identical, exact LOOP_DURATION (Section
# 2-5), the waveform's actual value at the very last sample essentially never matches
# its value at the very first sample - because the audio content itself (drums, bass,
# pad) simply isn't guaranteed to end where it began. Concatenating two copies at a
# point like that is a real, audible discontinuity - a click - right at the seam.
# Confirmed by testing: even after fixing the length mismatch above, occasional runs
# still produced a small click exactly at the loop boundary (varies because the drum
# sounds are randomized noise - see the note in Section 8 - so exactly how far from
# zero the waveform sits at that exact sample differs run to run).
#
# The fix is a standard loop-crossfading technique: instead of a hard cut, overlap
# each copy with the next by a short crossfade window - fade the tail of one copy out
# while fading the head of the next one in, and mix (add) them together over that
# overlap instead of concatenating. AudioBuffer doesn't have a built-in "crossfade
# loop" helper, but it's straightforward to build from the pieces it does have:
#   AudioBuffer.silence(duration, sr) -> a new buffer of pure silence to build into
#   AudioBuffer.copy()                -> an independent copy (so fades don't affect the original)
#   AudioBuffer.fade_in(duration) / fade_out(duration) -> smooth fade in/out, in-place
#   AudioBuffer.mix(other, at=seconds) -> additively mixes another buffer in at a time offset
XFADE = 0.015  # 15ms crossfade - short enough to be inaudible as a "fade", long enough to
                # smooth over the seam completely
repeats_needed = int(60 // LOOP_DURATION) + 2    # enough repeats to safely pass 60 seconds
total_len = LOOP_DURATION * repeats_needed + 1.0  # extra second of headroom before the final trim

def _build_crossfaded_loop(single_loop):
    target = single_loop.__class__.silence(total_len, single_loop.sr)
    cursor = 0.0
    for i in range(repeats_needed):
        seg = single_loop.copy()
        if i > 0:
            seg.fade_in(XFADE)                 # blend in on top of the previous copy's fade-out
        if i < repeats_needed - 1:
            seg.fade_out(XFADE)                 # taper this copy's tail into the next copy's head
        target.mix(seg, at=cursor)
        cursor += LOOP_DURATION - XFADE          # each copy overlaps the previous one by XFADE
    return target

left = _build_crossfaded_loop(left)
right = _build_crossfaded_loop(right)

TARGET_SECONDS = 60.0
left = left.slice(0.0, TARGET_SECONDS)
right = right.slice(0.0, TARGET_SECONDS)

left.fade_in(0.02).fade_out(1.5)     # a very short fade-in (avoids a click), a punchier/shorter fade-out
right.fade_in(0.02).fade_out(1.5)


# =====================================================================================
# Section 10: saving the output  ->  jaudiopy.audio_io.save_wav_stereo
# =====================================================================================
# save_wav_stereo(path, left, right): writes two AudioBuffer channels to a single
# stereo .wav file (16-bit PCM). The mono equivalent is save_wav_mono(path, buf).
OUTPUT_PATH = "heavy_gangsta_bass_beat.wav"
save_wav_stereo(OUTPUT_PATH, left, right)
print(f"saved: {OUTPUT_PATH}  |  duration: {left.duration:.2f} seconds")


# =====================================================================================
# Appendix: other parts of the library not used in this beat, worth knowing about
# =====================================================================================
"""
- AudioFile: for working on an *existing* audio file (local path or web link),
  not building one from scratch:
      from jaudiopy import AudioFile
      song = AudioFile("song.wav")   # or an http link, plus change_to_numpy=True/False
      song.play()                                          # blocking playback in the terminal
      song.add_effect("reverb", "out.wav", room_size=0.7, mix=0.4)
      song.change_speed(1.25, "out.wav", play_after=True)  # speed changes, pitch follows it
      song.tune(-2, "out.wav")                              # pitch shift without changing length (semitones)
      song.master("out.wav", preset="loud_edm")
      # other methods: trim, normalize, reverse, fade, gain_db, repeat, mix_with, to_mono
      # all of them take save_as (required) + play_after (default False)
      # song.repeat_before_save / song.repeat_after_save: repeat before/after saving (default 0)

- Sampler: for playing/re-pitching your own audio file instead of the built-in synths:
      from jaudiopy import Sampler
      sampler = Sampler.load("piano_note_C4.wav", base_note="C", base_octave=4)
      note = sampler.play_note("D#", 4, duration=1.0)  # re-pitched to D#4
      loop = sampler.loop_to(8.0)                      # looped to 8 seconds
      chops = sampler.chop(8)                          # split into 8 equal slices

- BeatBuilder: the fastest way to get a full, ready-made beat with one built-in style:
      from jaudiopy import BeatBuilder, list_styles, list_progressions
      bb = BeatBuilder(style="trap_808", bpm=140, key="E", change_to_numpy=True)
      left, right = bb.build(duration_minutes=2.5)
      bb.build_and_save("my_beat.wav", duration_minutes=2.5)
      # styles: trap_808, boombap_piano, street_hiphop, dark_melodic, minimal_bounce
      #   ("dark_melodic" is the closest built-in style to this hand-built beat:
      #    sub kick + piano + glide bass, moody and wide, "warm" master)
      # ready-made chord progressions: minor_epic, minor_simple, trap_dark, single_root

- vocal_chain / mix_vocal_with_beat: for putting a vocal (e.g. a rap verse) over a beat:
      from jaudiopy import mix_vocal_with_beat, load_audio
      vocal = load_audio("my_vocal.wav")
      final_l, final_r = mix_vocal_with_beat(beat_left, beat_right, vocal, vocal_at=8.0)
      # vocal_chain applies a ready-made EQ/de-ess/compress/limit chain to the vocal
      # vocal_at = how many seconds into the beat the vocal starts (the beat ducks under it from there)

- effects (the full list of functions, all operating on an AudioBuffer):
      lowpass, highpass          (freq, q)         low/high-pass filter
      eq_band, low_shelf, high_shelf (freq, gain_db, q)  parametric EQ/shelf
      distortion (drive, mix)   bitcrush (bit_depth, downsample)
      delay (time_sec, feedback, mix)   reverb (room_size, damping, mix)
      chorus, vibrato (rate, depth_ms, mix?)   tremolo (rate, depth)
      phaser (rate, depth, stages, mix)   wah (rate, min_freq, max_freq, ...)
      saturation (amount, mix)   exciter (freq, amount)
      compressor (threshold_db, ratio, attack, release, makeup_db)
      limiter (ceiling_db, release)   noise_gate (threshold_db, attack, release)
      pan_stereo (pan)   autopan (rate, depth)   stereo_widener (width)
      haas_widen (delay_ms, mix)   loudness_maximizer (target_crest_db, ceiling_db)
      sidechain (pattern, bpm, steps_per_beat, depth, release)  <- fully explained above
      de_esser (freq, threshold_db, ...)   duck_under (trigger, depth, ...)

- Song: for stitching several sections (intro/verse/chorus...) back to back:
      from jaudiopy import Song
      song = Song(sr=44100)
      song.add_section(intro_buf).add_section(verse_buf)
      full = song.render()

- MIDI: load_midi_notes(path) converts a MIDI file into a note list you can pass
  straight to MelodySequencer.render or bass_808_line.

- Raw waveform functions (jaudiopy.synth): sine_wave, square_wave, saw_wave,
  triangle_wave (freq, duration, sr, amp) + apply_envelope/envelope_adsr
  (attack, decay, sustain, release) for building a fully custom synth from scratch.

- Single drum hits (jaudiopy.drums) if you want to make sounds directly without StepSequencer:
      drums.kick(duration, start_freq, end_freq, sr) / kick_808 / kick_sub / ...
      drums.snare(...) / drums.hihat(duration, closed, sr) / drums.crash(...) etc.
      drums.get_sound(family, variant="default") returns a single sound-making function
"""
