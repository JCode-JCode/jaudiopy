from jaudiopy import drums, StepSequencer, MelodySequencer, AdvancedMixer, piano_synth, effects
from jaudiopy.audio_io import save_wav_stereo

BPM = 85

kit = drums.build_kit({
    "K": ("kick", "acoustic"),
    "S": ("snare", "tight"),
    "H": ("hihat", "closed"),
    "O": ("hihat", "open"),
})

drum_patterns = {
    "K": "X.......x..X....",
    "S": "....X.......X...",
    "H": "X.X.X.X.X.X.X.X.",
    "O": "..............X.",
}
drum_sounds = {
    "K": {"X": kit["K"], "x": (kit["K"], 0.6)},
    "S": {"X": kit["S"]},
    "H": {"X": kit["H"], "x": (kit["H"], 0.4)},
    "O": {"X": kit["O"]},
}

seq = StepSequencer(bpm=BPM, steps_per_beat=4)
drum_bus = seq.render_kit(drum_patterns, drum_sounds, bars=4, swing=0.05)

mel = MelodySequencer(bpm=BPM)
progression = [
    (["C", "D#", "G"], 3, 2, 0.55),
    (["G#", "C", "D#"], 2, 2, 0.55),
    (["A#", "D", "F"], 2, 2, 0.55),
    (["G", "A#", "D"], 2, 2, 0.55),
] * 2
piano_line = mel.render(progression, piano_synth)

effects.sidechain(piano_line, drum_patterns["K"] * 4, bpm=BPM, steps_per_beat=4, depth=0.55, release=0.18)

mixer = AdvancedMixer()
mixer.add_track("drums", drum_bus, volume=1.0)
mixer.add_track("piano", piano_line, volume=0.8)
mixer.add_bus("melodics", ["piano"], volume=1.0, pan=0.0)

left, right = mixer.render_and_master(preset="warm", add_saturation=0.15)

save_wav_stereo("melancholic_beat.wav", left, right)
print("saved: melancholic_beat.wav")
