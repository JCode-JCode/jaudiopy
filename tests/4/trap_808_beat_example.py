from jaudiopy import drums, StepSequencer, AdvancedMixer, bass_808_line, effects
from jaudiopy.audio_io import save_wav_stereo

BPM = 140

kit = drums.build_kit({
    "K": ("kick", "808"),
    "C": ("clap", "default"),
    "H": ("hihat", "trap"),
    "O": ("hihat", "open"),
})

kick_pattern = "X.......x.......X.......x......"
patterns = {
    "C": "................X...............",
    "H": "X.XxX.XxXXXxX.XxX.XxX.XxXXXxX.Xx",
    "O": "................................X",
}
sounds = {
    "C": {"X": kit["C"]},
    "H": {"X": kit["H"], "x": (kit["H"], 0.4)},
    "O": {"X": kit["O"]},
}

seq = StepSequencer(bpm=BPM, steps_per_beat=8)

kick_track = seq.render_pattern(kick_pattern * 2, {"X": kit["K"], "x": (kit["K"], 0.55)})
drum_bus = seq.render_kit(patterns, sounds, bars=2)
drum_bus.mix(kick_track, at=0.0)

bass_notes = [
    ("E", 1, 3), ("E", 1, 1), ("G", 1, 2), ("E", 1, 2),
    ("D", 1, 3), ("D", 1, 1), ("C", 1, 2), ("D", 1, 2),
] * 2
bass_line = bass_808_line(bass_notes, bpm=BPM, glide_time=0.09, drive=3.5)

full_kick_pattern = kick_pattern * 2
effects.sidechain(bass_line, full_kick_pattern, bpm=BPM, steps_per_beat=8, depth=0.5, release=0.12)

mixer = AdvancedMixer()
mixer.add_track("drums", drum_bus, volume=1.0)
mixer.add_track("bass808", bass_line, volume=0.95)
mixer.add_bus("low_end", ["bass808"], volume=1.15, pan=0.0)

left, right = mixer.render_and_master(preset="loud_edm", bass_gain=3.5, add_saturation=0.1)

save_wav_stereo("trap_808_beat.wav", left, right)
print("saved: trap_808_beat.wav")
