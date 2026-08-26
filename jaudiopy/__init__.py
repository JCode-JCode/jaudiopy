from .buffer import AudioBuffer, mix_buffers, white_noise, SAMPLE_RATE, seed_random
from .synth import (
    sine_wave, square_wave, saw_wave, triangle_wave,
    note_freq, apply_envelope, envelope_adsr,
    resample_linear, change_speed, pitch_shift_semitones,
)
from . import drums
from .instruments import bass_synth, pluck_synth, pad_synth, piano_synth, bass_808_line
from .sequencer import StepSequencer, MelodySequencer, Song
from .mixer import Mixer, AdvancedMixer, Track, Bus
from . import effects
from .mastering import master_chain, master_chain_advanced, master_with_preset, MASTER_PRESETS
from .audio_io import (
    save_wav_mono, save_wav_stereo, load_wav, load_audio, resolve_source, is_url,
    save_wav_mono_bytes, save_wav_stereo_bytes,
)
from .audio_file import AudioFile
from .sampler import Sampler
from .vocal import vocal_chain, mix_vocal_with_beat
from .beatbuilder import BeatBuilder, list_styles, list_progressions, build_many
from .player import play_buffer, play_stereo
from .midi import load_midi_notes

__version__ = "1.0.0"
__author__ = "J Code"
__license__ = "Apache-2.0"

__all__ = [
    "AudioFile",
    "AudioBuffer", "mix_buffers", "white_noise", "SAMPLE_RATE", "seed_random",
    "sine_wave", "square_wave", "saw_wave", "triangle_wave", "note_freq",
    "apply_envelope", "envelope_adsr",
    "resample_linear", "change_speed", "pitch_shift_semitones",
    "drums", "bass_synth", "pluck_synth", "pad_synth", "piano_synth", "bass_808_line",
    "StepSequencer", "MelodySequencer", "Song", "Mixer", "AdvancedMixer", "Track", "Bus", "effects",
    "master_chain", "master_chain_advanced", "master_with_preset", "MASTER_PRESETS",
    "save_wav_mono", "save_wav_stereo", "load_wav", "load_audio", "resolve_source", "is_url",
    "save_wav_mono_bytes", "save_wav_stereo_bytes",
    "Sampler",
    "vocal_chain", "mix_vocal_with_beat",
    "BeatBuilder", "list_styles", "list_progressions", "build_many",
    "play_buffer", "play_stereo",
    "load_midi_notes",
]