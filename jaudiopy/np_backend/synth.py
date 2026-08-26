import numpy as np
from typing import Optional, Dict, List, Union
from .buffer import NpAudioBuffer, SAMPLE_RATE

def sine_wave(freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 1.0,
              phase: float = 0.0) -> NpAudioBuffer:
    n = int(duration * sr)
    t = np.arange(n) / sr
    return NpAudioBuffer(amp * np.sin(2 * np.pi * freq * t + phase), sr)

def square_wave(freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 1.0,
                duty: float = 0.5) -> NpAudioBuffer:
    if freq <= 0:
        raise ValueError("'freq' must be > 0")
    n = int(duration * sr)
    t = np.arange(n) / sr
    phase = (t * freq) % 1.0
    return NpAudioBuffer(np.where(phase < duty, amp, -amp), sr)

def saw_wave(freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 1.0) -> NpAudioBuffer:
    if freq <= 0:
        raise ValueError("'freq' must be > 0")
    n = int(duration * sr)
    t = np.arange(n) / sr
    phase = (t * freq) % 1.0
    return NpAudioBuffer(amp * (2 * phase - 1), sr)

def triangle_wave(freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 1.0) -> NpAudioBuffer:
    if freq <= 0:
        raise ValueError("'freq' must be > 0")
    n = int(duration * sr)
    t = np.arange(n) / sr
    phase = (t * freq) % 1.0
    return NpAudioBuffer(amp * (4 * np.abs(phase - 0.5) - 1), sr)

def envelope_adsr(n_samples: int, sr: int = SAMPLE_RATE, attack: float = 0.01, decay: float = 0.1,
                  sustain: float = 0.7, release: float = 0.2) -> np.ndarray:
    a = int(attack * sr)
    d = int(decay * sr)
    r = int(release * sr)
    s = max(n_samples - a - d - r, 0)
    parts = []
    if a > 0:
        parts.append(np.linspace(0.0, 1.0, a, endpoint=False))
    if d > 0:
        parts.append(np.linspace(1.0, sustain, d, endpoint=False))
    parts.append(np.full(s, sustain))
    if r > 0:
        parts.append(np.linspace(sustain, 0.0, r, endpoint=False))
    env = np.concatenate(parts) if parts else np.zeros(0)
    if len(env) < n_samples:
        env = np.pad(env, (0, n_samples - len(env)))
    return env[:n_samples]

def apply_envelope(buf: NpAudioBuffer, **kwargs) -> NpAudioBuffer:
    env = envelope_adsr(len(buf.samples), sr=buf.sr, **kwargs)
    buf.samples = buf.samples * env
    return buf

NOTE_FREQS: Dict[str, float] = {
    "C": 16.35, "C#": 17.32, "D": 18.35, "D#": 19.45, "E": 20.60,
    "F": 21.83, "F#": 23.12, "G": 24.50, "G#": 25.96, "A": 27.50,
    "A#": 29.14, "B": 30.87,
}

def note_freq(note: str, octave: int = 4) -> float:
    if note not in NOTE_FREQS:
        raise ValueError(f"unknown note '{note}'. valid notes: {list(NOTE_FREQS)}")
    return NOTE_FREQS[note] * (2 ** octave)

def resample_linear(buf: NpAudioBuffer, ratio: float) -> NpAudioBuffer:
    if ratio <= 0:
        raise ValueError("'ratio' must be > 0")
    n_src = len(buf.samples)
    n_out = max(int(n_src / ratio), 1)
    if n_src == 0:
        return NpAudioBuffer(np.zeros(n_out, dtype=np.float64), buf.sr)
    src_idx = np.arange(n_src)
    out_idx = np.clip(np.arange(n_out) * ratio, 0, n_src - 1)
    out = np.interp(out_idx, src_idx, buf.samples, left=0.0, right=0.0)
    return NpAudioBuffer(out, buf.sr)

def change_speed(buf: NpAudioBuffer, factor: float) -> NpAudioBuffer:
    return resample_linear(buf, factor)

def pitch_shift_semitones(buf: NpAudioBuffer, semitones: float) -> NpAudioBuffer:
    ratio = 2 ** (semitones / 12)
    shifted = resample_linear(buf, ratio)
    n_target = len(buf.samples)
    if len(shifted.samples) >= n_target:
        shifted.samples = shifted.samples[:n_target]
    else:
        shifted.pad_to(n_target)
    return shifted