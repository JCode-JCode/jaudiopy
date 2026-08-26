import math
from array import array
from typing import Optional, Dict, Union, List, Any
from .buffer import AudioBuffer, SAMPLE_RATE

try:
    import numpy as _np
except ImportError:
    _np = None

def sine_wave(freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 1.0, phase: float = 0.0) -> AudioBuffer:
    n = int(duration * sr)
    if _np is not None:
        t = _np.arange(n) / sr
        return AudioBuffer(array('d', (amp * _np.sin(2 * _np.pi * freq * t + phase)).tobytes()), sr)
    return AudioBuffer([amp * math.sin(2 * math.pi * freq * i / sr + phase) for i in range(n)], sr)

def square_wave(freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 1.0, duty: float = 0.5) -> AudioBuffer:
    if freq <= 0:
        raise ValueError("'freq' must be > 0")
    n = int(duration * sr)
    if _np is not None:
        t = _np.arange(n) / sr
        phase = (t * freq) % 1.0
        out = _np.where(phase < duty, amp, -amp)
        return AudioBuffer(array('d', out.tobytes()), sr)
    out = []
    for i in range(n):
        phase = (i / sr * freq) % 1.0
        out.append(amp if phase < duty else -amp)
    return AudioBuffer(out, sr)

def saw_wave(freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 1.0) -> AudioBuffer:
    if freq <= 0:
        raise ValueError("'freq' must be > 0")
    n = int(duration * sr)
    if _np is not None:
        t = _np.arange(n) / sr
        phase = (t * freq) % 1.0
        out = amp * (2 * phase - 1)
        return AudioBuffer(array('d', out.tobytes()), sr)
    return AudioBuffer([amp * (2 * ((i / sr * freq) % 1.0) - 1) for i in range(n)], sr)

def triangle_wave(freq: float, duration: float, sr: int = SAMPLE_RATE, amp: float = 1.0) -> AudioBuffer:
    if freq <= 0:
        raise ValueError("'freq' must be > 0")
    n = int(duration * sr)
    if _np is not None:
        t = _np.arange(n) / sr
        phase = (t * freq) % 1.0
        out = amp * (4 * _np.abs(phase - 0.5) - 1)
        return AudioBuffer(array('d', out.tobytes()), sr)
    out = []
    for i in range(n):
        phase = (i / sr * freq) % 1.0
        val = 4 * abs(phase - 0.5) - 1
        out.append(amp * val)
    return AudioBuffer(out, sr)

def envelope_adsr(n_samples: int, sr: int = SAMPLE_RATE, attack: float = 0.01, decay: float = 0.1,
                  sustain: float = 0.7, release: float = 0.2) -> Union[List[float], "_np.ndarray"]:
    a = int(attack * sr)
    d = int(decay * sr)
    r = int(release * sr)
    s = max(n_samples - a - d - r, 0)
    if _np is not None:
        parts = []
        if a > 0:
            parts.append(_np.linspace(0.0, 1.0, a, endpoint=False))
        if d > 0:
            parts.append(_np.linspace(1.0, sustain, d, endpoint=False))
        if s > 0:
            parts.append(_np.full(s, sustain, dtype=_np.float64))
        if r > 0:
            parts.append(_np.linspace(sustain, 0.0, r, endpoint=False))
        env_np = _np.concatenate(parts) if parts else _np.zeros(0, dtype=_np.float64)
        if len(env_np) < n_samples:
            env_np = _np.pad(env_np, (0, n_samples - len(env_np)))
        return env_np[:n_samples]
    env = []
    if a > 0:
        env += [i / a for i in range(a)]
    if d > 0:
        env += [1 - (1 - sustain) * i / d for i in range(d)]
    env += [sustain] * s
    if r > 0:
        env += [sustain * (1 - i / r) for i in range(r)]
    if len(env) < n_samples:
        env += [0.0] * (n_samples - len(env))
    return env[:n_samples]

def apply_envelope(buf: AudioBuffer, **kwargs) -> AudioBuffer:
    env = envelope_adsr(len(buf.samples), sr=buf.sr, **kwargs)
    if _np is not None and len(buf.samples) > 0:
        s_view = _np.frombuffer(buf.samples, dtype=_np.float64)
        buf.samples = array('d', (s_view * _np.asarray(env, dtype=_np.float64)).tobytes())
    else:
        buf.samples = array('d', (s * e for s, e in zip(buf.samples, env)))
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

def resample_linear(buf: AudioBuffer, ratio: float) -> AudioBuffer:
    if ratio <= 0:
        raise ValueError("'ratio' must be > 0")
    src = buf.samples
    n_src = len(src)
    n_out = max(int(n_src / ratio), 1)
    if _np is not None and n_src > 0:
        src_view = _np.frombuffer(src, dtype=_np.float64)
        out_idx = _np.arange(n_out) * ratio
        out_idx = _np.clip(out_idx, 0, n_src - 1)
        out = _np.interp(out_idx, _np.arange(n_src), src_view)
        return AudioBuffer(array('d', out.tobytes()), buf.sr)
    out = [0.0] * n_out
    for i in range(n_out):
        pos = i * ratio
        idx0 = int(pos)
        idx1 = min(idx0 + 1, n_src - 1)
        frac = pos - idx0
        if idx0 >= n_src:
            out[i] = 0.0
        else:
            out[i] = src[idx0] * (1 - frac) + src[idx1] * frac
    return AudioBuffer(out, buf.sr)

def change_speed(buf: AudioBuffer, factor: float) -> AudioBuffer:
    return resample_linear(buf, factor)

def pitch_shift_semitones(buf: AudioBuffer, semitones: float) -> AudioBuffer:
    ratio = 2 ** (semitones / 12)
    shifted = resample_linear(buf, ratio)
    n_target = len(buf.samples)
    if len(shifted.samples) >= n_target:
        shifted.samples = shifted.samples[:n_target]
    else:
        shifted.pad_to(n_target)
    return shifted