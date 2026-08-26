import math
from array import array
from typing import List, Tuple, Any, Union
from .buffer import AudioBuffer, white_noise, SAMPLE_RATE
from .synth import saw_wave, sine_wave, square_wave, apply_envelope, note_freq
from .effects import lowpass

try:
    import numpy as _np
except ImportError:
    _np = None

def bass_synth(note: str, octave: int, duration: float, sr: int = 44100) -> AudioBuffer:
    freq = note_freq(note, octave)
    buf = saw_wave(freq, duration, sr, amp=0.8)
    buf.mix(sine_wave(freq / 2, duration, sr, amp=0.4), at=0.0)
    lowpass(buf, freq=800, q=0.9)
    apply_envelope(buf, attack=0.005, decay=0.1, sustain=0.7, release=0.1)
    buf.normalize(0.8)
    return buf

def pluck_synth(note: str, octave: int, duration: float, sr: int = 44100) -> AudioBuffer:
    freq = note_freq(note, octave)
    buf = square_wave(freq, duration, sr, amp=0.5)
    lowpass(buf, freq=3000, q=0.8)
    apply_envelope(buf, attack=0.002, decay=duration * 0.5, sustain=0.0, release=duration * 0.3)
    buf.normalize(0.8)
    return buf

def pad_synth(note: str, octave: int, duration: float, sr: int = 44100) -> AudioBuffer:
    freq = note_freq(note, octave)
    buf = sine_wave(freq, duration, sr, amp=0.5)
    buf.mix(sine_wave(freq * 1.005, duration, sr, amp=0.5), at=0.0)
    lowpass(buf, freq=2000, q=0.7)
    apply_envelope(buf, attack=0.3, decay=0.2, sustain=0.8, release=0.5)
    buf.normalize(0.7)
    return buf

PIANO_HARMONICS: List[Tuple[float, float, float]] = [
    (1, 1.00, 0.9), (2, 0.55, 1.6), (3, 0.35, 2.4), (4, 0.22, 3.4),
    (5, 0.14, 4.6), (6, 0.09, 6.0), (7, 0.05, 7.5), (8, 0.03, 9.0),
]

def piano_synth(note: str, octave: int, duration: float, sr: int = SAMPLE_RATE,
                velocity: float = 1.0) -> AudioBuffer:
    freq = note_freq(note, octave)
    n = int(duration * sr)
    if _np is not None and n > 0:
        t = _np.arange(n) / sr
        out = _np.zeros(n, dtype=_np.float64)
        for h, amp, decay_rate in PIANO_HARMONICS:
            hfreq = freq * h * (1 + 0.0004 * h * h)
            gain = amp * velocity
            out += gain * _np.exp(-decay_rate * t) * _np.sin(2 * _np.pi * hfreq * t)
        buf = AudioBuffer(array('d', out.tobytes()), sr)
    else:
        out = array('d', bytes(8 * n))
        for h, amp, decay_rate in PIANO_HARMONICS:
            hfreq = freq * h * (1 + 0.0004 * h * h)
            gain = amp * velocity
            dtheta = 2 * math.pi * hfreq / sr
            c_inc = math.cos(dtheta)
            s_inc = math.sin(dtheta)
            decay_per_sample = math.exp(-decay_rate / sr)
            cur_c, cur_s = 1.0, 0.0
            env = gain
            for i in range(n):
                out[i] += env * cur_s
                env *= decay_per_sample
                cur_c, cur_s = cur_c * c_inc - cur_s * s_inc, cur_s * c_inc + cur_c * s_inc
                if i & 1023 == 0 and i > 0:
                    norm = math.sqrt(cur_c * cur_c + cur_s * cur_s)
                    cur_c /= norm
                    cur_s /= norm
        buf = AudioBuffer(out, sr)
    buf.mix(white_noise(0.006, sr, amp=0.15 * velocity), at=0.0)
    buf.normalize(0.85)
    return buf

def _freq_curve(notes: List[Tuple[str, int, float]], bpm: float, sr: int,
                glide_time: float) -> Tuple[Union[List[float], "_np.ndarray"], int]:
    freqs = [note_freq(n, o) for n, o, _ in notes]
    durations = [b * 60 / bpm for _, _, b in notes]
    total = sum(durations)
    n_total = int(total * sr)
    cum = 0.0
    if _np is not None and n_total > 0:
        curve_np = _np.zeros(n_total, dtype=_np.float64)
        for i, (f, d) in enumerate(zip(freqs, durations)):
            seg_start = int(cum * sr)
            seg_end = min(int((cum + d) * sr), n_total)
            prev_f = freqs[i - 1] if i > 0 else f
            glide_samples = min(int(glide_time * sr), max(seg_end - seg_start, 0))
            if glide_samples > 0:
                ramp = _np.linspace(prev_f, f, glide_samples, endpoint=False)
                curve_np[seg_start:seg_start + glide_samples] = ramp
            curve_np[seg_start + glide_samples:seg_end] = f
            cum += d
        return curve_np, n_total
    curve = [0.0] * n_total
    for i, (f, d) in enumerate(zip(freqs, durations)):
        seg_start = int(cum * sr)
        seg_end = int((cum + d) * sr)
        prev_f = freqs[i - 1] if i > 0 else f
        glide_samples = min(int(glide_time * sr), max(seg_end - seg_start, 0))
        for s in range(seg_start, min(seg_end, n_total)):
            local = s - seg_start
            if glide_samples > 0 and local < glide_samples:
                t = local / glide_samples
                curve[s] = prev_f + (f - prev_f) * t
            else:
                curve[s] = f
        cum += d
    return curve, n_total

def bass_808_line(notes: List[Tuple[str, int, float]], bpm: float, sr: int = SAMPLE_RATE,
                  glide_time: float = 0.12, drive: float = 3.0) -> AudioBuffer:
    curve, n_total = _freq_curve(notes, bpm, sr, glide_time)
    if _np is not None and n_total > 0:
        curve_arr = _np.asarray(curve, dtype=_np.float64)
        phase_inc = 2 * _np.pi * curve_arr / sr
        phase = _np.cumsum(phase_inc) - phase_inc
        wave = _np.tanh(drive * _np.sin(phase))
        buf = AudioBuffer(array('d', wave.tobytes()), sr)
    else:
        out = array('d', [0.0] * n_total)
        phase = 0.0
        for i in range(n_total):
            out[i] = math.tanh(drive * math.sin(phase))
            phase += 2 * math.pi * curve[i] / sr
        buf = AudioBuffer(out, sr)
    apply_envelope(buf, attack=0.005, decay=0.05, sustain=0.9, release=0.08)
    buf.normalize(0.9)
    return buf