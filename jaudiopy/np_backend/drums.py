import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
from .buffer import NpAudioBuffer, white_noise, SAMPLE_RATE
from .synth import apply_envelope, sine_wave

def _diff_noise(duration: float, sr: int) -> NpAudioBuffer:
    buf = white_noise(duration, sr)
    diffed = np.diff(buf.samples, prepend=0.0)
    buf.samples = diffed
    return buf

def kick(duration: float = 0.4, start_freq: float = 150, end_freq: float = 45,
         sr: int = SAMPLE_RATE, click: bool = True, drive: float = 1.0) -> NpAudioBuffer:
    n = int(duration * sr)
    t = np.arange(n) / n
    freq = start_freq + (end_freq - start_freq) * t
    phase_inc = 2 * np.pi * freq / sr
    phase = np.cumsum(phase_inc) - phase_inc
    buf = NpAudioBuffer(np.tanh(drive * np.sin(phase)), sr)
    apply_envelope(buf, attack=0.001, decay=duration * 0.6, sustain=0.0, release=duration * 0.3)
    if click:
        buf.mix(white_noise(0.004, sr, amp=0.3), at=0.0)
    buf.normalize()
    return buf

def kick_808(duration: float = 0.9, start_freq: float = 120, end_freq: float = 35,
             sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    return kick(duration=duration, start_freq=start_freq, end_freq=end_freq, sr=sr, click=False, drive=1.4)

def kick_sub(duration: float = 0.6, freq: float = 45, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    buf = sine_wave(freq, duration, sr, amp=0.9)
    apply_envelope(buf, attack=0.002, decay=duration * 0.5, sustain=0.2, release=duration * 0.4)
    buf.normalize()
    return buf

def kick_acoustic(duration: float = 0.35, start_freq: float = 180, end_freq: float = 60,
                  sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    return kick(duration=duration, start_freq=start_freq, end_freq=end_freq, sr=sr, click=True, drive=1.0)

def kick_punchy(duration: float = 0.25, start_freq: float = 200, end_freq: float = 55,
                sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    return kick(duration=duration, start_freq=start_freq, end_freq=end_freq, sr=sr, click=True, drive=1.8)

def snare(duration: float = 0.2, sr: int = SAMPLE_RATE, tone_freq: float = 180,
          noise_amt: float = 1.0) -> NpAudioBuffer:
    n = int(duration * sr)
    t = np.arange(n) / sr
    tone = NpAudioBuffer(0.5 * np.sin(2 * np.pi * tone_freq * t), sr)
    tone.mix(white_noise(duration, sr, amp=noise_amt), at=0.0)
    apply_envelope(tone, attack=0.001, decay=duration * 0.5, sustain=0.05, release=duration * 0.3)
    tone.normalize()
    return tone

def snare_tight(duration: float = 0.12, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    return snare(duration=duration, sr=sr, tone_freq=220, noise_amt=0.8)

def snare_fat(duration: float = 0.28, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    return snare(duration=duration, sr=sr, tone_freq=150, noise_amt=1.1)

def snare_electro(duration: float = 0.18, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    n = int(duration * sr)
    buf = snare(duration=duration, sr=sr, tone_freq=200, noise_amt=0.9)
    period = int(sr / 90) or 1
    duty = int(sr / 180) or 1
    idx = np.arange(n)
    square = np.where((idx % period) < duty, 0.25, -0.25)
    buf.mix(NpAudioBuffer(square, sr), at=0.0)
    buf.normalize()
    return buf

def rimshot(duration: float = 0.1, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    n = int(duration * sr)
    t = np.arange(n) / sr
    tone = NpAudioBuffer(0.6 * np.sin(2 * np.pi * 400 * t), sr)
    tone.mix(white_noise(duration, sr, amp=0.4), at=0.0)
    apply_envelope(tone, attack=0.001, decay=duration * 0.3, sustain=0.0, release=duration * 0.1)
    tone.normalize()
    return tone

def hihat(duration: float = 0.08, closed: bool = True, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    buf = _diff_noise(duration, sr)
    decay = duration * (0.3 if closed else 0.9)
    apply_envelope(buf, attack=0.001, decay=decay, sustain=0.0, release=duration * 0.2)
    buf.normalize()
    return buf

def hihat_closed(duration: float = 0.06, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    return hihat(duration=duration, closed=True, sr=sr)

def hihat_open(duration: float = 0.35, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    return hihat(duration=duration, closed=False, sr=sr)

def hihat_trap(duration: float = 0.05, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    buf = _diff_noise(duration, sr)
    apply_envelope(buf, attack=0.0005, decay=duration * 0.2, sustain=0.0, release=duration * 0.1)
    buf.normalize()
    return buf

def hihat_pedal(duration: float = 0.1, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    buf = _diff_noise(duration, sr)
    apply_envelope(buf, attack=0.005, decay=duration * 0.5, sustain=0.0, release=duration * 0.4)
    buf.gain(0.6)
    buf.normalize(0.6)
    return buf

def ride(duration: float = 0.9, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    n = int(duration * sr)
    t = np.arange(n) / sr
    buf = NpAudioBuffer(np.zeros(n), sr)
    for f in (410, 630, 890, 1230):
        buf.samples += 0.15 * np.sin(2 * np.pi * f * t)
    buf.mix(_diff_noise(duration, sr).gain(0.3), at=0.0)
    apply_envelope(buf, attack=0.001, decay=duration * 0.7, sustain=0.05, release=duration * 0.3)
    buf.normalize(0.8)
    return buf

def crash(duration: float = 1.4, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    n = int(duration * sr)
    t = np.arange(n) / sr
    buf = NpAudioBuffer(np.zeros(n), sr)
    for f in (350, 520, 710, 990, 1400):
        buf.samples += 0.12 * np.sin(2 * np.pi * f * t)
    buf.mix(_diff_noise(duration, sr).gain(0.5), at=0.0)
    apply_envelope(buf, attack=0.001, decay=duration * 0.85, sustain=0.0, release=duration * 0.4)
    buf.normalize()
    return buf

def clap(duration: float = 0.25, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    buf = NpAudioBuffer.silence(duration, sr)
    for offset in (0, 0.01, 0.02, 0.035):
        burst = white_noise(0.05, sr)
        apply_envelope(burst, attack=0.001, decay=0.04, sustain=0.0, release=0.02)
        buf.mix(burst, at=offset)
    buf.normalize()
    return buf

def clap_tight(duration: float = 0.15, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    buf = NpAudioBuffer.silence(duration, sr)
    for offset in (0, 0.008, 0.016):
        burst = white_noise(0.035, sr)
        apply_envelope(burst, attack=0.001, decay=0.025, sustain=0.0, release=0.015)
        buf.mix(burst, at=offset)
    buf.normalize()
    return buf

def tom(pitch: float = 120, duration: float = 0.3, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    n = int(duration * sr)
    t = np.arange(n) / n
    freq = pitch * (1 - 0.3 * t)
    phase_inc = 2 * np.pi * freq / sr
    phase = np.cumsum(phase_inc) - phase_inc
    buf = NpAudioBuffer(np.sin(phase), sr)
    apply_envelope(buf, attack=0.001, decay=duration * 0.7, sustain=0.0, release=duration * 0.2)
    buf.normalize()
    return buf

def tom_low(duration: float = 0.4, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    return tom(pitch=90, duration=duration, sr=sr)

def tom_mid(duration: float = 0.32, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    return tom(pitch=140, duration=duration, sr=sr)

def tom_high(duration: float = 0.22, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    return tom(pitch=210, duration=duration, sr=sr)

def cowbell(duration: float = 0.3, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    n = int(duration * sr)
    t = np.arange(n) / sr
    buf = NpAudioBuffer(np.zeros(n), sr)
    for f in (587.0, 845.0):
        buf.samples += 0.5 * np.sin(2 * np.pi * f * t)
    apply_envelope(buf, attack=0.001, decay=duration * 0.4, sustain=0.1, release=duration * 0.3)
    buf.normalize()
    return buf

def shaker(duration: float = 0.12, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    buf = _diff_noise(duration, sr)
    apply_envelope(buf, attack=0.005, decay=duration * 0.4, sustain=0.1, release=duration * 0.4)
    buf.normalize(0.7)
    return buf

def tambourine(duration: float = 0.2, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    n = int(duration * sr)
    t = np.arange(n) / sr
    buf = _diff_noise(duration, sr).gain(0.6)
    for f in (2600, 3200, 4100):
        buf.samples += 0.08 * np.sin(2 * np.pi * f * t)
    apply_envelope(buf, attack=0.001, decay=duration * 0.5, sustain=0.05, release=duration * 0.3)
    buf.normalize(0.8)
    return buf

def conga(pitch: float = 300, duration: float = 0.22, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    n = int(duration * sr)
    t = np.arange(n) / sr
    buf = NpAudioBuffer(np.sin(2 * np.pi * pitch * t), sr)
    apply_envelope(buf, attack=0.002, decay=duration * 0.6, sustain=0.05, release=duration * 0.2)
    buf.normalize()
    return buf

def clave(pitch: float = 2500, duration: float = 0.09, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    n = int(duration * sr)
    t = np.arange(n) / sr
    buf = NpAudioBuffer(np.sin(2 * np.pi * pitch * t), sr)
    apply_envelope(buf, attack=0.0005, decay=duration * 0.4, sustain=0.0, release=duration * 0.2)
    buf.normalize()
    return buf

def woodblock(pitch: float = 1200, duration: float = 0.1, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    n = int(duration * sr)
    t = np.arange(n) / sr
    buf = NpAudioBuffer(np.sin(2 * np.pi * pitch * t), sr)
    apply_envelope(buf, attack=0.0005, decay=duration * 0.3, sustain=0.0, release=duration * 0.15)
    buf.normalize()
    return buf

def triangle_perc(pitch: float = 3800, duration: float = 0.6, sr: int = SAMPLE_RATE) -> NpAudioBuffer:
    n = int(duration * sr)
    t = np.arange(n) / sr
    buf = NpAudioBuffer(0.6 * np.sin(2 * np.pi * pitch * t), sr)
    apply_envelope(buf, attack=0.001, decay=duration * 0.8, sustain=0.05, release=duration * 0.3)
    buf.normalize(0.7)
    return buf

VARIANTS: Dict[str, Dict[str, Callable[..., NpAudioBuffer]]] = {
    "kick": {
        "default": kick, "808": kick_808, "sub": kick_sub,
        "acoustic": kick_acoustic, "punchy": kick_punchy,
    },
    "snare": {
        "default": snare, "tight": snare_tight, "fat": snare_fat,
        "electro": snare_electro, "rimshot": rimshot,
    },
    "hihat": {
        "closed": hihat_closed, "open": hihat_open,
        "trap": hihat_trap, "pedal": hihat_pedal,
    },
    "cymbal": {"ride": ride, "crash": crash},
    "clap": {"default": clap, "tight": clap_tight},
    "tom": {"low": tom_low, "mid": tom_mid, "high": tom_high},
    "perc": {
        "cowbell": cowbell, "shaker": shaker, "tambourine": tambourine,
        "conga": conga, "clave": clave, "woodblock": woodblock,
        "triangle": triangle_perc,
    },
}

def get_sound(family: str, variant: str = "default") -> Callable[..., NpAudioBuffer]:
    fam = VARIANTS.get(family)
    if fam is None:
        raise ValueError(f"unknown drum family '{family}'. available: {list(VARIANTS)}")
    fn = fam.get(variant)
    if fn is None:
        raise ValueError(f"unknown variant '{variant}' for '{family}'. available: {list(fam)}")
    return fn

def list_sounds() -> Dict[str, List[str]]:
    return {fam: list(v) for fam, v in VARIANTS.items()}

def build_kit(mapping: Dict[str, tuple]) -> Dict[str, Callable[..., NpAudioBuffer]]:
    return {sym: get_sound(*spec) for sym, spec in mapping.items()}

DEFAULT_KIT: Dict[str, Callable[..., NpAudioBuffer]] = {
    "K": kick, "S": snare,
    "H": lambda: hihat(closed=True),
    "O": lambda: hihat(duration=0.25, closed=False),
    "C": clap, "T": tom, "B": cowbell, "R": rimshot,
    "Y": ride, "Z": crash, "A": shaker, "M": tambourine,
    "G": conga, "V": clave, "W": woodblock,
}