import numpy as np
from typing import Optional, List, Union

SAMPLE_RATE = 44100

_module_rng = np.random.default_rng()


def seed_random(seed: Optional[int]) -> None:
    global _module_rng
    _module_rng = np.random.default_rng(seed)

class NpAudioBuffer:
    __slots__ = ("samples", "sr")

    def __init__(self, samples: Optional[Union[np.ndarray, List[float]]] = None, sr: int = SAMPLE_RATE):
        self.sr = sr
        if samples is None:
            self.samples = np.zeros(0, dtype=np.float64)
        else:
            self.samples = np.asarray(samples, dtype=np.float64)

    @classmethod
    def silence(cls, duration: float, sr: int = SAMPLE_RATE) -> 'NpAudioBuffer':
        return cls(np.zeros(int(duration * sr), dtype=np.float64), sr)

    @classmethod
    def from_function(cls, duration: float, fn: callable, sr: int = SAMPLE_RATE) -> 'NpAudioBuffer':
        n = int(duration * sr)
        t = np.arange(n) / sr
        try:
            out = np.asarray(fn(t), dtype=np.float64)
            if out.shape != t.shape:
                raise ValueError("fn did not return one value per sample")
        except Exception:
            out = np.fromiter((fn(ti) for ti in t), dtype=np.float64, count=n)
        return cls(out, sr)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr if len(self.samples) else 0.0

    def copy(self) -> 'NpAudioBuffer':
        return NpAudioBuffer(self.samples.copy(), self.sr)

    def gain(self, factor: float) -> 'NpAudioBuffer':
        self.samples = self.samples * factor
        return self

    def gain_db(self, db: float) -> 'NpAudioBuffer':
        return self.gain(10 ** (db / 20))

    def pad_to(self, n_samples: int) -> 'NpAudioBuffer':
        if len(self.samples) < n_samples:
            self.samples = np.pad(self.samples, (0, n_samples - len(self.samples)))
        return self

    def mix(self, other: 'NpAudioBuffer', at: float = 0.0, volume: float = 1.0) -> 'NpAudioBuffer':
        if at < 0:
            raise ValueError("'at' must be >= 0")
        start = int(at * self.sr)
        needed = start + len(other.samples)
        if needed > len(self.samples):
            self.pad_to(needed)
        self.samples[start:start + len(other.samples)] += other.samples * volume
        return self

    def append(self, other: 'NpAudioBuffer') -> 'NpAudioBuffer':
        self.samples = np.concatenate([self.samples, other.samples])
        return self

    def repeat(self, times: int) -> 'NpAudioBuffer':
        if times == 0:
            self.samples = np.zeros(0, dtype=np.float64)
            return self
        self.samples = np.tile(self.samples, max(times, 1))
        return self

    def slice(self, start_sec: float, end_sec: float) -> 'NpAudioBuffer':
        start = int(start_sec * self.sr)
        end = int(end_sec * self.sr)
        return NpAudioBuffer(self.samples[start:end].copy(), self.sr)

    def reverse(self) -> 'NpAudioBuffer':
        self.samples = self.samples[::-1].copy()
        return self

    def peak(self) -> float:
        return float(np.max(np.abs(self.samples))) if len(self.samples) else 0.0

    def normalize(self, peak: float = 0.95) -> 'NpAudioBuffer':
        m = self.peak()
        if m > 0:
            self.gain(peak / m)
        return self

    def clip(self, limit: float = 1.0) -> 'NpAudioBuffer':
        self.samples = np.clip(self.samples, -limit, limit)
        return self

    def fade_in(self, duration: float) -> 'NpAudioBuffer':
        n = min(int(duration * self.sr), len(self.samples))
        if n > 0:
            ramp = np.linspace(0.0, 1.0, n, endpoint=False)
            self.samples[:n] *= ramp
        return self

    def fade_out(self, duration: float) -> 'NpAudioBuffer':
        n = min(int(duration * self.sr), len(self.samples))
        if n > 0:
            ramp = (n - 1 - np.arange(n)) / n
            self.samples[-n:] *= ramp
        return self

    def to_list(self) -> List[float]:
        return self.samples.tolist()

def mix_buffers(buffers: List[NpAudioBuffer], volumes: Optional[List[float]] = None) -> NpAudioBuffer:
    if not buffers:
        raise ValueError("'buffers' must not be empty")
    if volumes is None:
        volumes = [1.0] * len(buffers)
    max_len = max(len(b.samples) for b in buffers)
    sr = buffers[0].sr
    result = np.zeros(max_len, dtype=np.float64)
    for b, v in zip(buffers, volumes):
        result[:len(b.samples)] += b.samples * v
    return NpAudioBuffer(result, sr)

def white_noise(duration: float, sr: int = SAMPLE_RATE, amp: float = 1.0,
                 seed: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None) -> NpAudioBuffer:
    n = int(duration * sr)
    if rng is not None:
        gen = rng
    elif seed is not None:
        gen = np.random.default_rng(seed)
    else:
        gen = _module_rng
    return NpAudioBuffer(gen.uniform(-amp, amp, n), sr)

AudioBuffer = NpAudioBuffer