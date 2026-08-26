from array import array
import random as _random_module
from typing import Optional, List, Union

SAMPLE_RATE = 44100

try:
    import numpy as _np
except ImportError:
    _np = None


def _as_np(samples: array):
    if _np is None or len(samples) == 0:
        return None
    return _np.frombuffer(samples, dtype=_np.float64)

_module_rng = _random_module.Random()


def seed_random(seed: Optional[int]) -> None:
    _module_rng.seed(seed)

class AudioBuffer:
    __slots__ = ("samples", "sr")

    def __init__(self, samples: Optional[Union[array, List[float]]] = None, sr: int = SAMPLE_RATE):
        self.sr = sr
        if samples is None:
            self.samples = array('d')
        elif isinstance(samples, array):
            self.samples = samples
        elif _np is not None and hasattr(samples, "dtype") and hasattr(samples, "astype"):
            self.samples = array('d', samples.astype(_np.float64, copy=False).tobytes())
        else:
            self.samples = array('d', samples)

    @classmethod
    def silence(cls, duration: float, sr: int = SAMPLE_RATE) -> 'AudioBuffer':
        return cls([0.0] * int(duration * sr), sr)

    @classmethod
    def from_function(cls, duration: float, fn: callable, sr: int = SAMPLE_RATE) -> 'AudioBuffer':
        n = int(duration * sr)
        return cls([fn(i / sr) for i in range(n)], sr)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr if self.samples else 0.0

    def copy(self) -> 'AudioBuffer':
        return AudioBuffer(array('d', self.samples), self.sr)

    def gain(self, factor: float) -> 'AudioBuffer':
        view = _as_np(self.samples)
        if view is not None:
            self.samples = array('d', (view * factor).tobytes())
        else:
            self.samples = array('d', (s * factor for s in self.samples))
        return self

    def gain_db(self, db: float) -> 'AudioBuffer':
        return self.gain(10 ** (db / 20))

    def pad_to(self, n_samples: int) -> 'AudioBuffer':
        if len(self.samples) < n_samples:
            self.samples.extend([0.0] * (n_samples - len(self.samples)))
        return self

    def mix(self, other: 'AudioBuffer', at: float = 0.0, volume: float = 1.0) -> 'AudioBuffer':
        if at < 0:
            raise ValueError("'at' must be >= 0")
        start = int(at * self.sr)
        needed = start + len(other.samples)
        if needed > len(self.samples):
            self.pad_to(needed)
        self_view = _as_np(self.samples)
        other_view = _as_np(other.samples)
        if self_view is not None and other_view is not None:
            self_view[start:start + len(other_view)] += other_view * volume
        else:
            for i, s in enumerate(other.samples):
                self.samples[start + i] += s * volume
        return self

    def append(self, other: 'AudioBuffer') -> 'AudioBuffer':
        self.samples.extend(other.samples)
        return self

    def repeat(self, times: int) -> 'AudioBuffer':
        if times == 0:
            self.samples = array('d')
            return self
        original = array('d', self.samples)
        for _ in range(times - 1):
            self.samples.extend(original)
        return self

    def slice(self, start_sec: float, end_sec: float) -> 'AudioBuffer':
        start = int(start_sec * self.sr)
        end = int(end_sec * self.sr)
        return AudioBuffer(self.samples[start:end], self.sr)

    def reverse(self) -> 'AudioBuffer':
        self.samples = array('d', reversed(self.samples))
        return self

    def peak(self) -> float:
        view = _as_np(self.samples)
        if view is not None:
            return float(_np.max(_np.abs(view)))
        return max(map(abs, self.samples), default=0.0)

    def normalize(self, peak: float = 0.95) -> 'AudioBuffer':
        m = self.peak()
        if m > 0:
            self.gain(peak / m)
        return self

    def clip(self, limit: float = 1.0) -> 'AudioBuffer':
        view = _as_np(self.samples)
        if view is not None:
            self.samples = array('d', _np.clip(view, -limit, limit).tobytes())
        else:
            self.samples = array('d', (max(-limit, min(limit, s)) for s in self.samples))
        return self

    def fade_in(self, duration: float) -> 'AudioBuffer':
        n = min(int(duration * self.sr), len(self.samples))
        if n == 0:
            return self
        if _np is not None:
            ramp = _np.arange(n, dtype=_np.float64) / n
            self.samples[0:n] = array('d', (_as_np(self.samples)[:n] * ramp).tobytes())
        else:
            for i in range(n):
                self.samples[i] *= i / n
        return self

    def fade_out(self, duration: float) -> 'AudioBuffer':
        n = min(int(duration * self.sr), len(self.samples))
        if n == 0:
            return self
        total = len(self.samples)
        if _np is not None:
            ramp = (n - 1 - _np.arange(n, dtype=_np.float64)) / n
            view = _as_np(self.samples)
            self.samples[total - n:total] = array('d', (view[total - n:total] * ramp).tobytes())
        else:
            for i in range(n):
                self.samples[total - 1 - i] *= i / n
        return self

    def to_list(self) -> List[float]:
        return list(self.samples)


def mix_buffers(buffers: List[AudioBuffer], volumes: Optional[List[float]] = None) -> AudioBuffer:
    if not buffers:
        raise ValueError("'buffers' must not be empty")
    if volumes is None:
        volumes = [1.0] * len(buffers)
    max_len = max(len(b.samples) for b in buffers)
    sr = buffers[0].sr
    if _np is not None and max_len > 0:
        acc = _np.zeros(max_len, dtype=_np.float64)
        for b, v in zip(buffers, volumes):
            view = _as_np(b.samples)
            if view is not None:
                acc[:len(view)] += view * v
        result = array('d', acc.tobytes())
    else:
        result = array('d', [0.0] * max_len)
        for b, v in zip(buffers, volumes):
            for i, s in enumerate(b.samples):
                result[i] += s * v
    return AudioBuffer(result, sr)


def white_noise(duration: float, sr: int = SAMPLE_RATE, amp: float = 1.0,
                 seed: Optional[int] = None,
                 rng: Optional[_random_module.Random] = None) -> AudioBuffer:
    n = int(duration * sr)
    if rng is not None:
        r = rng
    elif seed is not None:
        r = _random_module.Random(seed)
    else:
        r = _module_rng
    return AudioBuffer([r.uniform(-amp, amp) for _ in range(n)], sr)