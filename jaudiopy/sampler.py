from typing import Optional, List, Tuple, Union, Any
from .buffer import AudioBuffer
from .synth import note_freq, resample_linear

class Sampler:
    def __init__(self, buf: AudioBuffer, base_note: str = "C", base_octave: int = 4):
        self.buf = buf
        self.base_freq = note_freq(base_note, base_octave)

    @classmethod
    def load(cls, path_or_url: Union[str, bytes, bytearray, Any], base_note: str = "C",
             base_octave: int = 4) -> 'Sampler':
        from .audio_io import load_audio
        loaded = load_audio(path_or_url)
        buf = loaded[0] if isinstance(loaded, tuple) else loaded
        return cls(buf, base_note, base_octave)

    def _fit_to(self, shifted: AudioBuffer, duration: Optional[float]) -> AudioBuffer:
        if duration is None:
            return shifted
        n = int(duration * shifted.sr)
        if len(shifted.samples) > n:
            shifted.samples = shifted.samples[:n]
        else:
            shifted.pad_to(n)
        return shifted

    def play_note(self, note: str, octave: int, duration: Optional[float] = None) -> AudioBuffer:
        ratio = note_freq(note, octave) / self.base_freq
        shifted = resample_linear(self.buf, ratio)
        return self._fit_to(shifted, duration)

    def play_semitones(self, semitones: float, duration: Optional[float] = None) -> AudioBuffer:
        ratio = 2 ** (semitones / 12)
        shifted = resample_linear(self.buf, ratio)
        return self._fit_to(shifted, duration)

    def loop_to(self, duration: float) -> AudioBuffer:
        n_target = int(duration * self.buf.sr)
        out = self.buf.copy()
        if len(out.samples) == 0:
            return out
        while len(out.samples) < n_target:
            out.append(self.buf)
        out.samples = out.samples[:n_target]
        return out

    def chop(self, n_slices: int) -> List[AudioBuffer]:
        if n_slices <= 0:
            raise ValueError("'n_slices' must be > 0")
        n = len(self.buf.samples)
        size = max(n // n_slices, 1)
        slices = []
        for i in range(n_slices):
            start = i * size
            end = n if i == n_slices - 1 else start + size
            slices.append(AudioBuffer(self.buf.samples[start:end], self.buf.sr))
        return slices

    def slice_seconds(self, start_sec: float, end_sec: float) -> AudioBuffer:
        return self.buf.slice(start_sec, end_sec)