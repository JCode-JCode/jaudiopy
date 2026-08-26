from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from .buffer import AudioBuffer, mix_buffers

DEFAULT_VELOCITY: Dict[str, float] = {"X": 1.0, "x": 0.6}

class StepSequencer:
    def __init__(self, bpm: float = 120, steps_per_beat: int = 4, sr: int = 44100):
        self.bpm = bpm
        self.steps_per_beat = steps_per_beat
        self.sr = sr

    @property
    def step_duration(self) -> float:
        return 60 / self.bpm / self.steps_per_beat

    def _resolve(self, ch: str, sound_fn: Union[Callable[..., AudioBuffer], Dict[str, Any]]) -> Tuple[Optional[Callable[..., AudioBuffer]], float]:
        if ch == ".":
            return None, 0.0
        if isinstance(sound_fn, dict):
            entry = sound_fn.get(ch)
            if entry is None:
                return None, 0.0
            if isinstance(entry, tuple):
                fn, vel = entry
            else:
                fn, vel = entry, DEFAULT_VELOCITY.get(ch, 1.0 if ch.isupper() else 0.6)
            return fn, vel
        vel = 1.0 if ch.isupper() else 0.6
        return sound_fn, vel

    def render_pattern(self, pattern: str, sound_fn: Union[Callable[..., AudioBuffer], Dict[str, Any]],
                       swing: float = 0.0) -> AudioBuffer:
        step_dur = self.step_duration
        total_len = int(len(pattern) * step_dur * self.sr)
        track = AudioBuffer.silence(total_len / self.sr, self.sr)
        for i, ch in enumerate(pattern):
            fn, velocity = self._resolve(ch, sound_fn)
            if fn is None:
                continue
            offset = step_dur * swing if i % 2 == 1 else 0.0
            t = i * step_dur + offset
            sample = fn()
            sample.gain(velocity)
            track.mix(sample, at=t)
        return track

    def render_kit(self, patterns: Dict[str, str], sounds: Dict[str, Union[Callable[..., AudioBuffer], Dict[str, Any]]],
                   bars: int = 1, volumes: Optional[Dict[str, float]] = None, swing: float = 0.0) -> AudioBuffer:
        tracks: List[AudioBuffer] = []
        names: List[str] = []
        for name, pattern in patterns.items():
            full_pattern = pattern * bars
            tracks.append(self.render_pattern(full_pattern, sounds[name], swing=swing))
            names.append(name)
        volumes = volumes or {}
        return mix_buffers(tracks, [volumes.get(n, 1.0) for n in names])

class MelodySequencer:
    def __init__(self, bpm: float = 120, sr: int = 44100):
        self.bpm = bpm
        self.sr = sr

    def beat_seconds(self, beats: float) -> float:
        return beats * 60 / self.bpm

    def render(self, events: List[tuple], instrument_fn: Callable[..., AudioBuffer]) -> AudioBuffer:
        total_beats = sum(e[2] for e in events)
        total_len = int(self.beat_seconds(total_beats) * self.sr) + 1
        track = AudioBuffer.silence(total_len / self.sr, self.sr)
        t = 0.0
        for notes, octave, beats, velocity in events:
            dur = self.beat_seconds(beats)
            if notes is not None:
                note_list = notes if isinstance(notes, (list, tuple)) else [notes]
                if isinstance(octave, (list, tuple)):
                    oct_list = octave
                else:
                    oct_list = [octave] * len(note_list)
                for note, oct_ in zip(note_list, oct_list):
                    sample = instrument_fn(note, oct_, dur)
                    sample.gain(velocity)
                    track.mix(sample, at=t)
            t += dur
        return track

class Song:
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.sections: List[AudioBuffer] = []

    def add_section(self, buf: AudioBuffer) -> 'Song':
        self.sections.append(buf)
        return self

    def render(self) -> AudioBuffer:
        out = AudioBuffer.silence(0, self.sr)
        for section in self.sections:
            out.append(section)
        return out