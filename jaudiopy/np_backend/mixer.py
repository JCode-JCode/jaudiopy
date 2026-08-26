import numpy as np
from typing import Optional, List, Dict, Any, Callable, Tuple
from .buffer import NpAudioBuffer
from .effects import pan_stereo
from .mastering import master_chain_advanced, master_with_preset

class Track:
    def __init__(self, name: str, buf: NpAudioBuffer, volume: float = 1.0, pan: float = 0.0,
                 mute: bool = False, solo: bool = False):
        self.name = name
        self.buf = buf
        self.volume = volume
        self.pan = pan
        self.mute = mute
        self.solo = solo

class Bus:
    def __init__(self, name: str, track_names: List[str], volume: float = 1.0, pan: float = 0.0):
        self.name = name
        self.track_names = track_names
        self.volume = volume
        self.pan = pan

class Mixer:
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.tracks: List[Track] = []
        self.buses: List[Bus] = []

    def add_track(self, name: str, buf: NpAudioBuffer, volume: float = 1.0, pan: float = 0.0) -> 'Mixer':
        self.tracks.append(Track(name, buf, volume, pan))
        return self

    def get_track(self, name: str) -> Optional[Track]:
        for t in self.tracks:
            if t.name == name:
                return t
        return None

    def add_bus(self, name: str, track_names: List[str], volume: float = 1.0, pan: float = 0.0) -> 'Mixer':
        self.buses.append(Bus(name, track_names, volume, pan))
        return self

    def set_track_fx(self, name: str, fx_fn: Callable[[NpAudioBuffer], None]) -> 'Mixer':
        t = self.get_track(name)
        if t is not None:
            fx_fn(t.buf)
        return self

    def _bus_for_track(self, name: str) -> Optional[Bus]:
        for bus in self.buses:
            if name in bus.track_names:
                return bus
        return None

    def _effective_volume_pan(self, track: Track) -> Tuple[float, float]:
        bus = self._bus_for_track(track.name)
        if bus is None:
            return track.volume, track.pan
        combined_pan = max(-1.0, min(1.0, track.pan + bus.pan))
        return track.volume * bus.volume, combined_pan

    def render(self) -> Tuple[NpAudioBuffer, NpAudioBuffer]:
        active = self.tracks
        if any(t.solo for t in self.tracks):
            active = [t for t in self.tracks if t.solo]
        active = [t for t in active if not t.mute]
        if not active:
            return NpAudioBuffer.silence(0, self.sr), NpAudioBuffer.silence(0, self.sr)
        max_len = max(len(t.buf.samples) for t in active)
        left = np.zeros(max_len, dtype=np.float64)
        right = np.zeros(max_len, dtype=np.float64)
        for t in active:
            volume, pan = self._effective_volume_pan(t)
            b = t.buf.copy().gain(volume)
            b.pad_to(max_len)
            l, r = pan_stereo(b, pan)
            left += l
            right += r
        return NpAudioBuffer(left, self.sr), NpAudioBuffer(right, self.sr)

    def render_and_master(self, preset: str = "balanced", **overrides) -> Tuple[NpAudioBuffer, NpAudioBuffer]:
        left, right = self.render()
        return master_with_preset(left, right, preset=preset, **overrides)

AdvancedMixer = Mixer