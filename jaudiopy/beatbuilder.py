import os
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from .audio_io import save_wav_stereo
from .buffer import AudioBuffer, seed_random as _seed_python_engine

CHROMATIC: List[str] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def transpose(root: str, semitones: int) -> Tuple[str, int]:
    idx = CHROMATIC.index(root)
    total = idx + semitones
    return CHROMATIC[total % 12], total // 12

PROGRESSIONS: Dict[str, List[int]] = {
    "minor_epic": [0, 8, 10, 5],
    "minor_simple": [0, 5, 8, 7],
    "trap_dark": [0, 3, 8, 5],
    "single_root": [0, 0, 0, 0],
}
MINOR_TRIAD = [0, 3, 7]

STYLES: Dict[str, Dict[str, Any]] = {
    "trap_808": dict(
        kick_variant="808", snare_variant=None, clap_variant="default", hat_variant="trap",
        master_preset="loud_edm", bass="808", melody=None, swing=0.0, steps_per_beat=8,
        kick_pattern="X.......x.......X.......x......",
        clap_pattern="." * 16 + "X" + "." * 15,
        hat_pattern="X.XxX.XxXXXxX.XxX.XxX.XxXXXxX.Xx",
        open_pattern="." * 31 + "X",
    ),
    "boombap_piano": dict(
        kick_variant="acoustic", snare_variant="tight", clap_variant=None, hat_variant="closed",
        master_preset="warm", bass=None, melody="piano", swing=0.06, steps_per_beat=4,
        kick_pattern="X.......x..X....",
        snare_pattern="....X.......X...",
        hat_pattern="X.X.X.X.X.X.X.X.",
        open_pattern="." * 14 + "X.",
    ),
    "street_hiphop": dict(
        kick_variant="punchy", snare_variant="fat", clap_variant="default", hat_variant="closed",
        master_preset="balanced", bass="synth", melody="pluck", swing=0.08, steps_per_beat=8,
        kick_pattern="X..x..X...x.X...X..x..X...x.X.x.",
        snare_pattern="....X.......X.......X.......X...",
        clap_pattern="....X.......X.......X.......X...",
        hat_pattern="X.XxX.XxX.XxX.XxX.XxX.XxX.XxX.Xx",
    ),
    "dark_melodic": dict(
        kick_variant="sub", snare_variant="fat", clap_variant=None, hat_variant="pedal",
        master_preset="warm", bass="808", melody="piano", swing=0.03, steps_per_beat=8,
        kick_pattern="X.......x.......X.......x......",
        snare_pattern="." * 16 + "X" + "." * 15,
        hat_pattern="X...X...X...X...X...X...X...X..",
    ),
    "minimal_bounce": dict(
        kick_variant="default", snare_variant="tight", clap_variant=None, hat_variant="closed",
        master_preset="bright", bass="synth", melody=None, swing=0.0, steps_per_beat=4,
        kick_pattern="X.......x...X...",
        snare_pattern="....X.......X...",
        hat_pattern="X.X.X.X.X.X.X.X.",
    ),
}

def _load_backend(change_to_numpy: bool) -> Dict[str, Any]:
    if change_to_numpy:
        from .np_backend import drums as d, instruments as inst, effects as fx
        from .np_backend.sequencer import StepSequencer, MelodySequencer
        from .np_backend.mixer import AdvancedMixer
        from .np_backend.mastering import master_with_preset
    else:
        from . import drums as d, instruments as inst, effects as fx
        from .sequencer import StepSequencer, MelodySequencer
        from .mixer import AdvancedMixer
        from .mastering import master_with_preset
    return dict(
        drums=d, inst=inst, fx=fx,
        StepSequencer=StepSequencer, MelodySequencer=MelodySequencer,
        Mixer=AdvancedMixer, master_with_preset=master_with_preset,
    )

class BeatBuilder:
    def __init__(self, style: str = "trap_808", bpm: float = 140, key: str = "C",
                 progression: str = "minor_epic", loop_bars: int = 4, sr: int = 44100,
                 change_to_numpy: bool = False, seed: Optional[int] = None):
        if style not in STYLES:
            raise ValueError(f"unknown style '{style}'. available: {list(STYLES)}")
        if progression not in PROGRESSIONS:
            raise ValueError(f"unknown progression '{progression}'. available: {list(PROGRESSIONS)}")
        if key not in CHROMATIC:
            raise ValueError(f"unknown key '{key}'. valid keys: {CHROMATIC}")
        self.style_name = style
        self.style = STYLES[style]
        self.bpm = bpm
        self.key = key
        self.progression = PROGRESSIONS[progression]
        self.loop_bars = loop_bars
        self.sr = sr
        self.change_to_numpy = bool(change_to_numpy)
        self.seed = seed
        self.engine = "numpy" if self.change_to_numpy else "python"
        backend = _load_backend(self.change_to_numpy)
        self._drums = backend["drums"]
        self._inst = backend["inst"]
        self._fx = backend["fx"]
        self._StepSequencer = backend["StepSequencer"]
        self._MelodySequencer = backend["MelodySequencer"]
        self._Mixer = backend["Mixer"]
        self._master_with_preset = backend["master_with_preset"]

    def _kit(self) -> Dict[str, Callable[..., AudioBuffer]]:
        s = self.style
        mapping: Dict[str, Tuple[str, str]] = {"K": ("kick", s["kick_variant"])}
        if s.get("snare_variant"):
            mapping["S"] = ("snare", s["snare_variant"])
        if s.get("clap_variant"):
            mapping["C"] = ("clap", s["clap_variant"])
        mapping["H"] = ("hihat", s["hat_variant"])
        if s.get("open_pattern"):
            mapping["O"] = ("hihat", "open")
        return self._drums.build_kit(mapping)

    def _chord_notes(self, root_semitone_offset: int, octave: int) -> List[Tuple[str, int]]:
        notes = []
        for off in MINOR_TRIAD:
            note, oct_shift = transpose(self.key, root_semitone_offset + off)
            notes.append((note, octave + oct_shift))
        return notes

    def _build_drum_bus(self) -> AudioBuffer:
        s = self.style
        kit = self._kit()
        patterns: Dict[str, str] = {}
        sounds: Dict[str, Dict[str, Any]] = {}
        patterns["K"] = s["kick_pattern"]
        sounds["K"] = {"X": kit["K"], "x": (kit["K"], 0.55)}
        if s.get("snare_variant") and s.get("snare_pattern"):
            patterns["S"] = s["snare_pattern"]
            sounds["S"] = {"X": kit["S"]}
        if s.get("clap_variant") and s.get("clap_pattern"):
            patterns["C"] = s["clap_pattern"]
            sounds["C"] = {"X": kit["C"]}
        patterns["H"] = s["hat_pattern"]
        sounds["H"] = {"X": kit["H"], "x": (kit["H"], 0.45)}
        if s.get("open_pattern"):
            patterns["O"] = s["open_pattern"]
            sounds["O"] = {"X": kit["O"]}
        seq = self._StepSequencer(bpm=self.bpm, steps_per_beat=s["steps_per_beat"])
        drum_bus = seq.render_kit(patterns, sounds, bars=self.loop_bars, swing=s["swing"])
        return drum_bus

    def _beats_per_chord(self) -> float:
        return (self.loop_bars * 4) / len(self.progression)

    def _build_bass(self) -> Optional[AudioBuffer]:
        s = self.style
        if not s.get("bass"):
            return None
        beats_per_chord = self._beats_per_chord()
        if s["bass"] == "808":
            notes = []
            for offset in self.progression:
                note, oct_shift = transpose(self.key, offset)
                notes.append((note, 1 + oct_shift, beats_per_chord))
            bass_line = self._inst.bass_808_line(notes, bpm=self.bpm, glide_time=0.09, drive=3.2)
            full_kick_pattern = s["kick_pattern"] * self.loop_bars
            self._fx.sidechain(
                bass_line, full_kick_pattern, bpm=self.bpm,
                steps_per_beat=s["steps_per_beat"], depth=0.5, release=0.12,
            )
            return bass_line
        mel = self._MelodySequencer(bpm=self.bpm)
        events = []
        for offset in self.progression:
            note, oct_shift = transpose(self.key, offset)
            events.append((note, 2 + oct_shift, beats_per_chord, 0.85))
        return mel.render(events, self._inst.bass_synth)

    def _build_melody(self) -> Optional[AudioBuffer]:
        s = self.style
        if not s.get("melody"):
            return None
        beats_per_chord = self._beats_per_chord()
        mel = self._MelodySequencer(bpm=self.bpm)
        if s["melody"] == "piano":
            events = []
            for offset in self.progression:
                triad = self._chord_notes(offset, 3)
                note_names = [n for n, _ in triad]
                octaves = [o for _, o in triad]
                events.append((note_names, octaves, beats_per_chord, 0.5))
            return mel.render(events, self._inst.piano_synth)
        if s["melody"] == "pluck":
            events = []
            for offset in self.progression:
                note, oct_shift = transpose(self.key, offset)
                events.append((note, 4 + oct_shift, beats_per_chord, 0.6))
            return mel.render(events, self._inst.pluck_synth)
        return None

    def _apply_seed(self) -> None:
        if self.seed is None:
            return
        _seed_python_engine(self.seed)
        if self.change_to_numpy:
            from .np_backend.buffer import seed_random as _seed_numpy_engine
            _seed_numpy_engine(self.seed)

    def build_loop(self) -> Tuple[AudioBuffer, AudioBuffer]:
        self._apply_seed()
        drum_bus = self._build_drum_bus()
        bass_line = self._build_bass()
        melody_line = self._build_melody()
        mixer = self._Mixer()
        mixer.add_track("drums", drum_bus, volume=1.0)
        if bass_line is not None:
            mixer.add_track("bass", bass_line, volume=0.9)
            mixer.add_bus("low_end", ["bass"], volume=1.1, pan=0.0)
        if melody_line is not None:
            mixer.add_track("melody", melody_line, volume=0.7)
            mixer.add_bus("melodics", ["melody"], volume=1.0, pan=0.08)
        left, right = mixer.render_and_master(preset=self.style["master_preset"])
        return left, right

    def build(self, duration_seconds: Optional[float] = None,
              duration_minutes: Optional[float] = None) -> Tuple[AudioBuffer, AudioBuffer]:
        left, right = self.build_loop()
        if duration_seconds is None and duration_minutes is None:
            return left, right
        target = duration_seconds if duration_seconds is not None else duration_minutes * 60
        target_samples = int(target * self.sr)
        loop_samples = len(left.samples)
        if loop_samples == 0:
            return left, right
        if target_samples <= loop_samples:
            left.samples = left.samples[:target_samples]
            right.samples = right.samples[:target_samples]
            return left, right
        n_repeats = -(-target_samples // loop_samples)
        left.repeat(n_repeats)
        right.repeat(n_repeats)
        left.samples = left.samples[:target_samples]
        right.samples = right.samples[:target_samples]
        return left, right

    def build_and_save(self, path: str, duration_seconds: Optional[float] = None,
                       duration_minutes: Optional[float] = None) -> str:
        left, right = self.build(duration_seconds=duration_seconds, duration_minutes=duration_minutes)
        save_wav_stereo(path, left, right)
        return path

def list_styles() -> List[str]:
    return list(STYLES)

def list_progressions() -> List[str]:
    return list(PROGRESSIONS)


def _build_one(spec: Dict[str, Any]) -> Union[str, Tuple[AudioBuffer, AudioBuffer]]:
    """Worker used by build_many(). Must stay a top-level function (not a
    closure/method) so it can be pickled and sent to a worker process."""
    kwargs = dict(spec)
    duration_seconds = kwargs.pop("duration_seconds", None)
    duration_minutes = kwargs.pop("duration_minutes", None)
    save_as = kwargs.pop("save_as", None)
    builder = BeatBuilder(**kwargs)
    left, right = builder.build(duration_seconds=duration_seconds, duration_minutes=duration_minutes)
    if save_as:
        save_wav_stereo(save_as, left, right)
        return save_as
    return left, right


def build_many(specs: List[Dict[str, Any]], n_jobs: Optional[int] = None,
                parallel: bool = True) -> List[Union[str, Tuple[AudioBuffer, AudioBuffer]]]:
    if not specs:
        return []
    if not parallel or len(specs) == 1:
        return [_build_one(s) for s in specs]
    import multiprocessing as mp
    n_jobs = max(1, min(n_jobs or (os.cpu_count() or 1), len(specs)))
    try:
        with mp.Pool(processes=n_jobs) as pool:
            return pool.map(_build_one, specs)
    except (OSError, ValueError, RuntimeError, ImportError):
        return [_build_one(s) for s in specs]