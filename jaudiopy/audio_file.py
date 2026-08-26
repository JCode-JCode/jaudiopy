import os
import warnings
from typing import Optional, Any, Dict, Union, Tuple, Callable
from . import audio_io, player
from .buffer import AudioBuffer

class AudioFile:
    def __init__(self, source: Union[str, bytes, bytearray, Any], change_to_numpy: bool = False):
        self.source = source
        self.change_to_numpy = change_to_numpy
        self.repeat_before_save: int = 0
        self.repeat_after_save: int = 0
        self.sr: Optional[int] = None
        self.buf: Optional[AudioBuffer] = None
        self.left: Optional[AudioBuffer] = None
        self.right: Optional[AudioBuffer] = None
        self.is_stereo: bool = False
        self._load()

    def _engine(self) -> Tuple[Any, Any, Any, Any]:
        if self.change_to_numpy:
            from .np_backend import buffer as buf_mod, synth as synth_mod, \
                effects as fx_mod, mastering as master_mod
        else:
            from . import buffer as buf_mod, synth as synth_mod, \
                effects as fx_mod, mastering as master_mod
        return buf_mod, synth_mod, fx_mod, master_mod

    def _effect_fn(self, fx_mod: Any, name: str) -> Callable:
        fn = getattr(fx_mod, name, None)
        if fn is not None:
            return fn
        from . import effects as py_fx
        fn = getattr(py_fx, name, None)
        if fn is None:
            raise ValueError(f"effect '{name}' does not exist.")
        warnings.warn(
            f"effect '{name}' is not available in the numpy engine; "
            f"falling back to the pure-python implementation.",
            RuntimeWarning,
        )
        return fn

    def _load(self) -> 'AudioFile':
        wav_path, cleanup = audio_io.resolve_source(self.source)
        try:
            channels, sr = audio_io.load_wav_raw(wav_path)
        finally:
            for p in cleanup:
                try:
                    os.remove(p)
                except OSError:
                    pass
        self.sr = sr
        buf_mod, *_ = self._engine()
        if len(channels) == 2:
            self.is_stereo = True
            self.left = buf_mod.AudioBuffer(channels[0], sr)
            self.right = buf_mod.AudioBuffer(channels[1], sr)
            self.buf = None
        else:
            self.is_stereo = False
            self.buf = buf_mod.AudioBuffer(channels[0], sr)
            self.left = self.right = None
        return self

    @property
    def duration(self) -> float:
        if self.is_stereo:
            return self.left.duration if self.left else 0.0
        return self.buf.duration if self.buf else 0.0

    def _resolve_save_path(self, save_as: str) -> str:
        if not save_as:
            raise ValueError("'save_as' (output file name) is required.")
        if not save_as.lower().endswith(".wav"):
            save_as += ".wav"
        return save_as

    def _finalize(self, save_as: str, play_after: bool) -> str:
        path = self._resolve_save_path(save_as)
        if self.is_stereo:
            if self.repeat_after_save and self.repeat_after_save > 0:
                self.left.repeat(self.repeat_after_save)
                self.right.repeat(self.repeat_after_save)
            audio_io.save_wav_stereo(path, self.left, self.right)
            if play_after:
                pl, pr = self.left, self.right
                if self.repeat_before_save and self.repeat_before_save > 0:
                    pl = self.left.copy().repeat(self.repeat_before_save)
                    pr = self.right.copy().repeat(self.repeat_before_save)
                player.play_stereo(pl, pr)
        else:
            if self.repeat_after_save and self.repeat_after_save > 0:
                self.buf.repeat(self.repeat_after_save)
            audio_io.save_wav_mono(path, self.buf)
            if play_after:
                pb = self.buf
                if self.repeat_before_save and self.repeat_before_save > 0:
                    pb = self.buf.copy().repeat(self.repeat_before_save)
                player.play_buffer(pb)
        return path

    def play(self) -> 'AudioFile':
        if self.is_stereo:
            pl, pr = self.left, self.right
            if self.repeat_before_save and self.repeat_before_save > 0:
                pl = self.left.copy().repeat(self.repeat_before_save)
                pr = self.right.copy().repeat(self.repeat_before_save)
            player.play_stereo(pl, pr)
        else:
            pb = self.buf
            if self.repeat_before_save and self.repeat_before_save > 0:
                pb = self.buf.copy().repeat(self.repeat_before_save)
            player.play_buffer(pb)
        return self

    def add_effect(self, effect_name: str, save_as: str, play_after: bool = False, **params) -> str:
        _, _, fx_mod, _ = self._engine()
        fn = self._effect_fn(fx_mod, effect_name)
        if self.is_stereo:
            fn(self.left, **params)
            fn(self.right, **params)
        else:
            fn(self.buf, **params)
        return self._finalize(save_as, play_after)

    def change_speed(self, factor: float, save_as: str, play_after: bool = False) -> str:
        _, synth_mod, _, _ = self._engine()
        if self.is_stereo:
            self.left = synth_mod.change_speed(self.left, factor)
            self.right = synth_mod.change_speed(self.right, factor)
        else:
            self.buf = synth_mod.change_speed(self.buf, factor)
        return self._finalize(save_as, play_after)

    def tune(self, semitones: float, save_as: str, play_after: bool = False) -> str:
        _, synth_mod, _, _ = self._engine()
        if self.is_stereo:
            self.left = synth_mod.pitch_shift_semitones(self.left, semitones)
            self.right = synth_mod.pitch_shift_semitones(self.right, semitones)
        else:
            self.buf = synth_mod.pitch_shift_semitones(self.buf, semitones)
        return self._finalize(save_as, play_after)

    def trim(self, start_sec: float, end_sec: float, save_as: str, play_after: bool = False) -> str:
        if self.is_stereo:
            self.left = self.left.slice(start_sec, end_sec)
            self.right = self.right.slice(start_sec, end_sec)
        else:
            self.buf = self.buf.slice(start_sec, end_sec)
        return self._finalize(save_as, play_after)

    def normalize(self, save_as: str, peak: float = 0.95, play_after: bool = False) -> str:
        if self.is_stereo:
            self.left.normalize(peak)
            self.right.normalize(peak)
        else:
            self.buf.normalize(peak)
        return self._finalize(save_as, play_after)

    def reverse(self, save_as: str, play_after: bool = False) -> str:
        if self.is_stereo:
            self.left.reverse()
            self.right.reverse()
        else:
            self.buf.reverse()
        return self._finalize(save_as, play_after)

    def fade(self, save_as: str, fade_in: float = 0.0, fade_out: float = 0.0,
             play_after: bool = False) -> str:
        if self.is_stereo:
            if fade_in:
                self.left.fade_in(fade_in); self.right.fade_in(fade_in)
            if fade_out:
                self.left.fade_out(fade_out); self.right.fade_out(fade_out)
        else:
            if fade_in:
                self.buf.fade_in(fade_in)
            if fade_out:
                self.buf.fade_out(fade_out)
        return self._finalize(save_as, play_after)

    def gain_db(self, db: float, save_as: str, play_after: bool = False) -> str:
        if self.is_stereo:
            self.left.gain_db(db)
            self.right.gain_db(db)
        else:
            self.buf.gain_db(db)
        return self._finalize(save_as, play_after)

    def repeat(self, times: int, save_as: str, play_after: bool = False) -> str:
        if self.is_stereo:
            self.left.repeat(times)
            self.right.repeat(times)
        else:
            self.buf.repeat(times)
        return self._finalize(save_as, play_after)

    def mix_with(self, other: Union[str, 'AudioFile'], save_as: str, at: float = 0.0,
                 volume: float = 1.0, play_after: bool = False) -> str:
        other_file = other if isinstance(other, AudioFile) else AudioFile(other, self.change_to_numpy)
        if self.is_stereo and other_file.is_stereo:
            self.left.mix(other_file.left, at=at, volume=volume)
            self.right.mix(other_file.right, at=at, volume=volume)
        elif not self.is_stereo and not other_file.is_stereo:
            self.buf.mix(other_file.buf, at=at, volume=volume)
        else:
            raise ValueError("to mix, both files must be either mono or stereo.")
        return self._finalize(save_as, play_after)

    def master(self, save_as: str, preset: str = "balanced", play_after: bool = False,
               **overrides) -> str:
        _, _, _, master_mod = self._engine()
        if not self.is_stereo:
            self.left = self.buf.copy()
            self.right = self.buf.copy()
            self.buf = None
            self.is_stereo = True
        self.left, self.right = master_mod.master_with_preset(
            self.left, self.right, preset=preset, **overrides
        )
        return self._finalize(save_as, play_after)

    def to_mono(self, save_as: str, play_after: bool = False) -> str:
        if self.is_stereo:
            buf_mod, *_ = self._engine()
            l, r = self.left.samples, self.right.samples
            n = min(len(l), len(r))
            if hasattr(l, "dtype"):
                merged = (l[:n] + r[:n]) / 2
            else:
                from array import array
                merged = array('d', ((a + b) / 2 for a, b in zip(l, r)))
            self.buf = buf_mod.AudioBuffer(merged, self.sr)
            self.left = self.right = None
            self.is_stereo = False
        return self._finalize(save_as, play_after)

    def to_wav_bytes(self) -> bytes:
        if self.is_stereo:
            return audio_io.save_wav_stereo_bytes(self.left, self.right)
        return audio_io.save_wav_mono_bytes(self.buf)

    def __repr__(self) -> str:
        mode = "numpy" if self.change_to_numpy else "python"
        kind = "stereo" if self.is_stereo else "mono"
        return f"<AudioFile source={self.source!r} sr={self.sr} {kind} engine={mode} dur={self.duration:.2f}s>"