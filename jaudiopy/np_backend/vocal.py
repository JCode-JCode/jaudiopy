from typing import Optional, Any, Tuple
from .buffer import NpAudioBuffer
from .effects import highpass, eq_band, de_esser, compressor, limiter, duck_under

def vocal_chain(buf: NpAudioBuffer,
                hp_freq: float = 90,
                presence_freq: float = 3500,
                presence_db: float = 3.0,
                deess_freq: float = 6500,
                comp_threshold: float = -20,
                comp_ratio: float = 3.0,
                target_peak: float = 0.9) -> NpAudioBuffer:
    highpass(buf, freq=hp_freq, q=0.7)
    eq_band(buf, freq=presence_freq, gain_db=presence_db, q=1.0)
    de_esser(buf, freq=deess_freq)
    compressor(buf, threshold_db=comp_threshold, ratio=comp_ratio,
               attack=0.005, release=0.12, makeup_db=2.0)
    limiter(buf, ceiling_db=-0.5)
    buf.normalize(target_peak)
    return buf

def mix_vocal_with_beat(beat_left: NpAudioBuffer, beat_right: NpAudioBuffer, vocal: NpAudioBuffer,
                        vocal_volume: float = 1.0, duck_depth: float = 0.5,
                        apply_chain: bool = True, vocal_at: float = 0.0,
                        **chain_kwargs) -> Tuple[NpAudioBuffer, NpAudioBuffer]:
    """Mix a processed vocal into a stereo beat, ducking the beat under it.

    `vocal_at` is the vocal's start time in seconds within the beat
    (e.g. `vocal_at=8.0` to bring the vocal in 8 seconds into the
    instrumental, such as after an intro). Defaults to 0.0 (vocal starts
    at the top of the beat), which matches the previous behavior of this
    function exactly. The beat only ducks under the vocal where the
    vocal is actually sounding - not from the start of the beat - since
    the duck-under trigger is the vocal already placed at `vocal_at`.
    """
    if not all(isinstance(b, NpAudioBuffer) for b in (beat_left, beat_right, vocal)):
        raise ValueError("All buffers must be numpy NpAudioBuffer instances.")
    if vocal_at < 0:
        raise ValueError("'vocal_at' must be >= 0")
    v = vocal.copy()
    if apply_chain:
        vocal_chain(v, **chain_kwargs)
    sr = v.sr
    n = max(len(beat_left.samples), len(beat_right.samples),
            int(vocal_at * sr) + len(v.samples))
    bl = beat_left.copy()
    br = beat_right.copy()
    bl.pad_to(n)
    br.pad_to(n)
    v_placed = NpAudioBuffer.silence(n / sr, sr)
    v_placed.mix(v, at=vocal_at)
    duck_under(bl, v_placed, depth=duck_depth)
    duck_under(br, v_placed, depth=duck_depth)
    bl.mix(v_placed, at=0.0, volume=vocal_volume)
    br.mix(v_placed, at=0.0, volume=vocal_volume)
    peak = max(bl.peak(), br.peak())
    if peak > 0.98:
        scale = 0.98 / peak
        bl.gain(scale)
        br.gain(scale)
    return bl, br