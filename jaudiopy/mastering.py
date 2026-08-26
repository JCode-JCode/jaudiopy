from typing import Optional, Dict, Any, Tuple
from .buffer import AudioBuffer
from .effects import low_shelf, high_shelf, eq_band, compressor, limiter
from .effects import stereo_widener, haas_widen, loudness_maximizer, exciter, saturation

MASTER_PRESETS: Dict[str, Dict[str, Any]] = {
    "balanced":  dict(bass_gain=1.5, presence_gain=1.0, air_gain=0.5,
                       comp_threshold=-16, comp_ratio=3.0, limiter_ceiling=-0.3, width=1.0,
                       target_crest_db=11.0, haas_mix=0.18, haas_delay_ms=14),
    "warm":      dict(bass_gain=3.0, presence_gain=0.5, air_gain=0.0,
                       comp_threshold=-18, comp_ratio=2.5, limiter_ceiling=-0.5, width=1.05,
                       target_crest_db=12.0, haas_mix=0.15, haas_delay_ms=16),
    "bright":    dict(bass_gain=0.5, presence_gain=2.0, air_gain=2.5,
                       comp_threshold=-15, comp_ratio=3.5, limiter_ceiling=-0.2, width=1.15,
                       target_crest_db=10.0, haas_mix=0.22, haas_delay_ms=12),
    "loud_edm":  dict(bass_gain=4.0, presence_gain=1.5, air_gain=2.0,
                       comp_threshold=-20, comp_ratio=6.0, limiter_ceiling=-0.1, width=1.3,
                       target_crest_db=8.0, haas_mix=0.3, haas_delay_ms=10),
    "vocal_pop": dict(bass_gain=1.0, presence_gain=3.0, air_gain=1.5,
                       comp_threshold=-14, comp_ratio=3.0, limiter_ceiling=-0.3, width=1.0,
                       target_crest_db=9.5, haas_mix=0.2, haas_delay_ms=13),
}

def master_chain(left: AudioBuffer, right: AudioBuffer,
                 bass_gain: float = 1.5, presence_gain: float = 1.0, air_gain: float = 0.5,
                 comp_threshold: float = -16, comp_ratio: float = 3.0,
                 limiter_ceiling: float = -0.3, target_peak: float = 0.98) -> Tuple[AudioBuffer, AudioBuffer]:
    for buf in (left, right):
        low_shelf(buf, freq=100, gain_db=bass_gain)
        eq_band(buf, freq=2500, gain_db=presence_gain, q=1.0)
        high_shelf(buf, freq=10000, gain_db=air_gain)
        compressor(buf, threshold_db=comp_threshold, ratio=comp_ratio,
                   attack=0.01, release=0.15, makeup_db=2.0)
        limiter(buf, ceiling_db=limiter_ceiling)
    peak = max(left.peak(), right.peak())
    if peak > 0:
        scale = target_peak / peak
        left.gain(scale)
        right.gain(scale)
    return left, right

def master_chain_advanced(left: AudioBuffer, right: AudioBuffer,
                          bass_gain: float = 1.5, presence_gain: float = 1.0, air_gain: float = 0.5,
                          comp_threshold: float = -16, comp_ratio: float = 3.0,
                          limiter_ceiling: float = -0.3, target_peak: float = 0.98,
                          width: float = 1.0, add_exciter: bool = False,
                          add_saturation: float = 0.0,
                          target_crest_db: Optional[float] = None,
                          haas_mix: float = 0.0, haas_delay_ms: float = 15) -> Tuple[AudioBuffer, AudioBuffer]:
    for buf in (left, right):
        low_shelf(buf, freq=100, gain_db=bass_gain)
        eq_band(buf, freq=2500, gain_db=presence_gain, q=1.0)
        high_shelf(buf, freq=10000, gain_db=air_gain)
        if add_saturation > 0:
            saturation(buf, amount=add_saturation)
        if add_exciter:
            exciter(buf)
        compressor(buf, threshold_db=comp_threshold, ratio=comp_ratio,
                   attack=0.01, release=0.15, makeup_db=2.0)
        limiter(buf, ceiling_db=limiter_ceiling)
    if width != 1.0:
        left, right = stereo_widener(left, right, width=width)
    if haas_mix > 0:
        left, right = haas_widen(left, right, delay_ms=haas_delay_ms, mix=haas_mix)
    if target_crest_db is not None:
        loudness_maximizer(left, target_crest_db=target_crest_db, ceiling_db=limiter_ceiling)
        loudness_maximizer(right, target_crest_db=target_crest_db, ceiling_db=limiter_ceiling)
    peak = max(left.peak(), right.peak())
    if peak > 0:
        scale = target_peak / peak
        left.gain(scale)
        right.gain(scale)
    return left, right

def master_with_preset(left: AudioBuffer, right: AudioBuffer, preset: str = "balanced",
                       **overrides) -> Tuple[AudioBuffer, AudioBuffer]:
    if preset not in MASTER_PRESETS:
        raise ValueError(f"unknown preset '{preset}'. available: {list(MASTER_PRESETS)}")
    params = dict(MASTER_PRESETS[preset])
    params.update(overrides)
    return master_chain_advanced(left, right, **params)