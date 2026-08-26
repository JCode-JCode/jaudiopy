import math
from array import array as _array
import numpy as np
from typing import List, Tuple, Optional, Union, Any, Callable
from .buffer import NpAudioBuffer

try:
    from scipy.signal import lfilter as _lfilter
    _HAVE_SCIPY = True
except ImportError:
    _lfilter = None
    _HAVE_SCIPY = False

_FAST_IIR_MIN_SAMPLES = 20000


def _phase_matrix(x, delay_len: int):
    n = len(x)
    n_chunks = -(-n // delay_len)
    padded = np.zeros(n_chunks * delay_len, dtype=np.float64)
    padded[:n] = x
    return padded.reshape(n_chunks, delay_len).T, n


def _unphase(mat, n: int):
    return mat.T.reshape(-1)[:n]


def _fast_comb(samples, delay_len: int, feedback: float):
    if not _HAVE_SCIPY or len(samples) < _FAST_IIR_MIN_SAMPLES or delay_len < 1:
        return None
    T, n = _phase_matrix(samples, delay_len)
    z = _lfilter([1.0], [1.0, -feedback], T, axis=1)
    out = np.zeros_like(z)
    out[:, 1:] = z[:, :-1]
    return _unphase(out, n)


def _fast_allpass(samples, delay_len: int, gain: float):
    if not _HAVE_SCIPY or len(samples) < _FAST_IIR_MIN_SAMPLES or delay_len < 1:
        return None
    T, n = _phase_matrix(samples, delay_len)
    z = _lfilter([1.0 - gain * gain], [1.0, -gain], T, axis=1)
    zshift = np.zeros_like(z)
    zshift[:, 1:] = z[:, :-1]
    y = -gain * T + zshift
    return _unphase(y, n)


def _fast_delay_line(dry, delay_len: int, feedback: float, mix: float):
    if not _HAVE_SCIPY or len(dry) < _FAST_IIR_MIN_SAMPLES or delay_len < 1:
        return None
    T, n = _phase_matrix(dry, delay_len)
    z = _lfilter([1.0], [1.0, -feedback], T, axis=1)
    zshift = np.zeros_like(z)
    zshift[:, 1:] = z[:, :-1]
    out = T + mix * zshift
    return _unphase(out, n)


def _zeros_d(n: int) -> _array:
    return _array('d', bytes(8 * n))


def _fast_list(arr: np.ndarray) -> _array:
    return _array('d', np.ascontiguousarray(arr, dtype=np.float64).tobytes())


class Biquad:
    def __init__(self, b0: float, b1: float, b2: float, a0: float, a1: float, a2: float):
        self.b0, self.b1, self.b2 = b0 / a0, b1 / a0, b2 / a0
        self.a1, self.a2 = a1 / a0, a2 / a0

    def coeffs(self) -> Tuple[float, float, float, float, float]:
        return self.b0, self.b1, self.b2, self.a1, self.a2

    @classmethod
    def lowpass(cls, freq: float, sr: int, q: float = 0.707) -> 'Biquad':
        w0 = 2 * math.pi * freq / sr
        alpha = math.sin(w0) / (2 * q)
        cosw0 = math.cos(w0)
        b0 = (1 - cosw0) / 2
        b1 = 1 - cosw0
        b2 = (1 - cosw0) / 2
        return cls(b0, b1, b2, 1 + alpha, -2 * cosw0, 1 - alpha)

    @classmethod
    def highpass(cls, freq: float, sr: int, q: float = 0.707) -> 'Biquad':
        w0 = 2 * math.pi * freq / sr
        alpha = math.sin(w0) / (2 * q)
        cosw0 = math.cos(w0)
        b0 = (1 + cosw0) / 2
        b1 = -(1 + cosw0)
        b2 = (1 + cosw0) / 2
        return cls(b0, b1, b2, 1 + alpha, -2 * cosw0, 1 - alpha)

    @classmethod
    def peaking(cls, freq: float, sr: int, gain_db: float, q: float = 1.0) -> 'Biquad':
        A = 10 ** (gain_db / 40)
        w0 = 2 * math.pi * freq / sr
        alpha = math.sin(w0) / (2 * q)
        cosw0 = math.cos(w0)
        b0 = 1 + alpha * A
        b1 = -2 * cosw0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cosw0
        a2 = 1 - alpha / A
        return cls(b0, b1, b2, a0, a1, a2)

    @classmethod
    def lowshelf(cls, freq: float, sr: int, gain_db: float, q: float = 0.707) -> 'Biquad':
        A = 10 ** (gain_db / 40)
        w0 = 2 * math.pi * freq / sr
        alpha = math.sin(w0) / (2 * q)
        cosw0 = math.cos(w0)
        sq = 2 * math.sqrt(A) * alpha
        b0 = A * ((A + 1) - (A - 1) * cosw0 + sq)
        b1 = 2 * A * ((A - 1) - (A + 1) * cosw0)
        b2 = A * ((A + 1) - (A - 1) * cosw0 - sq)
        a0 = (A + 1) + (A - 1) * cosw0 + sq
        a1 = -2 * ((A - 1) + (A + 1) * cosw0)
        a2 = (A + 1) + (A - 1) * cosw0 - sq
        return cls(b0, b1, b2, a0, a1, a2)

    @classmethod
    def highshelf(cls, freq: float, sr: int, gain_db: float, q: float = 0.707) -> 'Biquad':
        A = 10 ** (gain_db / 40)
        w0 = 2 * math.pi * freq / sr
        alpha = math.sin(w0) / (2 * q)
        cosw0 = math.cos(w0)
        sq = 2 * math.sqrt(A) * alpha
        b0 = A * ((A + 1) + (A - 1) * cosw0 + sq)
        b1 = -2 * A * ((A - 1) + (A + 1) * cosw0)
        b2 = A * ((A + 1) + (A - 1) * cosw0 - sq)
        a0 = (A + 1) - (A - 1) * cosw0 + sq
        a1 = 2 * ((A - 1) - (A + 1) * cosw0)
        a2 = (A + 1) - (A - 1) * cosw0 - sq
        return cls(b0, b1, b2, a0, a1, a2)


def _apply_biquad_np(x: np.ndarray, biq: Biquad) -> np.ndarray:
    b0, b1, b2, a1, a2 = biq.coeffs()
    n = len(x)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if _HAVE_SCIPY:
        return _lfilter([b0, b1, b2], [1.0, a1, a2], x)
    xl = _fast_list(x)
    out = _zeros_d(n)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(n):
        xi = xl[i]
        yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[i] = yi
        x2, x1 = x1, xi
        y2, y1 = y1, yi
    return np.frombuffer(out, dtype=np.float64).copy()


def apply_filter(buf: NpAudioBuffer, biquad: Biquad) -> NpAudioBuffer:
    buf.samples = _apply_biquad_np(buf.samples, biquad)
    return buf


def lowpass(buf: NpAudioBuffer, freq: float, q: float = 0.707) -> NpAudioBuffer:
    return apply_filter(buf, Biquad.lowpass(freq, buf.sr, q))


def highpass(buf: NpAudioBuffer, freq: float, q: float = 0.707) -> NpAudioBuffer:
    return apply_filter(buf, Biquad.highpass(freq, buf.sr, q))


def eq_band(buf: NpAudioBuffer, freq: float, gain_db: float, q: float = 1.0) -> NpAudioBuffer:
    return apply_filter(buf, Biquad.peaking(freq, buf.sr, gain_db, q))


def low_shelf(buf: NpAudioBuffer, freq: float, gain_db: float, q: float = 0.707) -> NpAudioBuffer:
    return apply_filter(buf, Biquad.lowshelf(freq, buf.sr, gain_db, q))


def high_shelf(buf: NpAudioBuffer, freq: float, gain_db: float, q: float = 0.707) -> NpAudioBuffer:
    return apply_filter(buf, Biquad.highshelf(freq, buf.sr, gain_db, q))


def distortion(buf: NpAudioBuffer, drive: float = 5.0, mix: float = 1.0) -> NpAudioBuffer:
    wet = np.tanh(buf.samples * drive)
    buf.samples = wet * mix + buf.samples * (1 - mix)
    return buf


def bitcrush(buf: NpAudioBuffer, bit_depth: int = 8, downsample: int = 1) -> NpAudioBuffer:
    if downsample < 1:
        raise ValueError("'downsample' must be >= 1")
    levels = 2 ** bit_depth
    x = buf.samples
    if downsample > 1 and len(x) > 0:
        idx = (np.arange(len(x)) // downsample) * downsample
        idx = np.clip(idx, 0, len(x) - 1)
        x = x[idx]
    buf.samples = np.round(x * levels) / levels
    return buf


def delay(buf: NpAudioBuffer, time_sec: float = 0.3, feedback: float = 0.4,
          mix: float = 0.35, repeats_tail: int = 3) -> NpAudioBuffer:
    delay_samples = max(int(time_sec * buf.sr), 1)
    tail = delay_samples * repeats_tail
    dry_len = len(buf.samples)
    n = dry_len + tail
    dry_source = np.zeros(n, dtype=np.float64)
    dry_source[:dry_len] = buf.samples
    fast = _fast_delay_line(dry_source, delay_samples, feedback, mix)
    if fast is not None:
        buf.samples = np.ascontiguousarray(fast, dtype=np.float64)
        return buf
    dry = _fast_list(dry_source)
    out = _zeros_d(n)
    line = [0.0] * delay_samples
    idx = 0
    for i in range(n):
        d = dry[i]
        delayed = line[idx]
        out[i] = d + delayed * mix
        line[idx] = d + delayed * feedback
        idx += 1
        if idx == delay_samples:
            idx = 0
    buf.samples = np.frombuffer(out, dtype=np.float64).copy()
    return buf


def _comb_filter(x, delay_len: int, feedback: float):
    fast = _fast_comb(x, delay_len, feedback)
    if fast is not None:
        return fast
    xl = _fast_list(x) if isinstance(x, np.ndarray) else x
    n = len(xl)
    line = [0.0] * delay_len
    out = [0.0] * n
    idx = 0
    for i in range(n):
        y = line[idx]
        out[i] = y
        line[idx] = xl[i] + y * feedback
        idx += 1
        if idx == delay_len:
            idx = 0
    return out


def _allpass_filter(x, delay_len: int, gain: float = 0.5):
    fast = _fast_allpass(x, delay_len, gain)
    if fast is not None:
        return fast
    xl = _fast_list(x) if isinstance(x, np.ndarray) else x
    n = len(xl)
    line = [0.0] * delay_len
    out = [0.0] * n
    idx = 0
    for i in range(n):
        buffered = line[idx]
        y = -gain * xl[i] + buffered
        line[idx] = xl[i] + gain * y
        out[i] = y
        idx += 1
        if idx == delay_len:
            idx = 0
    return out


def reverb(buf: NpAudioBuffer, room_size: float = 0.6, damping: float = 0.5,
           mix: float = 0.3) -> NpAudioBuffer:
    comb_delays_ms = [29.7, 37.1, 41.1, 43.7]
    feedback = 0.28 + room_size * 0.35
    dry_list = _fast_list(buf.samples)
    n = len(dry_list)
    wet = np.zeros(n, dtype=np.float64)
    n_combs = len(comb_delays_ms)
    for ms in comb_delays_ms:
        d = max(int(buf.sr * ms / 1000), 1)
        comb_out = _comb_filter(dry_list, d, feedback * (1 - damping * 0.3))
        wet += np.asarray(comb_out, dtype=np.float64) / n_combs
    for ms in (5.0, 1.7):
        d = max(int(buf.sr * ms / 1000), 1)
        wet = _allpass_filter(wet, d, 0.5)
    wet_arr = np.asarray(wet, dtype=np.float64)
    buf.samples = buf.samples * (1 - mix) + wet_arr * mix
    return buf


def chorus(buf: NpAudioBuffer, rate: float = 1.5, depth_ms: float = 3.0,
           mix: float = 0.5) -> NpAudioBuffer:
    sr = buf.sr
    n = len(buf.samples)
    if n == 0:
        return buf
    i = np.arange(n)
    lfo = (np.sin(2 * np.pi * rate * i / sr) + 1) / 2
    delay_samples = depth_ms / 1000 * sr * lfo
    read_pos = i - delay_samples
    i_ext = np.concatenate(([-1], i))
    samples_ext = np.concatenate(([0.0], buf.samples))
    delayed = np.interp(read_pos, i_ext, samples_ext, left=0.0, right=0.0)
    buf.samples = buf.samples * (1 - mix) + delayed * mix
    return buf


def tremolo(buf: NpAudioBuffer, rate: float = 5.0, depth: float = 0.5) -> NpAudioBuffer:
    sr = buf.sr
    i = np.arange(len(buf.samples))
    lfo = 1 - depth * (0.5 + 0.5 * np.sin(2 * np.pi * rate * i / sr))
    buf.samples = buf.samples * lfo
    return buf


def vibrato(buf: NpAudioBuffer, rate: float = 5.0, depth_ms: float = 4.0) -> NpAudioBuffer:
    sr = buf.sr
    n = len(buf.samples)
    if n == 0:
        return buf
    i = np.arange(n)
    lfo = np.sin(2 * np.pi * rate * i / sr)
    delay_samples = depth_ms / 1000 * sr * (0.5 + 0.5 * lfo)
    read_pos = i - delay_samples
    i_ext = np.concatenate(([-1], i))
    samples_ext = np.concatenate(([0.0], buf.samples))
    buf.samples = np.interp(read_pos, i_ext, samples_ext, left=0.0, right=0.0)
    return buf


def phaser(buf: NpAudioBuffer, rate: float = 0.5, depth: float = 0.7, stages: int = 4,
           mix: float = 0.5) -> NpAudioBuffer:
    sr = buf.sr
    x = buf.samples
    n = len(x)
    if n == 0:
        return buf
    i_arr = np.arange(n)
    lfo = (np.sin(2 * np.pi * rate * i_arr / sr) + 1) / 2
    freq = 300 + depth * lfo * 3000
    tan_val = np.tan(np.pi * freq / sr)
    a_arr = (tan_val - 1) / (tan_val + 1)
    xl = _fast_list(x)
    al = _fast_list(a_arr)
    out = _zeros_d(n)
    stage_states = [0.0] * stages
    for i in range(n):
        xi = xl[i]
        a = al[i]
        wet = xi
        for s in range(stages):
            y = a * wet + stage_states[s]
            stage_states[s] = wet - a * y
            wet = y
        out[i] = xi * (1 - mix) + wet * mix
    buf.samples = np.frombuffer(out, dtype=np.float64).copy()
    return buf


def autopan(buf: NpAudioBuffer, rate: float = 1.0, depth: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    sr = buf.sr
    n = len(buf.samples)
    i = np.arange(n)
    lfo = np.sin(2 * np.pi * rate * i / sr) * depth
    angle = (lfo + 1) * np.pi / 4
    left = buf.samples * np.cos(angle)
    right = buf.samples * np.sin(angle)
    return left, right


def saturation(buf: NpAudioBuffer, amount: float = 0.3, mix: float = 1.0) -> NpAudioBuffer:
    drive = 1 + amount * 4
    wet = np.tanh(buf.samples * drive) / math.tanh(drive)
    buf.samples = wet * mix + buf.samples * (1 - mix)
    return buf


def wah(buf: NpAudioBuffer, rate: float = 2.0, min_freq: float = 400,
        max_freq: float = 2000, q: float = 3.0) -> NpAudioBuffer:
    sr = buf.sr
    x = buf.samples
    n = len(x)
    if n == 0:
        return buf
    i_arr = np.arange(n)
    lfo = (np.sin(2 * np.pi * rate * i_arr / sr) + 1) / 2
    freq = min_freq + (max_freq - min_freq) * lfo
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cosw0 = np.cos(w0)
    b0_arr = alpha
    a0_arr = 1 + alpha
    a1_arr = -2 * cosw0
    a2_arr = 1 - alpha
    xl = _fast_list(x)
    b0l = _fast_list(b0_arr)
    a0l = _fast_list(a0_arr)
    a1l = _fast_list(a1_arr)
    a2l = _fast_list(a2_arr)
    out = _zeros_d(n)
    x1 = x2 = y1 = y2 = 0.0
    for i in range(n):
        xi = xl[i]
        b0 = b0l[i]
        y = (b0 * xi - b0 * x2 - a1l[i] * y1 - a2l[i] * y2) / a0l[i]
        out[i] = y
        x2, x1 = x1, xi
        y2, y1 = y1, y
    buf.samples = np.frombuffer(out, dtype=np.float64).copy()
    return buf


def exciter(buf: NpAudioBuffer, freq: float = 3000, amount: float = 0.3) -> NpAudioBuffer:
    bright = buf.copy()
    high_shelf(bright, freq=freq, gain_db=6.0)
    saturation(bright, amount=0.5)
    buf.samples = buf.samples + bright.samples * amount
    return buf


def noise_gate(buf: NpAudioBuffer, threshold_db: float = -40, attack: float = 0.002,
               release: float = 0.15) -> NpAudioBuffer:
    sr = buf.sr
    threshold = 10 ** (threshold_db / 20)
    attack_coef = math.exp(-1 / (sr * attack))
    release_coef = math.exp(-1 / (sr * release))
    one_minus_release = 1 - release_coef
    xl = _fast_list(buf.samples)
    n = len(xl)
    out = _zeros_d(n)
    env = 0.0
    gain = 0.0
    for i in range(n):
        xi = xl[i]
        rectified = xi if xi >= 0 else -xi
        env = env * release_coef + rectified * one_minus_release if rectified < env else rectified
        target = 1.0 if env > threshold else 0.0
        coef = attack_coef if target > gain else release_coef
        gain = gain * coef + target * (1 - coef)
        out[i] = xi * gain
    buf.samples = np.frombuffer(out, dtype=np.float64).copy()
    return buf


def compressor(buf: NpAudioBuffer, threshold_db: float = -18, ratio: float = 4.0,
               attack: float = 0.005, release: float = 0.1, makeup_db: float = 0.0) -> NpAudioBuffer:
    sr = buf.sr
    threshold = 10 ** (threshold_db / 20)
    attack_coef = math.exp(-1 / (sr * attack))
    release_coef = math.exp(-1 / (sr * release))
    one_minus_attack = 1 - attack_coef
    one_minus_release = 1 - release_coef
    makeup = 10 ** (makeup_db / 20)
    k = 1 - 1 / ratio
    xl = _fast_list(buf.samples)
    n = len(xl)
    out = _zeros_d(n)
    env = 0.0
    for i in range(n):
        xi = xl[i]
        rectified = xi if xi >= 0 else -xi
        if rectified > env:
            env = attack_coef * env + one_minus_attack * rectified
        else:
            env = release_coef * env + one_minus_release * rectified
        if env > threshold:
            gain = (threshold / env) ** k
        else:
            gain = 1.0
        out[i] = xi * gain * makeup
    buf.samples = np.frombuffer(out, dtype=np.float64).copy()
    return buf


def limiter(buf: NpAudioBuffer, ceiling_db: float = -0.3, release: float = 0.05) -> NpAudioBuffer:
    ceiling = 10 ** (ceiling_db / 20)
    sr = buf.sr
    release_coef = math.exp(-1 / (sr * release))
    one_minus_release = 1 - release_coef
    xl = _fast_list(buf.samples)
    n = len(xl)
    out = _zeros_d(n)
    gain = 1.0
    for i in range(n):
        xi = xl[i]
        ax = xi if xi >= 0 else -xi
        target_gain = ceiling / ax if ax > ceiling else 1.0
        if target_gain < gain:
            gain = target_gain
        else:
            gain = gain * release_coef + target_gain * one_minus_release
        out[i] = xi * gain
    buf.samples = np.frombuffer(out, dtype=np.float64).copy()
    return buf


def pan_stereo(buf: NpAudioBuffer, pan: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    angle = (pan + 1) * math.pi / 4
    left = buf.samples * math.cos(angle)
    right = buf.samples * math.sin(angle)
    return left, right


def sidechain(buf: NpAudioBuffer, pattern: str, bpm: float, steps_per_beat: int = 4,
              depth: float = 0.6, release: float = 0.2) -> NpAudioBuffer:
    sr = buf.sr
    step_dur = 60 / bpm / steps_per_beat
    xl = _fast_list(buf.samples)
    n = len(xl)
    release_coef = math.exp(-1 / (sr * release))
    one_minus_release = 1 - release_coef
    trigger_flags = bytearray(n)
    for i, ch in enumerate(pattern):
        if ch != ".":
            pos = int(i * step_dur * sr)
            if 0 <= pos < n:
                trigger_flags[pos] = 1
    out = _zeros_d(n)
    gain = 1.0
    for i in range(n):
        if trigger_flags[i]:
            gain = 1 - depth
        else:
            gain = gain + (1 - gain) * one_minus_release
        out[i] = xl[i] * gain
    buf.samples = np.frombuffer(out, dtype=np.float64).copy()
    return buf


def de_esser(buf: NpAudioBuffer, freq: float = 6500, threshold_db: float = -26,
             ratio: float = 4.0, q: float = 1.0) -> NpAudioBuffer:
    high = buf.copy()
    highpass(high, freq=freq, q=q)
    compressor(high, threshold_db=threshold_db, ratio=ratio, attack=0.001, release=0.05)
    low = buf.copy()
    lowpass(low, freq=freq, q=q)
    buf.samples = low.samples + high.samples
    return buf


def duck_under(buf: NpAudioBuffer, trigger: NpAudioBuffer, depth: float = 0.6,
               attack: float = 0.01, release: float = 0.15, threshold: float = 0.05) -> NpAudioBuffer:
    sr = buf.sr
    xl = _fast_list(buf.samples)
    trig = trigger.samples
    n_trigger = len(trig)
    n = len(xl)
    attack_coef = math.exp(-1 / (sr * attack))
    release_coef = math.exp(-1 / (sr * release))
    one_minus_attack = 1 - attack_coef
    one_minus_release = 1 - release_coef
    peak = float(np.max(np.abs(trig))) if n_trigger else 1.0
    peak = peak or 1.0
    trig_l = _fast_list(np.abs(trig) / peak) if n_trigger else _array('d')
    out = _zeros_d(n)
    env = 0.0
    for i in range(n):
        t = trig_l[i] if i < n_trigger else 0.0
        if t > env:
            env = attack_coef * env + one_minus_attack * t
        else:
            env = release_coef * env + one_minus_release * t
        active = 1.0 if env > threshold else env / threshold
        gain = 1 - depth * active
        out[i] = xl[i] * gain
    buf.samples = np.frombuffer(out, dtype=np.float64).copy()
    return buf


def stereo_widener(left: NpAudioBuffer, right: NpAudioBuffer,
                   width: float = 1.3) -> Tuple[NpAudioBuffer, NpAudioBuffer]:
    n = min(len(left.samples), len(right.samples))
    l, r = left.samples[:n], right.samples[:n]
    mid = (l + r) / 2
    side = (l - r) / 2 * width
    new_left = mid + side
    new_right = mid - side
    return NpAudioBuffer(new_left, left.sr), NpAudioBuffer(new_right, left.sr)


def haas_widen(left: NpAudioBuffer, right: NpAudioBuffer, delay_ms: float = 15,
               mix: float = 0.35) -> Tuple[NpAudioBuffer, NpAudioBuffer]:
    sr = left.sr
    delay_samples = max(int(delay_ms * sr / 1000), 1)
    n = min(len(left.samples), len(right.samples))
    r = right.samples[:n]
    delayed_right = np.zeros(n, dtype=np.float64)
    if delay_samples < n:
        delayed_right[delay_samples:] = r[:n - delay_samples]
    new_left = left.samples[:n].copy()
    new_right = r * (1 - mix) + delayed_right * mix
    return NpAudioBuffer(new_left, sr), NpAudioBuffer(new_right, sr)


def loudness_maximizer(buf: NpAudioBuffer, target_crest_db: float = 10.0,
                       ceiling_db: float = -0.3, iterations: int = 8,
                       drive_step_db: float = 1.5) -> NpAudioBuffer:
    ceiling = 10 ** (ceiling_db / 20)
    prev_crest_db = None
    for _ in range(iterations):
        n = len(buf.samples)
        if n == 0:
            break
        peak = buf.peak()
        if peak <= 0:
            break
        rms = float(np.sqrt(np.mean(buf.samples ** 2)))
        if rms <= 0:
            break
        crest_db = 20 * math.log10(peak) - 20 * math.log10(rms)
        if crest_db <= target_crest_db:
            break
        if prev_crest_db is not None and crest_db >= prev_crest_db - 0.05:
            break
        prev_crest_db = crest_db
        buf.gain_db(drive_step_db)
        limiter(buf, ceiling_db=ceiling_db, release=0.05)
    m = buf.peak()
    if m > 0:
        buf.gain(ceiling / m)
    return buf
