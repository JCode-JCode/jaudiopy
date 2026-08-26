import math
from array import array
from typing import List, Tuple, Optional, Union, Any, Callable
from .buffer import AudioBuffer

try:
    import numpy as _np
    try:
        from scipy.signal import lfilter as _lfilter
        _HAVE_FAST_IIR = True
    except ImportError:
        _lfilter = None
        _HAVE_FAST_IIR = False
except ImportError:
    _np = None
    _lfilter = None
    _HAVE_FAST_IIR = False

_FAST_IIR_MIN_SAMPLES = 20000


def _phase_matrix(x: List[float], delay_len: int):
    n = len(x)
    n_chunks = -(-n // delay_len)
    padded = _np.zeros(n_chunks * delay_len, dtype=_np.float64)
    padded[:n] = x
    return padded.reshape(n_chunks, delay_len).T, n


def _unphase(mat, n: int):
    return mat.T.reshape(-1)[:n]


def _fast_comb(samples: List[float], delay_len: int, feedback: float) -> Optional['_np.ndarray']:
    if not _HAVE_FAST_IIR or len(samples) < _FAST_IIR_MIN_SAMPLES or delay_len < 1:
        return None
    T, n = _phase_matrix(samples, delay_len)
    z = _lfilter([1.0], [1.0, -feedback], T, axis=1)
    out = _np.zeros_like(z)
    out[:, 1:] = z[:, :-1]
    return _unphase(out, n)


def _fast_allpass(samples: List[float], delay_len: int, gain: float) -> Optional['_np.ndarray']:
    if not _HAVE_FAST_IIR or len(samples) < _FAST_IIR_MIN_SAMPLES or delay_len < 1:
        return None
    T, n = _phase_matrix(samples, delay_len)
    z = _lfilter([1.0 - gain * gain], [1.0, -gain], T, axis=1)
    zshift = _np.zeros_like(z)
    zshift[:, 1:] = z[:, :-1]
    y = -gain * T + zshift
    return _unphase(y, n)


def _fast_delay_line(dry: List[float], delay_len: int, feedback: float,
                      mix: float) -> Optional['_np.ndarray']:
    if not _HAVE_FAST_IIR or len(dry) < _FAST_IIR_MIN_SAMPLES or delay_len < 1:
        return None
    T, n = _phase_matrix(dry, delay_len)
    z = _lfilter([1.0], [1.0, -feedback], T, axis=1)
    zshift = _np.zeros_like(z)
    zshift[:, 1:] = z[:, :-1]
    out = T + mix * zshift
    return _unphase(out, n)


class Biquad:
    def __init__(self, b0: float, b1: float, b2: float, a0: float, a1: float, a2: float):
        self.b0, self.b1, self.b2 = b0 / a0, b1 / a0, b2 / a0
        self.a1, self.a2 = a1 / a0, a2 / a0
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def process(self, x: float) -> float:
        y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 - self.a1 * self.y1 - self.a2 * self.y2
        self.x2, self.x1 = self.x1, x
        self.y2, self.y1 = self.y1, y
        return y

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

def apply_filter(buf: AudioBuffer, biquad: Biquad) -> AudioBuffer:
    n = len(buf.samples)
    b0, b1, b2, a1, a2 = biquad.b0, biquad.b1, biquad.b2, biquad.a1, biquad.a2
    if (_HAVE_FAST_IIR and n >= _FAST_IIR_MIN_SAMPLES
            and biquad.x1 == 0.0 and biquad.x2 == 0.0
            and biquad.y1 == 0.0 and biquad.y2 == 0.0):
        x_arr = _np.frombuffer(buf.samples, dtype=_np.float64)
        y_arr = _lfilter([b0, b1, b2], [1.0, a1, a2], x_arr)
        biquad.x2, biquad.x1 = buf.samples[-2], buf.samples[-1]
        biquad.y2, biquad.y1 = float(y_arr[-2]), float(y_arr[-1])
        buf.samples = array('d', y_arr.tobytes())
        return buf
    out = array('d', bytes(8 * n))
    x1 = biquad.x1
    x2 = biquad.x2
    y1 = biquad.y1
    y2 = biquad.y2
    for i, x in enumerate(buf.samples):
        y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        out[i] = y
        x2, x1 = x1, x
        y2, y1 = y1, y
    biquad.x1 = x1
    biquad.x2 = x2
    biquad.y1 = y1
    biquad.y2 = y2
    buf.samples = out
    return buf

def lowpass(buf: AudioBuffer, freq: float, q: float = 0.707) -> AudioBuffer:
    return apply_filter(buf, Biquad.lowpass(freq, buf.sr, q))

def highpass(buf: AudioBuffer, freq: float, q: float = 0.707) -> AudioBuffer:
    return apply_filter(buf, Biquad.highpass(freq, buf.sr, q))

def eq_band(buf: AudioBuffer, freq: float, gain_db: float, q: float = 1.0) -> AudioBuffer:
    return apply_filter(buf, Biquad.peaking(freq, buf.sr, gain_db, q))

def low_shelf(buf: AudioBuffer, freq: float, gain_db: float, q: float = 0.707) -> AudioBuffer:
    return apply_filter(buf, Biquad.lowshelf(freq, buf.sr, gain_db, q))

def high_shelf(buf: AudioBuffer, freq: float, gain_db: float, q: float = 0.707) -> AudioBuffer:
    return apply_filter(buf, Biquad.highshelf(freq, buf.sr, gain_db, q))

def distortion(buf: AudioBuffer, drive: float = 5.0, mix: float = 1.0) -> AudioBuffer:
    if _np is not None and len(buf.samples) > 0:
        x = _np.frombuffer(buf.samples, dtype=_np.float64)
        wet = _np.tanh(x * drive)
        buf.samples = array('d', (wet * mix + x * (1 - mix)).tobytes())
    else:
        wet = [math.tanh(s * drive) for s in buf.samples]
        buf.samples = array('d', (w * mix + s * (1 - mix) for w, s in zip(wet, buf.samples)))
    return buf

def bitcrush(buf: AudioBuffer, bit_depth: int = 8, downsample: int = 1) -> AudioBuffer:
    if downsample < 1:
        raise ValueError("'downsample' must be >= 1")
    levels = 2 ** bit_depth
    if _np is not None and len(buf.samples) > 0:
        x = _np.frombuffer(buf.samples, dtype=_np.float64)
        if downsample > 1:
            idx = (_np.arange(len(x)) // downsample) * downsample
            idx = _np.clip(idx, 0, len(x) - 1)
            x = x[idx]
        buf.samples = array('d', (_np.round(x * levels) / levels).tobytes())
    else:
        out = []
        held = 0.0
        for i, s in enumerate(buf.samples):
            if i % downsample == 0:
                held = round(s * levels) / levels
            out.append(held)
        buf.samples = array('d', out)
    return buf

def delay(buf: AudioBuffer, time_sec: float = 0.3, feedback: float = 0.4,
          mix: float = 0.35, repeats_tail: int = 3) -> AudioBuffer:
    delay_samples = max(int(time_sec * buf.sr), 1)
    tail = delay_samples * repeats_tail
    dry_len = len(buf.samples)
    n_total = dry_len + tail
    dry_source = array('d', buf.samples)
    dry_source.extend([0.0] * tail)
    fast = _fast_delay_line(dry_source, delay_samples, feedback, mix)
    if fast is not None:
        buf.samples = array('d', _np.ascontiguousarray(fast, dtype=_np.float64).tobytes())
        return buf
    out = dry_source
    line = [0.0] * delay_samples
    idx = 0
    for i in range(n_total):
        dry = out[i] if i < dry_len else 0.0
        delayed = line[idx]
        out[i] = dry + delayed * mix
        line[idx] = dry + delayed * feedback
        idx = (idx + 1) % delay_samples
    buf.samples = out
    return buf

def _comb_filter(samples: List[float], delay_len: int, feedback: float) -> List[float]:
    fast = _fast_comb(samples, delay_len, feedback)
    if fast is not None:
        return fast
    line = [0.0] * delay_len
    out = [0.0] * len(samples)
    idx = 0
    for i, x in enumerate(samples):
        y = line[idx]
        out[i] = y
        line[idx] = x + y * feedback
        idx = (idx + 1) % delay_len
    return out

def _allpass_filter(samples: List[float], delay_len: int, gain: float = 0.5) -> List[float]:
    fast = _fast_allpass(samples, delay_len, gain)
    if fast is not None:
        return fast
    line = [0.0] * delay_len
    out = [0.0] * len(samples)
    idx = 0
    for i, x in enumerate(samples):
        buffered = line[idx]
        y = -gain * x + buffered
        line[idx] = x + gain * y
        out[i] = y
        idx = (idx + 1) % delay_len
    return out

def reverb(buf: AudioBuffer, room_size: float = 0.6, damping: float = 0.5, mix: float = 0.3) -> AudioBuffer:
    comb_delays_ms = [29.7, 37.1, 41.1, 43.7]
    feedback = 0.28 + room_size * 0.35
    n = len(buf.samples)
    if _np is not None and n > 0:
        wet_acc = _np.zeros(n, dtype=_np.float64)
        for ms in comb_delays_ms:
            d = max(int(buf.sr * ms / 1000), 1)
            comb_out = _comb_filter(buf.samples, d, feedback * (1 - damping * 0.3))
            wet_acc += _np.asarray(comb_out, dtype=_np.float64) / len(comb_delays_ms)
        wet = array('d', wet_acc.tobytes())
    else:
        wet = [0.0] * n
        for ms in comb_delays_ms:
            d = max(int(buf.sr * ms / 1000), 1)
            comb_out = _comb_filter(buf.samples, d, feedback * (1 - damping * 0.3))
            for i in range(len(wet)):
                wet[i] += comb_out[i] / len(comb_delays_ms)
    for ms in (5.0, 1.7):
        d = max(int(buf.sr * ms / 1000), 1)
        wet = _allpass_filter(wet, d, 0.5)
    if _np is not None and n > 0:
        wet_np = _np.asarray(wet, dtype=_np.float64)
        dry_np = _np.frombuffer(buf.samples, dtype=_np.float64)
        buf.samples = array('d', (dry_np * (1 - mix) + wet_np * mix).tobytes())
    else:
        buf.samples = array('d', (dry * (1 - mix) + w * mix for dry, w in zip(buf.samples, wet)))
    return buf

def chorus(buf: AudioBuffer, rate: float = 1.5, depth_ms: float = 3.0, mix: float = 0.5) -> AudioBuffer:
    sr = buf.sr
    n = len(buf.samples)
    if _np is not None and n > 0:
        i = _np.arange(n, dtype=_np.float64)
        lfo = (_np.sin(2 * _np.pi * rate * i / sr) + 1) / 2
        delay_samples = depth_ms / 1000 * sr * lfo
        read_pos = i - delay_samples
        i_ext = _np.concatenate(([-1], i))
        x = _np.frombuffer(buf.samples, dtype=_np.float64)
        samples_ext = _np.concatenate(([0.0], x))
        delayed = _np.interp(read_pos, i_ext, samples_ext, left=0.0, right=0.0)
        buf.samples = array('d', (x * (1 - mix) + delayed * mix).tobytes())
        return buf
    max_delay = int(depth_ms * 2 * sr / 1000) + 2
    history = [0.0] * max_delay
    out = array('d', bytes(8 * n))
    for i, x in enumerate(buf.samples):
        history[i % max_delay] = x
        lfo = (math.sin(2 * math.pi * rate * i / sr) + 1) / 2
        delay_samples = depth_ms / 1000 * sr * lfo
        read_pos = i - delay_samples
        idx0 = math.floor(read_pos) % max_delay
        idx1 = (idx0 + 1) % max_delay
        frac = read_pos - math.floor(read_pos)
        delayed = history[idx0] * (1 - frac) + history[idx1] * frac
        out[i] = x * (1 - mix) + delayed * mix
    buf.samples = out
    return buf

def tremolo(buf: AudioBuffer, rate: float = 5.0, depth: float = 0.5) -> AudioBuffer:
    sr = buf.sr
    n = len(buf.samples)
    if _np is not None and n > 0:
        i = _np.arange(n, dtype=_np.float64)
        lfo = 1 - depth * (0.5 + 0.5 * _np.sin(2 * _np.pi * rate * i / sr))
        x = _np.frombuffer(buf.samples, dtype=_np.float64)
        buf.samples = array('d', (x * lfo).tobytes())
    else:
        out = array('d', [0.0] * n)
        for i, x in enumerate(buf.samples):
            lfo = 1 - depth * (0.5 + 0.5 * math.sin(2 * math.pi * rate * i / sr))
            out[i] = x * lfo
        buf.samples = out
    return buf

def compressor(buf: AudioBuffer, threshold_db: float = -18, ratio: float = 4.0,
               attack: float = 0.005, release: float = 0.1, makeup_db: float = 0.0) -> AudioBuffer:
    sr = buf.sr
    threshold = 10 ** (threshold_db / 20)
    attack_coef = math.exp(-1 / (sr * attack))
    release_coef = math.exp(-1 / (sr * release))
    makeup = 10 ** (makeup_db / 20)
    one_minus_attack = 1 - attack_coef
    one_minus_release = 1 - release_coef
    k = 1 - 1 / ratio
    env = 0.0
    n = len(buf.samples)
    out = array('d', bytes(8 * n))
    for i, x in enumerate(buf.samples):
        rectified = x if x >= 0 else -x
        if rectified > env:
            env = attack_coef * env + one_minus_attack * rectified
        else:
            env = release_coef * env + one_minus_release * rectified
        if env > threshold:
            gain = (threshold / env) ** k
        else:
            gain = 1.0
        out[i] = x * gain * makeup
    buf.samples = out
    return buf

def limiter(buf: AudioBuffer, ceiling_db: float = -0.3, release: float = 0.05) -> AudioBuffer:
    ceiling = 10 ** (ceiling_db / 20)
    sr = buf.sr
    release_coef = math.exp(-1 / (sr * release))
    one_minus_release = 1 - release_coef
    gain = 1.0
    n = len(buf.samples)
    out = array('d', bytes(8 * n))
    for i, x in enumerate(buf.samples):
        ax = x if x >= 0 else -x
        target_gain = ceiling / ax if ax > ceiling else 1.0
        if target_gain < gain:
            gain = target_gain
        else:
            gain = gain * release_coef + target_gain * one_minus_release
        out[i] = x * gain
    buf.samples = out
    return buf

def _pan_gains_np(view: '_np.ndarray', pan: float):
    angle = (pan + 1) * math.pi / 4
    return view * math.cos(angle), view * math.sin(angle)

def pan_stereo(buf: AudioBuffer, pan: float = 0.0) -> Tuple[List[float], List[float]]:
    if _np is not None and len(buf.samples) > 0:
        view = _np.frombuffer(buf.samples, dtype=_np.float64)
        left, right = _pan_gains_np(view, pan)
        return left.tolist(), right.tolist()
    angle = (pan + 1) * math.pi / 4
    left_gain = math.cos(angle)
    right_gain = math.sin(angle)
    left = [s * left_gain for s in buf.samples]
    right = [s * right_gain for s in buf.samples]
    return left, right

def vibrato(buf: AudioBuffer, rate: float = 5.0, depth_ms: float = 4.0) -> AudioBuffer:
    sr = buf.sr
    n = len(buf.samples)
    if _np is not None and n > 0:
        i = _np.arange(n, dtype=_np.float64)
        lfo = _np.sin(2 * _np.pi * rate * i / sr)
        delay_samples = depth_ms / 1000 * sr * (0.5 + 0.5 * lfo)
        read_pos = i - delay_samples
        i_ext = _np.concatenate(([-1], i))
        x = _np.frombuffer(buf.samples, dtype=_np.float64)
        samples_ext = _np.concatenate(([0.0], x))
        buf.samples = array('d', _np.interp(read_pos, i_ext, samples_ext, left=0.0, right=0.0).tobytes())
        return buf
    max_delay = int(depth_ms * 2 * sr / 1000) + 2
    history = [0.0] * max_delay
    out = array('d', [0.0] * n)
    for i, x in enumerate(buf.samples):
        history[i % max_delay] = x
        lfo = math.sin(2 * math.pi * rate * i / sr)
        delay_samples = depth_ms / 1000 * sr * (0.5 + 0.5 * lfo)
        read_pos = i - delay_samples
        idx0 = math.floor(read_pos) % max_delay
        idx1 = (idx0 + 1) % max_delay
        frac = read_pos - math.floor(read_pos)
        out[i] = history[idx0] * (1 - frac) + history[idx1] * frac
    buf.samples = out
    return buf

def phaser(buf: AudioBuffer, rate: float = 0.5, depth: float = 0.7, stages: int = 4,
           mix: float = 0.5) -> AudioBuffer:
    sr = buf.sr
    n = len(buf.samples)
    out = array('d', [0.0] * n)
    stage_states = [0.0] * stages
    if _np is not None and n > 0:
        i_arr = _np.arange(n, dtype=_np.float64)
        lfo = (_np.sin(2 * _np.pi * rate * i_arr / sr) + 1) / 2
        freq = 300 + depth * lfo * 3000
        tan_val = _np.tan(_np.pi * freq / sr)
        a_arr = _np.frombuffer(array('d', ((tan_val - 1) / (tan_val + 1)).tobytes()), dtype=_np.float64)
        xl = array('d', _np.frombuffer(buf.samples, dtype=_np.float64).tobytes())
        al = array('d', a_arr.tobytes())
        for i in range(n):
            x = xl[i]
            a = al[i]
            wet = x
            for s in range(stages):
                y = a * wet + stage_states[s]
                stage_states[s] = wet - a * y
                wet = y
            out[i] = x * (1 - mix) + wet * mix
        buf.samples = out
        return buf
    for i, x in enumerate(buf.samples):
        lfo = (math.sin(2 * math.pi * rate * i / sr) + 1) / 2
        freq = 300 + depth * lfo * 3000
        tan_val = math.tan(math.pi * freq / sr)
        a = (tan_val - 1) / (tan_val + 1)
        wet = x
        for s in range(stages):
            y = a * wet + stage_states[s]
            stage_states[s] = wet - a * y
            wet = y
        out[i] = x * (1 - mix) + wet * mix
    buf.samples = out
    return buf

def autopan(buf: AudioBuffer, rate: float = 1.0, depth: float = 1.0) -> Tuple[List[float], List[float]]:
    sr = buf.sr
    n = len(buf.samples)
    if _np is not None and n > 0:
        i = _np.arange(n, dtype=_np.float64)
        lfo = _np.sin(2 * _np.pi * rate * i / sr) * depth
        angle = (lfo + 1) * _np.pi / 4
        x = _np.frombuffer(buf.samples, dtype=_np.float64)
        return (x * _np.cos(angle)).tolist(), (x * _np.sin(angle)).tolist()
    left = [0.0] * n
    right = [0.0] * n
    for i, x in enumerate(buf.samples):
        lfo = math.sin(2 * math.pi * rate * i / sr) * depth
        angle = (lfo + 1) * math.pi / 4
        left[i] = x * math.cos(angle)
        right[i] = x * math.sin(angle)
    return left, right

def noise_gate(buf: AudioBuffer, threshold_db: float = -40, attack: float = 0.002,
               release: float = 0.15) -> AudioBuffer:
    sr = buf.sr
    threshold = 10 ** (threshold_db / 20)
    attack_coef = math.exp(-1 / (sr * attack))
    release_coef = math.exp(-1 / (sr * release))
    one_minus_release = 1 - release_coef
    env = 0.0
    gain = 0.0
    n = len(buf.samples)
    out = array('d', bytes(8 * n))
    for i, x in enumerate(buf.samples):
        rectified = x if x >= 0 else -x
        env = env * release_coef + rectified * one_minus_release if rectified < env else rectified
        target = 1.0 if env > threshold else 0.0
        coef = attack_coef if target > gain else release_coef
        gain = gain * coef + target * (1 - coef)
        out[i] = x * gain
    buf.samples = out
    return buf

def saturation(buf: AudioBuffer, amount: float = 0.3, mix: float = 1.0) -> AudioBuffer:
    drive = 1 + amount * 4
    if _np is not None and len(buf.samples) > 0:
        x = _np.frombuffer(buf.samples, dtype=_np.float64)
        wet = _np.tanh(x * drive) / math.tanh(drive)
        buf.samples = array('d', (wet * mix + x * (1 - mix)).tobytes())
    else:
        wet = [math.tanh(s * drive) / math.tanh(drive) for s in buf.samples]
        buf.samples = array('d', (w * mix + s * (1 - mix) for w, s in zip(wet, buf.samples)))
    return buf

def wah(buf: AudioBuffer, rate: float = 2.0, min_freq: float = 400,
        max_freq: float = 2000, q: float = 3.0) -> AudioBuffer:
    sr = buf.sr
    n = len(buf.samples)
    out = array('d', [0.0] * n)
    x1 = x2 = y1 = y2 = 0.0
    if _np is not None and n > 0:
        i_arr = _np.arange(n, dtype=_np.float64)
        lfo = (_np.sin(2 * _np.pi * rate * i_arr / sr) + 1) / 2
        freq = min_freq + (max_freq - min_freq) * lfo
        w0 = 2 * _np.pi * freq / sr
        alpha = _np.sin(w0) / (2 * q)
        cosw0 = _np.cos(w0)
        b0_arr = alpha
        a0_arr = 1 + alpha
        a1_arr = -2 * cosw0
        a2_arr = 1 - alpha
        xl = array('d', _np.frombuffer(buf.samples, dtype=_np.float64).tobytes())
        b0l = array('d', b0_arr.tobytes())
        a0l = array('d', a0_arr.tobytes())
        a1l = array('d', a1_arr.tobytes())
        a2l = array('d', a2_arr.tobytes())
        for i in range(n):
            x = xl[i]
            b0 = b0l[i]
            y = (b0 * x - b0 * x2 - a1l[i] * y1 - a2l[i] * y2) / a0l[i]
            out[i] = y
            x2, x1 = x1, x
            y2, y1 = y1, y
        buf.samples = out
        return buf
    for i, x in enumerate(buf.samples):
        lfo = (math.sin(2 * math.pi * rate * i / sr) + 1) / 2
        freq = min_freq + (max_freq - min_freq) * lfo
        w0 = 2 * math.pi * freq / sr
        alpha = math.sin(w0) / (2 * q)
        cosw0 = math.cos(w0)
        b0, b1, b2 = alpha, 0.0, -alpha
        a0, a1, a2 = 1 + alpha, -2 * cosw0, 1 - alpha
        y = (b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0
        x2, x1 = x1, x
        y2, y1 = y1, y
        out[i] = y
    buf.samples = out
    return buf

def exciter(buf: AudioBuffer, freq: float = 3000, amount: float = 0.3) -> AudioBuffer:
    bright = buf.copy()
    high_shelf(bright, freq=freq, gain_db=6.0)
    saturation(bright, amount=0.5)
    if _np is not None and len(buf.samples) > 0:
        s = _np.frombuffer(buf.samples, dtype=_np.float64)
        h = _np.frombuffer(bright.samples, dtype=_np.float64)
        buf.samples = array('d', (s + h * amount).tobytes())
    else:
        buf.samples = array('d', (s + h * amount for s, h in zip(buf.samples, bright.samples)))
    return buf

def sidechain(buf: AudioBuffer, pattern: str, bpm: float, steps_per_beat: int = 4,
              depth: float = 0.6, release: float = 0.2) -> AudioBuffer:
    sr = buf.sr
    step_dur = 60 / bpm / steps_per_beat
    n = len(buf.samples)
    release_coef = math.exp(-1 / (sr * release))
    one_minus_release = 1 - release_coef
    trigger_flags = bytearray(n)
    for i, ch in enumerate(pattern):
        if ch != ".":
            pos = int(i * step_dur * sr)
            if 0 <= pos < n:
                trigger_flags[pos] = 1
    out = array('d', bytes(8 * n))
    gain = 1.0
    for i, x in enumerate(buf.samples):
        if trigger_flags[i]:
            gain = 1 - depth
        else:
            gain = gain + (1 - gain) * one_minus_release
        out[i] = x * gain
    buf.samples = out
    return buf

def de_esser(buf: AudioBuffer, freq: float = 6500, threshold_db: float = -26,
             ratio: float = 4.0, q: float = 1.0) -> AudioBuffer:
    high = buf.copy()
    highpass(high, freq=freq, q=q)
    compressor(high, threshold_db=threshold_db, ratio=ratio, attack=0.001, release=0.05)
    low = buf.copy()
    lowpass(low, freq=freq, q=q)
    if _np is not None and len(buf.samples) > 0:
        l_view = _np.frombuffer(low.samples, dtype=_np.float64)
        h_view = _np.frombuffer(high.samples, dtype=_np.float64)
        buf.samples = array('d', (l_view + h_view).tobytes())
    else:
        buf.samples = array('d', (l + h for l, h in zip(low.samples, high.samples)))
    return buf

def duck_under(buf: AudioBuffer, trigger: AudioBuffer, depth: float = 0.6,
               attack: float = 0.01, release: float = 0.15, threshold: float = 0.05) -> AudioBuffer:
    sr = buf.sr
    n_trigger = len(trigger.samples)
    attack_coef = math.exp(-1 / (sr * attack))
    release_coef = math.exp(-1 / (sr * release))
    one_minus_attack = 1 - attack_coef
    one_minus_release = 1 - release_coef
    peak = trigger.peak() or 1.0
    trig_samples = trigger.samples
    env = 0.0
    n = len(buf.samples)
    out = array('d', bytes(8 * n))
    for i, x in enumerate(buf.samples):
        t = abs(trig_samples[i]) / peak if i < n_trigger else 0.0
        if t > env:
            env = attack_coef * env + one_minus_attack * t
        else:
            env = release_coef * env + one_minus_release * t
        active = 1.0 if env > threshold else env / threshold
        gain = 1 - depth * active
        out[i] = x * gain
    buf.samples = out
    return buf

def stereo_widener(left: AudioBuffer, right: AudioBuffer, width: float = 1.3) -> Tuple[AudioBuffer, AudioBuffer]:
    from .buffer import AudioBuffer
    n = min(len(left.samples), len(right.samples))
    if _np is not None and n > 0:
        l = _np.frombuffer(left.samples, dtype=_np.float64)[:n]
        r = _np.frombuffer(right.samples, dtype=_np.float64)[:n]
        mid = (l + r) / 2
        side = (l - r) / 2 * width
        new_left = array('d', (mid + side).tobytes())
        new_right = array('d', (mid - side).tobytes())
        return AudioBuffer(new_left, left.sr), AudioBuffer(new_right, left.sr)
    mid = [0.0] * n
    side = [0.0] * n
    for i in range(n):
        m = (left.samples[i] + right.samples[i]) / 2
        s = (left.samples[i] - right.samples[i]) / 2
        mid[i] = m
        side[i] = s * width
    new_left = [m + s for m, s in zip(mid, side)]
    new_right = [m - s for m, s in zip(mid, side)]
    return AudioBuffer(new_left, left.sr), AudioBuffer(new_right, left.sr)

def haas_widen(left: AudioBuffer, right: AudioBuffer, delay_ms: float = 15,
               mix: float = 0.35) -> Tuple[AudioBuffer, AudioBuffer]:
    from .buffer import AudioBuffer
    sr = left.sr
    delay_samples = max(int(delay_ms * sr / 1000), 1)
    n = min(len(left.samples), len(right.samples))
    if _np is not None and n > 0:
        r = _np.frombuffer(right.samples, dtype=_np.float64)[:n]
        delayed_right = _np.zeros(n, dtype=_np.float64)
        if delay_samples < n:
            delayed_right[delay_samples:] = r[:n - delay_samples]
        new_left = left.samples[:n]
        new_right = array('d', (r * (1 - mix) + delayed_right * mix).tobytes())
        return AudioBuffer(new_left, sr), AudioBuffer(new_right, sr)
    delayed_right = [0.0] * n
    for i in range(n):
        src = i - delay_samples
        delayed_right[i] = right.samples[src] if src >= 0 else 0.0
    new_left = left.samples[:n]
    new_right = [r * (1 - mix) + d * mix for r, d in zip(right.samples[:n], delayed_right)]
    return AudioBuffer(new_left, sr), AudioBuffer(new_right, sr)

def loudness_maximizer(buf: AudioBuffer, target_crest_db: float = 10.0, ceiling_db: float = -0.3,
                       iterations: int = 8, drive_step_db: float = 1.5) -> AudioBuffer:
    ceiling = 10 ** (ceiling_db / 20)
    prev_crest_db = None
    for _ in range(iterations):
        n = len(buf.samples)
        if n == 0:
            break
        peak = buf.peak()
        if peak <= 0:
            break
        if _np is not None:
            view = _np.frombuffer(buf.samples, dtype=_np.float64)
            rms = float(_np.sqrt(_np.mean(view * view)))
        else:
            rms = math.sqrt(sum(s * s for s in buf.samples) / n)
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