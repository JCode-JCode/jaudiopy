import io
import os
import wave
from array import array
import tempfile
import urllib.request
from urllib.parse import urlparse
from typing import List, Tuple, Union, Any, Optional
from .buffer import AudioBuffer

_URL_TIMEOUT_SECONDS = 30


def _bytes_from_source(source: Any) -> Optional[bytes]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        return bytes(source)
    if hasattr(source, "read"):
        data = source.read()
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(
                "file-like 'source' must yield bytes from .read() "
                "(open it in binary mode, e.g. open(path, 'rb'))."
            )
        return bytes(data)
    return None


def _ensure_wav_from_temp(tmp_path: str) -> Tuple[str, bool]:
    try:
        with wave.open(tmp_path, "rb"):
            pass
        return tmp_path, False
    except (wave.Error, EOFError):
        pass
    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError(
            "the provided in-memory audio data is not a valid WAV file. "
            "install pydub (and ffmpeg) to load other formats from memory "
            "(pip install jaudiopy[formats]), or provide raw WAV bytes."
        )
    seg = AudioSegment.from_file(tmp_path)
    fd, out_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    seg.export(out_wav, format="wav")
    return out_wav, True


def _is_numpy_array(samples: Any) -> bool:
    return hasattr(samples, "dtype") and hasattr(samples, "astype")


def _to_int16(samples: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        if _is_numpy_array(samples):
            arr = samples
        elif isinstance(samples, array) and samples.typecode == 'd':
            arr = np.frombuffer(samples, dtype=np.float64)
        else:
            arr = np.asarray(samples, dtype=np.float64)
        clipped = np.clip(arr * 32767.0, -32767, 32767)
        return clipped.astype(np.int16)
    return array('h', (max(-32767, min(32767, int(s * 32767))) for s in samples))


def _int16_bytes(int16_arr: Any) -> bytes:
    return int16_arr.tobytes()


def save_wav_mono(path: Union[str, "io.BufferedIOBase"], buf: AudioBuffer) -> Union[str, None]:
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(buf.sr)
        f.writeframes(_int16_bytes(_to_int16(buf.samples)))
    return path if isinstance(path, str) else None


def save_wav_stereo(path: Union[str, "io.BufferedIOBase"], left: AudioBuffer,
                     right: AudioBuffer) -> Union[str, None]:
    n = min(len(left.samples), len(right.samples))
    with wave.open(path, "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(left.sr)
        l_samples = left.samples[:n]
        r_samples = right.samples[:n]
        li = _to_int16(l_samples)
        ri = _to_int16(r_samples)
        if _is_numpy_array(li):
            import numpy as np
            interleaved = np.empty(n * 2, dtype=np.int16)
            interleaved[0::2] = li
            interleaved[1::2] = ri
            f.writeframes(interleaved.tobytes())
        else:
            interleaved = array('h', bytes(4 * n))
            interleaved[0::2] = li
            interleaved[1::2] = ri
            f.writeframes(interleaved.tobytes())
    return path if isinstance(path, str) else None


def save_wav_mono_bytes(buf: AudioBuffer) -> bytes:
    bio = io.BytesIO()
    save_wav_mono(bio, buf)
    return bio.getvalue()


def save_wav_stereo_bytes(left: AudioBuffer, right: AudioBuffer) -> bytes:
    bio = io.BytesIO()
    save_wav_stereo(bio, left, right)
    return bio.getvalue()


def load_wav_raw(path: str) -> Tuple[List[List[float]], int]:
    with wave.open(path, "r") as f:
        sr = f.getframerate()
        n_channels = f.getnchannels()
        n_frames = f.getnframes()
        raw = f.readframes(n_frames)
    try:
        import numpy as np
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32767.0
        if n_channels == 1:
            return [arr], sr
        return [arr[0::2].copy(), arr[1::2].copy()], sr
    except ImportError:
        pass
    ints = array('h')
    ints.frombytes(raw)
    floats = [x / 32767 for x in ints]
    if n_channels == 1:
        return [floats], sr
    return [floats[0::2], floats[1::2]], sr


def load_wav(path: str) -> Union[AudioBuffer, Tuple[AudioBuffer, AudioBuffer]]:
    channels, sr = load_wav_raw(path)
    if len(channels) == 1:
        return AudioBuffer(channels[0], sr)
    return AudioBuffer(channels[0], sr), AudioBuffer(channels[1], sr)


def is_url(path_or_url: str) -> bool:
    try:
        return urlparse(path_or_url).scheme in ("http", "https")
    except Exception:
        return False


def download_to_temp(url: str, dest_dir: Optional[str] = None) -> str:
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1] or ".wav"
    fd, tmp_path = tempfile.mkstemp(suffix=ext, dir=dest_dir)
    os.close(fd)
    req = urllib.request.Request(url, headers={"User-Agent": "jaudiopy/1.0.0"})
    try:
        with urllib.request.urlopen(req, timeout=_URL_TIMEOUT_SECONDS) as resp, open(tmp_path, "wb") as out:
            out.write(resp.read())
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return tmp_path


def _ensure_wav(path: str) -> Tuple[str, bool]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        return path, False
    try:
        from pydub import AudioSegment
    except ImportError:
        raise RuntimeError(
            f"format '{ext}' is not supported directly. install pydub (and ffmpeg) "
            f"to support mp3/ogg/flac/... : pip install pydub -- or use a wav file instead."
        )
    seg = AudioSegment.from_file(path)
    fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    seg.export(tmp_wav, format="wav")
    return tmp_wav, True


def resolve_source(path_or_url: Any) -> Tuple[str, List[str]]:
    cleanup: List[str] = []
    in_memory = _bytes_from_source(path_or_url)
    if in_memory is not None:
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with open(tmp_path, "wb") as f:
            f.write(in_memory)
        cleanup.append(tmp_path)
        wav_path, converted = _ensure_wav_from_temp(tmp_path)
        if converted:
            cleanup.append(wav_path)
        return wav_path, cleanup
    if is_url(path_or_url):
        downloaded = download_to_temp(path_or_url)
        cleanup.append(downloaded)
        wav_path, converted = _ensure_wav(downloaded)
        if converted:
            cleanup.append(wav_path)
        return wav_path, cleanup
    if not os.path.exists(path_or_url):
        raise FileNotFoundError(f"file or link not found: {path_or_url}")
    wav_path, converted = _ensure_wav(path_or_url)
    if converted:
        cleanup.append(wav_path)
    return wav_path, cleanup


def load_audio(path_or_url: str) -> Union[AudioBuffer, Tuple[AudioBuffer, AudioBuffer]]:
    wav_path, cleanup = resolve_source(path_or_url)
    try:
        return load_wav(wav_path)
    finally:
        for p in cleanup:
            try:
                os.remove(p)
            except OSError:
                pass
