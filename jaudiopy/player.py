import os
import platform
import subprocess
import tempfile
from typing import Optional
from .audio_io import save_wav_mono, save_wav_stereo
from .buffer import AudioBuffer

def _play_with_simpleaudio(wav_path: str) -> bool:
    import simpleaudio as sa
    wave_obj = sa.WaveObject.from_wave_file(wav_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()
    return True

def _play_with_system_tool(wav_path: str) -> bool:
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.run(["afplay", wav_path], check=True)
        elif system == "Windows":
            ps_cmd = f"(New-Object Media.SoundPlayer '{wav_path}').PlaySync();"
            subprocess.run(["powershell", "-c", ps_cmd], check=True)
        else:
            for tool in ("paplay", "aplay", "ffplay"):
                if _tool_exists(tool):
                    args = [tool, wav_path]
                    if tool == "ffplay":
                        args = [tool, "-nodisp", "-autoexit", "-loglevel", "quiet", wav_path]
                    subprocess.run(args, check=True)
                    return True
            raise RuntimeError("none of paplay/aplay/ffplay were found on this system.")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"playback with a system tool failed: {e}")

def _tool_exists(name: str) -> bool:
    from shutil import which
    return which(name) is not None

def play_buffer(buf: AudioBuffer) -> None:
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        save_wav_mono(tmp_path, buf)
        _play(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def play_stereo(left: AudioBuffer, right: AudioBuffer) -> None:
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        save_wav_stereo(tmp_path, left, right)
        _play(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

def _play(wav_path: str) -> None:
    try:
        _play_with_simpleaudio(wav_path)
        return
    except ImportError:
        pass
    except Exception:
        pass
    _play_with_system_tool(wav_path)