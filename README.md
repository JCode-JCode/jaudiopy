# jaudiopy

[![Python Version](https://img.shields.io/pypi/pyversions/jaudiopy)](https://pypi.org/project/jaudiopy/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/jaudiopy)](https://pypi.org/project/jaudiopy/)
[![PyPI project](https://img.shields.io/badge/PyPI-jaudiopy-blue)](https://pypi.org/project/jaudiopy/)

<br>

<img src="docs/images/jaudiopy-logo.png" alt="jaudiopy logo">

<br>

**jaudiopy** is a complete, high‑performance library for beat making, sound synthesis, mixing, mastering, and processing existing audio files (local path or web link). It ships with a pure‑Python engine that has zero required dependencies and an optional numpy/scipy engine (user selectable) for maximum speed, plus drum machines, step sequencers, melodic synths, samplers, a vocal‑mixing chain, a one‑call full‑beat builder, and MIDI import — all through a clean, unified API.

---

## Quick Start – Load and Process an Audio File

```python
from jaudiopy import AudioFile

song = AudioFile("song.wav")
song.add_effect("reverb", "song_reverb.wav", room_size=0.7, mix=0.4)
song.play()
```

---

## Main Capabilities

**· Audio Files** – Load from a local path or a web link, play in the terminal, apply effects, change speed/pitch, tune, trim, normalize, fade, mix, and master — all through the `AudioFile` class.

**· Dual Engine** – Pure‑Python engine with zero required dependencies, and a numpy engine (`change_to_numpy=True`) that uses `scipy.signal.lfilter` when available; both paths are optimized end‑to‑end and produce equivalent results.

**· Drum Machine & Sequencing** – Built‑in drum kits (kick, snare, hihat, and more), the `X/x/.` step pattern language, a `StepSequencer`, and a `MelodySequencer` for chord progressions.

**· Synths & Instruments** – Bass, pluck, pad, and piano synths, plus a dedicated 808 bass line generator with continuous pitch glide.

**· Effects** – Filters, EQ, distortion, bitcrush, delay, reverb, chorus, tremolo, vibrato, phaser, wah, saturation, exciter, noise gate, compressor, limiter, panning, autopan, and stereo widening.

**· Mixing & Mastering** – A `Mixer`/`AdvancedMixer` with buses, and a mastering chain with ready‑made presets (`balanced`, `warm`, `bright`, `loud_edm`, `vocal_pop`) including loudness targeting and stereo widening.

**· Sampling** – Load your own one‑shots or recordings with `Sampler`, re‑pitch and loop them, or chop them into slices for breaks.

**· Vocals** – A vocal‑ready EQ/de‑ess/compress/limit chain and automatic sidechain ducking against a beat.

**· Full Beat Builder** – `BeatBuilder` generates a complete, mixed, and mastered beat from a style, BPM, key, and length in a single call, at any duration.

**· Utilities** – MIDI note import, WAV I/O (mono/stereo, file or in‑memory bytes), and URL/local‑path resolution built in.

---

## Installation

```bash
# Basic installation (core dependencies only)
pip install jaudiopy

# With the numpy engine
pip install jaudiopy[numpy]

# With numpy + scipy for maximum performance
pip install jaudiopy[fast]

# With more reliable terminal playback
pip install jaudiopy[playback]

# With mp3/ogg/... support (also requires ffmpeg)
pip install jaudiopy[formats]

# With MIDI import support
pip install jaudiopy[midi]

# All optional features
pip install jaudiopy[all]
```

## Debug

The core library has no external dependency — it only uses `array`, `math`,
`random`, and `wave` from the Python standard library. If an optional
feature (playback, formats, MIDI) isn't working, make sure the matching
extra above is installed.

For `jaudiopy[formats]` (mp3/ogg/... support), you also need `ffmpeg`
installed on your system:

· Ubuntu/Debian: `sudo apt install ffmpeg`
· Termux (Android): `pkg install ffmpeg`
· macOS: `brew install ffmpeg`

Or download from ffmpeg.org

---

## More Examples

## Engine switch: numpy vs pure‑Python

```python
from jaudiopy import AudioFile

song = AudioFile("song.wav", change_to_numpy=False)  # pure-python engine (default)
song = AudioFile("song.wav", change_to_numpy=True)    # numpy engine
```

## Drums and the X/x/. Pattern

```python
from jaudiopy import drums, StepSequencer

kit = drums.build_kit({
    "K": ("kick", "808"),
    "S": ("snare", "fat"),
    "H": ("hihat", "trap"),
})
seq = StepSequencer(bpm=140)
track = seq.render_kit(
    {"K": "X...x...X...x...", "S": "....X.......X...", "H": "X.X.X.X.X.X.X.X."},
    {
        "K": {"X": kit["K"], "x": (kit["K"], 0.5)},
        "S": {"X": kit["S"]},
        "H": {"X": kit["H"], "x": (kit["H"], 0.4)},
    },
)
```

## Building a Full Beat Fast

```python
from jaudiopy import BeatBuilder

bb = BeatBuilder(style="trap_808", bpm=140, key="E", change_to_numpy=True)
left, right = bb.build(duration_minutes=2.5)
bb.build_and_save("my_beat.wav", duration_minutes=2.5)
```

## Sampling

```python
from jaudiopy import Sampler

sampler = Sampler.load("piano_note_C4.wav", base_note="C", base_octave=4)
note = sampler.play_note("D#", 4, duration=1.0)   # re-pitched to D#4
loop = sampler.loop_to(8.0)                        # looped to 8 seconds
chops = sampler.chop(8)                             # 8 equal slices
```

## Adding Vocals Over a Beat

```python
from jaudiopy import mix_vocal_with_beat

final_left, final_right = mix_vocal_with_beat(
    beat_left, beat_right, vocal_buffer, vocal_at=8.0,
)
```

## Mastering With a Preset

```python
from jaudiopy import master_with_preset

left, right = master_with_preset(left, right, preset="loud_edm", target_crest_db=7.0, haas_mix=0.4)
```

---

## Issues and Contributions

You can report bugs via GitHub Issues or submit fixes via pull requests.

---

## Links

**· GitHub repository:**
https://github.com/JCode-JCode/jaudiopy

**· PyPI page:**
https://pypi.org/project/jaudiopy/

---

## License

This project is licensed under the Apache License 2.0 – see the LICENSE file for details.

---

Designed and built with love by **J Code**
