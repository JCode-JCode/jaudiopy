# Changelog

All notable changes to **jaudiopy** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [1.0.0] - 2026-08-23

### Added

**Core `AudioFile` API**
- New `AudioFile` class for loading, processing, and saving audio from a
  **local path or a web link** — playback, effects, speed/pitch change,
  tuning, trimming, normalizing, fading, mixing, and mastering, all through
  one object.
- `play`, `add_effect`, `change_speed`, `tune`, `trim`, `normalize`,
  `reverse`, `fade`, `gain_db`, `repeat`, `mix_with`, `master`, `to_mono`.
- `repeat_before_save` / `repeat_after_save` for repeating audio on
  playback only, or baking the repeat into the saved file.

**Dual processing engine**
- Pure‑Python engine (`change_to_numpy=False`, default) with **zero
  required dependencies** — only `array`, `math`, `random`, and `wave` from
  the standard library.
- Numpy engine (`change_to_numpy=True`) using `numpy` as native buffer
  storage, with `scipy.signal.lfilter` acceleration for EQ/filter stages
  when `scipy` is installed.
- Automatic, silent acceleration of the pure‑Python engine with numpy/scipy
  when they happen to be importable — with an identical pure‑Python
  fallback when they aren't.
- Vectorized WAV saving (`save_wav_mono`, `save_wav_stereo`) and
  `AudioFile.to_mono` for the numpy engine.

**Drum machine & sequencing**
- Built‑in drum kit builder (`drums.build_kit`) with multiple kick, snare,
  and hihat variants plus crash, rimshot, shaker, conga, clave, tambourine,
  and more (`drums.list_sounds`).
- `StepSequencer` with the `X`/`x`/`.` step‑pattern language for programming
  beats, including per‑step velocity variants.
- `MelodySequencer` for rendering chord progressions and rests over time.
- `Song` for arranging sequenced parts.

**Synths & instruments**
- `bass_synth`, `pluck_synth`, `pad_synth`, `piano_synth`.
- `bass_808_line` — 808 bass generator with **continuous pitch glide**
  between notes (classic trap‑style slide) instead of retriggering per note.
- `piano_synth` multi‑harmonic synthesis via a recursive complex‑rotation
  oscillator (pure engine) or vectorized `numpy.sin`/`numpy.exp` (numpy
  engine), matching output to float64 precision.

**Effects (`jaudiopy.effects`)**
- Filters & tone shaping: `lowpass`, `highpass`, `eq_band`, `low_shelf`,
  `high_shelf`.
- Character & drive: `distortion`, `bitcrush`, `saturation`, `exciter`.
- Time‑based: `delay`, `reverb`, `chorus`, `phaser`, `vibrato`, `tremolo`,
  `wah`, `autopan`.
- Dynamics: `compressor`, `limiter`, `noise_gate`, `duck_under`,
  `sidechain`.
- Stereo: `pan_stereo`, `stereo_widener`.
- Comb/all‑pass/delay lines behind `reverb`/`delay` vectorized via a
  phase‑splitting `lfilter` trick on the numpy engine.

**Mixing**
- `Mixer` and `AdvancedMixer` with track‑level volume/pan.
- Bus support (`add_bus`) — tracks assigned to a bus share the bus's volume
  (multiplied with the track's own) and pan.

**Mastering**
- `master_chain`, `master_chain_advanced`, and `master_with_preset` with
  ready‑made presets: `balanced`, `warm`, `bright`, `loud_edm`,
  `vocal_pop`.
- Per‑preset **loudness targeting** (`target_crest_db`) for a final mix
  closer to a commercially mastered reference.
- Automatic **Haas‑delay stereo widening** (`haas_mix`) so dead‑center mono
  renders still end up with audible stereo width; both parameters are
  overridable, and `target_crest_db=None` restores the older, more dynamic
  behavior.

**Sampling**
- `Sampler.load` from a local path, URL, or in‑memory bytes.
- Note‑accurate re‑pitching via `play_note`, time‑stretch looping via
  `loop_to`, and even‑slice chopping via `chop` (for chopping breaks).
- Zero‑arg callables can stand in for any built‑in drum/instrument function,
  so custom one‑shots plug directly into `StepSequencer` patterns.

**Vocals**
- `vocal_chain` — EQ, de‑ess, compress, and limit chain tuned for vocals.
- `mix_vocal_with_beat` — automatically ducks the beat under the vocal and
  mixes them together, with a `vocal_at` offset (seconds) for vocals that
  enter after an intro.

**Full beat builder**
- `BeatBuilder` — builds a complete, mixed, and mastered beat from a style,
  BPM, key, and length in a single call (`build` / `build_and_save`).
- Only one loop is ever synthesized and then tiled, so a 30‑second beat and
  a 5‑minute beat take about the same time to build.
- Five ready‑made styles (`list_styles`): `trap_808`, `boombap_piano`,
  `street_hiphop`, `dark_melodic`, `minimal_bounce`.
- Four ready‑made chord progressions (`list_progressions`): `minor_epic`,
  `minor_simple`, `trap_dark`, `single_root`.
- `build_many` for batch‑generating multiple beats.

**I/O & utilities**
- WAV I/O for mono and stereo, to file or in‑memory bytes:
  `save_wav_mono`, `save_wav_stereo`, `load_wav`, `load_audio`,
  `save_wav_mono_bytes`, `save_wav_stereo_bytes`.
- `resolve_source` / `is_url` for transparent local‑path‑or‑URL loading
  anywhere audio is accepted.
- `load_midi_notes` for importing note data from MIDI files.
- Terminal playback via `play_buffer` / `play_stereo`.

**Packaging**
- Optional extras: `jaudiopy[numpy]`, `jaudiopy[fast]` (numpy + scipy),
  `jaudiopy[playback]` (more reliable terminal playback via `simpleaudio`),
  `jaudiopy[formats]` (mp3/ogg/... via `pydub`, requires `ffmpeg`),
  `jaudiopy[midi]` (via `mido`), and `jaudiopy[all]`.
- Published under the Apache License 2.0.
