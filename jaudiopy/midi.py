from typing import List, Tuple, Optional, Dict

def load_midi_notes(midi_path: str, track: int = 0) -> List[Tuple[str, int, float]]:
    try:
        import mido
    except ImportError:
        raise ImportError(
            "mido is required for MIDI import. Install with: pip install jaudiopy[midi]"
        )
    mid = mido.MidiFile(midi_path)
    if track >= len(mid.tracks):
        raise ValueError(f"Track {track} not found. Available tracks: 0..{len(mid.tracks)-1}")
    tempo = 500000
    for msg in mid.tracks[track]:
        if msg.type == 'set_tempo':
            tempo = msg.tempo
            break
    else:
        for msg in mid.tracks[0]:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
    ticks_per_beat = mid.ticks_per_beat
    notes: List[Tuple[int, int, int]] = []
    pending: Dict[int, int] = {}
    abs_time = 0
    for msg in mid.tracks[track]:
        abs_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            pending[msg.note] = abs_time
        elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
            if msg.note in pending:
                start = pending.pop(msg.note)
                duration = abs_time - start
                notes.append((start, msg.note, duration))
    end_time = max(abs_time, max((start for start, _, _ in notes), default=0))
    for note_num, start in pending.items():
        notes.append((start, note_num, end_time - start))
    notes.sort(key=lambda x: x[0])
    result: List[Tuple[str, int, float]] = []
    for start_ticks, note_num, dur_ticks in notes:
        note_name, octave = _note_num_to_name(note_num)
        dur_beats = dur_ticks / ticks_per_beat
        result.append((note_name, octave, dur_beats))
    return result

def _note_num_to_name(note_num: int) -> Tuple[str, int]:
    NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = note_num // 12 - 1
    note = NOTES[note_num % 12]
    return note, octave
