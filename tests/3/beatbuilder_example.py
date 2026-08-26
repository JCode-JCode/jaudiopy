from jaudiopy import BeatBuilder
import time

t0 = time.time()

bb = BeatBuilder(style="trap_808", bpm=144, key="F#", progression="trap_dark", engine="numpy")
bb.build_and_save("samurai_style_beat.wav", duration_minutes=2)

print(f"saved: samurai_style_beat.wav in {time.time() - t0:.2f}s")

# want a different length or vibe? just change the arguments:
# bb = BeatBuilder(style="dark_melodic", bpm=90, key="A", progression="minor_epic")
# bb.build_and_save("another_beat.wav", duration_minutes=3.5)
