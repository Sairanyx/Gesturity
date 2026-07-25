# Future features

Ideas captured for later. Not built yet — parked here so they aren't lost.

## Custom unlock sound (upload + trim)

**Now (done):** `WavAction` plays a `.wav` file if present, else falls back to a beep.

**Later — the full idea:**
- Let the user pick/upload their own sound.
- Accept common formats (MP3, WAV, etc.), not just WAV.
- Cap the sound to 3 seconds: if longer, use only the first 3 seconds.
- Remember the choice in config so it persists between runs.

**Why it's not done now:** trimming and non-WAV playback need an audio library
(e.g. `pydub`) plus `ffmpeg`. That's a mini-project and off the critical path to
the portfolio-ready demo. `winsound` (built in) only plays WAV and can't trim.

## Thumb orientation (thumbs-up vs thumbs-down)

Current recognition is rotation-invariant, so it can't tell thumbs-up from
thumbs-down (same finger shape, rotated). Add an optional hand-orientation
signal (relative to the frame/gravity) used only where direction matters.

## Sequence step timeout

If the user holds one gesture too long (e.g. > 3 seconds) or pauses between
steps, reset progress to 0. Prevents a stale half-entered sequence from lingering.
Listed in the roadmap (Phase 6, "timeout for each step").

## Per-user calibration / recorded gestures

Let users record their own gestures (see roadmap Phases 10, 12, 13). Compare a
learned classifier against the rule-based baseline — a key academic result.
