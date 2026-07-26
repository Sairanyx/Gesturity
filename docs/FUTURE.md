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

## Quiet the startup logs

MediaPipe/TensorFlow print noisy INFO/WARNING lines and a harmless clearcut
telemetry error on startup. Suppress them (e.g. TF_CPP_MIN_LOG_LEVEL env var,
absl logging config) so the terminal is clean for a real product.

## Polished lock-screen visuals

Make the lock screen look like a real Windows lock/sign-in screen: background
image, centred user card/avatar, styled fonts, rounded PIN field, clean layout.
Raw OpenCV drawing is limited here — likely move the UI to a real GUI toolkit
(PyQt or Tkinter). This is presentation only; the recognition engine is untouched.

## Two-hand support (near-term, after swipes)

Enable num_hands=2 and let each hand hold its own gesture at the same time
(e.g. left FIST + right PEACE as one combined step). Design questions to settle:
do both hands need to match, how it interacts with sequences, combined vs
either-hand. Do this as a focused step right after swipes, not mixed in with them.

## Custom recorded motion gestures (fingertip drawing, shapes, arbitrary paths)

Let the user "draw" a gesture in the air with a specific finger (e.g. index
fingertip as a pen) — cleaner path than tracking the whole hand. They can record
a circle, square, triangle, letter, or any custom squiggle and save it as their
own gesture. Recognition matches new motion against the saved template.

After recording, play back a SLOW-MOTION TRACE of the captured path so the user
can see exactly what the system recorded, confirm it, and re-record if messy.
This visual feedback also helps them make repeatable gestures.

Needs: motion tracking (Phase 7 swipes) -> recording (Phase 10) -> template
matching / DTW or a small model (Phase 12/13). Simple angle-snapping works for
straight swipes; arbitrary shapes need the template/learned approach.

## Gesturity Centre (the product UI)

A desktop UI (installed app) where the user sets up everything: their PIN, their
gesture/swipe/shape sequence, records custom motions (with the slow-mo trace
preview above), picks the unlock sound, enables/disables the OS-login option.
Front-end of the whole product, comes near the end once its features exist.

## Per-user calibration / recorded gestures

Let users record their own gestures (see roadmap Phases 10, 12, 13). Compare a
learned classifier against the rule-based baseline — a key academic result.
