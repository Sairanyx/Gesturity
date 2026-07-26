"""This is the reusable gesture unlock pipeline and is shared by all scripts for convenience"""

import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from gesture_unlock.normalisation import landmarks_to_array, normalise
from gesture_unlock.static import recognise
from gesture_unlock.stability import GestureStabiliser
from gesture_unlock.sequence import SequenceEngine, SequenceEvent


@dataclass
class FrameResult:
    """What the pipeline see in one frame"""
    gesture: str          # the current stable gesture name
    step: int             # progress through the sequence
    total: int            # length of the sequence
    just_unlocked: bool   # True only on the frame the sequence completes


class GestureUnlocker:
    """Runs the full gesture pipeline and tracks sequence progress."""

    def __init__(self, model_path: str, sequence: list[str]):
        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
            )
        )
        self._stabiliser = GestureStabiliser(window_size=5, hold_seconds=0.4)
        self._sequence = SequenceEngine(sequence)
        self._total = len(sequence)

    def process(self, frame, now: float) -> FrameResult:
        """Feed one camera frame plus the current time. Returns the result."""
        gesture = "UNKNOWN"
        just_unlocked = False

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, int(now * 1000))

        if result.hand_landmarks:
            points = landmarks_to_array(result.hand_landmarks[0])
            stable = self._stabiliser.update(recognise(normalise(points)).name, now)
            gesture = stable.name
            if stable.is_stable:
                outcome = self._sequence.update(stable.name)
                if outcome.event == SequenceEvent.COMPLETED:
                    just_unlocked = True

        # Always report the engine's real remembered step, not a per-frame value.
        step = self._sequence.step
        return FrameResult(gesture=gesture, step=step, total=self._total,
                           just_unlocked=just_unlocked)
