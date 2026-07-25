"""Just a demo lock screen that gets unlocled by the correct sequence hand gesture.
In order to make development and testing safe it never touches OS logs etc."""

import cv2
import numpy as np

import time

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from gesture_unlock.normalisation import landmarks_to_array, normalise
from gesture_unlock.static import recognise
from gesture_unlock.stability import GestureStabiliser
from gesture_unlock.sequence import SequenceEngine, SequenceEvent
from gesture_unlock.actions import WavAction


WINDOW_NAME = "Gesturity Lock"
MODEL_PATH = "models/hand_landmarker.task"
PIN = "1234"   # fallback if the camera or gestures fail

def draw_screen(unlocked: bool, step: int, total: int, typed_pin: str = "",
                width: int = 1280, height: int = 720) -> np.ndarray:

    """Returns the lock screen image for the current state."""
    screen = np.zeros((height, width, 3), dtype=np.uint8)

    if unlocked:
        cv2.putText(screen, "UNLOCKED", (width // 2 - 220, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 4)
    else:
        cv2.putText(screen, "LOCKED", (width // 2 - 160, height // 2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 4)
        cv2.putText(screen, "Perform: FIST -> PEACE -> OPEN_PALM",
                    (width // 2 - 340, height // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(screen, f"Progress: {step}/{total}",
                    (width // 2 - 90, height // 2 + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        dots = "*" * len(typed_pin)
        cv2.putText(screen, f"PIN: {dots}   (type {len(PIN)} digits)",
                    (width // 2 - 200, height // 2 + 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    cv2.putText(screen, "Press ESC to exit", (30, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    return screen

def main() -> None:
    landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
    )
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Could not open camera 0.")

    stabiliser = GestureStabiliser(window_size=5, hold_seconds=0.4)
    sequence = SequenceEngine(["FIST", "PEACE", "OPEN_PALM"])
    unlock_sound = WavAction("sounds/unlock.wav")

    start = time.perf_counter()
    unlocked = False
    typed_pin = ""
    step = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            now = time.perf_counter() - start

            # Running the gesture pipeline on the camera frame and no camera window shown
            if not unlocked:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, int(now * 1000))

                if result.hand_landmarks:
                    points = landmarks_to_array(result.hand_landmarks[0])
                    stable = stabiliser.update(recognise(normalise(points)).name, now)
                    if stable.is_stable:
                        outcome = sequence.update(stable.name)
                        step = outcome.step
                        if outcome.event == SequenceEvent.COMPLETED:
                            unlocked = True
                            unlock_sound.run()

            # Making the lock screen with live updates
            screen = draw_screen(unlocked, step, total=3, typed_pin=typed_pin)
            cv2.imshow(WINDOW_NAME, screen)

            key = cv2.waitKey(30) & 0xFF
            if key == 27: # 27 is set as the ESC key
                break
            if not unlocked and 48 <= key <= 57: # Setting keys 0 to 9
                typed_pin += chr(key)
                if typed_pin == PIN:
                    unlocked = True
                    unlock_sound.run()
                elif len(typed_pin) >= len(PIN):
                    typed_pin = ""               # So wrong PIN and resets basically
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
                                     