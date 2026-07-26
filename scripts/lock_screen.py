"""Just a demo lock screen that gets unlocled by the correct sequence hand gesture.
In order to make development and testing safe it never touches OS logs etc."""

import time

import cv2
import numpy as np

from gesture_unlock.pipeline import GestureUnlocker
from gesture_unlock.actions import WavAction


WINDOW_NAME = "Gesturity Lock"
MODEL_PATH = "models/hand_landmarker.task"
PIN = "1234"   # fallback if the camera or gestures fail

def draw_screen(unlocked: bool, typed_pin: str = "",
                width: int = 1280, height: int = 720) -> np.ndarray:

    """Returns the lock screen image for the current state."""
    screen = np.zeros((height, width, 3), dtype=np.uint8)

    if unlocked:
        cv2.putText(screen, "UNLOCKED", (width // 2 - 220, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 4)
    else:
        cv2.putText(screen, "LOCKED", (width // 2 - 160, height // 2 - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 4)
        cv2.putText(screen, "Perform your gesture sequence to unlock",
                    (width // 2 - 360, height // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        # No progress shown on purpose: revealing steps would help an attacker guess.
        dots = "*" * len(typed_pin)
        cv2.putText(screen, f"PIN: {dots}   (type {len(PIN)} digits)",
                    (width // 2 - 200, height // 2 + 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    cv2.putText(screen, "Press ESC to exit", (30, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    return screen

def main() -> None:
    unlocker = GestureUnlocker(MODEL_PATH, ["FIST", "PEACE", "OPEN_PALM"])
    unlock_sound = WavAction("sounds/unlock.wav")

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Could not open camera 0.")

    start = time.perf_counter()
    unlocked = False
    typed_pin = ""

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            now = time.perf_counter() - start

            if not unlocked:
                result = unlocker.process(frame, now)
                if result.just_unlocked:
                    unlocked = True
                    unlock_sound.run()

            screen = draw_screen(unlocked, typed_pin=typed_pin)
            cv2.imshow(WINDOW_NAME, screen)

            key = cv2.waitKey(30) & 0xFF
            if key == 27:   # ESC
                break
            if not unlocked and 48 <= key <= 57:   # digits 0 to 9
                typed_pin += chr(key)
                if typed_pin == PIN:
                    unlocked = True
                    unlock_sound.run()
                elif len(typed_pin) >= len(PIN):
                    typed_pin = ""
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
                                     