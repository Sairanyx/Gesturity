""" This is a live gesture recognition: shows the detected gesture on the webcam feed. 
Mostly developed for testing and checking all works as expected and also for developing
purposes"""

import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from gesture_unlock.normalisation import landmarks_to_array, normalise
from gesture_unlock.static import recognise
from gesture_unlock.stability import GestureStabiliser
from gesture_unlock.sequence import SequenceEngine, SequenceEvent
from gesture_unlock.actions import WavAction
from gesture_unlock.movement import SwipeDetector




WINDOW_NAME = "Gesturity - gestures"
MODEL_PATH = "models/hand_landmarker.task"

def main():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Could not open camera 0.")
    
    start = time.perf_counter()
    stabiliser = GestureStabiliser(window_size=5, hold_seconds=0.4)
    sequence = SequenceEngine(["FIST", "PEACE", "OPEN_PALM"])
    unlocked_until = 0.0   # keeps the "UNLOCKED" message on screen briefly
    unlock_action = WavAction("sounds/unlock.wav")
    swipe_detector = SwipeDetector()
    last_swipe = "NONE"

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            now = time.perf_counter() - start

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.perf_counter() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                points = landmarks_to_array(hand)
                normalised = normalise(points)
                gesture = recognise(normalised)
                stable = stabiliser.update(gesture.name, now)

                # White while still settling and green once it has held long enough
                colour = (0, 255, 0) if stable.is_stable else (200, 200, 200)
                cv2.putText(frame, stable.name, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 2)
                swipe = swipe_detector.update(points, now)
                if swipe != "NONE":
                    last_swipe = swipe
                cv2.putText(frame, f"Swipe: {last_swipe}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 2)

                # Only feeds STABLE gestures into the sequence engine
                if stable.is_stable:
                    outcome = sequence.update(stable.name)
                    if outcome.event == SequenceEvent.COMPLETED:
                        unlocked_until = now + 2.0   # show success for 2 seconds
                        unlock_action.run()          #  dies the sound it was set to do

                    progress = f"Step {outcome.step}/{outcome.total}"
                    cv2.putText(frame, progress, (10, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
                y = 80

                for finger_name in gesture.fingers:
                    is_up = gesture.fingers[finger_name]
                    text = finger_name + ": " + ("UP" if is_up else "down")
                    cv2.putText(frame, text, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    y = y + 25

            else:
                # No hand in frame tells the system nothing is there so it resets cleanly
                stabiliser.update("UNKNOWN", now)
                sequence.update("UNKNOWN")
                last_swipe = "NONE"
            
            if now < unlocked_until:
                cv2.putText(frame, "UNLOCKED!", (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
