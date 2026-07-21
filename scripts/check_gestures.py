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

WINDOW_NAME = "Gesturity - gestures"
MODEL_PATH = "models/hand_landmarker.task"

def main():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Could not open camera 0.")
    
    start = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.perf_counter() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                points = landmarks_to_array(hand)
                normalised = normalise(points)
                gesture = recognise(normalised)

                cv2.putText(frame, gesture.name, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                
                y = 80
                for finger_name in gesture.fingers:
                    is_up = gesture.fingers[finger_name]
                    text = finger_name + ": " + ("UP" if is_up else "down")
                    cv2.putText(frame, text, (10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    y = y + 25

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
