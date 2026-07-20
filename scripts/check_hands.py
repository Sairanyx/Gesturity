"""Webcam preview with MediaPipe hand landmarks drawn on top."""

import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

WINDOW_NAME = "Gesturity - hand check"
MODEL_PATH = "models/hand_landmarker.task"


def draw_hand(frame, hand_landmarks) -> None:
    height, width = frame.shape[:2]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks]

    for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, points[connection.start], points[connection.end], (255, 255, 255), 2)
    for point in points:
        cv2.circle(frame, point, 4, (0, 255, 0), -1)


def main() -> None:
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
    prev_time = start

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.perf_counter() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            for i, hand_landmarks in enumerate(result.hand_landmarks):
                draw_hand(frame, hand_landmarks)
                handedness = result.handedness[i][0]
                label = f"{handedness.category_name} {handedness.score:.2f}"
                cv2.putText(frame, label, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            now = time.perf_counter()
            fps = 1.0 / (now - prev_time)
            prev_time = now
            cv2.putText(frame, f"{fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

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
