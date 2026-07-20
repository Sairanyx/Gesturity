"""Minimal webcam preview: shows video,
displays FPS, quits on Q, for dev purposes 
mostly to test and check"""

import time
import cv2

def main() -> None:
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Could not open camera 0. Is there another app using it or no camera 0?")
    
    prev_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Failed to read frame, stopping.")
                break

            now = time.perf_counter()
            fps = 1.0 / (now - prev_time)
            prev_time = now

            cv2.putText(
                frame,
                f"{fps:.1f} FPS",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Gesturity - camera check", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if cv2.getWindowProperty("Gesturity - camera check", cv2.WND_PROP_VISIBLE) < 1:
                 break
    finally:
         capture.release()
         cv2.destroyAllWindows()
        
if __name__ == "__main__":
     main()