"""Downloads the MediaPipe hand landmark model if it is not already"""

import urllib.request
import os

URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
PATH = "models/hand_landmarker.task"


def main():
    if os.path.exists(PATH):
        print("Model already here.")
        return
    print("Downloading model...")
    os.makedirs("models", exist_ok=True)
    urllib.request.urlretrieve(URL, PATH)
    print("Done.")


if __name__ == "__main__":
    main()
