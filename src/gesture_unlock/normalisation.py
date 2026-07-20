""" Turning raw MediaPipe landmarks into position and scaling-invariant coordinates."""

import numpy as np

WRIST = 0
MIDDLE_KNUCKLE = 9

def landmarks_to_array(hand_landmarks) -> np.ndarray:
    """Converting MediaPipe landmark objects to a (21, 3) NumPy array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)

def normalise(points: np.ndarray) -> np.ndarray:
    """Returns the landmarks relative to the wrist in units of palm lenghts"""
    centred = points - points[WRIST]
    scale = float(np.linalg.norm(centred[MIDDLE_KNUCKLE]))
    if scale == 0.0:
        raise ValueError("Not a hand: wrist and middle knuckle coincide")
    return centred / scale
