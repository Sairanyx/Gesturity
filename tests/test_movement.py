"""Swipe detection must report the correct direction and ignore small fluctuations"""

import numpy as np

from gesture_unlock.movement import SwipeDetector


def hand_at(x: float, y: float) -> np.ndarray:
    """A fake hand and the wrist is at (x, y) and knuckle one unit away (scale=1)."""
    points = np.zeros((21, 3), dtype=np.float32)
    points[0] = [x, y, 0.0]        # wrist
    points[9] = [x, y + 1.0, 0.0]  # mid knuckle 1 unit away so scale = 1
    return points


def test_swipe_right():
    d = SwipeDetector()
    d.update(hand_at(0.0, 0.0), now=0.0)
    result = d.update(hand_at(2.0, 0.0), now=0.1)   # moved +2 in x
    assert result == "SWIPE_RIGHT"


def test_swipe_up():
    d = SwipeDetector()
    d.update(hand_at(0.0, 5.0), now=0.0)
    result = d.update(hand_at(0.0, 3.0), now=0.1)   # y decreased = moved up
    assert result == "SWIPE_UP"


def test_small_movement_is_ignored():
    d = SwipeDetector()
    d.update(hand_at(0.0, 0.0), now=0.0)
    result = d.update(hand_at(0.1, 0.0), now=0.1)   # small move
    assert result == "NONE"

