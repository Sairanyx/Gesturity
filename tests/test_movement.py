"""Swipe detection must report the correct direction and ignore small fluctuations"""

import numpy as np

from gesture_unlock.movement import SwipeDetector


def hand_at(x: float, y: float) -> np.ndarray:
    """A fake hand centred at (x, y). All palm landmarks move together so the
    palm-centre (average of 0,5,9,13,17) sits at (x, y) too. Scale = 1."""
    points = np.zeros((21, 3), dtype=np.float32)
    for i in (0, 5, 13, 17):
        points[i] = [x, y, 0.0]        # wrist and finger base knuckles
    points[9] = [x, y + 1.0, 0.0]      # mid knuckle 1 unit away so scale = 1
    return points


def test_swipe_right():
    # The webcam view is mirrored, so a real rightward swipe moves the raw x
    # in the NEGATIVE direction. The detector negates x to correct this. Movement
    # is now measured in RAW image units, so ~0.15 is a clear swipe.
    d = SwipeDetector()
    d.update(hand_at(0.0, 0.0), now=0.0, pose="FIST")
    result = d.update(hand_at(-0.25, 0.0), now=0.1, pose="FIST")   # mirrored: -x is right
    assert result == "SWIPE_RIGHT"


def test_swipe_up():
    d = SwipeDetector()
    d.update(hand_at(0.0, 0.5), now=0.0, pose="FIST")
    result = d.update(hand_at(0.0, 0.25), now=0.1, pose="FIST")   # y decreased = up
    assert result == "SWIPE_UP"


def test_small_movement_is_ignored():
    d = SwipeDetector()
    d.update(hand_at(0.0, 0.0), now=0.0, pose="FIST")
    result = d.update(hand_at(0.03, 0.0), now=0.1, pose="FIST")   # tiny move
    assert result == "NONE"



