"""Static gesture recognition will classify hand-crafted synthetic hands"""

import numpy as np

from gesture_unlock.static import recognise

def synthetic_hand(extended: set[str]) -> np.ndarray:
    """This builds a fake normalised hand where the names fingers are extended
    The landmarks are placed in such was that extended fingertips sit far from the wrist
    and curled ones sit near it. Only the wrist, MCPs and tips need to be geometrically meaningfull
    for the distance-based rule."""

    points = np.zeros((21, 3), dtype=np.float32)
    tips = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    mcps = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}

    # Spreading the fingers along the x so the distances are independent per finger
    for i, name in enumerate(tips):
        x = float(i)
        points[mcps[name]] = [x, 1.0, 0.0]
        points[tips[name]] = [x, 3.0 if name in extended else 1.2, 0.0]
    return points

def test_open_palm():
    hand = synthetic_hand({"thumb", "index", "middle", "ring", "pinky"})
    assert recognise(hand).name == "OPEN_PALM"


def test_fist():
    hand = synthetic_hand(set())
    assert recognise(hand).name == "FIST"


def test_peace():
    hand = synthetic_hand({"index", "middle"})
    assert recognise(hand).name == "PEACE"


def test_thumbs_up():
    hand = synthetic_hand({"thumb"})
    assert recognise(hand).name == "THUMBS_UP"


def test_diagnostics_present():
    hand = synthetic_hand({"index", "middle"})
    result = recognise(hand)
    assert result.fingers["index"] is True
    assert result.fingers["ring"] is False

