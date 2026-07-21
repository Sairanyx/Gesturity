"""Rule based static gesture recognistion from normalised landmarks"""

from dataclasses import dataclass, field

import numpy as np

WRIST = 0

# Fingertip landmark indices in finger order
FINGERTIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
# Knuckle joints, MCP, and the base of each finger
FINGER_MCP = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}

# Setting that a finger counts as extended when its tip is minimum that many palm lenghts
# further from the wrist than its base knuckles
EXTENSION_MARGIN = 0.6

@dataclass
class GestureResult:
    name: str
    score: float
    fingers: dict[str, bool] = field(default_factory=dict)

def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a -b))

def finger_states(points: np.ndarray) -> dict[str, bool]:
    """Returns {finger_name: is_extended} for all the five fingers"""
    wrist = points[WRIST]
    states = {}
    for name in FINGERTIPS:
        tip_dist = _distance(points[FINGERTIPS[name]], wrist)
        mcp_dist = _distance(points[FINGER_MCP[name]], wrist)
        states[name] = tip_dist > mcp_dist + EXTENSION_MARGIN
    return states

GESTURE_TABLE = {
    ("thumb", "index", "middle", "ring", "pinky"): "OPEN_PALM",
    (): "FIST",
    ("index", "middle"): "PEACE",
    ("thumb",): "THUMBS_UP",
}

def recognise(points: np.ndarray) -> GestureResult:
    """Classifying a normalised hand into one of the known gestures"""
    states = finger_states(points)
    extended = tuple(name for name in FINGERTIPS if states[name])
    name = GESTURE_TABLE.get(extended, "UNKNOWN")
    score = 1.0 if name != "UNKOWN" else 0.0
    return GestureResult(name=name, score=score, fingers=states)

