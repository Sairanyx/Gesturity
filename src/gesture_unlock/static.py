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

# The thumb curls sideways and distance from the wrist was unreliable
# Measuring how the thumb is, so straight is near 180 degrees, curled is smaller
# Adding the three thumb joins according to the model from base to tip
THUMB_BASE = 1
THUMB_MID = 2
THUMB_TIP = 4
# The thumb means its exteneded when at least the below degrees
THUMB_ANGLE_THRESHOLD = 150.0

@dataclass
class GestureResult:
    name: str
    score: float
    fingers: dict[str, bool] = field(default_factory=dict)

def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a -b))

def angle_at(a: np.ndarray, b: np.ndarray, c:np.ndarray) -> float:
    """Returns the angle in degrees at point b so between the bones or joints b to a and b to c"""
    bone1 = a - b
    bone2 = c - b
    cosine = np.dot(bone1, bone2) / (np.linalg.norm(bone1) * np.linalg.norm(bone2))
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def finger_states(points: np.ndarray) -> dict[str, bool]:
    """Returns {finger_name: is_extended} for all the five fingers"""
    wrist = points[WRIST]
    states = {}
    for name in FINGERTIPS:
        if name == "thumb":
            # Thumb is extended when it is almost straight so a large bend angle
            angle = angle_at(points[THUMB_BASE], points[THUMB_MID], points[THUMB_TIP])
            states[name] =angle >= THUMB_ANGLE_THRESHOLD
        else:
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
    score = 1.0 if name != "UNKNOWN" else 0.0
    return GestureResult(name=name, score=score, fingers=states)

