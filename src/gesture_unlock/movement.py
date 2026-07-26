"""Swipe detection part and it recognises which direction the hand moved over recent frames"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

WRIST = 0
MIDDLE_KNUCKLE = 9

# Setting the minimum distance in hand widths for the wrist to count as a swipe
SWIPE_DISTANCE = 1.2
# Setting the seconds to look back when measureing the movement
LOOKBACK = 0.3
# After the swipe setting it to ignore news ones for a specific time to stop the repeats being counted and
# avoid resetting
COOLDOWN = 0.6

# Setting the 8 directions by the angle. Using atan2 gives 0 = right and 90 = up for example up to 360 degrees
DIRECTIONS = [
    (0, "SWIPE_RIGHT"),
    (45, "SWIPE_UP_RIGHT"),
    (90, "SWIPE_UP"),
    (135, "SWIPE_UP_LEFT"),
    (180, "SWIPE_LEFT"),
    (225, "SWIPE_DOWN_LEFT"),
    (270, "SWIPE_DOWN"),
    (315, "SWIPE_DOWN_RIGHT"),
]

@dataclass
class Sample:
    """One recorded wrist position at a moment in time"""
    x: float
    y: float
    t: float

def hand_scale(points: np.ndarray) -> float:
    """Distance from wrist to middle knuckle: our unit of hand size"""
    return float(np.linalg.norm(points[MIDDLE_KNUCKLE] - points[WRIST]))

def snap_direction(angle_deg: float) -> str:
    """Rounds a movement angle to the nearest of the 8 names directions"""
    best_name = "NONE"
    best_diff = 999.0
    for target, name in DIRECTIONS:
        # Setting the minimum difference between the angle and this direction
        diff = abs((angle_deg - target + 180) % 360 - 180)
        if diff < best_diff:
            best_diff = diff
            best_name = name
    return best_name

class SwipeDetector:
    """Tracks the wrist over time and reports directional swipes"""

    def __init__(self):
        self._history: deque[Sample] = deque()
        self._last_swipe_time = -999.0

    def update(self, points: np.ndarray, now: float) -> str:
        """Feeds the current hand landmarks + time and returns a swipe name or NONE"""
        scale = hand_scale(points)
        if scale == 0:
            return "NONE"

        # Records the wrist's position that is scaled by the hand size
        wrist = points[WRIST]
        self._history.append(Sample(x=wrist[0] / scale, y=wrist[1] / scale, t=now))

        # Drops the samples that are older then the lookback window
        while self._history and now - self._history[0].t > LOOKBACK:
            self._history.popleft()

        # If still in cooldown from the last swipe it does nothing
        if now - self._last_swipe_time < COOLDOWN:
            return "NONE"

        if len(self._history) < 2:
            return "NONE"

        # When theres movemene from the oldest sample to the newest
        start = self._history[0]
        end = self._history[-1]
        dx = end.x - start.x
        dy = end.y - start.y
        distance = math.hypot(dx, dy)

        if distance < SWIPE_DISTANCE:
            return "NONE"

        # atan 2 gives the movement angle and this negates the dy because image y is growing downward
        angle = math.degrees(math.atan2(-dy, dx)) % 360
        self._last_swipe_time = now
        self._history.clear()
        return snap_direction(angle)