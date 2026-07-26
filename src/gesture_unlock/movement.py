"""Swipe detection part and it recognises which direction the hand moved over recent frames"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

WRIST = 0
MIDDLE_KNUCKLE = 9

# Minimum RAW movement (fraction of image, before any scaling) to count as a swipe.
# The hand-size scale collapses when tracking wobbles and inflates noise, so we
# threshold the raw movement directly, which the data shows is clean and stable.
SWIPE_DISTANCE = 0.17
# Above this the movement is likely a tracking glitch or the hand left the frame.
SWIPE_DISTANCE_MAX = 0.45
# Setting the seconds to look back when measureing the movement
LOOKBACK = 0.5
# After the swipe setting it to ignore news ones for a specific time to stop the repeats being counted and
# avoid resetting
COOLDOWN = 0.4
# After a swipe the hand must slow below this raw speed (per sample) to read the next one
STILL_THRESHOLD = 0.03


# Setting the 8 directions by the angle. Using atan2 gives 0 = right and 90 = up for example up to 360 degrees
DIRECTIONS = [
    (0, "SWIPE_RIGHT"),
    #(45, "SWIPE_UP_RIGHT"),
    (90, "SWIPE_UP"),
    #(135, "SWIPE_UP_LEFT"),
    (180, "SWIPE_LEFT"),
    #(225, "SWIPE_DOWN_LEFT"),
    (270, "SWIPE_DOWN"),
    #(315, "SWIPE_DOWN_RIGHT"),
]

@dataclass
class Sample:
    """One recorded palm position at a moment in time"""
    x: float
    y: float
    t: float
    pose: str = "UNKNOWN"   # the hand shape at this moment

def hand_scale(points: np.ndarray) -> float:
    """A pose-stable hand size: average spread of the palm-base points from the
    palm centre. Landmarks 0, 5, 9, 13, 17 are the wrist and finger-base knuckles
    on the rigid palm, so they barely move when fingers extend or curl. This keeps
    the scale constant across poses (fist, open palm, thumbs up)."""
    palm_pts = points[[0, 5, 9, 13, 17]]
    center = palm_pts.mean(axis=0)
    spread = float(np.linalg.norm(palm_pts - center, axis=1).mean())
    return spread

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
        self._armed = True 

    def update(self, points: np.ndarray, now: float, pose: str = "UNKNOWN") -> str:
        """Feeds the current hand landmarks, time and pose. Returns a swipe name or NONE."""
        scale = hand_scale(points)
        if scale == 0:
            return "NONE"

        # Track the raw palm center (average of wrist and finger base knuckles) in
        # image coordinates -- NOT divided by hand size. The size scale collapses when
        # tracking wobbles and inflates noise; the raw movement is stable and clean.
        # Negate x: the webcam view is mirrored, so rightward motion must read as +x.
        palm = (points[0] + points[5] + points[9] + points[13] + points[17]) / 5.0
        self._history.append(Sample(x=-palm[0], y=palm[1], t=now, pose=pose))

        # Drops the samples that are older then the lookback window
        while self._history and now - self._history[0].t > LOOKBACK:
            self._history.popleft()

        # If the user just swiped this makes it stay disarmed until the hand goes nearly still
        if not self._armed:
            if len(self._history) >= 2:
                recent = self._history[-1]
                prev = self._history[-2]
                speed = math.hypot(recent.x - prev.x, recent.y - prev.y)
                if speed < STILL_THRESHOLD:
                    self._armed = True       # hand settled so ready again
            return "NONE"

        if len(self._history) < 2:
            return "NONE"

        # When there's movement from the oldest sample to the newest
        start = self._history[0]
        end = self._history[-1]
        dx = end.x - start.x
        dy = end.y - start.y
        distance = math.hypot(dx, dy)

        # Reject too small (jitter) or too big (tracking glitch / hand left frame).
        if distance < SWIPE_DISTANCE or distance > SWIPE_DISTANCE_MAX:
            return "NONE"

        # Pick direction by the DOMINANT axis: whichever of horizontal/vertical
        # moved more decides it. This matches intent better than a pure angle,
        # because swipes often arc (a "swipe right" drifts down but is still right).
        if abs(dx) >= abs(dy):
            direction = "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
        else:
            direction = "SWIPE_UP" if dy < 0 else "SWIPE_DOWN"
        self._armed = False          # disarm until the hand settles
        self._history.clear()
        return direction