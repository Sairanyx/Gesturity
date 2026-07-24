"""Temporal smoothing: turning noisy per-frame guesses into a stable reading/decision"""

from collections import deque
from dataclasses import dataclass

@dataclass
class StableGesture:
    """The current smoothed out result"""
    name: str        # here is the gesture the buffer takes and agrees on
    is_stable: bool  # this will be True once it has held for long enough to be accepted

class GestureStabiliser:
    """Smooths a stream of gesture guesses over a sliding time window"""

    def __init__(self, window_size: int = 5, hold_seconds: float = 0.4):
        # Latest window size guesses gets stored and the oldest drops off automatically from the deque
        self._history: deque[str] = deque(maxlen=window_size)
        # How long the majority gesture should be before it gets called stable
        self._hold_seconds = hold_seconds
        # The gesture that is now being timed and also when it first got seen
        self._current = "UNKKOWN"
        self._current_since = 0.0

    def update(self, guess:str, now:float) -> StableGesture:
        """Feeds one guess of frame plus the current time and gets the smoothed out result"""
        self._history.append(guess)
        majority = self._majority_vote()

        # If majority changes then restarts the stability timer
        if majority != self._current:
            self._current = majority
            self._current_since = now

        held_for = now - self._current_since
        is_stable = held_for >= self._hold_seconds
        return StableGesture(name=self._current, is_stable=is_stable)
        
    def _majority_vote(self) -> str:
        """Returns the guess that appears most often in the history buffer"""
        counts: dict[str, int] = {}
        for name in self._history:
            counts[name] = counts.get(name, 0) + 1

            best_name = "UNKNOWN"
            best_count = 0
            for name in counts:
                if counts[name] > best_count:
                    best_name = name
                    best_count = counts[name]
            return best_name