"""Gesture sequence engine: a finite state machine (FSM) for gesture passwords"""
from dataclasses import dataclass
from enum import Enum

class SequenceEvent(Enum):
    """Checks what happens after feeding one gesture to the engine"""
    NONE = "none"             # ignored
    ADVANCED = "advanced"     # one step further
    RESET = "reset"           # wrong, back to start
    COMPLETED = "completed"   # right sequence


@dataclass
class SequenceState:
    """The result of the engine after an update"""
    event: SequenceEvent
    step: int                   # amount of correct gestures to that point
    total: int                  # length of the whole sequence

class SequenceEngine:
    """It tracks progress through a fixed gesture sequence"""

    def __init__(self, sequence: list[str]):
        if not sequence:
            raise ValueError("Sequence must contain at least one gesture.")
        self._sequence = sequence
        self._step = 0
        self._last_gesture = "UNKNOWN"

    def update(self, gesture: str) -> SequenceState:
        """It gets feed one stable gesture and returns what happened"""
        # Ignores UNKNOWN so hand is gone or unrecognised shape for now
        if gesture == "UNKNOWN":
            return self._result(SequenceEvent.NONE)

        # Ignore repeats of a HELD pose (fist held for many frames = one fist).
        # Swipes are discrete events, so they are never treated as repeats.
        is_swipe = gesture.startswith("SWIPE_")
        if not is_swipe and gesture == self._last_gesture:
            return self._result(SequenceEvent.NONE)
        self._last_gesture = gesture

        expected = self._sequence[self._step]
        if gesture == expected:
            self._step += 1
            if self._step == len(self._sequence):
                self._step = 0       # is ready for next sequence
                return self._result(SequenceEvent.COMPLETED)
            return self._result(SequenceEvent.ADVANCED)
        else:
            # Only reset if this gesture is actually part of the sequence
            # (a real out oforder mistake). Ignores gestures not in the password
            # at all as those are just noise and not a wrong step. Had it like this for convenience.
            if gesture in self._sequence:
                self._step = 0
                return self._result(SequenceEvent.RESET)
            return self._result(SequenceEvent.NONE)



    @property
    def step(self) -> int:
        """The current progress, always available even between updates"""
        return self._step

    def _result(self, event: SequenceEvent) -> SequenceState:
        return SequenceState(event=event, step=self._step, total=len(self._sequence))

