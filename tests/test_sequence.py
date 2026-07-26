"""Testing the sequence engine works as expected, advances, resets and completes correctly"""

import pytest

from gesture_unlock.sequence import SequenceEngine, SequenceEvent

def test_full_sequence_completes():
    engine = SequenceEngine(["FIST", "PEACE", "OPEN_PALM"])
    assert engine.update("FIST").event == SequenceEvent.ADVANCED
    assert engine.update("PEACE").event == SequenceEvent.ADVANCED
    assert engine.update("OPEN_PALM").event == SequenceEvent.COMPLETED


def test_wrong_in_sequence_gesture_resets():
    # A gesture that IS in the password but out of order is a real mistake -> reset.
    engine = SequenceEngine(["FIST", "PEACE", "OPEN_PALM"])
    engine.update("FIST")                        # step 1
    result = engine.update("OPEN_PALM")          # in-sequence but wrong position
    assert result.event == SequenceEvent.RESET
    assert result.step == 0


def test_noise_gesture_is_ignored():
    # A gesture NOT in the password is just noise -> ignored, progress kept.
    engine = SequenceEngine(["FIST", "PEACE", "OPEN_PALM"])
    engine.update("FIST")                        # step 1
    result = engine.update("THUMBS_UP")          # not in sequence -> ignored
    assert result.event == SequenceEvent.NONE
    assert result.step == 1                       # progress kept



def test_repeated_gesture_is_ignored():
    engine = SequenceEngine(["FIST", "PEACE"])
    assert engine.update("FIST").event == SequenceEvent.ADVANCED
    assert engine.update("FIST").event == SequenceEvent.NONE   # same gesture, ignored


def test_unknown_is_ignored():
    engine = SequenceEngine(["FIST", "PEACE"])
    engine.update("FIST")
    assert engine.update("UNKNOWN").event == SequenceEvent.NONE
    # After UNKNOWN, PEACE should still advance (progress not lost).
    assert engine.update("PEACE").event == SequenceEvent.COMPLETED


def test_empty_sequence_rejected():
    with pytest.raises(ValueError):
        SequenceEngine([])