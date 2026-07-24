"""The stabiliser has to ignore the single frame noise and require a hold"""

from gesture_unlock.stability import GestureStabiliser

def test_single_blip_is_ignored():
    s = GestureStabiliser(window_size=5, hold_seconds=0.4)
    # Four FISTs and then one strat UNKNOWN
    s.update("FIST", now=0.0)
    s.update("FIST", now=0.1)
    s.update("FIST", now=0.2)
    s.update("FIST", now=0.3)
    result = s.update("UNKNOWN", now=0.4)
    # Majority is still FIST so the blip does not change the decision
    assert result.name == "FIST"

def test_gesture_becomes_stable_after_hold():
    s = GestureStabiliser(window_size=3, hold_seconds=0.4)
    early = s.update("PEACE", now=0.0)
    assert early.is_stable is False        # just appeared and not held yet
    s.update("PEACE", now=0.2)
    late = s.update("PEACE", now=0.5)
    assert late.is_stable is True          # held for 0.5s >= 0.4s

def test_changing_gesture_resets_timer():
    s = GestureStabiliser(window_size=3, hold_seconds=0.4)
    s.update("FIST", now=0.0)
    s.update("FIST", now=0.1)              # FIST is the established majority
    # Now feed enough PEACE frames to make PEACE the new majority.
    s.update("PEACE", now=0.5)             # history: FIST, FIST, PEACE -> still FIST
    s.update("PEACE", now=0.6)             # history: FIST, PEACE, PEACE -> PEACE wins now
    switched = s.update("PEACE", now=0.7)  # history: PEACE, PEACE, PEACE
    # PEACE only just became the majority so now its hold timer is fresh -> not stable yet.
    assert switched.name == "PEACE"
    assert switched.is_stable is False
