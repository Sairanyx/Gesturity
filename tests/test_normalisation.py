"""Normalisation must be invariant to the hand postions and have apparent size"""

import numpy as np
import pytest

from gesture_unlock.normalisation import normalise

def fake_hand() -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.random((21, 3), dtype=np.float32)

def test_wrist_is_at_origin():
    result = normalise(fake_hand())
    assert np.allclose(result[0], [0.0, 0.0, 0.0])

def test_translation_invariance():
    hand = fake_hand()
    moved = hand + np.array([0.3, -0.2, 0.1], dtype=np.float32)
    assert np.allclose(normalise(hand), normalise(moved), atol=1e-5)

def test_scale_invariance():
    hand = fake_hand()
    doubled = hand * 2.0
    assert np.allclose(normalise(hand), normalise(doubled), atol=1e-5)


def test_degenerate_hand_raises():
    flat = np.zeros((21, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        normalise(flat) 