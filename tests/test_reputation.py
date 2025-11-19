# pylint: disable=protected-access, duplicate-code
# pylint: disable=redefined-outer-name
"""
Unit tests for the ReputationManager class.
"""

import pytest
from src.reputation import ReputationManager, MAX_REPUTATION

@pytest.fixture
def rep_manager():
    """
    Pytest fixture to provide a ReputationManager instance for tests.
    """
    return ReputationManager(decay_rate=0) # Disable time decay for predictable tests

def test_apply_positive_increases_reputation(rep_manager):
    """
    Test that a positive event increases reputation.
    """
    initial_rep = MAX_REPUTATION / 2
    exp_pos = 0
    perf_ema = 0.5

    new_rep, new_exp_pos, _ = rep_manager.apply_positive(initial_rep, exp_pos, perf_ema)

    assert new_rep > initial_rep
    assert new_exp_pos == exp_pos + 1

def test_apply_negative_decreases_reputation(rep_manager):
    """
    Test that a negative event decreases reputation.
    """
    initial_rep = MAX_REPUTATION / 2
    exp_pos = 10  # Assume some positive history
    perf_ema = 0.5

    new_rep, new_exp_pos, _ = rep_manager.apply_negative(initial_rep, exp_pos, perf_ema)

    assert new_rep < initial_rep
    assert new_exp_pos == exp_pos # Should not change on negative event

def test_reputation_cannot_exceed_max(rep_manager):
    """
    Test that reputation is capped at MAX_REPUTATION.
    """
    initial_rep = MAX_REPUTATION - 1
    exp_pos = 1000 # A very high number of positive experiences
    perf_ema = 1.0

    new_rep, _, _ = rep_manager.apply_positive(initial_rep, exp_pos, perf_ema)

    # It should grow, but not exceed the max
    assert new_rep > initial_rep
    assert new_rep <= MAX_REPUTATION

def test_reputation_cannot_go_below_zero(rep_manager):
    """
    Test that reputation is floored at 0.
    """
    initial_rep = 1.0
    exp_pos = 0
    perf_ema = 0.0

    # Apply negative events repeatedly
    rep = initial_rep
    for _ in range(10):
        rep, _, _ = rep_manager.apply_negative(rep, exp_pos, perf_ema)

    assert rep >= 0.0

def test_gompertz_target(rep_manager):
    """
    Test the Gompertz target function behavior.
    """
    # With 0 positive experiences, target should be approximately neutral
    target_0 = rep_manager._gompertz_target(0)
    assert target_0 == pytest.approx(MAX_REPUTATION / 2, abs=0.01)

    # With many positive experiences, target should approach max_rep
    target_100 = rep_manager._gompertz_target(100)
    assert target_100 > MAX_REPUTATION * 0.99
    assert target_100 <= MAX_REPUTATION
