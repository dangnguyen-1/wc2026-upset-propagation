"""Tests for upset_propagation.l1_matcher.

Coverage focus: compute_l1_distance (the pure-math core of the matcher).
Hand-built synthetic survival dicts make the expected distances trivially
verifiable, which is what we want - the function's correctness IS the
matcher's correctness.

Skipped here: find_best_scenarios_l1 end-to-end. That path requires a
calibrated predictor (4-minute Nelder-Mead) plus a library of scenario
JSON files - too expensive for unit tests. The integration is exercised
via state_matcher tests (test_state_matcher.py) and via the cron pipeline
smoke tests we ran during item 7 development.
"""

from __future__ import annotations

import pytest

from upset_propagation.l1_matcher import compute_l1_distance


# compute_l1_distance: hand-computed cases


def test_l1_distance_identical_dicts_is_zero():
    """Same survival → L1 distance is exactly 0."""
    s = {
        "France": {"R16": 0.8, "QF": 0.5, "SF": 0.3, "F": 0.18, "Win": 0.10},
    }
    assert compute_l1_distance(s, s) == 0.0


def test_l1_distance_single_team_single_round_difference():
    """One team's R16 changes by 0.3 → distance = 0.3 (Win only)."""
    a = {"France": {"Win": 0.15}}
    b = {"France": {"Win": 0.45}}
    assert compute_l1_distance(a, b, rounds=("Win",)) == pytest.approx(0.30)


def test_l1_distance_sums_across_rounds():
    """Multiple rounds → distances sum."""
    a = {"France": {"R16": 0.8, "QF": 0.5, "Win": 0.10}}
    b = {"France": {"R16": 0.7, "QF": 0.5, "Win": 0.12}}
    # |0.1| + |0| + |0.02| = 0.12
    distance = compute_l1_distance(a, b, rounds=("R16", "QF", "Win"))
    assert distance == pytest.approx(0.12)


def test_l1_distance_sums_across_teams():
    """Multiple teams → all team distances summed."""
    a = {"France": {"Win": 0.10}, "Spain": {"Win": 0.15}}
    b = {"France": {"Win": 0.12}, "Spain": {"Win": 0.13}}
    # |0.02| + |0.02| = 0.04
    assert compute_l1_distance(a, b, rounds=("Win",)) == pytest.approx(0.04)


def test_l1_distance_team_missing_in_one_side():
    """Team present in only one side contributes its full probability mass.

    A team that appears only in the scenario (not the realised state)
    should add to the distance - this is what the docstring promises and
    why scenarios with different KO line-ups get correctly larger distances.
    """
    a = {"France": {"Win": 0.10}}  # only France
    b = {"France": {"Win": 0.10}, "Senegal": {"Win": 0.02}}  # France + Senegal
    # France: 0; Senegal: |0 - 0.02| = 0.02
    assert compute_l1_distance(a, b, rounds=("Win",)) == pytest.approx(0.02)


def test_l1_distance_round_missing_in_team_treated_as_zero():
    """Missing round-key on a team is treated as 0 probability for that round."""
    a = {"France": {"R16": 0.8, "Win": 0.10}}  # no QF
    b = {"France": {"R16": 0.8, "QF": 0.5, "Win": 0.10}}
    # R16: 0; QF: |0 - 0.5| = 0.5; Win: 0
    assert compute_l1_distance(a, b, rounds=("R16", "QF", "Win")) == pytest.approx(0.5)


def test_l1_distance_is_symmetric():
    """L1 is a metric - d(a, b) == d(b, a). Sanity guard."""
    a = {"France": {"Win": 0.10}, "Spain": {"Win": 0.15}}
    b = {"France": {"Win": 0.12}, "Senegal": {"Win": 0.02}}
    assert compute_l1_distance(a, b) == compute_l1_distance(b, a)


def test_l1_distance_non_negative():
    """L1 is always ≥ 0 (sum of absolute values)."""
    a = {"France": {"Win": 0.10}}
    b = {"France": {"Win": -0.05}}  # nonsensical but the function shouldn't crash
    distance = compute_l1_distance(a, b)
    assert distance >= 0


def test_l1_distance_default_rounds_covers_all_six():
    """Default rounds argument = all 6 KO rounds. Different from Win-only result."""
    a = {"France": {"R32": 1.0, "R16": 0.8, "Win": 0.10}}
    b = {"France": {"R32": 1.0, "R16": 0.7, "Win": 0.10}}
    # All-rounds distance includes R16 contribution; Win-only does not
    d_all = compute_l1_distance(a, b)  # default rounds
    d_win = compute_l1_distance(a, b, rounds=("Win",))
    assert d_all > d_win  # all-rounds picks up the R16 difference
    assert d_win == 0.0   # Win-only sees nothing


def test_l1_distance_empty_dicts():
    """Both dicts empty → distance is 0."""
    assert compute_l1_distance({}, {}) == 0.0