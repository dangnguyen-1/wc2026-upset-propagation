"""Tests for upset_propagation.ensemble_matcher.

Coverage focus: _assign_average_ranks (the Borda count's rank-resolution
core) and the EnsembleMatch dataclass. These are the pure-math pieces.

Skipped here: find_best_scenarios_ensemble end-to-end. That requires a
calibrated predictor + scenario JSON library - too expensive for unit
tests. The integration is exercised by item 3 smoke tests (state_matcher
ensemble path) and item 7 cron smoke tests.

Why fractional ranking matters: ties in distances must produce the same
rank for tied scenarios, otherwise our Borda sum would arbitrarily favor
one tied scenario over another - bug we want to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from upset_propagation.ensemble_matcher import (
    EnsembleMatch,
    _assign_average_ranks,
)


# _assign_average_ranks: pure-math tests


def test_average_ranks_strictly_sorted():
    """All-distinct distances → simple rank order 1, 2, 3, ..."""
    distances = [10.0, 20.0, 30.0, 40.0]
    ranks = _assign_average_ranks(distances)
    assert ranks == [1.0, 2.0, 3.0, 4.0]


def test_average_ranks_pairs_tied_at_top():
    """Two scenarios tied at lowest → both get rank 1.5 (mean of 1 and 2)."""
    distances = [10.0, 10.0, 30.0, 40.0]
    ranks = _assign_average_ranks(distances)
    assert ranks == [1.5, 1.5, 3.0, 4.0]


def test_average_ranks_three_tied():
    """Three scenarios tied at lowest → all rank 2 (mean of 1, 2, 3)."""
    distances = [10.0, 10.0, 10.0, 40.0]
    ranks = _assign_average_ranks(distances)
    assert ranks == [2.0, 2.0, 2.0, 4.0]


def test_average_ranks_all_tied():
    """All same distance → all rank N/2 (mean of all positions).

    Example with 4 elements: positions 1, 2, 3, 4 → mean 2.5.
    """
    distances = [15.0, 15.0, 15.0, 15.0]
    ranks = _assign_average_ranks(distances)
    assert ranks == [2.5, 2.5, 2.5, 2.5]


def test_average_ranks_preserves_input_order():
    """Returned ranks match input order - the i-th distance gets the i-th rank."""
    distances = [30.0, 10.0, 20.0]  # already reversed
    ranks = _assign_average_ranks(distances)
    # Position 0 holds distance 30 → it ranks 3rd
    # Position 1 holds distance 10 → ranks 1st
    # Position 2 holds distance 20 → ranks 2nd
    assert ranks == [3.0, 1.0, 2.0]


def test_average_ranks_empty_list():
    """Empty input → empty output (no crash)."""
    assert _assign_average_ranks([]) == []


def test_average_ranks_single_element():
    """One element → rank 1.0."""
    assert _assign_average_ranks([42.0]) == [1.0]


def test_average_ranks_floats():
    """Float-precision distances handled correctly."""
    distances = [0.001, 0.001, 0.002]
    ranks = _assign_average_ranks(distances)
    # First two tied → rank 1.5; third → rank 3
    assert ranks == [1.5, 1.5, 3.0]


def test_average_ranks_ties_in_middle():
    """Ties not at the top still get correct averaged ranks."""
    distances = [5.0, 15.0, 15.0, 30.0]
    ranks = _assign_average_ranks(distances)
    # Position 1 (15) and 2 (15) tied → mean of ranks 2 and 3 = 2.5
    assert ranks == [1.0, 2.5, 2.5, 4.0]


# EnsembleMatch dataclass mechanics


def test_ensemble_match_can_be_constructed():
    """EnsembleMatch with all fields populated. Verify the schema doesn't
    drift from what compute_top_ranking expects to consume."""
    match = EnsembleMatch(
        scenario_id="spain_runner_up_H",
        scenario_path=Path("/tmp/spain_runner_up_H.json"),
        borda_sum=2.0,
        score=1.0,
        per_matcher_ranks={"hamming": 1.0, "l1": 1.0},
        per_matcher_distances={"hamming": 0.0, "l1": 0.0},
    )
    assert match.scenario_id == "spain_runner_up_H"
    assert match.borda_sum == 2.0
    assert match.score == 1.0


def test_ensemble_match_defaults_for_optional_dicts():
    """per_matcher_ranks/distances default to empty dicts (don't crash on read)."""
    match = EnsembleMatch(
        scenario_id="baseline",
        scenario_path=Path("/tmp/baseline.json"),
        borda_sum=2.0,
        score=1.0,
    )
    assert match.per_matcher_ranks == {}
    assert match.per_matcher_distances == {}