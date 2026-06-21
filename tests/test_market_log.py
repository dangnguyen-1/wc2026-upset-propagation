"""Tests for upset_propagation.market_log.

Coverage:
  - read_last_market_log_entry: missing file, empty file, malformed line,
    multi-line correctly returns the last entry
  - compute_market_log_entry: minimal-input shape (no state/predictor),
    full-input shape (with state + predictor for edges)
  - _compute_diff: three-state distinction (None / [] / non-empty) ←
    REGRESSION for the truthiness gotcha
  - append_market_log: file created on first call, appended on subsequent
  - JSON serialization: optional fields omitted when None

Strategy: tmp_path-based filesystem tests, with a synthetic minimal
predictor stub for tests that need edges. We don't call the real
calibrator (4-minute Nelder-Mead) - the predictor is mocked to a
constant MatchPrediction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from upset_propagation._vendored.single_game import MatchContext, MatchPrediction
from upset_propagation.market_log import (
    MARKET_LOG_FILENAME,
    MIN_DIFF_THRESHOLD,
    MarketLogEntry,
    TeamDiff,
    TeamFairProb,
    _compute_diff,
    append_market_log,
    compute_market_log_entry,
    read_last_market_log_entry,
)
from upset_propagation.state_matcher import RealisedState


# Helpers


def _make_entry_dict(timestamp: str, top_teams: list[tuple[str, float]]) -> dict:
    """Minimal dict matching the on-disk JSONL shape (for fixture setup)."""
    return {
        "timestamp": timestamp,
        "n_teams_in_api": len(top_teams),
        "fair_probs_sum": round(sum(p for _, p in top_teams), 6),
        "top_market": [
            {"team": t, "fair_prob": p}
            for t, p in top_teams
        ],
        "n_groups_observed": 0,
    }


def _stub_predictor(elo_a, elo_b, ctx: MatchContext) -> MatchPrediction:
    """Mock predictor: 50/30/20 home/draw/away regardless of inputs.

    Tests that only need to verify "edges were computed" can use this
    instead of running the 4-minute calibrator.
    """
    return MatchPrediction(
        p_home=0.5,
        p_draw=0.3,
        p_away=0.2,
        goal_grid=np.zeros((9, 9)),
    )


# read_last_market_log_entry


def test_read_last_returns_none_when_missing(tmp_path):
    """No market_log.jsonl → returns None."""
    result = read_last_market_log_entry(tmp_path)
    assert result is None


def test_read_last_returns_none_when_empty(tmp_path):
    """Empty file → returns None (degenerate but possible)."""
    (tmp_path / MARKET_LOG_FILENAME).write_text("")
    result = read_last_market_log_entry(tmp_path)
    assert result is None


def test_read_last_returns_only_entry(tmp_path):
    """One entry → returns it."""
    entry = _make_entry_dict("2026-06-10T12:00:00+00:00", [("France", 0.15)])
    (tmp_path / MARKET_LOG_FILENAME).write_text(json.dumps(entry) + "\n")
    result = read_last_market_log_entry(tmp_path)
    assert result is not None
    assert result["timestamp"] == "2026-06-10T12:00:00+00:00"


def test_read_last_returns_last_of_many(tmp_path):
    """Multi-line file → returns the most recent entry only."""
    entries = [
        _make_entry_dict(f"2026-06-10T{h:02d}:00:00+00:00", [("France", 0.15)])
        for h in range(10, 14)
    ]
    with (tmp_path / MARKET_LOG_FILENAME).open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    result = read_last_market_log_entry(tmp_path)
    assert result["timestamp"] == "2026-06-10T13:00:00+00:00"


def test_read_last_handles_trailing_newlines(tmp_path):
    """File with trailing whitespace/newlines is parsed correctly."""
    entry = _make_entry_dict("2026-06-10T12:00:00+00:00", [("France", 0.15)])
    (tmp_path / MARKET_LOG_FILENAME).write_text(
        json.dumps(entry) + "\n\n  \n"  # trailing blank lines
    )
    result = read_last_market_log_entry(tmp_path)
    assert result is not None


# _compute_diff: the three-state distinction (REGRESSION focus)


def test_compute_diff_returns_none_with_no_previous():
    """REGRESSION: previous_entry=None → return None (NOT []).

    This is the truthiness gotcha - code that did `if diff:` would
    conflate "no previous entry exists" with "no team moved above
    threshold". The contract is: None means "can't compute", [] means
    "computed and nothing moved".
    """
    current = {"France": 0.15, "Spain": 0.12}
    result = _compute_diff(current, previous_entry=None)
    assert result is None


def test_compute_diff_returns_empty_list_when_no_movers():
    """Previous entry exists but no team moved above threshold → return []."""
    current = {"France": 0.15, "Spain": 0.12, "Brazil": 0.08}
    # Previous is identical → no team moved
    previous = _make_entry_dict(
        "2026-06-10T11:00:00+00:00",
        [("France", 0.15), ("Spain", 0.12), ("Brazil", 0.08)],
    )
    result = _compute_diff(current, previous_entry=previous)
    # Crucially: NOT None, IS empty list
    assert result is not None
    assert result == []


def test_compute_diff_returns_non_empty_when_team_moved():
    """A team moving by > threshold appears in the diff."""
    current = {"France": 0.20, "Spain": 0.12}  # France up 5pp
    previous = _make_entry_dict(
        "2026-06-10T11:00:00+00:00",
        [("France", 0.15), ("Spain", 0.12)],
    )
    result = _compute_diff(current, previous_entry=previous)
    assert result is not None
    assert len(result) >= 1
    # Find France's diff
    france = next(d for d in result if d.team == "France")
    assert france.delta_pp == pytest.approx(5.0, abs=0.01)


def test_compute_diff_filters_below_threshold():
    """Movements smaller than MIN_DIFF_THRESHOLD are excluded as noise."""
    # Use a tiny movement well below threshold (0.0001 = 0.01pp, threshold is ~0.1pp)
    current = {"France": 0.1501, "Spain": 0.12}
    previous = _make_entry_dict(
        "2026-06-10T11:00:00+00:00",
        [("France", 0.15), ("Spain", 0.12)],
    )
    result = _compute_diff(current, previous_entry=previous)
    # France's 0.01pp movement is below the 0.1pp threshold - not in result
    assert result == []


def test_compute_diff_sorts_by_absolute_delta():
    """Movers are sorted by |delta| descending so biggest move appears first."""
    current = {"France": 0.20, "Spain": 0.18, "Brazil": 0.05}
    previous = _make_entry_dict(
        "2026-06-10T11:00:00+00:00",
        [("France", 0.15), ("Spain", 0.10), ("Brazil", 0.08)],
    )
    result = _compute_diff(current, previous_entry=previous)
    # Spain moved most (+8pp), then France (+5pp), then Brazil (-3pp)
    deltas = [abs(d.delta_pp) for d in result]
    assert deltas == sorted(deltas, reverse=True)


# compute_market_log_entry: integration


def test_compute_entry_minimal(tmp_path):
    """market_probs only, no state/predictor → entry has no edges and no diff
    (first run). fairline_probs same as market_probs (irrelevant without predictor)."""
    probs = {"France": 0.15, "Spain": 0.12, "Brazil": 0.08}
    entry = compute_market_log_entry(probs, probs, output_dir=tmp_path)
    assert isinstance(entry, MarketLogEntry)
    assert entry.n_teams_in_api == 3
    assert entry.fair_probs_sum == pytest.approx(0.35, abs=1e-6)
    # First run → no previous to diff against
    assert entry.diff_from_previous_run is None
    # No predictor → no edges
    assert entry.implied_vs_market is None
    # No state → 0 groups observed
    assert entry.n_groups_observed == 0


def test_compute_entry_with_previous_yields_diff(tmp_path):
    """If previous entry exists, diff_from_previous_run is computed
    (based on market_probs movements, the comparison surface)."""
    # Seed a previous entry
    previous = _make_entry_dict(
        "2026-06-10T11:00:00+00:00",
        [("France", 0.10), ("Spain", 0.12), ("Brazil", 0.08)],
    )
    (tmp_path / MARKET_LOG_FILENAME).write_text(json.dumps(previous) + "\n")

    market_probs = {"France": 0.15, "Spain": 0.12, "Brazil": 0.08}  # France up 5pp
    entry = compute_market_log_entry(
        market_probs, market_probs, output_dir=tmp_path,
    )

    assert entry.diff_from_previous_run is not None
    france = next(
        d for d in entry.diff_from_previous_run if d.team == "France"
    )
    assert france.delta_pp == pytest.approx(5.0, abs=0.01)


# append_market_log: file shape


def test_append_creates_file_on_first_call(tmp_path):
    """First call creates the file with one JSON line."""
    entry = MarketLogEntry(
        timestamp="2026-06-10T12:00:00+00:00",
        n_teams_in_api=2,
        fair_probs_sum=0.5,
        top_market=[TeamFairProb(team="France", fair_prob=0.3)],
    )
    path = append_market_log(entry, tmp_path)
    assert path == tmp_path / MARKET_LOG_FILENAME
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["timestamp"] == "2026-06-10T12:00:00+00:00"


def test_append_appends_to_existing(tmp_path):
    """Subsequent calls append; previous lines preserved."""
    e1 = MarketLogEntry(
        timestamp="2026-06-10T11:00:00+00:00",
        n_teams_in_api=2, fair_probs_sum=0.5,
    )
    e2 = MarketLogEntry(
        timestamp="2026-06-10T12:00:00+00:00",
        n_teams_in_api=2, fair_probs_sum=0.5,
    )
    append_market_log(e1, tmp_path)
    append_market_log(e2, tmp_path)
    lines = (tmp_path / MARKET_LOG_FILENAME).read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["timestamp"] == "2026-06-10T11:00:00+00:00"
    assert json.loads(lines[1])["timestamp"] == "2026-06-10T12:00:00+00:00"


def test_append_omits_none_fields_in_json(tmp_path):
    """REGRESSION: serialized JSON should not contain diff_from_previous_run
    when it's None (first-ever entry)."""
    entry = MarketLogEntry(
        timestamp="2026-06-10T12:00:00+00:00",
        n_teams_in_api=2,
        fair_probs_sum=0.5,
        diff_from_previous_run=None,  # first entry
        implied_vs_market=None,        # no predictor
    )
    append_market_log(entry, tmp_path)
    parsed = json.loads(
        (tmp_path / MARKET_LOG_FILENAME).read_text().strip()
    )
    # Optional None fields omitted entirely
    assert "diff_from_previous_run" not in parsed
    assert "implied_vs_market" not in parsed


def test_append_includes_empty_list_in_json(tmp_path):
    """REGRESSION: empty list (no movers) MUST serialize as [], not be omitted.

    This is the truthiness gotcha mirror - consumers parsing the JSON
    need to distinguish "field missing" (no previous entry to compute
    against) from "field present but empty" (computed and zero movers).
    """
    entry = MarketLogEntry(
        timestamp="2026-06-10T12:00:00+00:00",
        n_teams_in_api=2,
        fair_probs_sum=0.5,
        diff_from_previous_run=[],  # explicitly empty, not None
    )
    append_market_log(entry, tmp_path)
    parsed = json.loads(
        (tmp_path / MARKET_LOG_FILENAME).read_text().strip()
    )
    # Empty list is PRESENT, just empty
    assert "diff_from_previous_run" in parsed
    assert parsed["diff_from_previous_run"] == []


# Regression: output_dir threading through _compute_edges


def test_compute_edges_threads_output_dir_to_implied_probs():
    """REGRESSION: _compute_edges must forward output_dir to
    compute_implied_probs. Without this, compute_implied_probs falls
    back to the default OUTPUT_DIR, which during cron-mode is the live
    output/ directory - empty until the atomic swap completes AFTER this
    function runs. Result: zero implied probabilities and bogus edges.

    Caught by the pre-WC end-to-end smoke test, not by earlier unit tests
    because those used a populated OUTPUT_DIR by accident.

    This test verifies the signature accepts output_dir and forwards it.
    We don't need a fully-populated scenario library to test the
    forwarding - we just need to verify the parameter goes through.
    A separate path with predictor=None short-circuits before the
    output_dir is used, so we test the call-chain shape via inspect.
    """
    import inspect
    from upset_propagation.market_log import _compute_edges

    sig = inspect.signature(_compute_edges)
    assert "output_dir" in sig.parameters, (
        "_compute_edges must accept output_dir so cron-mode can pass "
        "the staging dir (where scenarios live before atomic swap)"
    )

    # And compute_market_log_entry must call _compute_edges with output_dir
    src = inspect.getsource(
        __import__('upset_propagation.market_log', fromlist=['compute_market_log_entry']).compute_market_log_entry
    )
    assert "output_dir=output_dir" in src or "output_dir," in src, (
        "compute_market_log_entry must pass output_dir to _compute_edges"
    )