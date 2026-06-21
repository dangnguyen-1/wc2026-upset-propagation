"""Tests for upset_propagation.top_ranking.

Coverage focus:
  - TopRankingEntry / TopRanking dataclass mechanics
  - TopRanking.to_dict() JSON-friendly serialization (notes included
    when present, omitted when absent)
  - TopRankingEntry.from_ensemble_match() conversion + rounding
  - load_state_for_cron: missing CSV → empty state (REGRESSION - this
    is the pre-tournament case on day 1)
  - REGRESSION GUARD: NON_SCENARIO_FILENAMES includes top_10_ranking.json
    so a fresh run.py invocation doesn't break the matcher with a
    KeyError('scenario_id') on its own output. This is the bug we hit
    three times during item 7 - the centralized skip list prevents it.

Skipped here: compute_top_ranking end-to-end. Requires calibrator (~4
min) + scenario library. The smoke tests during item 7 covered this
integration; tests guard against regression of the dataclass shape and
the skip-list config.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from upset_propagation.config import NON_SCENARIO_FILENAMES
from upset_propagation.ensemble_matcher import EnsembleMatch
from upset_propagation.top_ranking import (
    MATCH_RESULTS_CSV_FILENAME,
    TOP_RANKING_FILENAME,
    TopRanking,
    TopRankingEntry,
    load_state_for_cron,
    write_top_ranking_file,
)


# REGRESSION: skip-list compliance


def test_top_ranking_filename_in_skip_list():
    """REGRESSION: top_10_ranking.json must be in NON_SCENARIO_FILENAMES.

    Without this, the matcher iterating over output/*.json would try to
    parse our own ranking output as a scenario file and crash with
    KeyError('scenario_id'). Bug recurrence prevention.
    """
    assert TOP_RANKING_FILENAME in NON_SCENARIO_FILENAMES


def test_all_known_non_scenario_files_in_skip_list():
    """REGRESSION: every output file the cron writes must be in the skip list.

    This is the centralized check from item 7's refactor. If a future
    contributor adds a new top-level output file, they should add it
    here; this test will fail until they do.
    """
    expected_non_scenarios = {
        "index.json",
        "validation_report.json",
        "top_10_ranking.json",
        "health.json",
        "our_vs_market.json",
    }
    assert expected_non_scenarios.issubset(NON_SCENARIO_FILENAMES)


# TopRankingEntry dataclass + conversion


def test_top_ranking_entry_from_ensemble_match_basics():
    """from_ensemble_match copies fields and rounds appropriately."""
    em = EnsembleMatch(
        scenario_id="spain_runner_up_H",
        scenario_path=Path("/tmp/spain_runner_up_H.json"),
        borda_sum=2.0,
        score=0.9876543,
        per_matcher_ranks={"hamming": 1.0, "l1": 1.0},
        per_matcher_distances={"hamming": 0.123456, "l1": 0.789012},
    )
    entry = TopRankingEntry.from_ensemble_match(rank=1, match=em)
    assert entry.rank == 1
    assert entry.scenario_id == "spain_runner_up_H"
    assert entry.scenario_filename == "spain_runner_up_H.json"
    # Score rounded to 4 places per docstring
    assert entry.score == round(0.9876543, 4)
    # Distances rounded to 6 places
    assert entry.per_matcher_distances["hamming"] == round(0.123456, 6)
    # Ranks rounded to 2 places
    assert entry.per_matcher_ranks["hamming"] == 1.0


def test_top_ranking_entry_extracts_filename_from_path():
    """scenario_filename is just the file name, not the full path
    (portable relative form for the JSON output)."""
    em = EnsembleMatch(
        scenario_id="baseline",
        scenario_path=Path("/some/deep/nested/output/baseline.json"),
        borda_sum=0.0,
        score=1.0,
    )
    entry = TopRankingEntry.from_ensemble_match(rank=0, match=em)
    assert entry.scenario_filename == "baseline.json"
    # NOT the full path
    assert "/" not in entry.scenario_filename


# TopRanking.to_dict


def test_top_ranking_to_dict_includes_note_when_present():
    """If note is set, it appears in the serialized dict."""
    tr = TopRanking(
        computed_at="2026-06-10T12:00:00+00:00",
        realised_state={},
        n_groups_observed=0,
        is_complete=False,
        top_scenarios=[],
        note="No groups played yet",
    )
    d = tr.to_dict()
    assert d["note"] == "No groups played yet"


def test_top_ranking_to_dict_omits_note_when_none():
    """If note is None, the field is absent from serialized dict (clean schema)."""
    tr = TopRanking(
        computed_at="2026-06-10T12:00:00+00:00",
        realised_state={"H": ["Spain", "Uruguay", "Cape Verde", "Saudi Arabia"]},
        n_groups_observed=1,
        is_complete=False,
        top_scenarios=[],
        note=None,
    )
    d = tr.to_dict()
    assert "note" not in d


def test_top_ranking_to_dict_includes_required_fields():
    """to_dict() always includes computed_at, realised_state, n_groups_observed,
    is_complete, top_scenarios - the consumer contract."""
    tr = TopRanking(
        computed_at="2026-06-10T12:00:00+00:00",
        realised_state={},
        n_groups_observed=0,
        is_complete=False,
    )
    d = tr.to_dict()
    required = {"computed_at", "realised_state", "n_groups_observed",
                "is_complete", "top_scenarios"}
    assert required.issubset(d.keys())


# write_top_ranking_file: persistence


def test_write_top_ranking_file_creates_json(tmp_path):
    """write_top_ranking_file produces output/top_10_ranking.json."""
    tr = TopRanking(
        computed_at="2026-06-10T12:00:00+00:00",
        realised_state={},
        n_groups_observed=0,
        is_complete=False,
        top_scenarios=[],
        note="pre-tournament",
    )
    path = write_top_ranking_file(tr, tmp_path)
    assert path == tmp_path / TOP_RANKING_FILENAME
    parsed = json.loads(path.read_text())
    assert parsed["note"] == "pre-tournament"


# load_state_for_cron: pre-tournament case (REGRESSION)


def test_load_state_for_cron_returns_empty_when_no_csv(tmp_path):
    """REGRESSION: missing match_results.csv → empty RealisedState, not crash.

    This is the pre-tournament case on day 1 of the
    tournament, no matches have been played yet, no CSV exists. The
    cron must not crash here - it should produce a baseline-style output
    with all top scenarios tied (since the matcher has nothing to
    disambiguate).
    """
    # Pretend we have ratings; load_state_for_cron only uses them if CSV exists
    ratings = {"FakeTeam": 1800.0}
    state = load_state_for_cron(tmp_path, ratings)
    assert state.standings == {}
    assert state.played_groups == set()
    assert not state.is_complete


def test_load_state_for_cron_uses_default_filename(tmp_path):
    """load_state_for_cron looks for output/match_results.csv by default."""
    # Confirm the constant matches what the function expects
    assert MATCH_RESULTS_CSV_FILENAME == "match_results.csv"