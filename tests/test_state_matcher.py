"""Tests for upset_propagation.state_matcher.

Coverage:
  - Severity enum and classify_group: per-group tier classification (EXACT,
    SWAP, PARTIAL, DISJOINT) for every relevant case
  - compute_distance: Hamming distance + sorted-descending severity vector
  - parse_state_from_dict: input validation
  - find_best_scenarios: end-to-end ranking with real (temp) JSON files
  - get_scenario_table_for_state: consumer-helper integration including
    ambiguity detection on partial states

Strategy: use pytest's tmp_path fixture to materialize a small library of
scenario JSON files for the matcher to read. This tests the actual disk
I/O path, not a mock. Each test creates only the scenarios it needs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from upset_propagation.state_matcher import (
    MatchedScenarioForState,
    RealisedState,
    ScenarioMatch,
    Severity,
    classify_group,
    compute_distance,
    find_best_scenarios,
    get_scenario_table_for_state,
    parse_state_from_dict,
)


# Fixtures: a tiny synthetic scenario library


def _baseline_standings() -> dict[str, list[str]]:
    """A canonical 12-group seeded baseline (TeamX1 > TeamX2 > TeamX3 > TeamX4)."""
    return {
        letter: [f"Team{letter}{i}" for i in (1, 2, 3, 4)]
        for letter in "ABCDEFGHIJKL"
    }


def _swap_top_two(standings: dict[str, list[str]], letter: str) -> dict[str, list[str]]:
    """Return a fresh copy of standings with letter's 1st/2nd swapped."""
    out = {g: list(s) for g, s in standings.items()}
    g = out[letter]
    out[letter] = [g[1], g[0], g[2], g[3]]
    return out


def _write_scenario_file(
    output_dir: Path,
    scenario_id: str,
    standings: dict[str, list[str]],
    survival: dict | None = None,
) -> Path:
    """Write a minimal scenario JSON to disk for the matcher to read.

    Includes only the fields the matcher actually consumes (scenario_id,
    standings). Survival is provided when needed for load_survival_table tests.
    """
    payload = {
        "scenario_id": scenario_id,
        "description": f"synthetic test scenario {scenario_id}",
        "standings": standings,
        "survival": survival or {},
    }
    path = output_dir / f"{scenario_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def baseline_standings() -> dict[str, list[str]]:
    return _baseline_standings()


@pytest.fixture
def scenario_library(tmp_path: Path) -> Path:
    """Build a tiny library of scenario files in tmp_path/output/.

    Contains:
      - baseline.json (no deviation)
      - spain_runner_up_H.json (single-deviation, swap Group H top 2)
      - mexico_runner_up_A.json (single-deviation, swap Group A top 2)
      - spain_mexico_runner_up_AH.json (pairwise, swap A and H)

    Returns the path of the output dir (for passing to matcher functions).
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    base = _baseline_standings()
    _write_scenario_file(output_dir, "baseline", base)
    _write_scenario_file(output_dir, "spain_runner_up_H", _swap_top_two(base, "H"))
    _write_scenario_file(output_dir, "mexico_runner_up_A", _swap_top_two(base, "A"))
    # Pairwise: swap both H and A
    pairwise = _swap_top_two(_swap_top_two(base, "A"), "H")
    _write_scenario_file(output_dir, "spain_mexico_runner_up_AH", pairwise)
    return output_dir


# classify_group


def test_classify_exact_match_returns_EXACT():
    """Same top-2 in same order → EXACT."""
    assert classify_group(("Spain", "Uruguay"), ("Spain", "Uruguay")) == Severity.EXACT


def test_classify_swap_returns_SWAP():
    """Same top-2 teams in opposite order → SWAP."""
    assert classify_group(("Spain", "Uruguay"), ("Uruguay", "Spain")) == Severity.SWAP


def test_classify_partial_one_overlap():
    """Top-2 sets share exactly one team → PARTIAL."""
    assert classify_group(
        ("Spain", "Uruguay"),
        ("Spain", "Cape Verde"),
    ) == Severity.PARTIAL


def test_classify_disjoint_no_overlap():
    """Top-2 sets share no teams → DISJOINT."""
    assert classify_group(
        ("Spain", "Uruguay"),
        ("Cape Verde", "Saudi Arabia"),
    ) == Severity.DISJOINT


def test_severity_ordering_is_intentional():
    """EXACT < SWAP < PARTIAL < DISJOINT for natural comparisons."""
    assert Severity.EXACT < Severity.SWAP
    assert Severity.SWAP < Severity.PARTIAL
    assert Severity.PARTIAL < Severity.DISJOINT


# parse_state_from_dict


def test_parse_state_accepts_full_state(baseline_standings):
    """Full 12-group dict parses and reports is_complete=True."""
    state = parse_state_from_dict(baseline_standings)
    assert state.is_complete is True
    assert len(state.played_groups) == 12


def test_parse_state_accepts_partial_state():
    """Subset of groups is valid (game stage in progress)."""
    state = parse_state_from_dict({"H": ["Spain", "Uruguay", "Cape Verde", "Saudi Arabia"]})
    assert state.is_complete is False
    assert state.played_groups == {"H"}


def test_parse_state_rejects_invalid_group_letter():
    """Group letter outside A-L raises."""
    with pytest.raises(ValueError, match="Unknown group letter"):
        parse_state_from_dict({"Z": ["A", "B", "C", "D"]})


def test_parse_state_rejects_wrong_team_count():
    """Group standings with anything but 4 teams raises."""
    with pytest.raises(ValueError, match="must be a list of 4 teams"):
        parse_state_from_dict({"A": ["Mexico", "South Korea", "Czechia"]})


# compute_distance


def test_distance_zero_for_exact_match(baseline_standings):
    """Identical state and scenario → distance 0, no disagreement vector."""
    state = parse_state_from_dict(baseline_standings)
    match = compute_distance(
        state, "baseline", baseline_standings, Path("/dev/null"),
    )
    assert match.hamming_distance == 0
    assert match.disagreement_vector == ()


def test_distance_one_for_single_swap(baseline_standings):
    """State matches scenario except for Group H swap → distance 1, SWAP severity."""
    state_standings = _swap_top_two(baseline_standings, "H")
    state = parse_state_from_dict(state_standings)
    # Compare against the un-swapped baseline scenario
    match = compute_distance(
        state, "baseline", baseline_standings, Path("/dev/null"),
    )
    assert match.hamming_distance == 1
    assert match.disagreement_vector == (Severity.SWAP,)


def test_distance_two_for_two_swaps(baseline_standings):
    """Two disagreeing groups (both SWAP) → distance 2, two-SWAP vector."""
    state_standings = _swap_top_two(_swap_top_two(baseline_standings, "H"), "J")
    state = parse_state_from_dict(state_standings)
    match = compute_distance(
        state, "baseline", baseline_standings, Path("/dev/null"),
    )
    assert match.hamming_distance == 2
    assert match.disagreement_vector == (Severity.SWAP, Severity.SWAP)


def test_distance_disagreement_vector_sorted_descending(baseline_standings):
    """When severities differ, disagreement_vector lists them WORST FIRST.

    This is important for lex comparison - bigger severities should
    dominate the ordering.
    """
    state_standings = _swap_top_two(baseline_standings, "H")  # SWAP in H
    # Modify Group A to be DISJOINT (different top-2 entirely)
    state_standings["A"] = ["NewTeam1", "NewTeam2", "TeamA3", "TeamA4"]
    state = parse_state_from_dict(state_standings)
    match = compute_distance(
        state, "baseline", baseline_standings, Path("/dev/null"),
    )
    assert match.hamming_distance == 2
    # DISJOINT (3) comes before SWAP (1) when sorted descending
    assert match.disagreement_vector == (Severity.DISJOINT, Severity.SWAP)


def test_distance_skips_unplayed_groups(baseline_standings):
    """Groups absent from state are NOT counted as disagreements."""
    # State observes only Group H, swapped
    state = parse_state_from_dict({
        "H": _swap_top_two(baseline_standings, "H")["H"],
    })
    match = compute_distance(
        state, "baseline", baseline_standings, Path("/dev/null"),
    )
    # Only 1 group observed, and it disagrees (SWAP) → distance 1
    assert match.hamming_distance == 1
    assert match.disagreement_vector == (Severity.SWAP,)


# find_best_scenarios


def test_find_best_returns_exact_match_first(scenario_library, baseline_standings):
    """When state matches a scenario exactly, that scenario is rank #1 with distance 0."""
    state_standings = _swap_top_two(baseline_standings, "H")
    state = parse_state_from_dict(state_standings)
    top = find_best_scenarios(state, scenario_library, k=3)
    assert top[0].scenario_id == "spain_runner_up_H"
    assert top[0].hamming_distance == 0


def test_find_best_returns_k_matches(scenario_library, baseline_standings):
    """find_best_scenarios respects the k parameter."""
    state = parse_state_from_dict(baseline_standings)
    top_k2 = find_best_scenarios(state, scenario_library, k=2)
    top_k4 = find_best_scenarios(state, scenario_library, k=4)
    assert len(top_k2) == 2
    assert len(top_k4) == 4  # we have 4 scenarios in the library


def test_find_best_ranks_lower_distance_first(scenario_library, baseline_standings):
    """A scenario with smaller Hamming distance always ranks before a larger one."""
    state = parse_state_from_dict(baseline_standings)
    top = find_best_scenarios(state, scenario_library, k=4)
    # Baseline → matches baseline.json exactly (d=0)
    # All other scenarios have at least one swap (d>=1)
    assert top[0].hamming_distance == 0
    assert top[0].scenario_id == "baseline"
    for m in top[1:]:
        assert m.hamming_distance > 0


def test_find_best_includes_baseline_as_candidate(scenario_library, baseline_standings):
    """baseline.json is one of the candidate scenarios, not silently skipped."""
    state = parse_state_from_dict(baseline_standings)
    top = find_best_scenarios(state, scenario_library, k=10)
    matched_ids = [m.scenario_id for m in top]
    assert "baseline" in matched_ids


# get_scenario_table_for_state


def test_get_scenario_table_marks_exact_match(scenario_library, baseline_standings):
    """Exact match → is_exact_match=True, is_ambiguous=False."""
    state_standings = _swap_top_two(baseline_standings, "H")
    state = parse_state_from_dict(state_standings)
    result = get_scenario_table_for_state(state, scenario_library)
    assert result.is_exact_match is True
    assert result.is_ambiguous is False
    assert result.best_match.scenario_id == "spain_runner_up_H"


def test_get_scenario_table_detects_ambiguity_on_partial_state(scenario_library):
    """Partial state matching multiple scenarios → is_ambiguous=True with reason."""
    # State observes only Group A, swapped. Both "mexico_runner_up_A" AND
    # the pairwise "spain_mexico_runner_up_AH" match Group A with d=0 - the
    # matcher can't distinguish them because Group H isn't observed.
    base = _baseline_standings()
    state = parse_state_from_dict({"A": _swap_top_two(base, "A")["A"]})
    result = get_scenario_table_for_state(state, scenario_library)
    assert result.is_ambiguous is True
    assert len(result.ambiguous_alternatives) >= 1
    assert "1/12 groups" in result.ambiguity_reason


def test_get_scenario_table_returns_survival_table(scenario_library, baseline_standings):
    """The matched scenario's full JSON payload is returned in survival_table."""
    state_standings = _swap_top_two(baseline_standings, "H")
    state = parse_state_from_dict(state_standings)
    result = get_scenario_table_for_state(state, scenario_library)
    # Synthetic scenarios have empty survival dicts but the payload itself
    # should load and contain the expected fields
    assert result.survival_table["scenario_id"] == "spain_runner_up_H"
    assert "standings" in result.survival_table