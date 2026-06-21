"""Tests for upset_propagation.match_results.

Coverage:
  - state_from_matches: happy path (full 12 groups), partial-group skip,
    inter-group ValueError, unknown-team KeyError, over-full group
    duplicate detection
  - load_matches_from_csv: column validation, optional card-field defaults
  - _construct_seeded_baseline_matches: produces matches equivalent to
    build_baseline_standings (the regression we caught during item 5
    smoke testing)

Strategy: uses the real groups + ratings + fair_probs from config (so
team-name aliases work), constructs synthetic match lists via the
helper, then asserts on the output's group standings. FIFA tiebreaker
correctness itself is the vendored responsibility (his vendored tests);
we test the wrapper plumbing.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from upset_propagation.match_results import (
    MATCHES_PER_FULL_GROUP,
    MatchResult,
    _construct_seeded_baseline_matches,
    load_matches_from_csv,
    state_from_matches,
    state_from_matches_csv,
)
from upset_propagation.scenarios import (
    build_baseline_standings,
    load_groups,
    load_latest_elo,
)


# Fixtures


@pytest.fixture(scope="module")
def groups() -> dict[str, list[str]]:
    return load_groups()


@pytest.fixture(scope="module")
def ratings() -> dict[str, float]:
    return load_latest_elo()


@pytest.fixture(scope="module")
def fair_probs(groups) -> dict[str, float]:
    """Synthetic fair_probs that mirror real-API shape - favourite per group
    has highest, second-favourite next, etc. Avoids network dependency.
    """
    probs: dict[str, float] = {}
    # Roughly mimic real API: top teams ~15%, midfield ~3%, weakest <1%
    for letter, teams in groups.items():
        # Within each group, assign descending probs to the 4 teams
        for rank, team in enumerate(teams):
            probs[team] = 0.10 / (rank + 1)  # rank 0 → 0.10, rank 3 → 0.025
    return probs


@pytest.fixture(scope="module")
def baseline_matches(groups, fair_probs) -> list[MatchResult]:
    """The 72-match list that produces the baseline standings."""
    return _construct_seeded_baseline_matches(groups, fair_probs)


# Happy path: full 12 groups


def test_state_from_matches_baseline_produces_full_standings(
    baseline_matches, ratings, groups, fair_probs
):
    """Feeding 72 favourite-wins matches yields all 12 groups in result."""
    state = state_from_matches(baseline_matches, ratings, groups)
    assert state.is_complete
    assert len(state.standings) == 12
    assert state.played_groups == set(groups.keys())


def test_state_from_matches_matches_baseline_standings(
    baseline_matches, ratings, groups, fair_probs
):
    """REGRESSION: helper output must equal build_baseline_standings exactly.

    Original Elo-sorted helper disagreed with build_baseline_standings in
    7/12 groups (Elo-favourite ≠ fair_prob-favourite). Fixed to use
    build_baseline_standings directly. This test guards against re-introducing
    the proxy-mismatch bug.
    """
    state = state_from_matches(baseline_matches, ratings, groups)
    baseline = build_baseline_standings(groups, fair_probs)
    for letter in groups:
        assert state.standings[letter] == baseline[letter], (
            f"Group {letter} mismatch: state_from_matches got "
            f"{state.standings[letter]}, build_baseline got {baseline[letter]}"
        )


# Partial states


def test_state_from_matches_skips_partial_groups(
    baseline_matches, ratings, groups
):
    """Groups with <6 matches are silently dropped from result."""
    # Take only the matches from groups A and B (12 of the 72)
    a_b_matches = [
        m for m in baseline_matches
        if m.home in (set(groups["A"]) | set(groups["B"]))
        and m.away in (set(groups["A"]) | set(groups["B"]))
    ]
    state = state_from_matches(a_b_matches, ratings, groups)
    # Should have exactly A and B
    assert set(state.standings.keys()) == {"A", "B"}
    assert not state.is_complete


def test_state_from_matches_empty_input(ratings, groups):
    """Zero matches → empty state."""
    state = state_from_matches([], ratings, groups)
    assert len(state.standings) == 0
    assert not state.is_complete


def test_state_from_matches_partial_single_match(ratings, groups):
    """One match (group has 5 missing) → that group is skipped silently."""
    a = groups["A"]
    matches = [MatchResult(home=a[0], away=a[1], home_goals=2, away_goals=0)]
    state = state_from_matches(matches, ratings, groups)
    assert "A" not in state.standings
    assert len(state.standings) == 0


# Error cases


def test_state_from_matches_raises_on_unknown_team(ratings, groups):
    """Unknown team name raises KeyError with helpful message."""
    matches = [
        MatchResult(home="Atlantis", away=groups["A"][0], home_goals=1, away_goals=0)
    ]
    with pytest.raises(KeyError, match="Atlantis"):
        state_from_matches(matches, ratings, groups)


def test_state_from_matches_raises_on_inter_group_match(ratings, groups):
    """Match between teams in different groups raises ValueError."""
    matches = [
        MatchResult(
            home=groups["A"][0],
            away=groups["H"][0],
            home_goals=1,
            away_goals=0,
        )
    ]
    with pytest.raises(ValueError, match="different groups"):
        state_from_matches(matches, ratings, groups)


def test_state_from_matches_raises_on_too_many_matches(ratings, groups):
    """More than 6 matches in one group raises ValueError (duplicate signal)."""
    a = groups["A"]
    # 7 matches in group A - one duplicate
    matches = [
        MatchResult(home=a[0], away=a[1], home_goals=2, away_goals=0),
        MatchResult(home=a[0], away=a[2], home_goals=2, away_goals=0),
        MatchResult(home=a[0], away=a[3], home_goals=2, away_goals=0),
        MatchResult(home=a[1], away=a[2], home_goals=2, away_goals=0),
        MatchResult(home=a[1], away=a[3], home_goals=2, away_goals=0),
        MatchResult(home=a[2], away=a[3], home_goals=2, away_goals=0),
        # Duplicate of the first one - should trigger over-full
        MatchResult(home=a[0], away=a[1], home_goals=3, away_goals=0),
    ]
    with pytest.raises(ValueError, match="duplicate"):
        state_from_matches(matches, ratings, groups)


# Spain-H upset


def test_state_from_matches_handles_upset(
    baseline_matches, ratings, groups, fair_probs
):
    """REGRESSION: real upset case from item 5 smoke test.

    Spain (favourite of Group H by fair_prob) loses 1-0 to Uruguay (2nd).
    Expected result: Group H = [Uruguay, Spain, Cape Verde, Saudi Arabia]
    (Uruguay 9pts, Spain 6pts after the upset).
    """
    h_seeded = build_baseline_standings(groups, fair_probs)["H"]
    spain = h_seeded[0]
    uruguay = h_seeded[1]

    # Strip the Spain-Uruguay match (was Spain 2-0 Uruguay in baseline)
    # and replace with Uruguay 1-0 Spain
    upset_matches = [
        m for m in baseline_matches
        if not {m.home, m.away} == {spain, uruguay}
    ]
    upset_matches.append(
        MatchResult(home=uruguay, away=spain, home_goals=1, away_goals=0)
    )

    state = state_from_matches(upset_matches, ratings, groups)
    assert state.standings["H"][0] == uruguay
    assert state.standings["H"][1] == spain
    # Other groups should be unchanged
    baseline = build_baseline_standings(groups, fair_probs)
    for letter in groups:
        if letter == "H":
            continue
        assert state.standings[letter] == baseline[letter]


# CSV loader


def test_load_matches_from_csv_required_columns(tmp_path):
    """Missing required columns raises ValueError listing them."""
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("home,away\nFrance,Senegal\n")  # missing goals
    with pytest.raises(ValueError, match="missing required columns"):
        load_matches_from_csv(csv_path)


def test_load_matches_from_csv_minimal(tmp_path):
    """Required cols only - card fields default to 0."""
    csv_path = tmp_path / "minimal.csv"
    csv_path.write_text(
        "home,away,home_goals,away_goals\n"
        "France,Senegal,2,1\n"
        "Brazil,Switzerland,3,0\n"
    )
    matches = load_matches_from_csv(csv_path)
    assert len(matches) == 2
    assert matches[0].home == "France"
    assert matches[0].away == "Senegal"
    assert matches[0].home_goals == 2
    assert matches[0].away_goals == 1
    # Card fields default to 0
    assert matches[0].home_yellows == 0
    assert matches[0].home_indirect_reds == 0
    assert matches[1].home == "Brazil"


def test_load_matches_from_csv_with_cards(tmp_path):
    """Card fields, when present, are parsed as ints."""
    csv_path = tmp_path / "with_cards.csv"
    csv_path.write_text(
        "home,away,home_goals,away_goals,home_yellows,away_direct_reds\n"
        "France,Senegal,2,1,3,1\n"
    )
    matches = load_matches_from_csv(csv_path)
    assert matches[0].home_yellows == 3
    assert matches[0].away_direct_reds == 1


def test_load_matches_from_csv_empty_card_cells_default_to_zero(tmp_path):
    """Empty string in a card column → 0 (not crash)."""
    csv_path = tmp_path / "blank_cards.csv"
    csv_path.write_text(
        "home,away,home_goals,away_goals,home_yellows\n"
        "France,Senegal,2,1,\n"  # empty home_yellows
    )
    matches = load_matches_from_csv(csv_path)
    assert matches[0].home_yellows == 0


def test_state_from_matches_csv_integration(
    tmp_path, ratings, groups, fair_probs, baseline_matches
):
    """state_from_matches_csv reads the CSV, then runs state_from_matches."""
    csv_path = tmp_path / "matches.csv"
    with csv_path.open("w") as f:
        w = csv.writer(f)
        w.writerow(["home", "away", "home_goals", "away_goals"])
        for m in baseline_matches:
            w.writerow([m.home, m.away, m.home_goals, m.away_goals])

    state = state_from_matches_csv(csv_path, ratings, groups)
    assert state.is_complete
    assert len(state.standings) == 12


# MatchResult dataclass


def test_match_result_defaults():
    """All card fields default to 0."""
    m = MatchResult(home="A", away="B", home_goals=1, away_goals=0)
    assert m.home_yellows == 0
    assert m.home_indirect_reds == 0
    assert m.home_direct_reds == 0
    assert m.home_yellow_plus_reds == 0
    assert m.away_yellows == 0
    assert m.away_indirect_reds == 0
    assert m.away_direct_reds == 0
    assert m.away_yellow_plus_reds == 0


def test_matches_per_full_group_constant():
    """4-team round-robin = 6 matches. Locked constant."""
    assert MATCHES_PER_FULL_GROUP == 6