"""Tests for upset_propagation.scenarios.

These tests use synthetic fixtures (no API calls, no CSV reads) so they
run in milliseconds and can't be broken by data changes upstream.

Coverage:
  - resolve_fair_prob_for_group_team: handles direct names + aliases + errors
  - rank_group_by_fair_prob: correctly orders teams
  - build_scenario_for_group: swaps 1st and 2nd, leaves other positions
    and other groups untouched
  - build_all_scenarios: produces exactly 12 scenarios with unique IDs
  - _slug: handles diacritics, ampersands, spaces consistently
  - build_pairwise_scenario: swaps 1st/2nd in TWO distinct groups,
    rejects same-group pairs, canonical ordering for stable IDs
  - build_all_pairwise_scenarios: produces exactly 66 unique pairs
  - Scenario dataclass tuple fields: correctly populated for both
    single-deviation and pairwise scenarios
"""

from __future__ import annotations

import pytest

from upset_propagation.scenarios import (
    Scenario,
    _slug,
    build_all_pairwise_scenarios,
    build_all_scenarios,
    build_pairwise_scenario,
    build_scenario_for_group,
    rank_group_by_fair_prob,
    resolve_fair_prob_for_group_team,
)


# Fixtures: synthetic 12-group + fair_probs setup


@pytest.fixture
def synthetic_groups() -> dict[str, list[str]]:
    """Minimal 12-group fixture mirroring real WC2026 structure shape."""
    return {
        "A": ["TeamA1", "TeamA2", "TeamA3", "TeamA4"],
        "B": ["TeamB1", "TeamB2", "TeamB3", "TeamB4"],
        "C": ["TeamC1", "TeamC2", "TeamC3", "TeamC4"],
        "D": ["TeamD1", "TeamD2", "TeamD3", "TeamD4"],
        "E": ["TeamE1", "TeamE2", "TeamE3", "TeamE4"],
        "F": ["TeamF1", "TeamF2", "TeamF3", "TeamF4"],
        "G": ["TeamG1", "TeamG2", "TeamG3", "TeamG4"],
        "H": ["TeamH1", "TeamH2", "TeamH3", "TeamH4"],
        "I": ["TeamI1", "TeamI2", "TeamI3", "TeamI4"],
        "J": ["TeamJ1", "TeamJ2", "TeamJ3", "TeamJ4"],
        "K": ["TeamK1", "TeamK2", "TeamK3", "TeamK4"],
        "L": ["TeamL1", "TeamL2", "TeamL3", "TeamL4"],
    }


@pytest.fixture
def synthetic_fair_probs(synthetic_groups) -> dict[str, float]:
    """Fair_probs with team1 always strongest, team4 always weakest.

    Sums to ~1 across all 48 teams; values are arbitrary but ordered.
    """
    probs = {}
    for letter, teams in synthetic_groups.items():
        # Within each group, descending: 0.040, 0.030, 0.020, 0.010
        for i, team in enumerate(teams):
            probs[team] = 0.040 - i * 0.010
    return probs


# resolve_fair_prob_for_group_team


def test_resolve_direct_name():
    """Direct lookup works when name is already in the dict."""
    fair_probs = {"Spain": 0.16, "France": 0.15}
    assert resolve_fair_prob_for_group_team("Spain", fair_probs) == 0.16


def test_resolve_via_alias_czechia():
    """Czechia (groups.json) resolves via 'Czech Republic' (API)."""
    fair_probs = {"Czech Republic": 0.002}
    assert resolve_fair_prob_for_group_team("Czechia", fair_probs) == 0.002


def test_resolve_via_alias_bosnia():
    """'Bosnia and Herzegovina' resolves to 'Bosnia & Herzegovina'."""
    fair_probs = {"Bosnia & Herzegovina": 0.002}
    assert resolve_fair_prob_for_group_team("Bosnia and Herzegovina", fair_probs) == 0.002


def test_resolve_via_alias_united_states():
    """'United States' resolves to 'USA'."""
    fair_probs = {"USA": 0.012}
    assert resolve_fair_prob_for_group_team("United States", fair_probs) == 0.012


def test_resolve_via_alias_curacao():
    """'Curacao' (no cedilla) resolves to 'Curaçao' (with)."""
    fair_probs = {"Curaçao": 0.001}
    assert resolve_fair_prob_for_group_team("Curacao", fair_probs) == 0.001


def test_resolve_unknown_raises_keyerror():
    """Unknown team raises KeyError with helpful message."""
    with pytest.raises(KeyError, match="No fair_prob found for 'NonexistentTeam'"):
        resolve_fair_prob_for_group_team("NonexistentTeam", {"Spain": 0.16})


# rank_group_by_fair_prob


def test_rank_group_sorts_descending():
    """Teams are returned with highest fair_prob first."""
    teams = ["TeamA", "TeamB", "TeamC", "TeamD"]
    probs = {"TeamA": 0.01, "TeamB": 0.04, "TeamC": 0.03, "TeamD": 0.02}
    ranked = rank_group_by_fair_prob(teams, probs)
    assert ranked == ["TeamB", "TeamC", "TeamD", "TeamA"]


def test_rank_group_preserves_all_teams():
    """All input teams are present in the output."""
    teams = ["X1", "X2", "X3", "X4"]
    probs = {"X1": 0.1, "X2": 0.2, "X3": 0.3, "X4": 0.4}
    ranked = rank_group_by_fair_prob(teams, probs)
    assert sorted(ranked) == sorted(teams)


# build_scenario_for_group


def test_scenario_swaps_first_and_second_in_deviating_group(
    synthetic_groups, synthetic_fair_probs
):
    """The deviating group's 1st becomes 2nd and 2nd becomes 1st."""
    s = build_scenario_for_group("H", synthetic_groups, synthetic_fair_probs)
    standing_h = s.standings["H"]
    # In our fixture, TeamH1 was 1st by fair_prob, TeamH2 was 2nd
    assert standing_h[0] == "TeamH2"  # now wins the group
    assert standing_h[1] == "TeamH1"  # slips to 2nd
    assert standing_h[2] == "TeamH3"  # unchanged
    assert standing_h[3] == "TeamH4"  # unchanged


def test_scenario_metadata_identifies_favourite_and_upset(
    synthetic_groups, synthetic_fair_probs
):
    """favourite and upset_winner reference the right teams."""
    s = build_scenario_for_group("H", synthetic_groups, synthetic_fair_probs)
    assert s.favourite == "TeamH1"
    assert s.upset_winner == "TeamH2"
    assert s.deviating_group == "H"


def test_scenario_other_groups_unchanged_from_baseline(
    synthetic_groups, synthetic_fair_probs
):
    """Groups other than the deviating one keep their seeded order."""
    s = build_scenario_for_group("H", synthetic_groups, synthetic_fair_probs)
    for letter in "ABCDEFGIJKL":  # all except H
        standing = s.standings[letter]
        # In synthetic fixture, team1 always strongest → standing[0] = letter+"1"
        assert standing == [f"Team{letter}{i}" for i in [1, 2, 3, 4]], (
            f"Group {letter} should match seeded order"
        )


def test_scenario_id_includes_favourite_and_group(
    synthetic_groups, synthetic_fair_probs
):
    """scenario_id has the form '<favourite_slug>_runner_up_<group>'."""
    s = build_scenario_for_group("H", synthetic_groups, synthetic_fair_probs)
    assert s.scenario_id == "teamh1_runner_up_H"


def test_scenario_unknown_group_raises_valueerror(
    synthetic_groups, synthetic_fair_probs
):
    """Asking for a non-existent group raises ValueError."""
    with pytest.raises(ValueError, match="Unknown group 'Z'"):
        build_scenario_for_group("Z", synthetic_groups, synthetic_fair_probs)


# build_all_scenarios


def test_build_all_scenarios_returns_twelve(
    synthetic_groups, synthetic_fair_probs
):
    """Exactly 12 scenarios are built - one per group A through L."""
    scenarios = build_all_scenarios(synthetic_fair_probs, groups=synthetic_groups)
    assert len(scenarios) == 12


def test_build_all_scenarios_one_per_group(
    synthetic_groups, synthetic_fair_probs
):
    """Each of the 12 groups appears as a deviating_group exactly once."""
    scenarios = build_all_scenarios(synthetic_fair_probs, groups=synthetic_groups)
    deviating_groups = sorted(s.deviating_group for s in scenarios)
    assert deviating_groups == list("ABCDEFGHIJKL")


def test_build_all_scenarios_unique_ids(
    synthetic_groups, synthetic_fair_probs
):
    """All 12 scenario IDs are unique."""
    scenarios = build_all_scenarios(synthetic_fair_probs, groups=synthetic_groups)
    ids = [s.scenario_id for s in scenarios]
    assert len(set(ids)) == 12


# _slug


def test_slug_strips_diacritics():
    """Diacritics like ç become ASCII (c)."""
    assert _slug("Curaçao") == "curacao"
    assert _slug("Côte d'Ivoire") == "cote_d'ivoire"


def test_slug_replaces_spaces_with_underscores():
    """Multi-word names become underscore-separated."""
    assert _slug("South Korea") == "south_korea"
    assert _slug("United States") == "united_states"


def test_slug_replaces_ampersand_with_and():
    """Ampersand becomes 'and' to match the FairLine API naming."""
    assert _slug("Bosnia & Herzegovina") == "bosnia_and_herzegovina"


def test_slug_simple_name_unchanged():
    """A name with no special chars just lowercases."""
    assert _slug("Spain") == "spain"


# v1 Scenario tuple fields populated correctly


def test_single_scenario_populates_tuple_fields(
    synthetic_groups, synthetic_fair_probs
):
    """Single-deviation scenarios populate the v2 tuple fields as 1-tuples.

    Backward-compat invariant: existing singular fields stay populated,
    AND the new tuple fields are populated as 1-tuples so v2 code can
    iterate uniformly across v1 and v2 scenarios.
    """
    s = build_scenario_for_group("H", synthetic_groups, synthetic_fair_probs)
    # Singular fields still work
    assert s.deviating_group == "H"
    assert s.favourite == "TeamH1"
    assert s.upset_winner == "TeamH2"
    # Tuple fields populated as 1-tuples
    assert s.deviating_groups == ("H",)
    assert s.favourites == ("TeamH1",)
    assert s.upset_winners == ("TeamH2",)


# build_pairwise_scenario


def test_pairwise_scenario_swaps_both_groups(
    synthetic_groups, synthetic_fair_probs
):
    """Both deviating groups get their 1st/2nd swapped; others stay seeded."""
    s = build_pairwise_scenario("H", "J", synthetic_groups, synthetic_fair_probs)
    # Group H: TeamH1/2 swapped
    assert s.standings["H"] == ["TeamH2", "TeamH1", "TeamH3", "TeamH4"]
    # Group J: TeamJ1/2 swapped
    assert s.standings["J"] == ["TeamJ2", "TeamJ1", "TeamJ3", "TeamJ4"]
    # Group A: unchanged from seeded
    assert s.standings["A"] == ["TeamA1", "TeamA2", "TeamA3", "TeamA4"]


def test_pairwise_scenario_metadata_uses_tuples(
    synthetic_groups, synthetic_fair_probs
):
    """Pairwise scenarios populate the v2 tuple fields with both teams."""
    s = build_pairwise_scenario("H", "J", synthetic_groups, synthetic_fair_probs)
    # Sorted alphabetically: H < J, so the tuple is (H, J)
    assert s.deviating_groups == ("H", "J")
    assert s.favourites == ("TeamH1", "TeamJ1")
    assert s.upset_winners == ("TeamH2", "TeamJ2")


def test_pairwise_scenario_id_is_canonical(
    synthetic_groups, synthetic_fair_probs
):
    """The scenario_id is invariant under argument-order swap.

    build_pairwise_scenario('H', 'J', ...) and ('J', 'H', ...) must produce
    identical scenario_id so they collapse to one file in output/ instead
    of accidentally duplicating.
    """
    s_hj = build_pairwise_scenario("H", "J", synthetic_groups, synthetic_fair_probs)
    s_jh = build_pairwise_scenario("J", "H", synthetic_groups, synthetic_fair_probs)
    assert s_hj.scenario_id == s_jh.scenario_id


def test_pairwise_scenario_id_format(
    synthetic_groups, synthetic_fair_probs
):
    """scenario_id format: '<fav_slug1>_<fav_slug2>_runner_up_<g1g2>'.

    Favourite slugs sorted alphabetically; group letters sorted alphabetically.
    """
    s = build_pairwise_scenario("H", "J", synthetic_groups, synthetic_fair_probs)
    # TeamH1 < TeamJ1 alphabetically (h < j), and H < J as letters
    assert s.scenario_id == "teamh1_teamj1_runner_up_HJ"


def test_pairwise_scenario_singular_fields_take_first(
    synthetic_groups, synthetic_fair_probs
):
    """For pairwise scenarios, deviating_group/favourite/upset_winner hold
    the first (canonical) of the two - for backward-compat with v1 readers.
    """
    s = build_pairwise_scenario("H", "J", synthetic_groups, synthetic_fair_probs)
    assert s.deviating_group == "H"
    assert s.favourite == "TeamH1"
    assert s.upset_winner == "TeamH2"


def test_pairwise_scenario_same_group_raises(
    synthetic_groups, synthetic_fair_probs
):
    """Same-group pair raises - v2 explicitly excludes single-group
    multi-deviations per the scoping decision."""
    with pytest.raises(ValueError, match="two distinct groups"):
        build_pairwise_scenario("H", "H", synthetic_groups, synthetic_fair_probs)


def test_pairwise_scenario_unknown_group_raises(
    synthetic_groups, synthetic_fair_probs
):
    """Unknown group letter raises a clear ValueError."""
    with pytest.raises(ValueError, match="Unknown group"):
        build_pairwise_scenario("H", "Z", synthetic_groups, synthetic_fair_probs)


# build_all_pairwise_scenarios


def test_build_all_pairwise_returns_66(
    synthetic_groups, synthetic_fair_probs
):
    """12 choose 2 = 66 unordered pairs."""
    pairwise = build_all_pairwise_scenarios(
        synthetic_fair_probs, groups=synthetic_groups
    )
    assert len(pairwise) == 66


def test_build_all_pairwise_unique_ids(
    synthetic_groups, synthetic_fair_probs
):
    """All 66 scenario IDs are unique (no collisions from group ordering)."""
    pairwise = build_all_pairwise_scenarios(
        synthetic_fair_probs, groups=synthetic_groups
    )
    ids = [s.scenario_id for s in pairwise]
    assert len(set(ids)) == 66


def test_build_all_pairwise_no_same_group_pairs(
    synthetic_groups, synthetic_fair_probs
):
    """No pairwise scenario has the same group twice."""
    pairwise = build_all_pairwise_scenarios(
        synthetic_fair_probs, groups=synthetic_groups
    )
    for s in pairwise:
        assert len(s.deviating_groups) == 2
        assert s.deviating_groups[0] != s.deviating_groups[1]


def test_build_all_pairwise_groups_canonically_ordered(
    synthetic_groups, synthetic_fair_probs
):
    """Each pairwise scenario's deviating_groups is sorted (g1 < g2 alphabetically).

    Ensures the canonical-ordering contract holds for every scenario
    produced - not just the H-J case verified above.
    """
    pairwise = build_all_pairwise_scenarios(
        synthetic_fair_probs, groups=synthetic_groups
    )
    for s in pairwise:
        g1, g2 = s.deviating_groups
        assert g1 < g2, (
            f"deviating_groups not canonically ordered in {s.scenario_id}: "
            f"({g1}, {g2})"
        )