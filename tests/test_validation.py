"""Tests for upset_propagation.validation.

Two scopes:
  - Pure functions: graph_distance math (no propagator needed, runs in ms)
  - Integration: directional_sanity + sensitivity_check end-to-end against
    a small synthetic scenario, using uniform/dominant predictors to make
    expected outputs derivable.

The integration tests use real WC2026 data files (groups.json, seeding
table) as fixtures - they're vendored and stable, treating them as test
fixtures is fine.
"""

from __future__ import annotations

import numpy as np
import pytest

from upset_propagation._vendored.single_game import MatchContext, MatchPrediction
from upset_propagation._vendored.simulator import Predictor
from upset_propagation.propagator import propagate
from upset_propagation.scenarios import (
    Scenario,
    build_baseline_standings,
    build_scenario_for_group,
    load_groups,
)
from upset_propagation.validation import (
    _match_round,
    _r32_match_to_round_path,
    deviating_r32_matches,
    directional_sanity,
    graph_distance,
    sensitivity_check,
    team_distance_from_deviation,
    team_to_r32_match,
)


# Pure functions: round mapping + graph distance


def test_match_round_R32():
    """Match IDs 73-88 are R32 matches (round 0)."""
    assert _match_round(73) == 0
    assert _match_round(80) == 0
    assert _match_round(88) == 0


def test_match_round_R16():
    """Match IDs 89-96 are R16 matches (round 1)."""
    assert _match_round(89) == 1
    assert _match_round(92) == 1
    assert _match_round(96) == 1


def test_match_round_QF_SF_F():
    """Quarterfinals 97-100, Semis 101-102, Final 104."""
    assert _match_round(97) == 2
    assert _match_round(100) == 2
    assert _match_round(101) == 3
    assert _match_round(102) == 3
    assert _match_round(104) == 4


def test_match_round_unknown_raises():
    """Any other match ID raises (defensive - bracket only has these IDs)."""
    with pytest.raises(ValueError, match="Unknown match_id"):
        _match_round(999)


def test_r32_paths_have_four_ancestors():
    """Every R32 match has exactly 4 ancestors going up to the Final.

    R32 → R16 → QF → SF → F = 4 ancestor matches above the R32 leaf.
    """
    paths = _r32_match_to_round_path()
    for r32_id in range(73, 89):
        assert len(paths[r32_id]) == 4, (
            f"R32 match {r32_id} should have 4 ancestors, got {len(paths[r32_id])}"
        )


def test_r32_paths_terminate_at_final():
    """Every R32 path's final entry is the Final match (104)."""
    paths = _r32_match_to_round_path()
    for r32_id in range(73, 89):
        assert paths[r32_id][-1] == 104, (
            f"R32 match {r32_id} should reach Final (104), got {paths[r32_id][-1]}"
        )


def test_graph_distance_same_match_is_zero():
    """Two teams in the same R32 match have distance 0."""
    assert graph_distance(73, 73) == 0


def test_graph_distance_sibling_r32_is_one():
    """Two R32 matches that feed the same R16 have distance 1.

    M73 and M75 both feed into M89 (per LATER_ROUNDS). They meet in R16.
    """
    assert graph_distance(73, 75) == 1


def test_graph_distance_opposite_halves_is_four():
    """Two R32 matches on opposite halves of the bracket have distance 4 (Final).

    M73 is in one half, M88 is in the opposite half - they only meet in the Final.
    """
    assert graph_distance(73, 88) == 4


def test_graph_distance_is_symmetric():
    """graph_distance(a, b) == graph_distance(b, a) - bracket is undirected."""
    for a in (73, 80, 88):
        for b in (74, 82, 87):
            assert graph_distance(a, b) == graph_distance(b, a)


def test_graph_distance_in_valid_range():
    """For any two R32 matches, distance is in [0, 4]."""
    for a in range(73, 89):
        for b in range(73, 89):
            d = graph_distance(a, b)
            assert 0 <= d <= 4, f"graph_distance({a}, {b}) = {d} out of range"


# Integration tests: synthetic predictors + real WC2026 bracket


def _make_uniform_predictor() -> Predictor:
    """Predictor that returns P_home = P_away = 0.45, P_draw = 0.10 always.

    With P_advance = 0.50 for both sides, every KO match is a coinflip.
    Useful for testing structural code paths without numerical favouritism.
    """
    def predictor(elo_a: float, elo_b: float, ctx: MatchContext) -> MatchPrediction:
        grid = np.zeros((9, 9))
        grid[0, 0] = 0.10
        grid[1, 0] = 0.45
        grid[0, 1] = 0.45
        return MatchPrediction(p_home=0.45, p_draw=0.10, p_away=0.45, goal_grid=grid)
    return predictor


@pytest.fixture
def real_groups() -> dict[str, list[str]]:
    """Real WC2026 groups for scenario construction."""
    return load_groups()


@pytest.fixture
def synthetic_ratings(real_groups) -> dict[str, float]:
    """Synthetic Elo for all 48 group teams. Spread enough that ordering
    is unambiguous (no ties)."""
    ratings: dict[str, float] = {}
    for li, (letter, teams) in enumerate(real_groups.items()):
        for ti, team in enumerate(teams):
            ratings[team] = 1800.0 - li * 5.0 - ti * 100.0
    return ratings


@pytest.fixture
def synthetic_fair_probs(real_groups) -> dict[str, float]:
    """fair_probs descending within each group."""
    probs: dict[str, float] = {}
    for letter, teams in real_groups.items():
        for i, team in enumerate(teams):
            probs[team] = 0.04 - i * 0.01
    return probs


# deviating_r32_matches + team mapping


def test_deviating_r32_matches_single_scenario_returns_two(
    real_groups, synthetic_fair_probs, synthetic_ratings
):
    """A single-deviation scenario H produces 2 deviating R32 matches.

    The favourite (Spain) goes into one slot; the upset winner (Uruguay)
    goes into another. Each of those matches contains one of them.
    """
    scenario = build_scenario_for_group("H", real_groups, synthetic_fair_probs)
    matches = deviating_r32_matches(scenario, synthetic_ratings)
    assert len(matches) == 2


def test_team_to_r32_match_covers_all_32_knockout_teams(
    real_groups, synthetic_fair_probs, synthetic_ratings
):
    """team_to_r32_match assigns every R32 team to exactly one match."""
    scenario = build_scenario_for_group("H", real_groups, synthetic_fair_probs)
    mapping = team_to_r32_match(scenario, synthetic_ratings)
    # 32 teams reach R32 (12 winners + 12 runners-up + 8 best thirds)
    assert len(mapping) == 32


def test_team_distance_returns_none_for_eliminated_team(
    real_groups, synthetic_fair_probs, synthetic_ratings
):
    """Teams eliminated in group stage have distance = None.

    Group-stage 4th-place teams don't appear in any R32 match.
    """
    scenario = build_scenario_for_group("H", real_groups, synthetic_fair_probs)
    # Team in 4th place (lowest fair_prob in some group) is eliminated.
    # Find one such team:
    eliminated = scenario.standings["A"][3]  # 4th-place in Group A
    assert team_distance_from_deviation(eliminated, scenario, synthetic_ratings) is None


# directional_sanity


def test_directional_sanity_returns_valid_report(
    real_groups, synthetic_fair_probs, synthetic_ratings
):
    """directional_sanity produces a well-structured report."""
    standings = build_baseline_standings(real_groups, synthetic_fair_probs)
    baseline = Scenario(
        scenario_id="baseline", description="", deviating_group="",
        favourite="", upset_winner="", standings=standings,
    )
    deviation = build_scenario_for_group("H", real_groups, synthetic_fair_probs)
    predictor = _make_uniform_predictor()
    baseline_result = propagate(baseline, synthetic_ratings, predictor=predictor)
    deviation_result = propagate(deviation, synthetic_ratings, predictor=predictor)

    report = directional_sanity(
        deviation, deviation_result, baseline_result, synthetic_ratings
    )
    assert report.scenario_id == deviation.scenario_id
    # Verdict is either PASS or starts with FAIL:
    assert report.verdict == "PASS" or report.verdict.startswith("FAIL:")


def test_directional_sanity_uniform_predictor_has_zero_deltas(
    real_groups, synthetic_fair_probs, synthetic_ratings
):
    """Under a uniform predictor, every team has P(Win) = 1/32 regardless of
    scenario - so deltas are all zero, and the verdict is trivially PASS.

    This test verifies the report machinery survives the degenerate case
    where all deltas are exactly zero.
    """
    standings = build_baseline_standings(real_groups, synthetic_fair_probs)
    baseline = Scenario(
        scenario_id="baseline", description="", deviating_group="",
        favourite="", upset_winner="", standings=standings,
    )
    deviation = build_scenario_for_group("H", real_groups, synthetic_fair_probs)
    predictor = _make_uniform_predictor()
    baseline_result = propagate(baseline, synthetic_ratings, predictor=predictor)
    deviation_result = propagate(deviation, synthetic_ratings, predictor=predictor)

    report = directional_sanity(
        deviation, deviation_result, baseline_result, synthetic_ratings
    )
    # All deltas should be 0 → all distance-buckets show mean 0 → PASS
    for mean in report.mean_abs_delta_by_distance.values():
        assert mean < 1e-9


# sensitivity_check


def test_sensitivity_check_returns_valid_report(
    real_groups, synthetic_fair_probs, synthetic_ratings
):
    """sensitivity_check produces a well-structured report."""
    standings = build_baseline_standings(real_groups, synthetic_fair_probs)
    baseline = Scenario(
        scenario_id="baseline", description="", deviating_group="",
        favourite="", upset_winner="", standings=standings,
    )
    predictor = _make_uniform_predictor()
    # Pick a team that's in the knockout - the favourite of any group works
    perturb_team = standings["H"][0]
    report = sensitivity_check(
        baseline, predictor, synthetic_ratings,
        perturb_team=perturb_team, delta_elo=50.0,
    )
    assert report.perturbed_team == perturb_team
    assert report.delta_elo == 50.0
    # Verdict is PASS or starts with FAIL:
    assert report.verdict == "PASS" or report.verdict.startswith("FAIL:")


def test_sensitivity_conservation_under_uniform_predictor(
    real_groups, synthetic_fair_probs, synthetic_ratings
):
    """Σ ΔP(Win) is exactly 0 - probability is conserved.

    Doesn't matter what predictor we use; conservation must hold.
    """
    standings = build_baseline_standings(real_groups, synthetic_fair_probs)
    baseline = Scenario(
        scenario_id="baseline", description="", deviating_group="",
        favourite="", upset_winner="", standings=standings,
    )
    predictor = _make_uniform_predictor()
    perturb_team = standings["H"][0]
    report = sensitivity_check(
        baseline, predictor, synthetic_ratings,
        perturb_team=perturb_team, delta_elo=50.0,
    )
    # Conservation within tight numerical tolerance
    assert abs(report.global_conservation_pp) < 1e-6


def test_sensitivity_uniform_predictor_fails_min_movement(
    real_groups, synthetic_fair_probs, synthetic_ratings
):
    """Under a uniform predictor, an Elo bump has NO effect on outcomes
    (every match is still a coinflip).

    So `team_movement_pp` will be ~0, well below the default 0.5pp
    threshold, and the verdict should be FAIL. This is a deliberate
    smoke test that the FAIL path is exercised in our test suite.
    """
    standings = build_baseline_standings(real_groups, synthetic_fair_probs)
    baseline = Scenario(
        scenario_id="baseline", description="", deviating_group="",
        favourite="", upset_winner="", standings=standings,
    )
    predictor = _make_uniform_predictor()
    perturb_team = standings["H"][0]
    report = sensitivity_check(
        baseline, predictor, synthetic_ratings,
        perturb_team=perturb_team, delta_elo=50.0,
        min_movement_pp=0.5,
    )
    # Movement should be near zero - well below 0.5pp threshold
    assert abs(report.team_movement_pp) < 0.1
    # Verdict should reflect the failure
    assert report.verdict.startswith("FAIL:")