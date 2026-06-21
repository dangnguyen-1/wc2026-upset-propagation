"""Tests for upset_propagation.propagator.

These tests verify the analytical bracket-walking math without depending on
the vendored specific Elo model. We inject synthetic predictors with known
win probabilities so the expected outputs are derivable by hand.

Coverage:
  - select_third_placers_by_elo: picks the top 8 by Elo
  - resolve_r32_pairings: produces 16 matches with 32 distinct teams
  - propagate, internal invariants:
      * each match's winner distribution sums to 1
      * every R32 participant has P(R32)=1
      * survival probs are monotonically non-increasing across rounds
      * Σ P(Win) across all teams = 1
      * teams not in the knockout are absent from output
  - propagate, behavioural check:
      * a "dominant team" predictor (one team always beats everyone) yields
        a tournament-win probability of exactly 1 for that team

Real WC2026 data files ARE used as fixtures (groups.json, seeding table),
since they're vendored and stable. The Elo *values* and the predictor are
synthetic - only the bracket structure comes from real data.
"""

from __future__ import annotations

import pytest

from upset_propagation._vendored.single_game import MatchContext, MatchPrediction
from upset_propagation._vendored.simulator import Predictor
from upset_propagation.config import KNOCKOUT_ROUNDS
from upset_propagation.propagator import (
    propagate,
    resolve_r32_pairings,
    select_third_placers_by_elo,
)
from upset_propagation.scenarios import (
    Scenario,
    build_baseline_standings,
    build_scenario_for_group,
    load_groups,
)


# Synthetic predictors


def make_uniform_predictor() -> Predictor:
    """Predictor that returns P_home = P_away = 0.45, P_draw = 0.10 always.

    With P_advance = P_home + 0.5·P_draw = 0.50 for both sides, every match
    is a coinflip. Useful for testing that probability propagation works
    structurally without numerical favouritism.
    """
    def predictor(elo_a: float, elo_b: float, ctx: MatchContext) -> MatchPrediction:
        # Build a 2-cell goal grid that produces these probs. We don't
        # actually use the grid here (KO matches only use p_home/p_draw/p_away
        # via build_ko_advance_table), so a minimal valid grid is fine.
        import numpy as np
        grid = np.zeros((9, 9))
        grid[0, 0] = 0.10  # diagonal (draw)
        grid[1, 0] = 0.45  # home wins
        grid[0, 1] = 0.45  # away wins
        return MatchPrediction(p_home=0.45, p_draw=0.10, p_away=0.45, goal_grid=grid)
    return predictor


def make_dominant_predictor(dominant_team: str) -> Predictor:
    """Predictor where `dominant_team` always wins with P=1.

    Lets us test the bracket walking deterministically - the dominant team
    should reach the final with probability 1 (and win with probability 1).
    """
    def predictor(elo_a: float, elo_b: float, ctx: MatchContext) -> MatchPrediction:
        import numpy as np
        grid = np.zeros((9, 9))
        if ctx.home_country == dominant_team:
            grid[1, 0] = 1.0
            return MatchPrediction(p_home=1.0, p_draw=0.0, p_away=0.0, goal_grid=grid)
        if ctx.away_country == dominant_team:
            grid[0, 1] = 1.0
            return MatchPrediction(p_home=0.0, p_draw=0.0, p_away=1.0, goal_grid=grid)
        # Neither side is the dominant team - fall back to coinflip so other
        # matches still resolve and the dominant team faces *someone* in each
        # round.
        grid[0, 0] = 0.10
        grid[1, 0] = 0.45
        grid[0, 1] = 0.45
        return MatchPrediction(p_home=0.45, p_draw=0.10, p_away=0.45, goal_grid=grid)
    return predictor


# Fixtures: real WC2026 data + synthetic Elo + a synthetic scenario


@pytest.fixture
def real_groups() -> dict[str, list[str]]:
    """The actual 12 WC2026 groups from data/mc_simu/wc2026_groups.json."""
    return load_groups()


@pytest.fixture
def synthetic_ratings(real_groups) -> dict[str, float]:
    """Synthetic ratings where strength decreases with group position.

    Each group A's team1 = 1800, team2 = 1700, team3 = 1600, team4 = 1500.
    Group A's team1 has the highest rating of all team1s (just for clean
    third-placer selection); strength decreases as we go through letters.
    """
    ratings: dict[str, float] = {}
    for li, (letter, teams) in enumerate(real_groups.items()):
        for ti, team in enumerate(teams):
            # Higher group letter → slightly weaker pool, plus within-group
            # decrement. Spread enough that ordering is unambiguous.
            ratings[team] = 1800.0 - li * 5.0 - ti * 100.0
    return ratings


@pytest.fixture
def synthetic_fair_probs(real_groups) -> dict[str, float]:
    """fair_probs matching the synthetic_ratings ordering.

    Within each group, descending: 0.04, 0.03, 0.02, 0.01. Sum across all
    48 teams = 4.8 (intentionally not normalised - we only need the
    relative ordering for build_scenario_for_group, not absolute values).
    """
    probs: dict[str, float] = {}
    for letter, teams in real_groups.items():
        for i, team in enumerate(teams):
            probs[team] = 0.04 - i * 0.01
    return probs


@pytest.fixture
def baseline_scenario(real_groups, synthetic_fair_probs) -> Scenario:
    """An empty scenario - every group resolves with team1 winning."""
    standings = build_baseline_standings(real_groups, synthetic_fair_probs)
    return Scenario(
        scenario_id="baseline",
        description="test baseline",
        deviating_group="",
        favourite="",
        upset_winner="",
        standings=standings,
    )


# select_third_placers_by_elo


def test_third_placers_selects_eight(real_groups, synthetic_ratings, synthetic_fair_probs):
    """Selects exactly 8 third-placers from 12 candidates."""
    standings = build_baseline_standings(real_groups, synthetic_fair_probs)
    selected = select_third_placers_by_elo(standings, synthetic_ratings)
    assert len(selected) == 8


def test_third_placers_returns_team_group_tuples(real_groups, synthetic_ratings, synthetic_fair_probs):
    """Each selected entry is a (team, group_letter) tuple."""
    standings = build_baseline_standings(real_groups, synthetic_fair_probs)
    selected = select_third_placers_by_elo(standings, synthetic_ratings)
    for entry in selected:
        assert len(entry) == 2
        team, group = entry
        assert isinstance(team, str)
        assert group in real_groups


def test_third_placers_picks_strongest(real_groups, synthetic_ratings, synthetic_fair_probs):
    """The 8 picked are the 8 with highest Elo among third-placers."""
    standings = build_baseline_standings(real_groups, synthetic_fair_probs)
    selected = select_third_placers_by_elo(standings, synthetic_ratings)
    selected_teams = {team for team, _g in selected}

    # All 12 third-placers, ranked by Elo
    all_thirds = [(standings[g][2], g) for g in real_groups]
    all_thirds.sort(key=lambda tg: -synthetic_ratings[tg[0]])

    top_8_expected = {team for team, _g in all_thirds[:8]}
    assert selected_teams == top_8_expected


# resolve_r32_pairings


def test_r32_pairings_returns_sixteen_matches(baseline_scenario, synthetic_ratings):
    """The R32 has exactly 16 matches."""
    pairings = resolve_r32_pairings(baseline_scenario, synthetic_ratings)
    assert len(pairings) == 16


def test_r32_pairings_have_32_distinct_teams(baseline_scenario, synthetic_ratings):
    """Across all 16 R32 matches, exactly 32 distinct teams participate."""
    pairings = resolve_r32_pairings(baseline_scenario, synthetic_ratings)
    teams = {t for _id, a, b in pairings for t in (a, b)}
    assert len(teams) == 32


def test_r32_match_ids_are_73_to_88(baseline_scenario, synthetic_ratings):
    """The match IDs match the vendored R32_BRACKET constants (73-88)."""
    pairings = resolve_r32_pairings(baseline_scenario, synthetic_ratings)
    match_ids = sorted(mid for mid, _a, _b in pairings)
    assert match_ids == list(range(73, 89))


def test_r32_no_team_plays_itself(baseline_scenario, synthetic_ratings):
    """No match has the same team on both sides - a real bug guard."""
    pairings = resolve_r32_pairings(baseline_scenario, synthetic_ratings)
    for _mid, a, b in pairings:
        assert a != b, f"Team {a} can't play itself"


# propagate: invariants on synthetic inputs


def test_propagate_match_winner_distributions_sum_to_one(
    baseline_scenario, synthetic_ratings
):
    """Every match in the propagation result has winner-probs summing to 1."""
    predictor = make_uniform_predictor()
    result = propagate(baseline_scenario, synthetic_ratings, predictor=predictor)
    for match_id, dist in result.match_winners.items():
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-9, (
            f"Match {match_id} winner distribution sums to {total}, expected 1.0"
        )


def test_propagate_all_r32_teams_have_full_r32_probability(
    baseline_scenario, synthetic_ratings
):
    """All 32 R32 participants have P(reach R32) = 1.0."""
    predictor = make_uniform_predictor()
    result = propagate(baseline_scenario, synthetic_ratings, predictor=predictor)
    assert len(result.survival) == 32
    for team, probs in result.survival.items():
        assert probs["R32"] == 1.0, f"{team} has P(R32) = {probs['R32']}, expected 1.0"


def test_propagate_survival_is_monotonically_nonincreasing(
    baseline_scenario, synthetic_ratings
):
    """P(reach R32) >= P(reach R16) >= ... >= P(Win) for every team."""
    predictor = make_uniform_predictor()
    result = propagate(baseline_scenario, synthetic_ratings, predictor=predictor)
    for team, probs in result.survival.items():
        for i in range(len(KNOCKOUT_ROUNDS) - 1):
            this_round, next_round = KNOCKOUT_ROUNDS[i], KNOCKOUT_ROUNDS[i + 1]
            assert probs[this_round] >= probs[next_round] - 1e-9, (
                f"{team}: P({this_round})={probs[this_round]} < "
                f"P({next_round})={probs[next_round]} violates monotonicity"
            )


def test_propagate_sum_of_p_win_is_one(baseline_scenario, synthetic_ratings):
    """The sum of P(Win) over all teams equals 1.0 - total prob conserved."""
    predictor = make_uniform_predictor()
    result = propagate(baseline_scenario, synthetic_ratings, predictor=predictor)
    total_win = sum(probs["Win"] for probs in result.survival.values())
    assert abs(total_win - 1.0) < 1e-9, f"Σ P(Win) = {total_win}, expected 1.0"


def test_propagate_excludes_group_stage_eliminated_teams(
    baseline_scenario, synthetic_ratings
):
    """Teams that don't make the knockout aren't in the output."""
    predictor = make_uniform_predictor()
    result = propagate(baseline_scenario, synthetic_ratings, predictor=predictor)
    # 48 total teams; 32 in knockout; 16 eliminated
    assert len(result.survival) == 32
    # Teams in result are a subset of all 48 group teams
    all_teams = {t for teams in baseline_scenario.standings.values() for t in teams}
    eliminated = all_teams - set(result.survival.keys())
    assert len(eliminated) == 16


# propagate: behavioural check with a dominant predictor


def test_propagate_dominant_team_wins_with_certainty(
    baseline_scenario, synthetic_ratings, real_groups
):
    """If one team always wins, its P(Win) is exactly 1.0.

    Uses the favourite of Group A (which is in the knockout under our
    baseline scenario - top of Group A always reaches R32).
    """
    dominant_team = real_groups["A"][0]
    predictor = make_dominant_predictor(dominant_team)
    result = propagate(baseline_scenario, synthetic_ratings, predictor=predictor)

    probs = result.survival[dominant_team]
    for round_name in KNOCKOUT_ROUNDS:
        assert abs(probs[round_name] - 1.0) < 1e-9, (
            f"Dominant team {dominant_team} has P({round_name})={probs[round_name]}, "
            f"expected 1.0"
        )


# propagate: deviation scenario differs from baseline


def test_propagate_deviation_scenario_changes_outputs(
    real_groups, synthetic_fair_probs, synthetic_ratings
):
    """A deviation scenario produces *different* survival probs than baseline.

    This is a regression guard: if the propagator silently ignored the
    scenario's standings, baseline and deviation would be identical.

    We use the DOMINANT predictor (one team always wins) rather than the
    uniform one. Reason: with uniform coinflips every team has P(Win) = 1/32
    by symmetry, regardless of bracket position - so swapping Spain and
    Uruguay in Group H changes nothing observable. Under the dominant
    predictor, non-dominant teams' P(reach round R) DOES depend on when
    they meet the dominant team, which depends on bracket position, which
    depends on the scenario standings. Result: at least some non-dominant
    team's probabilities differ between baseline and deviation.
    """
    standings = build_baseline_standings(real_groups, synthetic_fair_probs)
    baseline = Scenario(
        scenario_id="baseline", description="", deviating_group="",
        favourite="", upset_winner="", standings=standings,
    )
    deviation = build_scenario_for_group("H", real_groups, synthetic_fair_probs)

    # Pick a dominant team from a non-deviating group so the dominant team
    # itself is in the same bracket slot under both scenarios. The diffs
    # come from *other* teams' interaction with the dominant team's path.
    dominant_team = real_groups["I"][0]  # I-favourite (France in real data)
    predictor = make_dominant_predictor(dominant_team)
    baseline_result = propagate(baseline, synthetic_ratings, predictor=predictor)
    deviation_result = propagate(deviation, synthetic_ratings, predictor=predictor)

    # At least *some* non-dominant team's P(reach R16) or P(reach QF) should
    # differ - these are the rounds where bracket-half placement matters.
    n_changed = 0
    for team in baseline_result.survival:
        if team == dominant_team or team not in deviation_result.survival:
            continue
        for round_name in ("R16", "QF", "SF", "F", "Win"):
            diff = abs(
                baseline_result.survival[team][round_name]
                - deviation_result.survival[team][round_name]
            )
            if diff > 1e-6:
                n_changed += 1
                break
    assert n_changed > 0, "Deviation scenario produced identical outputs to baseline"