"""Analytical bracket propagation - the core math of the framework.

Given:
  - A scenario specifying group standings (12 groups × 4 teams each)
  - A match-prediction function (P(team_a beats team_b))
  - Elo ratings (used to pick the 8 best third-placers per a fixed heuristic)

Produces:
  - For every team in the knockout: P(team reaches round R) for R in
    {R32, R16, QF, SF, F, Win}

Algorithm:
  1. Read the scenario's standings to identify R32 participants:
       - 12 group winners (1st place per group)
       - 12 runners-up (2nd place per group)
       - 8 best third-placers, picked by Elo descending from the 12 3rd-place teams
  2. Build the R32 bracket using the vendored `R32_BRACKET` constants + his
     `r32_seeding_table.json` for the 8 winner-vs-third pairings.
  3. Precompute `P(team_a beats team_b)` for every ordered pair via the vendored
     `build_ko_advance_table` (regulation-time goals → P_home + 0.5·P_draw).
  4. Walk the bracket round-by-round, propagating probability distributions.
     At each match, the winner distribution is the convolution of (who's here
     from the left side) and (who's here from the right side) weighted by
     pairwise win probabilities.

The walk is fully deterministic - no Monte Carlo sampling. For WC 2026's
binary-tree bracket, this is exact (no approximation error beyond what's in
the match predictor itself).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from upset_propagation._vendored.simulator import (
    HostInfo,
    Predictor,
    build_ko_advance_table,
    make_elo_predictor,
)
from upset_propagation._vendored.single_game import ModelParams
from upset_propagation._vendored.tournaments.wc2026 import (
    ELO_HISTORY_NAME_ALIASES,
    LATER_ROUNDS,
    R32_BRACKET,
    WC2026_HOST_CONFEDERATION,
    WC2026_HOST_COUNTRIES,
    WC2026Bundle,
)
from upset_propagation.config import KNOCKOUT_ROUNDS
from upset_propagation.scenarios import (
    Scenario,
    load_groups,
    resolve_elo_for_wc_team,
)


# Bracket → round mapping


# Maps the vendored match_id ranges to round names.
# R32 = 73-88, R16 = 89-96, QF = 97-100, SF = 101-102, Final = 104
# (3rd-place playoff M103 is skipped in v1 per the vendored wc2026.py)
def _round_for_match(match_id: int) -> str:
    if 73 <= match_id <= 88:
        return "R32"
    if 89 <= match_id <= 96:
        return "R16"
    if 97 <= match_id <= 100:
        return "QF"
    if 101 <= match_id <= 102:
        return "SF"
    if match_id == 104:
        return "F"
    raise ValueError(f"Unknown match_id: {match_id}")


# The "round a team reaches" by *participating* in a match. R32 participants
# have reached R32; their match outcome determines whether they reach R16; etc.
_ROUND_AFTER: dict[str, str] = {
    "R32": "R16",
    "R16": "QF",
    "QF": "SF",
    "SF": "F",
    "F": "Win",
}


# R32 participant resolution


def select_third_placers_by_elo(
    standings: dict[str, list[str]],
    ratings: dict[str, float],
) -> list[tuple[str, str]]:
    """Pick the 8 best third-place finishers by Elo descending.

    By design, any reasonable heuristic is fine for v1. Elo descending
    is the simplest defensible choice - consistent with using calibrated
    strengths elsewhere in the framework. If two thirds have identical Elo
    (vanishingly unlikely) the tie is broken by group letter.

    Returns: list of (team, group_letter) tuples, ordered best-Elo first.
    Length is always 8 (out of 12 third-place finishers).
    """
    thirds = [(standings[g][2], g) for g in standings]
    thirds_with_elo = [
        (team, group, resolve_elo_for_wc_team(team, ratings))
        for team, group in thirds
    ]
    thirds_with_elo.sort(key=lambda item: (-item[2], item[1]))
    return [(team, group) for team, group, _elo in thirds_with_elo[:8]]


def resolve_r32_participants(
    scenario: Scenario,
    ratings: dict[str, float],
) -> tuple[
    dict[str, str],            # winners: {group_letter: team}
    dict[str, str],            # runners_up: {group_letter: team}
    dict[str, str],            # third_by_group: {group_letter: team} (only 8 entries)
    dict[str, str],            # r32_seeding: {"1A": "3X", ...} from the lookup table
]:
    """Determine all 32 teams entering R32 + the seeding-table lookup.

    Returns 4 dicts that together fully specify the R32 bracket:
      - winners[g]: which team won group g
      - runners_up[g]: which team finished 2nd
      - third_by_group[g]: which team finished 3rd AND advanced (only 8 of 12)
      - r32_seeding: the FIFA seeding-table row, mapping each W-vs-3rd slot
        (e.g. "1A") to the 3rd-place group it pairs with (e.g. "3E")
    """
    standings = scenario.standings
    winners = {g: standings[g][0] for g in standings}
    runners_up = {g: standings[g][1] for g in standings}

    top8_thirds = select_third_placers_by_elo(standings, ratings)
    third_by_group = {group: team for team, group in top8_thirds}

    # Compose the seeding-table key: sorted 8-letter string of advancing groups
    advancing_groups = sorted(third_by_group.keys())
    r32_key = "".join(advancing_groups)

    # Load the seeding table (vendored data file)
    from upset_propagation.config import R32_SEEDING_TABLE
    import json

    with R32_SEEDING_TABLE.open() as f:
        r32_table = json.load(f)
    r32_seeding = r32_table[r32_key]

    return winners, runners_up, third_by_group, r32_seeding


def resolve_r32_pairings(
    scenario: Scenario,
    ratings: dict[str, float],
) -> list[tuple[int, str, str]]:
    """Return the 16 R32 matches as (match_id, team_a, team_b) tuples.

    Walks the vendored `R32_BRACKET` constant, resolving each source spec
    (("W", letter) | ("RU", letter) | ("T3", letter)) to a concrete team
    based on the scenario standings and the seeding-table lookup.
    """
    winners, runners_up, third_by_group, r32_seeding = resolve_r32_participants(
        scenario, ratings
    )

    def resolve_source(source: tuple[str, str]) -> str:
        kind, key = source
        if kind == "W":
            return winners[key]
        if kind == "RU":
            return runners_up[key]
        if kind == "T3":
            # The W-vs-3rd slot is indexed by the winner's group; the
            # seeding table tells us which 3rd-place group pairs with it.
            third_slot = r32_seeding[f"1{key}"]   # e.g. "3E"
            third_group = third_slot[1]            # "E"
            return third_by_group[third_group]
        raise ValueError(f"Unknown source kind: {kind!r}")

    pairings: list[tuple[int, str, str]] = []
    for match_id, left_src, right_src in R32_BRACKET:
        team_a = resolve_source(left_src)
        team_b = resolve_source(right_src)
        pairings.append((match_id, team_a, team_b))
    return pairings


# Probability propagation


@dataclass
class PropagationResult:
    """The output of `propagate_scenario`.

    Attributes:
        survival: {team: {round: P(team reaches round)}}, where round is one
            of KNOCKOUT_ROUNDS. Teams not in the knockout (those eliminated
            in groups under this scenario) are not included.
        match_winners: {match_id: {team: P(team wins this match)}}. Useful
            for diagnostics and for higher-resolution outputs later.
    """

    survival: dict[str, dict[str, float]]
    match_winners: dict[int, dict[str, float]]


def propagate(
    scenario: Scenario,
    ratings: dict[str, float],
    predictor: Optional[Predictor] = None,
) -> PropagationResult:
    """Walk the bracket analytically and return round-by-round survival probs.

    Args:
        scenario: the bracket scenario (12 groups × 4 teams standings)
        ratings: {team: Elo rating}; used by predictor AND by third-placer
            selection
        predictor: optional `(elo_a, elo_b, ctx) -> MatchPrediction` callable.
            Defaults to the vendored Elo+Poisson model with stock params. The
            calibrator will hand us a calibrated variant later.

    Returns: PropagationResult.
    """
    if predictor is None:
        predictor = make_elo_predictor(ModelParams())

    pairings = resolve_r32_pairings(scenario, ratings)
    teams_in_ko = sorted({t for _id, a, b in pairings for t in (a, b)})

    # Precompute pairwise P(a beats b) for all (team, team) in knockout.
    host = HostInfo(
        host_countries=list(WC2026_HOST_COUNTRIES),
        host_confederation=WC2026_HOST_CONFEDERATION,
    )

    # build_ko_advance_table expects ratings keyed by names matching the team
    # strings. Apply the WC2026→elo_history alias mapping so lookups succeed.
    ratings_for_predictor: dict[str, float] = {}
    for team in teams_in_ko:
        if team in ratings:
            ratings_for_predictor[team] = ratings[team]
        elif team in ELO_HISTORY_NAME_ALIASES and ELO_HISTORY_NAME_ALIASES[team] in ratings:
            # Inject the rating under the WC2026 name so the predictor lookup works.
            ratings_for_predictor[team] = ratings[ELO_HISTORY_NAME_ALIASES[team]]
        else:
            raise KeyError(
                f"No Elo rating found for knockout team {team!r}"
            )

    ko_advance = build_ko_advance_table(
        teams=teams_in_ko,
        ratings=ratings_for_predictor,
        host=host,
        predictor=predictor,
    )

    # R32: each match has two certain participants
    # match_winners[id] = {team: P(team wins this match)}
    match_winners: dict[int, dict[str, float]] = {}

    for match_id, team_a, team_b in pairings:
        p_a = ko_advance[(team_a, team_b)]
        match_winners[match_id] = {team_a: p_a, team_b: 1.0 - p_a}

    # R16 and later: convolve previous-round distributions
    for match_id, prev_left_id, prev_right_id in LATER_ROUNDS:
        left_dist = match_winners[prev_left_id]
        right_dist = match_winners[prev_right_id]
        result: dict[str, float] = {}
        for team_a, p_a in left_dist.items():
            for team_b, p_b in right_dist.items():
                # team_a and team_b are on opposite sides of the bracket so
                # they're never the same team; no self-match case to handle.
                p_a_beats_b = ko_advance[(team_a, team_b)]
                # P(team_a wins this match) gets contribution from THIS
                # particular (team_a vs team_b) combination.
                joint = p_a * p_b
                result[team_a] = result.get(team_a, 0.0) + joint * p_a_beats_b
                result[team_b] = result.get(team_b, 0.0) + joint * (1.0 - p_a_beats_b)
        match_winners[match_id] = result

    # Aggregate to per-team survival probabilities
    # P(team reaches R) is the sum of P(team wins match m) for any m in the
    # round PRECEDING R. R32 participants all reach R32 by definition.
    survival: dict[str, dict[str, float]] = {
        team: {r: 0.0 for r in KNOCKOUT_ROUNDS} for team in teams_in_ko
    }
    for team in teams_in_ko:
        survival[team]["R32"] = 1.0

    for match_id, dist in match_winners.items():
        round_played = _round_for_match(match_id)
        round_reached = _ROUND_AFTER[round_played]
        for team, prob in dist.items():
            survival[team][round_reached] += prob

    return PropagationResult(survival=survival, match_winners=match_winners)


# CLI helpers


def format_survival_table(result: PropagationResult, top_n: int = 10) -> str:
    """Pretty-print top-N teams by P(Win) with full survival columns."""
    rounds = KNOCKOUT_ROUNDS
    header = ["Team".ljust(28)] + [r.rjust(7) for r in rounds]
    lines = ["  ".join(header)]
    lines.append("-" * len("  ".join(header)))
    ranked = sorted(
        result.survival.items(), key=lambda kv: -kv[1]["Win"]
    )[:top_n]
    for team, probs in ranked:
        row = [team.ljust(28)] + [
            f"{probs[r]:.4f}".rjust(7) for r in rounds
        ]
        lines.append("  ".join(row))
    return "\n".join(lines)


if __name__ == "__main__":
    # Manual smoke test - `python -m upset_propagation.propagator`
    from upset_propagation.baseline import fetch_baseline_fair_probs
    from upset_propagation.scenarios import build_all_scenarios, load_latest_elo

    print("Loading inputs...")
    fair_probs = fetch_baseline_fair_probs()
    ratings = load_latest_elo()
    scenarios = build_all_scenarios(fair_probs)

    # Run scenario H (Spain drops to 2nd) as the canonical demo
    sH = next(s for s in scenarios if s.deviating_group == "H")
    print(f"\nScenario: {sH.description}\n")
    result = propagate(sH, ratings)

    print("Top 10 teams by P(Win):\n")
    print(format_survival_table(result, top_n=10))

    # Sanity: every match's distribution should sum to 1
    print("\n--- Sanity checks ---")
    for mid, dist in list(result.match_winners.items())[:3]:
        total = sum(dist.values())
        print(f"  M{mid} match-winner dist sums to {total:.6f}")

    # Sanity: P(reach R32) is 1 for all 32 knockout teams
    n_in_r32 = sum(1 for probs in result.survival.values() if probs["R32"] == 1.0)
    print(f"  Teams with P(R32)=1: {n_in_r32}/32")

    # Sanity: P(Win) sums to ~1 across all teams
    total_win = sum(probs["Win"] for probs in result.survival.values())
    print(f"  Sum of P(Win) across all teams: {total_win:.6f}")
