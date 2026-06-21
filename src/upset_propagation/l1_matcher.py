"""L1 propagation-similarity matcher.

The existing state_matcher.find_best_scenarios() ranks scenarios by Hamming
distance over GROUP STANDINGS (the input side). This module ranks them by
L1 distance over PROPAGATION OUTPUTS (the output side).

Why both matter:
  - Two scenarios with similar standings can produce very different
    propagations (bracket position matters as much as identity)
  - Two scenarios with different standings can produce similar
    propagations (interaction effects can cancel)
  - The Hamming matcher and L1 matcher are complementary signals, and
    v3's ensemble (Borda count over matchers) combines them

Cost: ~60ms extra per match call to propagate the realised state, vs
the ~0ms of the Hamming matcher. Worth it when sub-second latency
isn't required (e.g., 2-hour cron cycle or trader-initiated on-demand
lookups).

Public API:
    load_calibrated_predictor_from_index(index_path) -> Predictor
    propagate_realised_state(state, predictor, ratings) -> PropagationResult
    compute_l1_distance(state_survival, scenario_survival) -> float
    find_best_scenarios_l1(state, output_dir, predictor, ratings, k=10)
        -> list[L1Match]
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from upset_propagation._vendored.simulator import Predictor, make_elo_predictor
from upset_propagation._vendored.single_game import ModelParams
from upset_propagation.calibrator import make_offset_predictor
from upset_propagation.config import NON_SCENARIO_FILENAMES, OUTPUT_DIR
from upset_propagation.propagator import PropagationResult, propagate
from upset_propagation.scenarios import (
    Scenario,
    build_baseline_standings,
    load_groups,
    load_latest_elo,
)
from upset_propagation.state_matcher import RealisedState


# Reconstructing a calibrated predictor from disk


def load_calibrated_predictor_from_index(
    index_path: Path,
) -> Predictor:
    """Reconstruct the calibrated predictor from a previously-saved index.json.

    The calibration step in run.py fits per-team Elo offsets, applies them
    via make_offset_predictor, then writes the offsets into index.json under
    `calibration.offsets`. This function reverses that - reads the offsets,
    wraps a fresh base predictor, returns the wrapped Predictor.

    Avoids the ~4-minute cost of re-running calibration when the user only
    needs to do a one-off similarity lookup against existing scenarios.

    Args:
        index_path: path to index.json (typically output/index.json)

    Returns: a Predictor identical (up to 2-decimal offset rounding) to the
        one that produced the precomputed scenario JSONs.

    Raises:
        FileNotFoundError: if index_path doesn't exist
        KeyError: if index.json doesn't contain calibration.offsets
    """
    if not index_path.exists():
        raise FileNotFoundError(
            f"index.json not found at {index_path}. "
            f"Run `python -m upset_propagation.run` first."
        )

    with index_path.open() as f:
        index = json.load(f)

    try:
        offsets = index["calibration"]["offsets"]
    except KeyError as exc:
        raise KeyError(
            f"index.json at {index_path} missing 'calibration.offsets'. "
            f"This may be an older format - re-run the pipeline to refresh."
        ) from exc

    base_predictor = make_elo_predictor(ModelParams())
    return make_offset_predictor(base_predictor, offsets)


# Propagating a realised state


def _build_scenario_for_realised_state(
    state: RealisedState,
    fair_probs_for_baseline: dict[str, float],
    groups: Optional[dict[str, list[str]]] = None,
) -> Scenario:
    """Convert a (possibly partial) RealisedState into a full Scenario object.

    Unknown groups (not present in state.standings) are filled with seeded
    baseline standings - the order each group would resolve to if every
    favourite won. This is the framework's convention: scenarios always
    cover all 12 groups, with "no deviation" groups using the seeded order.

    Args:
        state: realised tournament state
        fair_probs_for_baseline: needed to determine the seeded ordering
            for unobserved groups
        groups: optional override for the 12-group dict (default: load from disk)
    """
    if groups is None:
        groups = load_groups()

    # Start with the full seeded baseline
    seeded = build_baseline_standings(groups, fair_probs_for_baseline)

    # Overwrite observed groups with the realised standings
    full_standings = {g: list(seeded[g]) for g in seeded}
    for letter, standing in state.standings.items():
        full_standings[letter] = list(standing)

    return Scenario(
        scenario_id="__realised_state__",
        description=(
            f"Realised state with {len(state.played_groups)}/12 groups observed; "
            f"unobserved groups filled with seeded baseline"
        ),
        deviating_group="",
        favourite="",
        upset_winner="",
        standings=full_standings,
    )


def propagate_realised_state(
    state: RealisedState,
    predictor: Predictor,
    ratings: dict[str, float],
    fair_probs_for_baseline: dict[str, float],
) -> PropagationResult:
    """Run the full propagator on a (possibly partial) realised state.

    For partial states (groups not yet played), the unknown standings are
    filled with the seeded baseline order. This means early in the tournament
    the propagation result will look baseline-like; that's correct given
    limited info.

    Returns: PropagationResult with survival probabilities - same shape as
        any precomputed scenario's `survival` field.
    """
    scenario = _build_scenario_for_realised_state(state, fair_probs_for_baseline)
    return propagate(scenario, ratings, predictor=predictor)


# L1 distance computation


def compute_l1_distance(
    state_survival: dict[str, dict[str, float]],
    scenario_survival: dict[str, dict[str, float]],
    rounds: tuple[str, ...] = ("R32", "R16", "QF", "SF", "F", "Win"),
) -> float:
    """Sum of |Δp| across all team-round entries in either propagation.

    A team appearing in only one propagation contributes its full probability
    mass (treated as p=0 on the missing side). This means scenarios with
    different R32 line-ups (e.g., different 3rd-placers) get appropriately
    larger distances.

    Args:
        state_survival: survival dict from the realised state's propagation
        scenario_survival: survival dict from a precomputed scenario
        rounds: which rounds to include in the L1 sum. Default: all 6.
            For Win-only ranking, pass ("Win",).

    Returns: L1 distance as a non-negative float (sum of probability units).
    """
    all_teams = set(state_survival.keys()) | set(scenario_survival.keys())
    total = 0.0
    for team in all_teams:
        state_probs = state_survival.get(team, {})
        scen_probs = scenario_survival.get(team, {})
        for rnd in rounds:
            p_state = state_probs.get(rnd, 0.0)
            p_scen = scen_probs.get(rnd, 0.0)
            total += abs(p_state - p_scen)
    return total


# L1 Match result type


@dataclass
class L1Match:
    """One scenario's L1 distance from the realised state.

    Mirrors the shape of state_matcher.ScenarioMatch so the ensemble layer can treat both matcher outputs uniformly.

    Attributes:
        scenario_id: the matched scenario's id
        scenario_path: path to the full scenario JSON
        l1_distance: sum of |Δp| across (team, round) entries; lower = closer
    """
    scenario_id: str
    scenario_path: Path
    l1_distance: float

    @property
    def sort_key(self) -> float:
        """For sorting: lower l1_distance = better match."""
        return self.l1_distance


# Main entry point


def find_best_scenarios_l1(
    state: RealisedState,
    predictor: Predictor,
    ratings: dict[str, float],
    fair_probs_for_baseline: dict[str, float],
    output_dir: Optional[Path] = None,
    k: int = 10,
    rounds: tuple[str, ...] = ("R32", "R16", "QF", "SF", "F", "Win"),
) -> list[L1Match]:
    """Rank precomputed scenarios by L1 distance to the realised state's
    propagation.

    Pipeline:
      1. Propagate the realised state (using filled-in baseline for unknown
         groups) to get its survival vector
      2. Load each precomputed scenario's survival vector from disk
      3. Compute L1 distance between the two vectors
      4. Sort by distance ascending, return top k

    Args:
        state: realised tournament state (full or partial)
        predictor: calibrated predictor (typically from
            load_calibrated_predictor_from_index)
        ratings: team Elo ratings
        fair_probs_for_baseline: used to fill unobserved groups with seeded
            ordering when state is partial. Pass the same fair_probs used
            for the precomputed scenarios for consistency.
        output_dir: where the scenario JSONs live (default: ./output/)
        k: number of top matches to return
        rounds: rounds to include in L1 distance. Default: all 6.

    Returns: top-k scenarios sorted by L1 distance, lowest first.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    # Step 1: propagate the realised state
    state_result = propagate_realised_state(
        state, predictor, ratings, fair_probs_for_baseline
    )
    state_survival = state_result.survival

    # Step 2-3: load each scenario and compute L1
    matches: list[L1Match] = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name in NON_SCENARIO_FILENAMES:
            continue
        with path.open() as f:
            payload = json.load(f)
        scenario_id = payload.get("scenario_id")
        scenario_survival = payload.get("survival")
        if scenario_id is None or scenario_survival is None:
            # Skip malformed files (shouldn't happen but be defensive)
            continue
        distance = compute_l1_distance(
            state_survival, scenario_survival, rounds=rounds
        )
        matches.append(L1Match(
            scenario_id=scenario_id,
            scenario_path=path,
            l1_distance=distance,
        ))

    # Step 4: sort and return top k
    matches.sort(key=lambda m: m.sort_key)
    return matches[:k]


# Pretty-printing for CLI / diagnostic use


def format_l1_match(match: L1Match) -> str:
    """One-line summary of an L1 match for logs / smoke tests."""
    return (
        f"  L1={match.l1_distance:.4f}  "
        f"{match.scenario_id:42s}"
    )


# CLI smoke test


if __name__ == "__main__":
    # Manual smoke test - `python -m upset_propagation.l1_matcher`
    #
    # Reuses the existing output/ directory; requires that
    # `python -m upset_propagation.run` has been run at least once.
    from upset_propagation.baseline import fetch_baseline_fair_probs
    from upset_propagation.state_matcher import parse_state_from_dict

    fair_probs = fetch_baseline_fair_probs()
    ratings = load_latest_elo()
    groups = load_groups()
    predictor = load_calibrated_predictor_from_index(OUTPUT_DIR / "index.json")

    # Test 1: full baseline state
    print("Test 1: state == baseline (all favourites win)")
    baseline_standings = build_baseline_standings(groups, fair_probs)
    state1 = parse_state_from_dict(baseline_standings)
    top = find_best_scenarios_l1(state1, predictor, ratings, fair_probs, k=5)
    for m in top:
        print(format_l1_match(m))
    print()

    # Test 2: Spain finishes 2nd in H
    print("Test 2: Spain finishes 2nd in H - should match spain_runner_up_H")
    state2_standings = {g: list(s) for g, s in baseline_standings.items()}
    h = state2_standings["H"]
    state2_standings["H"] = [h[1], h[0], h[2], h[3]]
    state2 = parse_state_from_dict(state2_standings)
    top = find_best_scenarios_l1(state2, predictor, ratings, fair_probs, k=5)
    for m in top:
        print(format_l1_match(m))
    print()

    # Test 3: Spain-H AND Argentina-J both slip
    print("Test 3: Spain-H AND Argentina-J - should match HJ pairwise")
    state3_standings = {g: list(s) for g, s in baseline_standings.items()}
    h = state3_standings["H"]
    state3_standings["H"] = [h[1], h[0], h[2], h[3]]
    j = state3_standings["J"]
    state3_standings["J"] = [j[1], j[0], j[2], j[3]]
    state3 = parse_state_from_dict(state3_standings)
    top = find_best_scenarios_l1(state3, predictor, ratings, fair_probs, k=5)
    for m in top:
        print(format_l1_match(m))