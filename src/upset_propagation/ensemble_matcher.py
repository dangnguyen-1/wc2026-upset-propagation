"""Ensemble similarity matcher.

Combines the Hamming + L1 matchers into one unified ranking via Borda
count. Two genuinely-complementary signals:

  - Hamming matcher: input identity (which teams are 1st/2nd in each
    group). Cheap, works on partial states, captures "did the right
    teams advance?"
  - L1 matcher: propagation output (downstream effect on team
    Win/SF/QF/... probabilities). Captures "do the bracket dynamics
    play out similarly?"

We tested two additional matchers (feature-vector L2 and LLM embeddings
via OpenAI) during development and found they didn't add meaningful
signal over Hamming + L1:
  - feature-vector: ties 12 single-deviation scenarios at identical L2
    distance because z-score normalization makes any swap equivalent.
    Coarse buckets, not a ranking.
  - LLM embeddings: ranks by entity overlap in scenario descriptions,
    which approximates structural similarity but isn't directly about
    bracket geometry. Information overlaps with feature-vector.
Both were dropped from the production ensemble; the earlier
implementations are preserved in git history if needed.

Borda count math:
  For each matcher, sort scenarios ascending by distance, assign ranks
  1..N. For ties, assign the AVERAGE of the rank positions occupied.
  Per scenario: borda_sum = rank_hamming + rank_l1.
  Lower borda_sum = better match.

Normalised score:
  score = (2N - borda_sum) / (2N - 2) where N = number of scenarios.
  Best (rank 1 in both) = 1.0; worst (rank N in both) = 0.0.
  Linear in between, stable across tournaments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from upset_propagation._vendored.simulator import Predictor
from upset_propagation.config import OUTPUT_DIR
from upset_propagation.l1_matcher import L1Match, find_best_scenarios_l1
from upset_propagation.state_matcher import (
    RealisedState,
    ScenarioMatch,
    find_best_scenarios,
)


# Number of matchers in the ensemble - used for Borda normalization.
N_MATCHERS = 2


# Result type


@dataclass
class EnsembleMatch:
    """One scenario's combined ranking across the Hamming + L1 matchers.

    Attributes:
        scenario_id: scenario identifier
        scenario_path: path to the scenario JSON (for loading the survival table)
        borda_sum: sum of ranks across the two matchers (lower = better)
        score: normalized in [0, 1] where 1.0 = unanimous best match across
            both matchers, 0.0 = unanimous worst match
        per_matcher_ranks: {"hamming": rank_h, "l1": rank_l}
            for debugging - which matcher contributed what is visible
            to the final rank
        per_matcher_distances: {"hamming": d_h, "l1": d_l}
            raw distances from each matcher, for inspection
    """
    scenario_id: str
    scenario_path: Path
    borda_sum: float
    score: float
    per_matcher_ranks: dict[str, float] = field(default_factory=dict)
    per_matcher_distances: dict[str, float] = field(default_factory=dict)


# Average-rank assignment for ties


def _assign_average_ranks(distances: list[float]) -> list[float]:
    """Convert a list of distances into average-rank positions.

    Uses fractional ranking semantics: ties get the mean of the rank
    positions they collectively occupy.

    Example: distances [10, 10, 10, 20] → ranks [2.0, 2.0, 2.0, 4.0]
    (the three 10s occupy positions 1, 2, 3 → mean 2.0; the 20 occupies
    position 4.)
    """
    n = len(distances)
    if n == 0:
        return []
    sorted_indices = sorted(range(n), key=lambda i: distances[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and distances[sorted_indices[j]] == distances[sorted_indices[i]]:
            j += 1
        # Tied indices i..j-1 occupy rank positions (i+1)..j; average = (i+1+j)/2
        avg_rank = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[sorted_indices[k]] = avg_rank
        i = j
    return ranks


# Main entry point


def find_best_scenarios_ensemble(
    state: RealisedState,
    predictor: Predictor,
    ratings: dict[str, float],
    fair_probs: dict[str, float],
    output_dir: Optional[Path] = None,
    k: int = 10,
) -> list[EnsembleMatch]:
    """Rank precomputed scenarios using Borda count across Hamming + L1.

    Pipeline:
      1. Run both matchers, requesting the FULL ranking
      2. Assign average-rank for each matcher to break ties
      3. Sum ranks per scenario → borda_sum
      4. Normalize to score in [0, 1]
      5. Sort by (borda_sum, scenario_id) - alpha tiebreak for determinism
      6. Return top k

    Args:
        state: realised tournament state (full or partial)
        predictor: calibrated predictor (from
            l1_matcher.load_calibrated_predictor_from_index)
        ratings: team Elo ratings
        fair_probs: current API fair_probs (used by L1 matcher to fill
            unobserved groups)
        output_dir: where the scenario JSONs live (default: ./output/)
        k: number of top matches to return

    Returns: top-k EnsembleMatch, sorted by borda_sum ascending, then
        scenario_id ascending for determinism.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    # Get full rankings from both matchers. k=200 is safely larger than
    # the actual scenario count (~80).
    K_ALL = 200
    hamming_matches: list[ScenarioMatch] = find_best_scenarios(
        state, output_dir, k=K_ALL
    )
    l1_matches: list[L1Match] = find_best_scenarios_l1(
        state, predictor, ratings, fair_probs, output_dir, k=K_ALL
    )

    # Build {scenario_id -> {distances, path}}
    by_id: dict[str, dict[str, object]] = {}
    for m in hamming_matches:
        by_id.setdefault(m.scenario_id, {"path": m.scenario_path})
        by_id[m.scenario_id]["d_hamming"] = float(m.hamming_distance)
    for m in l1_matches:
        by_id.setdefault(m.scenario_id, {"path": m.scenario_path})
        by_id[m.scenario_id]["d_l1"] = m.l1_distance

    # Only scenarios that appeared in both matchers - defensive guard.
    # In normal operation all scenarios appear in both since they scan
    # the same output dir.
    scenario_ids = [
        sid for sid, data in by_id.items()
        if "d_hamming" in data and "d_l1" in data
    ]

    if not scenario_ids:
        return []

    # Compute average ranks per matcher
    d_h = [float(by_id[sid]["d_hamming"]) for sid in scenario_ids]
    d_l = [float(by_id[sid]["d_l1"]) for sid in scenario_ids]
    ranks_h = _assign_average_ranks(d_h)
    ranks_l = _assign_average_ranks(d_l)

    # Borda sums + normalized scores. With 2 matchers, min sum = 2 (rank 1
    # in both), max sum = 2N (rank N in both).
    n = len(scenario_ids)
    min_borda = float(N_MATCHERS)
    max_borda = float(N_MATCHERS * n)
    span = max_borda - min_borda

    ensemble_matches = []
    for i, sid in enumerate(scenario_ids):
        borda_sum = ranks_h[i] + ranks_l[i]
        score = 1.0 - (borda_sum - min_borda) / span if span > 0 else 1.0
        ensemble_matches.append(EnsembleMatch(
            scenario_id=sid,
            scenario_path=by_id[sid]["path"],  # type: ignore[arg-type]
            borda_sum=borda_sum,
            score=score,
            per_matcher_ranks={
                "hamming": ranks_h[i],
                "l1": ranks_l[i],
            },
            per_matcher_distances={
                "hamming": d_h[i],
                "l1": d_l[i],
            },
        ))

    # Deterministic sort
    ensemble_matches.sort(key=lambda m: (m.borda_sum, m.scenario_id))
    return ensemble_matches[:k]


# Pretty-printing for CLI


def format_ensemble_match(match: EnsembleMatch) -> str:
    """One-line summary for smoke tests / diagnostics."""
    return (
        f"  score={match.score:.4f}  borda={match.borda_sum:6.1f}  "
        f"{match.scenario_id:42s}  "
        f"ranks=[H:{match.per_matcher_ranks['hamming']:5.1f}, "
        f"L1:{match.per_matcher_ranks['l1']:5.1f}]"
    )


# CLI smoke test


if __name__ == "__main__":
    # Manual smoke test - `python -m upset_propagation.ensemble_matcher`
    from upset_propagation.baseline import fetch_baseline_fair_probs
    from upset_propagation.l1_matcher import load_calibrated_predictor_from_index
    from upset_propagation.scenarios import (
        build_baseline_standings,
        load_groups,
        load_latest_elo,
    )
    from upset_propagation.state_matcher import parse_state_from_dict

    fair_probs = fetch_baseline_fair_probs()
    ratings = load_latest_elo()
    groups = load_groups()
    predictor = load_calibrated_predictor_from_index(OUTPUT_DIR / "index.json")
    baseline_standings = build_baseline_standings(groups, fair_probs)

    # Test 1: baseline state
    print("Test 1: state == baseline (all favourites win)")
    state1 = parse_state_from_dict(baseline_standings)
    top = find_best_scenarios_ensemble(
        state1, predictor, ratings, fair_probs, k=5
    )
    for m in top:
        print(format_ensemble_match(m))
    print()

    # Test 2: Spain-H
    print("Test 2: Spain finishes 2nd in H - should match spain_runner_up_H")
    state2_standings = {g: list(s) for g, s in baseline_standings.items()}
    h = state2_standings["H"]
    state2_standings["H"] = [h[1], h[0], h[2], h[3]]
    state2 = parse_state_from_dict(state2_standings)
    top = find_best_scenarios_ensemble(
        state2, predictor, ratings, fair_probs, k=5
    )
    for m in top:
        print(format_ensemble_match(m))
    print()

    # Test 3: HJ pairwise
    print("Test 3: Spain-H AND Argentina-J - should match HJ pairwise")
    state3_standings = {g: list(s) for g, s in baseline_standings.items()}
    h = state3_standings["H"]
    state3_standings["H"] = [h[1], h[0], h[2], h[3]]
    j = state3_standings["J"]
    state3_standings["J"] = [j[1], j[0], j[2], j[3]]
    state3 = parse_state_from_dict(state3_standings)
    top = find_best_scenarios_ensemble(
        state3, predictor, ratings, fair_probs, k=5
    )
    for m in top:
        print(format_ensemble_match(m))