"""Implied fair_probs from the framework.

Computes the framework's own view of each team's tournament probabilities
by taking a weighted average across all 79 precomputed scenarios, using
each scenario's ensemble-matcher score as its weight.

The math:

    implied_P(team reaches round) = Σ_scenarios [score × P(team reaches round | scenario)]
                                    -------------------------------------------------------
                                                  Σ_scenarios score

where:
  - score is the ensemble matcher's normalized [0, 1] score for each
    scenario relative to the realised state (1.0 = exact match)
  - P(team reaches round | scenario) is read from each scenario's
    survival[team][round] field
  - the sum runs over ALL 79 scenarios - tail scenarios still contribute
    their weighted bit even when score is low

Teams missing from a scenario's KO bracket (because the 8 best-third-
placers differ across scenarios) contribute 0 to the numerator for that
scenario but the scenario's weight still counts in the denominator. This
correctly treats "not in KO" as "P(reaching any KO round) = 0" rather
than excluding the scenario from the average - which would inflate
probabilities of teams that are rarely in KO.

What this is useful for:

  - **Compare to the FairLine model's baseline**: where do we disagree with the
    sportsbook-devigged numbers? Big gaps = potential trading edges.
  - **Sharpen as the tournament progresses**: pre-tournament, all
    scenarios have similar ensemble scores so implied ≈ uniform average.
    As groups play out and the ensemble sharpens onto a small set of
    scenarios, implied probabilities collapse onto the matched
    scenario's values.

Public API:
    ImpliedProbs dataclass
    compute_implied_probs(state, predictor, ratings, fair_probs, output_dir)
        -> ImpliedProbs
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from upset_propagation._vendored.simulator import Predictor
from upset_propagation.config import OUTPUT_DIR
from upset_propagation.ensemble_matcher import find_best_scenarios_ensemble
from upset_propagation.state_matcher import RealisedState


# Rounds we compute implied probs for, in tournament order.
ROUNDS = ("R32", "R16", "QF", "SF", "F", "Win")


# Result type


@dataclass
class ImpliedProbs:
    """The framework's implied fair_probs for a given realised state.

    Attributes:
        probs: {team: {round: probability}} - nested dict matching the
            scenario JSON survival format. Each team has 6 round entries
            (R32, R16, QF, SF, F, Win). Teams with implied_P=0 across all
            rounds (never appeared in any positively-weighted scenario's
            KO) are still present with all zeros for completeness.
        total_weight: sum of ensemble scores across all scenarios used.
            For sanity checks: if this is near 0, the matcher couldn't
            find anything to weight on (shouldn't happen with normal
            scenarios) and the probs are meaningless.
        n_scenarios_used: how many scenarios contributed positive weight.
            Lower than 79 only if a scenario's score is exactly 0 (rare).
    """
    probs: dict[str, dict[str, float]] = field(default_factory=dict)
    total_weight: float = 0.0
    n_scenarios_used: int = 0


# Main entry point


def compute_implied_probs(
    state: RealisedState,
    predictor: Predictor,
    ratings: dict[str, float],
    fair_probs: dict[str, float],
    output_dir: Optional[Path] = None,
    weight_exponent: float = 4.0,
) -> ImpliedProbs:
    """Compute the framework's implied fair_probs for a realised state.

    Pipeline:
      1. Get ensemble scores for all 79 scenarios
      2. Apply weight = score ** weight_exponent
      3. Load each scenario's survival table
      4. Accumulate weighted sums per (team, round)
      5. Normalize by total weight

    Args:
        state: realised tournament state (full or partial)
        predictor: calibrated predictor (from
            l1_matcher.load_calibrated_predictor_from_index)
        ratings: team Elo ratings (passed to the ensemble matcher)
        fair_probs: API fair_probs (passed to the ensemble matcher;
            also used for partial-state group fill-in)
        output_dir: where the scenario JSONs live
        weight_exponent: power applied to ensemble scores before
            weighting. Default 4.0, selected empirically on 2026-06-10
            from a sweep over p ∈ {1, 2, 4, 8}. p=1 (linear) gave the
            baseline scenario only ~2.5% of total weight even when the
            matcher found it as an exact match, leaving implied probs
            ~1pp off from the matched scenario. p=4 brings tracking to
            ~0.3pp per team on the Win round while preserving smoothing
            across tied scenarios in ambiguous-state cases. p=8 tracked
            even tighter but at the cost of effectively becoming a
            top-K cutoff (rank-40 contributes 0.5^8 = 0.4% weight, vs
            6.25% at p=4).

    Returns: ImpliedProbs.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    # Step 1: get ensemble matches with scores for all 79 scenarios.
    # k=200 ensures we get the complete ranking (always more than 79
    # actual scenarios, so the matcher returns everything it has).
    matches = find_best_scenarios_ensemble(
        state, predictor, ratings, fair_probs,
        output_dir=output_dir, k=200,
    )

    if not matches:
        return ImpliedProbs()

    # Step 2: load each scenario's survival table and accumulate.
    # weighted_sums[team][round] = Σ (score ** exp) × P(team reaches round | scenario)
    weighted_sums: dict[str, dict[str, float]] = {}
    total_weight = 0.0
    n_scenarios_used = 0

    for match in matches:
        weight = match.score ** weight_exponent
        if weight <= 0.0:
            # Skip exact-zero-weight scenarios (saves work, doesn't change
            # the math).
            continue

        # Load the scenario's survival table
        with match.scenario_path.open() as f:
            payload = json.load(f)
        survival = payload.get("survival", {})

        # Accumulate
        for team, round_probs in survival.items():
            if team not in weighted_sums:
                weighted_sums[team] = {rnd: 0.0 for rnd in ROUNDS}
            for rnd in ROUNDS:
                p = round_probs.get(rnd, 0.0)
                weighted_sums[team][rnd] += weight * p

        total_weight += weight
        n_scenarios_used += 1

    # Step 3: normalize by total weight
    if total_weight <= 0.0:
        # Degenerate case - no scenarios contributed. Return empty.
        return ImpliedProbs(total_weight=0.0, n_scenarios_used=0)

    implied: dict[str, dict[str, float]] = {}
    for team, round_sums in weighted_sums.items():
        implied[team] = {
            rnd: round_sums[rnd] / total_weight
            for rnd in ROUNDS
        }

    return ImpliedProbs(
        probs=implied,
        total_weight=total_weight,
        n_scenarios_used=n_scenarios_used,
    )


# Sanity-check helpers


def implied_probs_sanity_check(
    implied: ImpliedProbs,
    tol: float = 1e-6,
) -> list[str]:
    """Return a list of warning strings if implied probs look malformed.

    Checks:
      - Each team's probs are monotonically non-increasing across rounds
        (P(R32) >= P(R16) >= P(QF) >= ... >= P(Win)). A team can't reach
        SF without reaching QF first.
      - Σ_team P(team wins) ≈ 1.0 (exactly one team wins the tournament)
      - All probabilities in [0, 1]

    Returns empty list if all checks pass.
    """
    warnings: list[str] = []

    # Check 1: monotonic decrease
    for team, round_probs in implied.probs.items():
        prev_p = 1.0
        for rnd in ROUNDS:
            p = round_probs.get(rnd, 0.0)
            if p > prev_p + tol:
                warnings.append(
                    f"{team}: P({rnd})={p:.6f} > P(previous round)={prev_p:.6f}"
                )
            if p < -tol or p > 1.0 + tol:
                warnings.append(
                    f"{team}: P({rnd})={p:.6f} out of [0, 1] range"
                )
            prev_p = p

    # Check 2: Σ P(Win) ≈ 1.0
    total_win = sum(rp.get("Win", 0.0) for rp in implied.probs.values())
    if abs(total_win - 1.0) > 0.01:  # 1pp tolerance for numerical noise
        warnings.append(
            f"Σ P(Win) = {total_win:.6f}, expected ≈ 1.0 (off by "
            f"{(total_win - 1.0) * 100:+.2f}pp)"
        )

    return warnings


def top_teams_by_round(
    implied: ImpliedProbs,
    rnd: str = "Win",
    k: int = 10,
) -> list[tuple[str, float]]:
    """Return the top-k teams by implied probability for a given round.

    Useful for CLI inspection and the comparison view.

    Args:
        implied: ImpliedProbs result
        rnd: which round to rank by ('Win', 'F', 'SF', etc.)
        k: how many top teams

    Returns: [(team, prob), ...] sorted by prob descending.
    """
    if rnd not in ROUNDS:
        raise ValueError(f"Unknown round {rnd!r}; expected one of {ROUNDS}")
    ranked = sorted(
        ((team, rp.get(rnd, 0.0)) for team, rp in implied.probs.items()),
        key=lambda x: -x[1],
    )
    return ranked[:k]


# CLI smoke test


if __name__ == "__main__":
    # Manual smoke test - `python -m upset_propagation.implied_probs`
    #
    # Verifies that the default weight_exponent (4.0) produces sensible
    # implied fair_probs on three test states. The exponent sweep that
    # picked p=4 is preserved in git history (run with the explicit
    # weight_exponent= override if you want to reproduce it).
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

    # Load reference (the FairLine model's baseline ≈ baseline scenario's Win probs since
    # the calibrator fit to her numbers)
    with (OUTPUT_DIR / "baseline.json").open() as f:
        fairline_win = {
            t: rp.get("Win", 0.0)
            for t, rp in json.load(f)["survival"].items()
        }

    # Test 1: state = baseline
    print("Test 1: state == baseline (sanity - should track FairLine closely)")
    state1 = parse_state_from_dict(baseline_standings)
    implied1 = compute_implied_probs(state1, predictor, ratings, fair_probs)
    print(f"  total_weight={implied1.total_weight:.4f}, "
          f"n_scenarios_used={implied1.n_scenarios_used}")
    print(f"  {'Team':25s}  {'Implied':>8s}  {'FairLine':>8s}  {'Δpp':>8s}")
    for team, p in top_teams_by_round(implied1, "Win", k=8):
        delta = (p - fairline_win.get(team, 0.0)) * 100
        print(f"  {team:25s}  {p*100:7.2f}%  {fairline_win.get(team, 0.0)*100:7.2f}%  {delta:+7.2f}pp")
    warnings = implied_probs_sanity_check(implied1)
    if warnings:
        print("  Sanity warnings:")
        for w in warnings:
            print(f"    ⚠ {w}")
    else:
        print("  Sanity check: PASS")
    print()

    # Test 2: state = Spain runs 2nd in H
    print("Test 2: Spain finishes 2nd in H (the conditional trade signal)")
    state2_standings = {g: list(s) for g, s in baseline_standings.items()}
    h = state2_standings["H"]
    state2_standings["H"] = [h[1], h[0], h[2], h[3]]
    state2 = parse_state_from_dict(state2_standings)
    implied2 = compute_implied_probs(state2, predictor, ratings, fair_probs)
    print(f"  total_weight={implied2.total_weight:.4f}, "
          f"n_scenarios_used={implied2.n_scenarios_used}")
    print(f"  {'Team':25s}  {'Implied':>8s}  {'FairLine':>8s}  {'Δpp':>8s}")
    for team, p in top_teams_by_round(implied2, "Win", k=8):
        delta = (p - fairline_win.get(team, 0.0)) * 100
        print(f"  {team:25s}  {p*100:7.2f}%  {fairline_win.get(team, 0.0)*100:7.2f}%  {delta:+7.2f}pp")
    warnings = implied_probs_sanity_check(implied2)
    if warnings:
        print("  Sanity warnings:")
        for w in warnings:
            print(f"    ⚠ {w}")
    else:
        print("  Sanity check: PASS")
    print()

    # -- Test 3: HJ pairwise - the case where v2 found the +0.32pp Argentina edge
    print("Test 3: Spain-H AND Argentina-J both slip (v2's interaction-effect case)")
    state3_standings = {g: list(s) for g, s in baseline_standings.items()}
    h = state3_standings["H"]
    state3_standings["H"] = [h[1], h[0], h[2], h[3]]
    j = state3_standings["J"]
    state3_standings["J"] = [j[1], j[0], j[2], j[3]]
    state3 = parse_state_from_dict(state3_standings)
    implied3 = compute_implied_probs(state3, predictor, ratings, fair_probs)
    print(f"  total_weight={implied3.total_weight:.4f}, "
          f"n_scenarios_used={implied3.n_scenarios_used}")
    print(f"  {'Team':25s}  {'Implied':>8s}  {'FairLine':>8s}  {'Δpp':>8s}")
    for team, p in top_teams_by_round(implied3, "Win", k=8):
        delta = (p - fairline_win.get(team, 0.0)) * 100
        print(f"  {team:25s}  {p*100:7.2f}%  {fairline_win.get(team, 0.0)*100:7.2f}%  {delta:+7.2f}pp")
    warnings = implied_probs_sanity_check(implied3)
    if warnings:
        print("  Sanity warnings:")
        for w in warnings:
            print(f"    ⚠ {w}")
    else:
        print("  Sanity check: PASS")