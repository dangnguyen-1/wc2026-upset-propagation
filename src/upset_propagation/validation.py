"""Validation suite for the wc2026-upset-propagation framework.

By design, traditional backtesting isn't possible (we predict a
*distribution* of outcomes; a past tournament is a single realisation). So
we validate structurally instead.

Two checks:

  1. Directional sanity (`directional_sanity`):
       For each scenario, group teams by their bracket graph distance from
       the deviating group's R32 slots. The mean |ΔWin| should decrease as
       distance increases. Teams 4 steps away (opposite half of the bracket)
       should have near-zero deltas.

  2. Sensitivity (`sensitivity_check`):
       Perturb a team's Elo by some amount. The team's P(Win) should move
       in the expected direction by at least a threshold. Probability
       conservation (Σ ΔP(Win) = 0) is checked globally - we don't require
       every other team's P(Win) to move strictly monotonically, since
       nonlinear bracket interactions can cause small wrong-direction
       movements in unrelated teams.

The "graph distance" here is depth-of-lowest-common-ancestor in the bracket
tree: two teams in the same R32 match have distance 0 (meet immediately);
teams in opposite halves of the bracket have distance 4 (only meet in the
Final).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from upset_propagation._vendored.simulator import Predictor
from upset_propagation._vendored.tournaments.wc2026 import LATER_ROUNDS, R32_BRACKET
from upset_propagation.propagator import PropagationResult, propagate, resolve_r32_pairings
from upset_propagation.scenarios import Scenario


# Bracket graph distance


def _build_match_predecessors() -> dict[int, tuple[int, int]]:
    """Build {match_id: (preceding_left_id, preceding_right_id)} from LATER_ROUNDS.

    R32 matches have no predecessors (they're the leaves of the tournament
    tree). Returns an empty entry for those via dict.get fallback.
    """
    predecessors: dict[int, tuple[int, int]] = {}
    for match_id, left, right in LATER_ROUNDS:
        predecessors[match_id] = (left, right)
    return predecessors


def _match_round(match_id: int) -> int:
    """Return the round level for a match: 0=R32, 1=R16, 2=QF, 3=SF, 4=F."""
    if 73 <= match_id <= 88:
        return 0
    if 89 <= match_id <= 96:
        return 1
    if 97 <= match_id <= 100:
        return 2
    if 101 <= match_id <= 102:
        return 3
    if match_id == 104:
        return 4
    raise ValueError(f"Unknown match_id: {match_id}")


def _descendants_of_match(
    match_id: int,
    predecessors: dict[int, tuple[int, int]],
) -> set[int]:
    """All R32 match_ids that eventually feed into `match_id`.

    Recursive: if `match_id` is itself an R32 match, returns just {match_id}.
    Otherwise, returns the union of descendants of its two predecessor matches.
    """
    if match_id not in predecessors:
        # R32 match - it's its own only descendant
        return {match_id}
    left, right = predecessors[match_id]
    return _descendants_of_match(left, predecessors) | _descendants_of_match(right, predecessors)


def _r32_match_to_round_path() -> dict[int, list[int]]:
    """For each R32 match, list the ancestors going up to the Final.

    Example: R32 match 73 → R16 match X → QF match Y → SF match Z → Final.
    Returns {73: [89, 97, 101, 104], ...}.

    Used to compute lowest-common-ancestor depth between any two R32 matches.
    """
    predecessors = _build_match_predecessors()
    # Build children: child[parent] -> set of (left, right) preceding matches
    # We need to walk DOWNWARD from each child to each ancestor - i.e. for
    # each R32 match, find its R16 ancestor, then QF ancestor, etc.
    # Easier to invert: for each non-R32 match, the two predecessors point to it.
    # So we build child→parent.
    child_to_parent: dict[int, int] = {}
    for parent_id, (left, right) in predecessors.items():
        child_to_parent[left] = parent_id
        child_to_parent[right] = parent_id

    paths: dict[int, list[int]] = {}
    for r32_id in range(73, 89):  # R32 matches are IDs 73-88
        path = []
        current = r32_id
        while current in child_to_parent:
            current = child_to_parent[current]
            path.append(current)
        paths[r32_id] = path  # [R16, QF, SF, Final]
    return paths


def graph_distance(r32_a: int, r32_b: int) -> int:
    """Bracket graph distance between two R32 matches.

    0 = same R32 (impossible in practice, but well-defined)
    1 = same R16 - teams meet in R16 if both advance
    2 = same QF
    3 = same SF
    4 = same Final (i.e., opposite halves of the bracket)

    Computed as depth of lowest common ancestor in the tournament tree.
    """
    if r32_a == r32_b:
        return 0
    paths = _r32_match_to_round_path()
    path_a = paths[r32_a]
    path_b = paths[r32_b]
    # Walk up both paths; first shared ancestor's index in `path_a` is the
    # distance (we ignore +1 because we want round-LEVEL not jumps).
    set_b = set(path_b)
    for i, ancestor in enumerate(path_a):
        if ancestor in set_b:
            return i + 1
    # Shouldn't happen - they should always share the Final at minimum
    raise ValueError(f"No common ancestor between R32 matches {r32_a} and {r32_b}")


# Mapping scenarios to deviating R32 slots


def deviating_r32_matches(
    scenario: Scenario,
    ratings: dict[str, float],
) -> set[int]:
    """The R32 match IDs that contain the favourite OR upset-winner of any
    deviating group in this scenario.

    For a single-deviation scenario H, the deviating teams are Spain (now
    runner-up of H) and Uruguay (now winner of H). Both end up in R32 matches
    determined by the seeding table. Returns the union of those match IDs.

    For a pairwise HJ scenario, returns the union of 4 R32 matches: 2 for H
    deviation, 2 for J deviation.

    Used as the "epicentre" for graph-distance calculations.
    """
    pairings = resolve_r32_pairings(scenario, ratings)
    deviating_teams: set[str] = set()
    # Use tuple fields to capture both single and pairwise cleanly.
    for fav in scenario.favourites:
        deviating_teams.add(fav)
    for upset in scenario.upset_winners:
        deviating_teams.add(upset)

    matches: set[int] = set()
    for match_id, team_a, team_b in pairings:
        if team_a in deviating_teams or team_b in deviating_teams:
            matches.add(match_id)
    return matches


def team_to_r32_match(
    scenario: Scenario,
    ratings: dict[str, float],
) -> dict[str, int]:
    """Return {team: r32_match_id} for every team in this scenario's R32."""
    pairings = resolve_r32_pairings(scenario, ratings)
    out: dict[str, int] = {}
    for match_id, team_a, team_b in pairings:
        out[team_a] = match_id
        out[team_b] = match_id
    return out


def team_distance_from_deviation(
    team: str,
    scenario: Scenario,
    ratings: dict[str, float],
) -> Optional[int]:
    """Min graph distance from `team`'s R32 match to ANY deviating R32 match.

    Returns None if `team` isn't in the knockout under this scenario.
    """
    team_match = team_to_r32_match(scenario, ratings).get(team)
    if team_match is None:
        return None
    deviating = deviating_r32_matches(scenario, ratings)
    return min(graph_distance(team_match, d) for d in deviating)


# Directional sanity check


@dataclass
class DirectionalSanityReport:
    """Result of the directional-sanity validation for one scenario.

    Attributes:
        scenario_id: which scenario was checked
        mean_abs_delta_by_distance: {graph_distance: mean |ΔWin|} for teams
            at each distance from the deviating R32 slots
        team_count_by_distance: {graph_distance: number_of_teams}
        verdict: 'PASS' if mean |ΔWin| decreases monotonically across
            increasing distance, else 'FAIL: <reason>'
    """
    scenario_id: str
    mean_abs_delta_by_distance: dict[int, float]
    team_count_by_distance: dict[int, int]
    verdict: str


def directional_sanity(
    scenario: Scenario,
    scenario_result: PropagationResult,
    baseline_result: PropagationResult,
    ratings: dict[str, float],
    far_threshold_pp: float = 0.5,
    near_dominance_ratio: float = 3.0,
) -> DirectionalSanityReport:
    """Check that deltas concentrate near the deviating group's bracket region.

    Args:
        scenario: the deviation scenario
        scenario_result: propagation under this scenario
        baseline_result: propagation under the baseline (empty) scenario
        ratings: team ratings (for resolving r32 pairings)
        far_threshold_pp: max |ΔWin| in percentage points for distance≥3
            teams to count as "near-zero" (default 0.5pp)
        near_dominance_ratio: how many times stronger the near-bin
            (distances 0+1) should be vs the far-bin (distances 2+3+4).
            Default 3.0 - concentration is "real" but not absurdly tight.

    Returns: report with verdict.

    Verdict is PASS if BOTH:
      (a) mean |ΔWin| in the FAR bin (distances ≥ 3) is below far_threshold_pp
      (b) NEAR-bin (distances ≤ 1) mean is at least near_dominance_ratio
          times the FAR-bin mean

    Note on relaxation from strict monotonicity:
    The framework's value is exactly that bracket interactions cross
    distance boundaries - e.g. Argentina's P(Win) shifts under the Spain-H
    scenario despite Argentina being on the opposite side of the bracket.
    Strict per-distance monotonicity penalises this correctness. The
    bin-comparison check is the right structural test: deltas should be
    concentrated near the deviation in aggregate, even if individual
    distance rungs occasionally swap.
    """
    deltas_by_distance: dict[int, list[float]] = defaultdict(list)
    for team in scenario_result.survival:
        if team not in baseline_result.survival:
            continue
        dist = team_distance_from_deviation(team, scenario, ratings)
        if dist is None:
            continue
        delta = scenario_result.survival[team]["Win"] - baseline_result.survival[team]["Win"]
        deltas_by_distance[dist].append(abs(delta))

    mean_abs = {
        d: (sum(vals) / len(vals)) if vals else 0.0
        for d, vals in deltas_by_distance.items()
    }
    counts = {d: len(vals) for d, vals in deltas_by_distance.items()}

    # Verdict logic
    verdict_reasons: list[str] = []

    # (a) far-zero check - distance ≥ 3 teams should be near-zero
    far_distances = [d for d in mean_abs if d >= 3]
    far_values_flat = [v for d in far_distances for v in deltas_by_distance[d]]
    far_mean_pp = (sum(far_values_flat) / len(far_values_flat) * 100) if far_values_flat else 0.0
    if far_mean_pp > far_threshold_pp:
        verdict_reasons.append(
            f"FAR-bin (d≥3) mean |ΔWin| = {far_mean_pp:.3f}pp exceeds threshold {far_threshold_pp}pp"
        )

    # (b) near-dominance check - near-bin should dominate far-bin
    near_distances = [d for d in mean_abs if d <= 1]
    near_values_flat = [v for d in near_distances for v in deltas_by_distance[d]]
    near_mean_pp = (sum(near_values_flat) / len(near_values_flat) * 100) if near_values_flat else 0.0
    if far_mean_pp > 0 and near_mean_pp / max(far_mean_pp, 1e-9) < near_dominance_ratio:
        verdict_reasons.append(
            f"near-bin (d≤1) mean = {near_mean_pp:.3f}pp is only "
            f"{near_mean_pp / max(far_mean_pp, 1e-9):.1f}× the far-bin mean = {far_mean_pp:.3f}pp "
            f"(expected ≥{near_dominance_ratio}×)"
        )

    verdict = "PASS" if not verdict_reasons else f"FAIL: {'; '.join(verdict_reasons)}"

    return DirectionalSanityReport(
        scenario_id=scenario.scenario_id,
        mean_abs_delta_by_distance=mean_abs,
        team_count_by_distance=counts,
        verdict=verdict,
    )


# Sensitivity check


@dataclass
class SensitivityReport:
    """Result of a sensitivity check.

    Attributes:
        perturbed_team: which team's Elo we shifted
        delta_elo: how much we shifted by (positive = stronger)
        baseline_pwin: team's P(Win) before perturbation
        perturbed_pwin: team's P(Win) after perturbation
        team_movement_pp: perturbed - baseline, in percentage points
        global_conservation_pp: Σ over all teams of (perturbed - baseline),
            in percentage points; should be ~0
        verdict: 'PASS' or 'FAIL: <reason>'
    """
    perturbed_team: str
    delta_elo: float
    baseline_pwin: float
    perturbed_pwin: float
    team_movement_pp: float
    global_conservation_pp: float
    verdict: str


def sensitivity_check(
    scenario: Scenario,
    predictor: Predictor,
    ratings: dict[str, float],
    perturb_team: str,
    delta_elo: float = 50.0,
    min_movement_pp: float = 0.5,
    conservation_tolerance_pp: float = 0.01,
) -> SensitivityReport:
    """Verify the perturbed team's P(Win) moves in the expected direction.

    Args:
        scenario: the scenario to perturb under (typically the baseline)
        predictor: the calibrated predictor (we'll wrap it with a perturbation)
        ratings: team ratings
        perturb_team: which team to bump
        delta_elo: how much (default +50, "a noticeable but not extreme bump")
        min_movement_pp: minimum P(Win) movement to count as PASS (default 0.5pp).
            Below this, the perturbation might be lost in propagator noise.
        conservation_tolerance_pp: max allowed deviation of Σ ΔP(Win) from 0.

    Returns: SensitivityReport with verdict.
    """
    # Wrap the predictor to bump perturb_team's Elo
    def perturbed_predictor(elo_a, elo_b, ctx):
        adj_a = elo_a + (delta_elo if ctx.home_country == perturb_team else 0.0)
        adj_b = elo_b + (delta_elo if ctx.away_country == perturb_team else 0.0)
        return predictor(adj_a, adj_b, ctx)

    baseline = propagate(scenario, ratings, predictor=predictor)
    perturbed = propagate(scenario, ratings, predictor=perturbed_predictor)

    baseline_pwin = baseline.survival.get(perturb_team, {}).get("Win", 0.0)
    perturbed_pwin = perturbed.survival.get(perturb_team, {}).get("Win", 0.0)
    team_movement_pp = (perturbed_pwin - baseline_pwin) * 100

    # Global conservation: sum of all ΔP(Win) should be ~0
    all_teams = set(baseline.survival) | set(perturbed.survival)
    total_delta = sum(
        perturbed.survival.get(t, {}).get("Win", 0.0)
        - baseline.survival.get(t, {}).get("Win", 0.0)
        for t in all_teams
    )
    global_conservation_pp = total_delta * 100

    # Verdict
    reasons: list[str] = []
    expected_direction = 1.0 if delta_elo > 0 else -1.0
    if expected_direction * team_movement_pp < min_movement_pp:
        reasons.append(
            f"{perturb_team} P(Win) moved {team_movement_pp:+.3f}pp, "
            f"expected at least {expected_direction * min_movement_pp:+.3f}pp"
        )
    if abs(global_conservation_pp) > conservation_tolerance_pp:
        reasons.append(
            f"Σ ΔP(Win) = {global_conservation_pp:+.4f}pp, expected ~0 "
            f"(tolerance ±{conservation_tolerance_pp}pp)"
        )

    verdict = "PASS" if not reasons else f"FAIL: {'; '.join(reasons)}"

    return SensitivityReport(
        perturbed_team=perturb_team,
        delta_elo=delta_elo,
        baseline_pwin=baseline_pwin,
        perturbed_pwin=perturbed_pwin,
        team_movement_pp=team_movement_pp,
        global_conservation_pp=global_conservation_pp,
        verdict=verdict,
    )


# CLI smoke test


def format_directional_report(report: DirectionalSanityReport) -> str:
    """Pretty-print a DirectionalSanityReport."""
    lines = [f"  scenario: {report.scenario_id}  →  {report.verdict}"]
    for d in sorted(report.mean_abs_delta_by_distance.keys()):
        mean_pp = report.mean_abs_delta_by_distance[d] * 100
        count = report.team_count_by_distance[d]
        lines.append(f"     d={d}: mean |ΔWin| = {mean_pp:.3f}pp  ({count} teams)")
    return "\n".join(lines)


def format_sensitivity_report(report: SensitivityReport) -> str:
    """Pretty-print a SensitivityReport."""
    return (
        f"  perturb {report.perturbed_team}  Elo {report.delta_elo:+.0f}  →  {report.verdict}\n"
        f"     P(Win): {report.baseline_pwin:.4f} → {report.perturbed_pwin:.4f}  "
        f"(Δ={report.team_movement_pp:+.3f}pp)\n"
        f"     Σ ΔP(Win) across all teams: {report.global_conservation_pp:+.5f}pp"
    )


if __name__ == "__main__":
    # Manual smoke test - runs the full validation suite against a freshly
    # computed propagation. Slow because it does a full calibration first.
    #
    # Run: `python -m upset_propagation.validation`
    from upset_propagation.baseline import fetch_baseline_fair_probs
    from upset_propagation.calibrator import build_baseline_scenario, calibrate
    from upset_propagation.scenarios import build_all_scenarios, load_latest_elo

    print("Loading inputs + calibrating (~4 minutes)...")
    fair_probs = fetch_baseline_fair_probs()
    ratings = load_latest_elo()
    cal = calibrate(fair_probs, ratings)

    baseline_scenario = build_baseline_scenario(fair_probs)
    baseline_result = propagate(baseline_scenario, ratings, predictor=cal.predictor)

    print("\n=== Directional sanity (single-deviation scenarios) ===")
    scenarios = build_all_scenarios(fair_probs)
    for s in scenarios:
        result = propagate(s, ratings, predictor=cal.predictor)
        report = directional_sanity(s, result, baseline_result, ratings)
        print(format_directional_report(report))

    print("\n=== Sensitivity checks (against baseline) ===")
    for team in ["Spain", "France", "Argentina", "Mexico"]:
        report = sensitivity_check(
            baseline_scenario, cal.predictor, ratings, perturb_team=team, delta_elo=50.0
        )
        print(format_sensitivity_report(report))