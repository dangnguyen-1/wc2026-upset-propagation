"""Calibrator - fit per-team Elo offsets so the propagator matches the FairLine model's baseline.

The framework's invariant: when the propagator is run with the baseline
(empty) scenario, the output must reproduce the FairLine model's market view. Out of the
box this won't be true - the vendored Elo-based predictor has its own view
that disagrees with the sportsbook market.

The calibrator solves the inverse problem:

    θ̂ = argmin_θ  Σ_i (p_prop_Win(i | θ) − p_base(i))²

    subject to Σ_i θ_i = 0    (zero-sum identification constraint)

where:
    p_prop_Win(i | θ) = propagator's tournament-winner prob for team i
                       under the BASELINE scenario, with team i's Elo
                       shifted by θ_i
    p_base(i)         = the FairLine model's market view from the FairLine API

Solver: Nelder-Mead (gradient-free, no Jacobian needed). Converges
in 200-500 function evaluations on a 32-dim problem.

The output is a `CalibratedPredictor` callable that wraps the vendored
default predictor with the fitted Elo offsets applied. The propagator
treats it like any other predictor.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from upset_propagation._vendored.simulator import Predictor, make_elo_predictor
from upset_propagation._vendored.single_game import MatchContext, MatchPrediction, ModelParams
from upset_propagation.config import CALIBRATION_TOLERANCE


logger = logging.getLogger(__name__)
from upset_propagation.propagator import propagate
from upset_propagation.scenarios import (
    GROUPS_TO_API_NAME,
    Scenario,
    build_baseline_standings,
    load_groups,
    resolve_elo_for_wc_team,
    resolve_fair_prob_for_group_team,
)


# Baseline-as-scenario (used as the calibration target)


def build_baseline_scenario(
    fair_probs: dict[str, float],
    groups: Optional[dict[str, list[str]]] = None,
) -> Scenario:
    """The 'empty' scenario - every group resolves with its fair_prob favourite.

    This is the reference point for calibration: when we run the propagator
    on this scenario, the output's `Win` column should match the FairLine model's baseline
    (after calibration converges).

    Reuses the same standings shape as the 12 deviation scenarios so the
    propagator code path is identical.
    """
    if groups is None:
        groups = load_groups()
    standings = build_baseline_standings(groups, fair_probs)
    return Scenario(
        scenario_id="baseline",
        description="Seeded baseline - every group's fair_prob favourite wins.",
        deviating_group="",
        favourite="",
        upset_winner="",
        standings=standings,
    )


# Baseline filtering + renormalization


# Six playoff losers from March 31 2026 are still in the FairLine API with
# near-zero residual fair_probs. They didn't qualify; the calibrator must
# exclude them. See conversation history for derivation.
_PLAYOFF_LOSERS = frozenset({
    "Denmark", "Italy", "Kosovo", "Poland", "Bolivia", "Jamaica",
})


def filter_and_renormalize(
    fair_probs: dict[str, float],
    qualified_teams_in_api_names: set[str],
) -> dict[str, float]:
    """Drop non-qualified teams and renormalize remaining probs to sum to 1.

    Drops:
      - any team whose API name isn't in `qualified_teams_in_api_names`
      - the six known playoff losers (explicit set, in case the team-name
        intersection misses one due to a spelling drift)

    Then renormalizes the remaining 48 entries to sum to exactly 1.0.
    """
    filtered = {
        team: prob
        for team, prob in fair_probs.items()
        if team in qualified_teams_in_api_names
        and team not in _PLAYOFF_LOSERS
    }
    total = sum(filtered.values())
    if total <= 0:
        raise ValueError(
            f"Sum of filtered fair_probs is {total} (no positive mass); "
            f"check team-name alignment between groups.json and API."
        )
    return {team: prob / total for team, prob in filtered.items()}


def build_calibration_targets(
    fair_probs: dict[str, float],
    groups: dict[str, list[str]],
) -> dict[str, float]:
    """Return {api_team_name: target_fair_prob} for the 48 qualified teams.

    The keys are in the API's spelling (e.g. "Czech Republic" not "Czechia")
    so we can directly compare against the propagator's output, which uses
    groups.json spelling - we apply the reverse alias before comparison.
    """
    # Build set of qualified teams in API spelling
    qualified_in_api = set()
    for letter, teams in groups.items():
        for team in teams:
            qualified_in_api.add(GROUPS_TO_API_NAME.get(team, team))
    return filter_and_renormalize(fair_probs, qualified_in_api)


# The offset-adjusted predictor


def make_offset_predictor(
    base_predictor: Predictor,
    offsets: dict[str, float],
) -> Predictor:
    """Wrap a predictor with per-team Elo offsets.

    Returns a new callable matching the Predictor signature
    `(elo_a, elo_b, ctx) -> MatchPrediction` that shifts each team's Elo by
    its offset before delegating to the base predictor.

    Implementation: we modify the Elo inputs at the boundary. The ctx tells
    us which country is home/away, so we look up offsets by country name.
    """
    def predictor(elo_a: float, elo_b: float, ctx: MatchContext) -> MatchPrediction:
        delta_a = offsets.get(ctx.home_country, 0.0)
        delta_b = offsets.get(ctx.away_country, 0.0)
        return base_predictor(elo_a + delta_a, elo_b + delta_b, ctx)
    return predictor


# Calibration


@dataclass
class CalibrationResult:
    """Outcome of a calibration run.

    Attributes:
        offsets: {team_name (groups.json spelling): elo_offset_delta}
        predictor: the calibrated predictor, ready to feed to propagator
        final_loss: SSE between propagated and target probs
        max_residual: max |propagated_i - target_i| in probability units
        n_iterations: number of optimizer function evaluations
        elapsed_sec: wall time for the fit
    """
    offsets: dict[str, float]
    predictor: Predictor
    final_loss: float
    max_residual: float
    n_iterations: int
    elapsed_sec: float


def _team_to_api_name(team: str) -> str:
    """Convert from groups.json spelling to API spelling (for residual lookup)."""
    return GROUPS_TO_API_NAME.get(team, team)


def calibrate(
    fair_probs: dict[str, float],
    ratings: dict[str, float],
    max_iter: int = 3000,
    initial_step: float = 25.0,
    verbose: bool = False,
) -> CalibrationResult:
    """Fit per-team Elo offsets to match the FairLine model's baseline.

    Args:
        fair_probs: raw API output from `baseline.fetch_baseline_fair_probs()`
            (unfiltered - calibrator handles the filtering)
        ratings: {team: latest_elo} from `scenarios.load_latest_elo()`
        max_iter: Nelder-Mead iteration cap. Default 3000 - Nelder-Mead is
            slow to converge in ~48 dims, and we have headroom (~70s per
            1000 iters), so generous is fine. The bracket is small enough
            that even 5000 iters takes <6 minutes.
        initial_step: initial simplex size in Elo points. 25 is roughly "the
            difference between two adjacent FIFA-ranked teams."
        verbose: print progress every ~50 iterations if True

    Returns: CalibrationResult.
    """
    groups = load_groups()
    targets = build_calibration_targets(fair_probs, groups)

    # The free parameters: one offset per knockout participant. Under the
    # baseline scenario, the 32 knockout teams are the top 2 from each group
    # plus the 8 best 3rd-placers. We free-fit all 48 group teams - the
    # propagator will only use the 32 actual KO teams, but freeing all 48
    # makes the parameter vector size constant and dead-simple.
    baseline_scenario = build_baseline_scenario(fair_probs, groups)
    all_teams_in_order = sorted({t for g in groups.values() for t in g})
    team_to_idx = {team: i for i, team in enumerate(all_teams_in_order)}
    n_params = len(all_teams_in_order)  # 48

    base_predictor = make_elo_predictor(ModelParams())

    n_evals = [0]   # boxed counter for the closure

    def loss(theta: np.ndarray) -> float:
        n_evals[0] += 1
        # Enforce zero-sum constraint
        theta_zs = theta - theta.mean()
        offsets = {team: float(theta_zs[i]) for team, i in team_to_idx.items()}

        # Build offset-adjusted predictor and propagate
        pred = make_offset_predictor(base_predictor, offsets)
        result = propagate(baseline_scenario, ratings, predictor=pred)

        # Compute squared error vs targets
        sse = 0.0
        for team in all_teams_in_order:
            api_name = _team_to_api_name(team)
            target = targets.get(api_name, 0.0)
            propagated = result.survival.get(team, {}).get("Win", 0.0)
            sse += (propagated - target) ** 2

        if verbose and n_evals[0] % 50 == 0:
            logger.info(f"  iter {n_evals[0]:4d}: loss = {sse:.6e}")
        return sse

    # Initial guess: all zeros (no shift). Nelder-Mead's initial simplex is
    # built around this with edge length controlled by `initial_simplex`.
    x0 = np.zeros(n_params)
    # Build a non-degenerate initial simplex manually for better behaviour
    # than the default tiny-perturbation one (Nelder-Mead's default uses
    # 0.05*x0 which is zero everywhere here).
    initial_simplex = np.vstack([x0, np.eye(n_params) * initial_step + x0])

    if verbose:
        logger.info(f"Calibrating {n_params} team-strength offsets (Nelder-Mead)...")
    t0 = time.time()
    result = minimize(
        loss,
        x0,
        method="Nelder-Mead",
        options={
            "initial_simplex": initial_simplex,
            "maxiter": max_iter,
            "xatol": 1e-3,     # convergence: Elo offsets stable to 1 mE-point
            "fatol": 1e-7,     # convergence: loss stable to ~10 bps² total
            "adaptive": True,   # Nelder-Mead adaptive variant - better for n_dim > 5
            "disp": verbose,
        },
    )
    elapsed = time.time() - t0

    # Final residuals (using the converged offsets, post zero-sum)
    theta_final = result.x - result.x.mean()
    offsets_final = {team: float(theta_final[i]) for team, i in team_to_idx.items()}
    pred_final = make_offset_predictor(base_predictor, offsets_final)
    propagation_final = propagate(baseline_scenario, ratings, predictor=pred_final)

    max_resid = 0.0
    for team in all_teams_in_order:
        api_name = _team_to_api_name(team)
        target = targets.get(api_name, 0.0)
        propagated = propagation_final.survival.get(team, {}).get("Win", 0.0)
        max_resid = max(max_resid, abs(propagated - target))

    return CalibrationResult(
        offsets=offsets_final,
        predictor=pred_final,
        final_loss=float(result.fun),
        max_residual=max_resid,
        n_iterations=n_evals[0],
        elapsed_sec=elapsed,
    )


# CLI / smoke test


if __name__ == "__main__":
    from upset_propagation.baseline import fetch_baseline_fair_probs
    from upset_propagation.scenarios import load_latest_elo

    print("Loading inputs...")
    fair_probs = fetch_baseline_fair_probs()
    ratings = load_latest_elo()

    print("Running calibration (this takes a few minutes)...\n")
    cal = calibrate(fair_probs, ratings, verbose=True)

    print(f"\nDone in {cal.elapsed_sec:.1f}s, {cal.n_iterations} evaluations")
    print(f"Final SSE: {cal.final_loss:.6e}")
    print(f"Max residual: {cal.max_residual:.4f} (target: < {CALIBRATION_TOLERANCE})")

    if cal.max_residual < CALIBRATION_TOLERANCE:
        print("PASS - calibration converged within tolerance")
    else:
        print("WARN - calibration did not converge; check max_iter or residual breakdown")

    # Show the 10 biggest offsets, both directions
    print("\nLargest positive offsets (model under-rated these teams):")
    by_offset = sorted(cal.offsets.items(), key=lambda kv: -kv[1])
    for team, offset in by_offset[:10]:
        print(f"  {team:32s}  {offset:+7.1f}")
    print("\nLargest negative offsets (model over-rated these teams):")
    for team, offset in by_offset[-10:]:
        print(f"  {team:32s}  {offset:+7.1f}")