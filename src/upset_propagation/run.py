"""End-to-end orchestrator - produces the 12 scenario propagation tables.

Workflow:
    1. Fetch the FairLine model's baseline fair_probs from the FairLine API.
    2. Load latest Elo ratings.
    3. Build the 12 single-deviation scenarios.
    4. Calibrate per-team Elo offsets so the propagator's empty-scenario
       output matches the FairLine model's baseline (~4 min).
    5. Propagate all 13 scenarios (12 deviations + 1 baseline) using the
       calibrated predictor.
    6. Write one JSON file per scenario to output/ plus a summary index.

The Δ-from-baseline is computed at write time - it's the actually-useful
trading signal. P(team reaches round | deviation) − P(team reaches round
| baseline). Positive means the scenario is *good* for that team.

Usage:
    python -m upset_propagation.run
    python -m upset_propagation.run --max-iter 5000   # tighter fit
    python -m upset_propagation.run --quiet           # suppress per-iter logging
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from upset_propagation._vendored.simulator import Predictor
from upset_propagation.baseline import (
    fetch_baseline_fair_probs,
    fetch_baseline_with_snapshot,
    load_baseline_from_snapshot,
)
from upset_propagation.calibrator import (
    CalibrationResult,
    build_baseline_scenario,
    calibrate,
)
from upset_propagation.config import (
    CALIBRATION_TOLERANCE,
    KNOCKOUT_ROUNDS,
    OUTPUT_DIR,
)
from upset_propagation.cron_utils import (
    LOCK_FILENAME,
    LockBusyError,
    atomic_output_dir,
    force_unlock,
    lockfile_acquired,
    write_health,
)
from upset_propagation.logging_config import (
    configure_cron_logging,
    configure_interactive_logging,
)
from upset_propagation.propagator import PropagationResult, propagate
from upset_propagation.scenarios import (
    Scenario,
    build_all_pairwise_scenarios,
    build_all_scenarios,
    load_groups,
    load_latest_elo,
)
from upset_propagation.validation import (
    DirectionalSanityReport,
    SensitivityReport,
    directional_sanity,
    sensitivity_check,
)


logger = logging.getLogger(__name__)


# Composing scenario outputs


def survival_table_to_dict(result: PropagationResult) -> dict[str, dict[str, float]]:
    """Convert PropagationResult.survival into a JSON-friendly nested dict.

    Keys ordered by P(Win) descending so the JSON is human-readable at the top.
    """
    ranked = sorted(
        result.survival.items(),
        key=lambda kv: -kv[1]["Win"],
    )
    return {
        team: {r: round(probs[r], 6) for r in KNOCKOUT_ROUNDS}
        for team, probs in ranked
    }


def compute_deltas(
    scenario_table: dict[str, dict[str, float]],
    baseline_table: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """For every team in either table, compute Δfair_prob_per_round.

    Δ = scenario − baseline. Positive = team benefits from the deviation.
    Teams present in one table but not the other get 0 for the missing side.
    """
    teams = set(scenario_table) | set(baseline_table)
    deltas: dict[str, dict[str, float]] = {}
    for team in teams:
        scen = scenario_table.get(team, {r: 0.0 for r in KNOCKOUT_ROUNDS})
        base = baseline_table.get(team, {r: 0.0 for r in KNOCKOUT_ROUNDS})
        deltas[team] = {r: round(scen[r] - base[r], 6) for r in KNOCKOUT_ROUNDS}
    # Sort by absolute change in P(Win) descending - biggest movers first
    return dict(sorted(deltas.items(), key=lambda kv: -abs(kv[1]["Win"])))


def write_scenario_file(
    scenario: Scenario,
    result: PropagationResult,
    baseline_table: dict[str, dict[str, float]],
    output_dir: Path,
    computed_at: str,
) -> Path:
    """Write one scenario JSON file. Returns the path written."""
    scenario_table = survival_table_to_dict(result)
    deltas = compute_deltas(scenario_table, baseline_table)
    payload = {
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "deviating_group": scenario.deviating_group,
        "favourite": scenario.favourite,
        "upset_winner": scenario.upset_winner,
        "computed_at": computed_at,
        "standings": scenario.standings,
        "survival": scenario_table,
        "delta_from_baseline": deltas,
    }
    path = output_dir / f"{scenario.scenario_id}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_baseline_file(
    baseline_scenario: Scenario,
    result: PropagationResult,
    output_dir: Path,
    computed_at: str,
) -> tuple[Path, dict[str, dict[str, float]]]:
    """Write the baseline (empty-scenario) propagation table.

    Returns (path, baseline_table_dict). The table is reused by every
    deviation scenario's Δ computation.
    """
    table = survival_table_to_dict(result)
    payload = {
        "scenario_id": "baseline",
        "description": baseline_scenario.description,
        "computed_at": computed_at,
        "standings": baseline_scenario.standings,
        "survival": table,
    }
    path = output_dir / "baseline.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path, table


def write_index(
    scenario_paths: list[Path],
    baseline_path: Path,
    calibration: CalibrationResult,
    computed_at: str,
    output_dir: Path,
) -> Path:
    """Write index.json - directory of all outputs + calibration diagnostics."""
    payload = {
        "computed_at": computed_at,
        "calibration": {
            "final_loss": calibration.final_loss,
            "max_residual": calibration.max_residual,
            "n_iterations": calibration.n_iterations,
            "elapsed_sec": round(calibration.elapsed_sec, 2),
            "within_tolerance": calibration.max_residual < CALIBRATION_TOLERANCE,
            "tolerance": CALIBRATION_TOLERANCE,
            "offsets": {team: round(v, 2) for team, v in calibration.offsets.items()},
        },
        "baseline_file": baseline_path.name,
        "scenario_files": [p.name for p in scenario_paths],
    }
    path = output_dir / "index.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_validation_report(
    sanity_reports: list[DirectionalSanityReport],
    sensitivity_reports: list[SensitivityReport],
    computed_at: str,
    output_dir: Path,
) -> Path:
    """Write validation_report.json - per-scenario sanity + sensitivity verdicts.

    Bundles all validation results into a single JSON to scan and see if
    the run is geometrically sane. Each verdict is a string ('PASS' or
    'FAIL: <reason>') so callers can detect failures by simple string match.
    """
    payload = {
        "computed_at": computed_at,
        "summary": {
            "directional_sanity_scenarios_checked": len(sanity_reports),
            "directional_sanity_pass_count": sum(
                1 for r in sanity_reports if r.verdict == "PASS"
            ),
            "sensitivity_checks_performed": len(sensitivity_reports),
            "sensitivity_pass_count": sum(
                1 for r in sensitivity_reports if r.verdict == "PASS"
            ),
            "overall_pass": (
                all(r.verdict == "PASS" for r in sanity_reports)
                and all(r.verdict == "PASS" for r in sensitivity_reports)
            ),
        },
        "directional_sanity": [
            {
                "scenario_id": r.scenario_id,
                "verdict": r.verdict,
                "mean_abs_delta_by_distance_pp": {
                    str(d): round(v * 100, 4)
                    for d, v in r.mean_abs_delta_by_distance.items()
                },
                "team_count_by_distance": {str(d): c for d, c in r.team_count_by_distance.items()},
            }
            for r in sanity_reports
        ],
        "sensitivity": [
            {
                "perturbed_team": r.perturbed_team,
                "delta_elo": r.delta_elo,
                "verdict": r.verdict,
                "baseline_pwin": round(r.baseline_pwin, 6),
                "perturbed_pwin": round(r.perturbed_pwin, 6),
                "team_movement_pp": round(r.team_movement_pp, 4),
                "global_conservation_pp": round(r.global_conservation_pp, 6),
            }
            for r in sensitivity_reports
        ],
    }
    path = output_dir / "validation_report.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# The orchestrator


def _build_run_directory_name(computed_at: str) -> str:
    """Convert an ISO-8601 timestamp into a filesystem-safe directory name.

    Colons in ISO timestamps break on Windows/some filesystems, so we
    replace them with dashes. Drops timezone info ('+00:00' tail) for
    readability since runs are all in UTC by convention.

    Example: '2026-06-08T05:20:12+00:00' -> '2026-06-08T05-20-12Z'
    """
    # Strip timezone tail and split date/time
    base = computed_at.split("+")[0].split("Z")[0]
    return base.replace(":", "-") + "Z"


def _snapshot_outputs_to_runs_dir(
    output_dir: Path,
    computed_at: str,
    verbose: bool = True,
) -> Path:
    """Copy the current contents of output/ into output/runs/<timestamp>/.

    Provides a historical audit trail for comparing today's outputs
    against yesterday's by inspecting different snapshots. The working set
    in output/ remains overwriteable by future runs; only the snapshots
    in output/runs/ persist.

    Returns the path of the snapshot directory just created.
    """
    import shutil

    runs_root = output_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    snapshot_dir = runs_root / _build_run_directory_name(computed_at)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Copy every JSON file from output/ (the working set) into the snapshot.
    # Skip the runs/ subdirectory itself so we don't recurse into ourselves.
    n_copied = 0
    for src in output_dir.glob("*.json"):
        shutil.copy2(src, snapshot_dir / src.name)
        n_copied += 1
    logger.info(f"      Snapshot: {n_copied} files saved to {snapshot_dir.relative_to(output_dir.parent)}/")
    return snapshot_dir


def run(
    max_iter: int = 3000,
    output_dir: Optional[Path] = None,
    verbose: bool = True,
    snapshot: bool = True,
    from_snapshot: Optional[Path] = None,
) -> dict[str, Path]:
    """End-to-end run: fetch → calibrate → propagate all scenarios → write JSONs.

    Args:
        max_iter: Nelder-Mead iteration cap
        output_dir: where to write outputs (default: ./output/)
        verbose: print per-iteration logging
        snapshot: if True, also copy the run's outputs into
            output/runs/<timestamp>/ for historical audit, and save the
            raw API response as api_snapshot.json alongside. Default True.
        from_snapshot: if given, read fair_probs from this JSON file
            instead of hitting the FairLine API. Used to reproduce a past
            run exactly. When set, an outgoing snapshot is NOT recorded
            (it would just be a copy of the input).

    Returns: {scenario_id: path_written} including 'baseline' and 'index'.
        Paths point at the working set in output/, not the snapshot copy.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # If no logging has been configured (i.e. this run() was called from
    # tests or imported as a library), fall back to interactive logging
    # so the user/test sees the progress output. Idempotent - main() will
    # have already configured logging before calling run() in the CLI path.
    if not logging.getLogger().handlers:
        configure_interactive_logging(quiet=not verbose)

    computed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # 1. Inputs
    logger.info("=" * 70)
    logger.info(f"WC2026 Bracket Scenarios - run at {computed_at}")
    logger.info("=" * 70)

    # Stage the API snapshot path BEFORE fetching, so we can write the raw
    # response into the versioned snapshot directory directly. When
    # snapshot=False, we still fetch but don't save the raw response.
    api_snapshot_path: Optional[Path] = None
    if snapshot and from_snapshot is None:
        snapshot_dir_name = _build_run_directory_name(computed_at)
        api_snapshot_path = output_dir / "runs" / snapshot_dir_name / "api_snapshot.json"

    if from_snapshot is not None:
        logger.info(f"\n[1/4] Loading baseline fair_probs from snapshot: {from_snapshot}")
        fair_probs = load_baseline_from_snapshot(from_snapshot)
    elif api_snapshot_path is not None:
        logger.info(f"\n[1/4] Fetching baseline fair_probs from FairLine API "
              f"(+ saving snapshot)...")
        fair_probs = fetch_baseline_with_snapshot(api_snapshot_path)
    else:
        logger.info("\n[1/4] Fetching baseline fair_probs from FairLine API...")
        fair_probs = fetch_baseline_fair_probs()

    if verbose:
        n = len(fair_probs)
        total = sum(fair_probs.values())
        logger.info(f"      {n} teams retrieved (Σ={total:.4f})")

    logger.info("\n[2/4] Loading Elo ratings from elo_history.csv...")
    ratings = load_latest_elo()
    logger.info(f"      {len(ratings)} teams with Elo history")

    # Input validation
    # Catches bad inputs BEFORE the ~4-minute calibration. WARN-level issues
    # log but don't kill the run; FAIL-level issues raise InputValidationError
    # which propagates to cron-mode's failure-health emission.
    #
    # Import deferred to keep core dependencies minimal - input_validation
    # pulls in pandas which would slow `python -m upset_propagation` startup.
    from upset_propagation.input_validation import (
        assert_validation_passes,
        validate_elo_history,
        validate_fair_probs,
    )
    logger.info("\n      Validating inputs...")
    fp_report = validate_fair_probs(fair_probs)
    logger.info(
        f"      fair_probs: {fp_report.verdict} "
        f"({len(fp_report.issues)} issues, Σ={fp_report.diagnostics.get('fair_probs_sum', 0):.4f})"
    )
    assert_validation_passes(fp_report)
    elo_report = validate_elo_history(ratings)
    logger.info(
        f"      elo_history: {elo_report.verdict} "
        f"({len(elo_report.issues)} issues, range=[{elo_report.diagnostics.get('elo_min', 0):.0f}, "
        f"{elo_report.diagnostics.get('elo_max', 0):.0f}])"
    )
    assert_validation_passes(elo_report)

    logger.info("\n[3/4] Calibrating predictor (Nelder-Mead, may take a few minutes)...")
    cal = calibrate(fair_probs, ratings, max_iter=max_iter, verbose=verbose)
    if verbose:
        within = "PASS" if cal.max_residual < CALIBRATION_TOLERANCE else "WARN"
        logger.info(
            f"      Done in {cal.elapsed_sec:.1f}s, {cal.n_iterations} evaluations. "
            f"max_residual={cal.max_residual:.4f} [{within}]"
        )

    # 2. Build scenarios + baseline
    logger.info("\n[4/4] Propagating baseline + 12 single + 66 pairwise scenarios...")
    scenarios = build_all_scenarios(fair_probs)
    pairwise_scenarios = build_all_pairwise_scenarios(fair_probs)
    baseline_scenario = build_baseline_scenario(fair_probs)

    # 3. Propagate baseline first (deltas need it)
    baseline_result = propagate(baseline_scenario, ratings, predictor=cal.predictor)
    baseline_path, baseline_table = write_baseline_file(
        baseline_scenario, baseline_result, output_dir, computed_at
    )
    if verbose:
        top_team, top_probs = next(iter(survival_table_to_dict(baseline_result).items()))
        logger.info(f"      baseline → {baseline_path.name} (top: {top_team} {top_probs['Win']:.4f})")

    # 4. Propagate each single-deviation scenario
    # We retain the PropagationResult per scenario in memory (alongside the
    # scenario itself) so the validation step below can run directional
    # sanity without re-propagating.
    scenario_paths: list[Path] = []
    single_results: list[tuple[Scenario, PropagationResult]] = []
    for s in scenarios:
        t0 = time.time()
        result = propagate(s, ratings, predictor=cal.predictor)
        path = write_scenario_file(s, result, baseline_table, output_dir, computed_at)
        scenario_paths.append(path)
        single_results.append((s, result))
        if verbose:
            table = survival_table_to_dict(result)
            top_team, top_probs = next(iter(table.items()))
            elapsed = time.time() - t0
            logger.info(
                f"      [{s.deviating_group}] {s.scenario_id:35s} "
                f"top: {top_team} {top_probs['Win']:.4f} "
                f"({elapsed:.2f}s)"
            )

    # 5. Propagate each pairwise compound scenario
    # Reuses the same calibrated predictor as the single-deviation loop -
    # calibration depends on the BASELINE empty-scenario matching the FairLine model's
    # market, not on the deviation type, so one calibration covers all
    # 78 scenarios. Output naming matches the v1 convention; the index
    # file lists all scenarios together.
    pairwise_paths: list[Path] = []
    logger.info(f"\n      Pairwise scenarios ({len(pairwise_scenarios)}):")
    pairwise_start = time.time()
    for s in pairwise_scenarios:
        result = propagate(s, ratings, predictor=cal.predictor)
        path = write_scenario_file(s, result, baseline_table, output_dir, computed_at)
        pairwise_paths.append(path)
    if verbose:
        pairwise_elapsed = time.time() - pairwise_start
        logger.info(
            f"      Done - {len(pairwise_paths)} pairwise scenarios written "
            f"({pairwise_elapsed:.1f}s, ~{pairwise_elapsed/len(pairwise_paths)*1000:.0f}ms each)"
        )

    # All scenario paths (single + pairwise) for the index file
    all_scenario_paths = scenario_paths + pairwise_paths

    # 6. Index file
    index_path = write_index(all_scenario_paths, baseline_path, cal, computed_at, output_dir)
    logger.info(
        f"\nWrote {len(all_scenario_paths)} scenarios "
        f"({len(scenario_paths)} single + {len(pairwise_paths)} pairwise) "
        f"+ baseline + index to {output_dir}/"
    )

    # 7. Auto-validation: directional sanity + sensitivity
    # Bundles the validation suite into the standard run so every output
    # ships with a pass/fail report; scan validation_report.json
    # right after the run and know whether to trust the outputs.
    #
    # Scope: directional sanity on all 12 single scenarios (the geometry-
    # carrying ones - pairwise inherits from the singles compositionally),
    # plus sensitivity on 4 top-favourite teams. Total wall time ~3s.
    logger.info("\n[5/5] Running validation suite (directional sanity + sensitivity)...")
    val_t0 = time.time()
    sanity_reports = [
        directional_sanity(s, result, baseline_result, ratings)
        for s, result in single_results
    ]
    sensitivity_reports = [
        sensitivity_check(
            baseline_scenario, cal.predictor, ratings,
            perturb_team=team, delta_elo=50.0,
        )
        for team in ["Spain", "France", "Argentina", "Mexico"]
    ]
    val_report_path = write_validation_report(
        sanity_reports, sensitivity_reports, computed_at, output_dir
    )
    n_sanity_pass = sum(1 for r in sanity_reports if r.verdict == "PASS")
    n_sens_pass = sum(1 for r in sensitivity_reports if r.verdict == "PASS")
    val_elapsed = time.time() - val_t0
    if verbose:
        logger.info(
            f"      Directional sanity: {n_sanity_pass}/{len(sanity_reports)} PASS  |  "
            f"Sensitivity: {n_sens_pass}/{len(sensitivity_reports)} PASS  "
            f"({val_elapsed:.1f}s)"
        )
        if n_sanity_pass < len(sanity_reports) or n_sens_pass < len(sensitivity_reports):
            logger.info("      ⚠  See validation_report.json for failure details")

    # 7. Real-time top-10 scenario ranking
    # Reads match results from output/match_results.csv if present (via
    # state_from_matches_csv), otherwise uses an empty state. Writes the
    # result to output/top_10_ranking.json.
    #
    # Imports are deferred to avoid cycles: top_ranking imports
    # ensemble_matcher → l1_matcher → ... which all exist by now.
    state = None
    top_ranking_path = None
    try:
        from upset_propagation.top_ranking import (
            compute_top_ranking,
            load_state_for_cron,
            write_top_ranking_file,
        )
        state = load_state_for_cron(output_dir, ratings)
        top_ranking = compute_top_ranking(
            state, cal.predictor, ratings, fair_probs,
            output_dir=output_dir,
        )
        top_ranking_path = write_top_ranking_file(top_ranking, output_dir)
    except Exception as exc:
        # Failure here is non-fatal - the scenarios + validation succeeded,
        # we don't want to scuttle the whole run because the ranking step
        # hit a problem. Log loudly so ops sees it.
        logger.warning(f"Failed to write top-10 ranking: {exc}", exc_info=True)

    # 8. Market snapshot logging
    # Appends one entry to output/market_log.jsonl with top-15 market
    # fair_probs, diff from previous entry, and implied-vs-market edges.
    # Complementary to api_snapshot.json (which is per-run raw data for
    # reproducibility): market_log is a cross-run historical record for
    # trade research.
    # 8a. Market log + implied-vs-market signal
    # Appends one entry to output/market_log.jsonl with top-15 market
    # prices, diff from previous entry, and implied-vs-market edges.
    # Complementary to api_snapshot.json (which is per-run raw data for
    # reproducibility): market_log is a cross-run historical record for
    # trade research.
    #
    # Two probability inputs:
    #   market_probs - Polymarket-derived view, what we compare against
    #   fair_probs   - the FairLine model's calibrated view, what the matcher uses internally
    # See market_log._compute_edges docstring for why we need both.
    #
    # market_probs is always fetched live (NOT snapshotted with the API
    # response). Snapshot-mode runs reproduce fair_probs exactly but not
    # market_probs - Polymarket prices change continuously and historical
    # reproducibility there is out of scope for v3.
    market_log_path = None
    try:
        from upset_propagation.baseline import fetch_market_prices
        from upset_propagation.market_log import (
            append_market_log,
            compute_market_log_entry,
        )
        from upset_propagation.state_matcher import RealisedState

        groups = load_groups()
        canonical_teams = {t for teams in groups.values() for t in teams}
        try:
            market_probs = fetch_market_prices(canonical_teams=canonical_teams)
            logger.info(
                f"      Fetched {len(market_probs)} market prices "
                f"(Σ={sum(market_probs.values()):.4f} after devig)"
            )
        except Exception as exc:
            logger.warning(
                f"Failed to fetch market prices; skipping market_log this run: "
                f"{exc}",
                exc_info=True,
            )
            market_probs = None

        if market_probs is not None:
            market_state = state if state is not None else RealisedState(standings={})
            entry = compute_market_log_entry(
                market_probs, fair_probs,
                market_state, cal.predictor, ratings, output_dir,
            )
            market_log_path = append_market_log(entry, output_dir)
    except Exception as exc:
        logger.warning(f"Failed to append market_log: {exc}", exc_info=True)

    # 8b. Analysis snapshot view
    # our_vs_market.json - derived from the just-appended market_log entry.
    # Pure projection of existing data, no new compute.
    our_vs_market_path = None
    try:
        from upset_propagation.our_vs_market import (
            build_snapshot as build_our_vs_market_snapshot,
            write_snapshot_file as write_our_vs_market_file,
        )
        our_vs_market_path = write_our_vs_market_file(
            build_our_vs_market_snapshot(output_dir), output_dir
        )
    except Exception as exc:
        logger.warning(f"Failed to write analysis snapshot: {exc}", exc_info=True)

    # 8c. Knockout stage - out of scope for v3
    # The 79-scenario library + calibration are built on group-stage
    # outcomes. Knockout-stage trading requires a different model
    # (different inputs, different calibration target). That's a separate
    # build, not a v3 feature.

    # 9. Snapshot historical copy
    # The working set in output/ overwrites on each run; snapshots in
    # output/runs/<timestamp>/ persist forever for historical comparison.
    if snapshot:
        _snapshot_outputs_to_runs_dir(output_dir, computed_at, verbose=verbose)

    return {
        **{p.stem: p for p in all_scenario_paths},
        "baseline": baseline_path,
        "index": index_path,
        "validation_report": val_report_path,
        **({"top_10_ranking": top_ranking_path} if top_ranking_path else {}),
        **({"market_log": market_log_path} if market_log_path else {}),
        **({"our_vs_market": our_vs_market_path} if our_vs_market_path else {}),
    }


# CLI


def _output_is_fresh(output_dir: Path, freshness_hours: float) -> tuple[bool, Optional[str]]:
    """Check whether the output directory has fresh results.

    "Fresh" = index.json exists and was modified within `freshness_hours`
    of now. Returns (is_fresh, human_readable_reason).
    """
    index_path = output_dir / "index.json"
    if not index_path.exists():
        return False, f"{index_path} does not exist"
    age_seconds = time.time() - index_path.stat().st_mtime
    age_hours = age_seconds / 3600
    if age_hours <= freshness_hours:
        return True, f"index.json is {age_hours:.1f}h old (threshold: {freshness_hours}h)"
    return False, f"index.json is {age_hours:.1f}h old (older than threshold {freshness_hours}h)"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3000,
        help="Nelder-Mead iteration cap for calibration (default: 3000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: ./output/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO-level logging (only WARN+ shown to stdout)",
    )
    parser.add_argument(
        "--skip-if-fresh",
        type=float,
        default=None,
        metavar="HOURS",
        help=(
            "Skip the full pipeline if output/index.json is younger than HOURS. "
            "Useful for avoiding wasted re-calibrations during the tournament. "
            "Example: --skip-if-fresh 6 skips if outputs are <6 hours old."
        ),
    )
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help=(
            "Skip the historical snapshot copy. By default each run copies "
            "its outputs to output/runs/<timestamp>/ for audit trail. Disable "
            "if you don't need history or are short on disk."
        ),
    )
    parser.add_argument(
        "--from-snapshot",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Read fair_probs from a previously-saved API snapshot JSON "
            "instead of hitting the live FairLine API. Used to reproduce a "
            "past run exactly. Example: --from-snapshot output/runs/"
            "2026-06-08T05-20-12Z/api_snapshot.json"
        ),
    )
    parser.add_argument(
        "--cron-mode",
        action="store_true",
        help=(
            "Enable production cron behavior: structured timestamped logging "
            "to output/logs/run-<ts>.log, WARN+ to stderr for cron-email "
            "alerting, exclusive lockfile (no concurrent runs), atomic output "
            "swap (partial writes are quarantined to output.pending/), and "
            "health.json emission. See docs/DEPLOYMENT.md for full deployment."
        ),
    )
    parser.add_argument(
        "--force-unlock",
        action="store_true",
        help=(
            "Operator command: clear a stale lockfile from a previous run "
            "that crashed without releasing it. Refuses to clear if the "
            "holding PID is still alive (use --force-unlock-dangerous to "
            "override that check). Then exits without running."
        ),
    )
    parser.add_argument(
        "--force-unlock-dangerous",
        action="store_true",
        help=(
            "Like --force-unlock but skips the PID-liveness check. ONLY use "
            "if you're certain no run is in progress; otherwise you may "
            "corrupt an active calibration."
        ),
    )
    args = parser.parse_args()

    output_dir = args.output_dir or OUTPUT_DIR

    # --force-unlock / --force-unlock-dangerous (early exit)
    if args.force_unlock or args.force_unlock_dangerous:
        configure_interactive_logging(quiet=args.quiet)
        lock_path = output_dir / LOCK_FILENAME
        cleared = force_unlock(
            lock_path, only_if_stale=not args.force_unlock_dangerous
        )
        sys.exit(0 if cleared else 1)

    # Configure logging based on mode
    log_path: Optional[Path] = None
    if args.cron_mode:
        # Set up the log file BEFORE acquiring the lock so we can record
        # the lock-busy error if it occurs.
        log_path = configure_cron_logging(output_dir)
        logger.info(f"Cron-mode run started. Log: {log_path}")
    else:
        configure_interactive_logging(quiet=args.quiet)

    # Cron-mode wraps the run in lockfile + atomic-output + health
    start_time = time.time()
    if args.cron_mode:
        lock_path = output_dir / LOCK_FILENAME
        try:
            with lockfile_acquired(lock_path):
                _run_with_cron_safeguards(args, output_dir, start_time)
        except LockBusyError as exc:
            logger.error(f"Lock busy: {exc}")
            # Even on lock-busy, write a failure health so monitoring sees it
            try:
                write_health(
                    output_dir,
                    exit_status="failure",
                    duration_sec=time.time() - start_time,
                    exit_reason=str(exc),
                )
            except Exception:
                pass
            sys.exit(2)
    else:
        # Interactive path - no lock, no atomic swap, no health emission.
        # Behaviour matches v1/v2 exactly.
        _run_interactive(args, output_dir)


def _run_interactive(args: argparse.Namespace, output_dir: Path) -> None:
    """Direct, non-atomic, non-locked execution.

    Used by interactive/test invocations. Identical to v2's run() behavior
    (skip-if-fresh check, then run with the supplied args). Existing tests
    that import and call this don't see any new behavior.
    """
    if args.skip_if_fresh is not None:
        is_fresh, reason = _output_is_fresh(output_dir, args.skip_if_fresh)
        if is_fresh:
            logger.info(f"Skipping run - {reason}")
            logger.info("Use --skip-if-fresh 0 or omit the flag to force a re-run.")
            return
        else:
            logger.info(f"Proceeding with run - {reason}")

    run(
        max_iter=args.max_iter,
        output_dir=args.output_dir,
        verbose=not args.quiet,
        snapshot=not args.no_snapshot,
        from_snapshot=args.from_snapshot,
    )


def _run_with_cron_safeguards(
    args: argparse.Namespace,
    output_dir: Path,
    start_time: float,
) -> None:
    """Inside lockfile_acquired() - perform skip check, atomic-output swap,
    catch any exception, write health.json on either path.

    On success: health.json reports exit_status=success with calibration +
    validation diagnostics.

    On failure: health.json reports exit_status=failure with exit_reason
    set to the exception message. The framework's existing exception
    handling (validation failures cause warnings, calibration failures
    raise) determines what counts as failure.
    """
    try:
        if args.skip_if_fresh is not None:
            is_fresh, reason = _output_is_fresh(output_dir, args.skip_if_fresh)
            if is_fresh:
                logger.info(f"Skipping run - {reason}")
                # Skip is a success outcome from a monitoring perspective -
                # the framework's outputs are fresh and consumable.
                write_health(
                    output_dir,
                    exit_status="success",
                    duration_sec=time.time() - start_time,
                    extra={"skipped": True, "skip_reason": reason},
                )
                return
            else:
                logger.info(f"Proceeding with run - {reason}")

        # Atomic output: write to output.pending/ then swap into output/
        # on success. On failure, output/ is untouched and the staging
        # dir remains for forensics.
        with atomic_output_dir(output_dir) as staging_dir:
            written = run(
                max_iter=args.max_iter,
                output_dir=staging_dir,
                verbose=not args.quiet,
                snapshot=not args.no_snapshot,
                from_snapshot=args.from_snapshot,
            )

        # Read back the validation report from the (now-swapped) output dir
        # to populate health.json with diagnostics.
        n_scenarios = 0
        max_residual: Optional[float] = None
        validation_pass: Optional[bool] = None
        try:
            index_path = output_dir / "index.json"
            if index_path.exists():
                index = json.loads(index_path.read_text())
                # scenario_files lists the 78 deviation scenarios (12 single +
                # 66 pairwise). The baseline scenario is tracked separately
                # as baseline_file. Total library size = 79.
                n_scenarios = len(index.get("scenario_files", []))
                if index.get("baseline_file"):
                    n_scenarios += 1
                max_residual = index.get("calibration", {}).get("max_residual")
            vr_path = output_dir / "validation_report.json"
            if vr_path.exists():
                vr = json.loads(vr_path.read_text())
                # overall_pass lives inside the summary section, not at
                # top level (top level has computed_at/summary/directional_
                # sanity/sensitivity).
                validation_pass = vr.get("summary", {}).get("overall_pass")
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Could not read diagnostics for health.json: {exc}")

        write_health(
            output_dir,
            exit_status="success",
            duration_sec=time.time() - start_time,
            n_scenarios=n_scenarios,
            calibration_max_residual=max_residual,
            validation_pass=validation_pass,
        )

    except Exception as exc:
        logger.error(f"Cron run failed: {exc}", exc_info=True)
        # Don't let write_health failures mask the original exception
        try:
            write_health(
                output_dir,
                exit_status="failure",
                duration_sec=time.time() - start_time,
                exit_reason=f"{type(exc).__name__}: {exc}",
            )
        except Exception as health_exc:
            logger.error(f"Also failed to write health.json: {health_exc}")
        raise


if __name__ == "__main__":
    main()