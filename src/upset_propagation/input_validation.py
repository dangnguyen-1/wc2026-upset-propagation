"""Pre-calibration input validation.

Catches data problems BEFORE the ~4-minute calibration runs, so failures
are clear and cheap. Two validators:

  - validate_elo_history(ratings) → ValidationReport
      Catches: missing teams, NaN/inf/negative Elos, implausible values
      (<1200 or >2300 warns; <1000 or >3000 fails), stale data
      (last update >90 days old warns)

  - validate_fair_probs(fair_probs) → ValidationReport
      Catches: wrong team count, sum outside reasonable range, max
      team prob too high (>0.30 warns, >0.50 fails), exact-zero team
      probabilities (FAIL - means the API has no view on that team).
      Note: small but nonzero values (e.g. Haiti ~0.07%) are NORMAL
      for 48-team WC, not a problem worth WARNing about.

Both return a structured ValidationReport with PASS/WARN/FAIL verdict.
The run.py wiring treats FAIL as fatal (raises, triggers cron-mode's
failure-health emission); WARN logs but proceeds.

The bounds were chosen from historical observation:
  - WC participant Elos: 1300-2100 typical (qualified by definition)
  - Pre-tournament market top favorite: 16-18% (Spain 2026, Brazil 2022,
    France 2018). 30%+ would be unprecedented.
  - API sum at observation time: 0.9965 (small overround built in).
    Range [0.95, 1.10] covers any plausible market state.

Public API:
    ValidationIssue dataclass
    ValidationReport dataclass
    validate_elo_history(ratings, groups=None) → ValidationReport
    validate_fair_probs(fair_probs, groups=None) → ValidationReport
    assert_validation_passes(report, *, allow_warn=True) → None
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import pandas as pd

from upset_propagation.config import ELO_HISTORY
from upset_propagation.scenarios import (
    load_groups,
    resolve_elo_for_wc_team,
    resolve_fair_prob_for_group_team,
)


logger = logging.getLogger(__name__)


# Thresholds (single source of truth)

# Elo bounds - chosen from historical World Cup data.
ELO_WARN_LOW = 1200.0   # warn if below
ELO_WARN_HIGH = 2300.0  # warn if above
ELO_FAIL_LOW = 1000.0   # fail if below
ELO_FAIL_HIGH = 3000.0  # fail if above

# How recent the most recent Elo entry should be.
ELO_STALENESS_WARN_DAYS = 90

# Fair_probs bounds.
FAIR_PROBS_SUM_WARN_LOW = 0.99
FAIR_PROBS_SUM_WARN_HIGH = 1.01
FAIR_PROBS_SUM_FAIL_LOW = 0.95
FAIR_PROBS_SUM_FAIL_HIGH = 1.10

FAIR_PROB_MAX_WARN = 0.30
FAIR_PROB_MAX_FAIL = 0.50
# We don't enforce a lower bound - even WC qualifiers can legitimately have
# <0.1% probability (the field has 48 teams and Σ ≈ 1, so the bottom ~20
# teams average <1% each, with the weakest below 0.1%). The only true
# failure mode is fair_prob == 0 exactly, which would mean the API has no
# view on the team - caught separately as FAIR_PROB_ZERO.

# WC2026 fixed field size.
EXPECTED_TEAM_COUNT = 48


# Shared types


@dataclass
class ValidationIssue:
    """One specific problem found by a validator.

    Attributes:
        severity: "warn" or "fail"
        code: short stable identifier for the issue type (e.g.
            "elo_too_low", "sum_out_of_range"). Stable across releases
            so monitoring can filter on it.
        message: human-readable description including the specific value
            that triggered the issue
    """
    severity: Literal["warn", "fail"]
    code: str
    message: str


@dataclass
class ValidationReport:
    """Aggregate result of one validator.

    Attributes:
        report_type: "elo_history" or "fair_probs"
        verdict: "PASS" (no issues), "WARN" (only warnings), "FAIL"
            (any fail-severity issue)
        issues: full list of issues found
        n_checked: how many items were inspected (teams, rows)
        diagnostics: extra context like "max_elo=2150 (France)"; not
            failure-relevant but useful for the log
    """
    report_type: str
    verdict: Literal["PASS", "WARN", "FAIL"]
    issues: list[ValidationIssue] = field(default_factory=list)
    n_checked: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def has_fail(self) -> bool:
        return any(i.severity == "fail" for i in self.issues)

    @property
    def has_warn(self) -> bool:
        return any(i.severity == "warn" for i in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_type": self.report_type,
            "verdict": self.verdict,
            "n_checked": self.n_checked,
            "n_warn": sum(1 for i in self.issues if i.severity == "warn"),
            "n_fail": sum(1 for i in self.issues if i.severity == "fail"),
            "issues": [asdict(i) for i in self.issues],
            "diagnostics": self.diagnostics,
        }


def _finalize_verdict(issues: list[ValidationIssue]) -> Literal["PASS", "WARN", "FAIL"]:
    """Derive the overall verdict from the issue list."""
    if any(i.severity == "fail" for i in issues):
        return "FAIL"
    if any(i.severity == "warn" for i in issues):
        return "WARN"
    return "PASS"


# Elo validator (item 15)


def validate_elo_history(
    ratings: dict[str, float],
    groups: Optional[dict[str, list[str]]] = None,
) -> ValidationReport:
    """Validate that the Elo ratings dict is healthy for calibration.

    Checks:
      1. Every WC team resolves to an Elo (via resolve_elo_for_wc_team)
      2. No NaN, inf, or negative values in `ratings`
      3. Every Elo for a WC team is in plausible range
      4. elo_history.csv was updated within ELO_STALENESS_WARN_DAYS

    Args:
        ratings: from load_latest_elo() - full ratings dict (all teams
            in elo_history.csv, not just WC ones)
        groups: optional override; defaults to load_groups()

    Returns: ValidationReport with verdict + issues + diagnostics.
    """
    if groups is None:
        groups = load_groups()

    issues: list[ValidationIssue] = []
    diagnostics: dict[str, Any] = {}
    wc_teams = [t for ts in groups.values() for t in ts]
    n_checked = len(wc_teams)

    # Check 1: every WC team resolvable
    wc_elos: dict[str, float] = {}
    for team in wc_teams:
        try:
            elo = resolve_elo_for_wc_team(team, ratings)
            wc_elos[team] = elo
        except KeyError as exc:
            issues.append(ValidationIssue(
                severity="fail",
                code="elo_team_missing",
                message=(
                    f"WC team {team!r} has no Elo entry "
                    f"(resolver raised KeyError: {exc}). "
                    f"Check elo_history.csv for the team name and any aliases."
                ),
            ))

    # Check 2: no NaN/inf/negative in any of the WC teams' Elos
    for team, elo in wc_elos.items():
        if math.isnan(elo) or math.isinf(elo):
            issues.append(ValidationIssue(
                severity="fail",
                code="elo_not_finite",
                message=f"Elo for {team} is {elo} (NaN or inf - propagator will produce garbage).",
            ))
        elif elo < 0:
            issues.append(ValidationIssue(
                severity="fail",
                code="elo_negative",
                message=f"Elo for {team} is {elo} (negative - impossible value).",
            ))

    # Check 3: plausible-range checks. Only run if value is finite.
    if wc_elos:
        for team, elo in wc_elos.items():
            if math.isnan(elo) or math.isinf(elo) or elo < 0:
                continue  # already flagged above
            if elo < ELO_FAIL_LOW or elo > ELO_FAIL_HIGH:
                issues.append(ValidationIssue(
                    severity="fail",
                    code="elo_out_of_range_fatal",
                    message=(
                        f"Elo for {team} is {elo:.0f}, outside hard bounds "
                        f"[{ELO_FAIL_LOW:.0f}, {ELO_FAIL_HIGH:.0f}]. "
                        f"Calibration will not produce sensible results."
                    ),
                ))
            elif elo < ELO_WARN_LOW or elo > ELO_WARN_HIGH:
                issues.append(ValidationIssue(
                    severity="warn",
                    code="elo_out_of_range",
                    message=(
                        f"Elo for {team} is {elo:.0f}, outside typical WC-participant "
                        f"range [{ELO_WARN_LOW:.0f}, {ELO_WARN_HIGH:.0f}]. "
                        f"Verify the source data."
                    ),
                ))

        diagnostics["elo_min"] = round(min(wc_elos.values()), 1)
        diagnostics["elo_min_team"] = min(wc_elos, key=wc_elos.get)
        diagnostics["elo_max"] = round(max(wc_elos.values()), 1)
        diagnostics["elo_max_team"] = max(wc_elos, key=wc_elos.get)
        diagnostics["elo_mean"] = round(sum(wc_elos.values()) / len(wc_elos), 1)

    # Check 4: staleness of elo_history.csv (most recent date across all rows)
    try:
        df = pd.read_csv(ELO_HISTORY)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            latest = df["date"].max()
            if pd.notna(latest):
                age_days = (datetime.now(timezone.utc) - latest.to_pydatetime().replace(tzinfo=timezone.utc)).days
                diagnostics["latest_elo_date"] = latest.strftime("%Y-%m-%d")
                diagnostics["latest_elo_age_days"] = age_days
                if age_days > ELO_STALENESS_WARN_DAYS:
                    issues.append(ValidationIssue(
                        severity="warn",
                        code="elo_stale",
                        message=(
                            f"Most recent elo_history.csv entry is {age_days} days old "
                            f"(latest: {latest.strftime('%Y-%m-%d')}). "
                            f"Threshold is {ELO_STALENESS_WARN_DAYS} days. "
                            f"Consider re-vendoring or pulling fresh data."
                        ),
                    ))
    except (OSError, ValueError, KeyError) as exc:
        # Non-fatal - we can validate Elo values without knowing the date column
        logger.debug(f"Could not check Elo staleness ({exc})")

    return ValidationReport(
        report_type="elo_history",
        verdict=_finalize_verdict(issues),
        issues=issues,
        n_checked=n_checked,
        diagnostics=diagnostics,
    )


# Fair_probs validator (item 16)


def validate_fair_probs(
    fair_probs: dict[str, float],
    groups: Optional[dict[str, list[str]]] = None,
) -> ValidationReport:
    """Validate that the API's fair_probs dict is sane.

    Checks:
      1. Exactly EXPECTED_TEAM_COUNT (48) teams
      2. Sum is within bounds (FAIL outside [0.95, 1.10], WARN outside [0.99, 1.01])
      3. Max team prob ≤ 0.30 (WARN) / ≤ 0.50 (FAIL)
      4. No team has fair_prob exactly 0 (FAIL) - small but nonzero values
         are normal for the field's weakest teams
      5. No NaN/inf
      6. Every WC team resolves to a fair_prob entry

    Args:
        fair_probs: from fetch_baseline_fair_probs()
        groups: optional override; defaults to load_groups()

    Returns: ValidationReport.
    """
    if groups is None:
        groups = load_groups()

    issues: list[ValidationIssue] = []
    diagnostics: dict[str, Any] = {}
    n_checked = len(fair_probs)

    # Check 1: team count
    if n_checked != EXPECTED_TEAM_COUNT:
        issues.append(ValidationIssue(
            severity="fail",
            code="team_count_wrong",
            message=(
                f"API returned {n_checked} teams, expected exactly {EXPECTED_TEAM_COUNT}. "
                f"Either the API is misconfigured or the WC field has changed."
            ),
        ))

    # Check 2: sum bounds
    total = sum(fair_probs.values())
    diagnostics["fair_probs_sum"] = round(total, 6)
    if total < FAIR_PROBS_SUM_FAIL_LOW or total > FAIR_PROBS_SUM_FAIL_HIGH:
        issues.append(ValidationIssue(
            severity="fail",
            code="sum_out_of_range_fatal",
            message=(
                f"fair_probs sum is {total:.4f}, outside hard bounds "
                f"[{FAIR_PROBS_SUM_FAIL_LOW:.2f}, {FAIR_PROBS_SUM_FAIL_HIGH:.2f}]. "
                f"Indicates API or parser malfunction."
            ),
        ))
    elif total < FAIR_PROBS_SUM_WARN_LOW or total > FAIR_PROBS_SUM_WARN_HIGH:
        issues.append(ValidationIssue(
            severity="warn",
            code="sum_out_of_range",
            message=(
                f"fair_probs sum is {total:.4f}, outside the tight range "
                f"[{FAIR_PROBS_SUM_WARN_LOW:.2f}, {FAIR_PROBS_SUM_WARN_HIGH:.2f}]. "
                f"Larger-than-usual market overround? Check API status."
            ),
        ))

    # Check 3-5: per-team checks (NaN/inf, max, min)
    if fair_probs:
        non_finite_teams = [
            t for t, p in fair_probs.items() if math.isnan(p) or math.isinf(p)
        ]
        if non_finite_teams:
            issues.append(ValidationIssue(
                severity="fail",
                code="fair_prob_not_finite",
                message=(
                    f"{len(non_finite_teams)} teams have non-finite fair_prob: "
                    f"{non_finite_teams[:5]}{'...' if len(non_finite_teams) > 5 else ''}"
                ),
            ))

        finite_probs = {
            t: p for t, p in fair_probs.items()
            if not (math.isnan(p) or math.isinf(p))
        }
        if finite_probs:
            max_team = max(finite_probs, key=finite_probs.get)
            max_p = finite_probs[max_team]
            min_team = min(finite_probs, key=finite_probs.get)
            min_p = finite_probs[min_team]
            diagnostics["max_team"] = max_team
            diagnostics["max_prob"] = round(max_p, 6)
            diagnostics["min_team"] = min_team
            diagnostics["min_prob"] = round(min_p, 6)

            if max_p > FAIR_PROB_MAX_FAIL:
                issues.append(ValidationIssue(
                    severity="fail",
                    code="max_prob_too_high_fatal",
                    message=(
                        f"{max_team} has fair_prob={max_p:.4f}, above hard ceiling "
                        f"{FAIR_PROB_MAX_FAIL:.2f}. No pre-tournament favorite has "
                        f"ever been this high. Likely an API bug."
                    ),
                ))
            elif max_p > FAIR_PROB_MAX_WARN:
                issues.append(ValidationIssue(
                    severity="warn",
                    code="max_prob_too_high",
                    message=(
                        f"{max_team} has fair_prob={max_p:.4f}, above the typical "
                        f"top-favorite range (>{FAIR_PROB_MAX_WARN:.2f}). "
                        f"Verify market state."
                    ),
                ))

            # Bottom-prob check: we deliberately don't WARN on small but
            # nonzero values. With 48 teams and Σ ≈ 1, the field's weakest
            # team legitimately sits around 0.05-0.10% (Haiti, Saudi Arabia,
            # etc.). The only true red flag is fair_prob == 0 exactly,
            # which means the API has no view on the team - suspicious
            # post-draw.
            zero_prob_teams = [t for t, p in finite_probs.items() if p == 0.0]
            if zero_prob_teams:
                issues.append(ValidationIssue(
                    severity="fail",
                    code="fair_prob_zero",
                    message=(
                        f"{len(zero_prob_teams)} teams have fair_prob=0 exactly: "
                        f"{zero_prob_teams[:5]}. Means the API has no view on "
                        f"these teams. Post-draw, every WC participant should "
                        f"have a nonzero probability."
                    ),
                ))

    # Check 6: every WC team resolves
    wc_teams = [t for ts in groups.values() for t in ts]
    unresolved: list[str] = []
    for team in wc_teams:
        try:
            resolve_fair_prob_for_group_team(team, fair_probs)
        except KeyError:
            unresolved.append(team)
    if unresolved:
        issues.append(ValidationIssue(
            severity="fail",
            code="fair_prob_team_missing",
            message=(
                f"{len(unresolved)} WC teams have no fair_prob entry: "
                f"{unresolved[:5]}{'...' if len(unresolved) > 5 else ''}. "
                f"Check team-name aliases in scenarios.resolve_fair_prob_for_group_team."
            ),
        ))

    return ValidationReport(
        report_type="fair_probs",
        verdict=_finalize_verdict(issues),
        issues=issues,
        n_checked=n_checked,
        diagnostics=diagnostics,
    )


# Enforcement helper


class InputValidationError(RuntimeError):
    """Raised when a validation report has FAIL-severity issues.

    Carries the report itself so callers can inspect for logging or
    health.json emission. Used by assert_validation_passes() to convert
    a structured report into a fatal error for the cron pipeline.
    """
    def __init__(self, report: ValidationReport):
        self.report = report
        n_fail = sum(1 for i in report.issues if i.severity == "fail")
        msgs = "; ".join(
            f"[{i.code}] {i.message}"
            for i in report.issues if i.severity == "fail"
        )
        super().__init__(
            f"{report.report_type} validation FAILED with {n_fail} issue(s): {msgs}"
        )


def assert_validation_passes(
    report: ValidationReport,
    *,
    allow_warn: bool = True,
) -> None:
    """Raise InputValidationError if the report has FAIL issues.

    Logs every issue at appropriate level:
      - FAIL → logger.error
      - WARN → logger.warning

    When allow_warn=False, WARN issues also trigger the exception (strict mode).
    Default allow_warn=True matches the v3 policy: WARN logs but doesn't kill,
    FAIL kills.

    Args:
        report: ValidationReport from one of the validators
        allow_warn: if False, WARN issues escalate to InputValidationError
    """
    for issue in report.issues:
        if issue.severity == "fail":
            logger.error(f"[{report.report_type}] [{issue.code}] {issue.message}")
        elif issue.severity == "warn":
            logger.warning(f"[{report.report_type}] [{issue.code}] {issue.message}")

    if report.has_fail or (not allow_warn and report.has_warn):
        raise InputValidationError(report)


# CLI smoke test


if __name__ == "__main__":
    # `python -m upset_propagation.input_validation` - runs both validators
    # against current data and prints the reports.
    from upset_propagation.baseline import fetch_baseline_fair_probs
    from upset_propagation.logging_config import configure_interactive_logging
    from upset_propagation.scenarios import load_latest_elo

    configure_interactive_logging()

    print("-" * 70)
    print("Validating Elo history...")
    print("-" * 70)
    ratings = load_latest_elo()
    elo_report = validate_elo_history(ratings)
    print(f"Verdict: {elo_report.verdict}  (n_checked={elo_report.n_checked})")
    for k, v in elo_report.diagnostics.items():
        print(f"  {k}: {v}")
    if elo_report.issues:
        print(f"\nIssues ({len(elo_report.issues)}):")
        for i in elo_report.issues:
            print(f"  [{i.severity.upper()}] [{i.code}] {i.message}")
    else:
        print("  No issues.")
    print()

    print("-" * 70)
    print("Validating fair_probs from FairLine API...")
    print("-" * 70)
    fair_probs = fetch_baseline_fair_probs()
    fp_report = validate_fair_probs(fair_probs)
    print(f"Verdict: {fp_report.verdict}  (n_checked={fp_report.n_checked})")
    for k, v in fp_report.diagnostics.items():
        print(f"  {k}: {v}")
    if fp_report.issues:
        print(f"\nIssues ({len(fp_report.issues)}):")
        for i in fp_report.issues:
            print(f"  [{i.severity.upper()}] [{i.code}] {i.message}")
    else:
        print("  No issues.")