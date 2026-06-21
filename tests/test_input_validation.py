"""Tests for upset_propagation.input_validation.

Coverage:
  - ValidationReport / ValidationIssue / verdict computation
  - validate_elo_history: missing teams, NaN/inf/negative, range bounds,
    staleness (warn-only)
  - validate_fair_probs: team count, sum bounds, max/min thresholds,
    NaN/inf, exact-zero detection, team resolution via aliases
  - assert_validation_passes: raises on FAIL, allows WARN by default,
    strict mode escalates WARN to FAIL

Strategy: use synthetic ratings + fair_probs dicts; both validators are
pure functions over dict inputs so no filesystem or network needed.
For tests that need a real WC2026 team list we use the actual groups
config (load_groups()) so team-name aliases work realistically.
"""

from __future__ import annotations

import math

import pytest

from upset_propagation.input_validation import (
    EXPECTED_TEAM_COUNT,
    InputValidationError,
    ValidationIssue,
    ValidationReport,
    assert_validation_passes,
    validate_elo_history,
    validate_fair_probs,
)
from upset_propagation.scenarios import load_groups


# Fixtures


@pytest.fixture
def wc_groups() -> dict[str, list[str]]:
    """Real WC2026 group config - 12 groups × 4 teams."""
    return load_groups()


@pytest.fixture
def wc_teams(wc_groups) -> list[str]:
    """Flat list of all 48 WC2026 team names."""
    return [t for ts in wc_groups.values() for t in ts]


@pytest.fixture
def healthy_ratings(wc_teams) -> dict[str, float]:
    """A ratings dict where every WC team has a sane Elo (1800).

    Uses the WC2026 spelling directly; this is what we expect after
    resolve_elo_for_wc_team normalization. For tests that need to exercise
    the alias path, pass a ratings dict keyed by elo_history.csv spelling.
    """
    return {team: 1800.0 for team in wc_teams}


@pytest.fixture
def healthy_fair_probs(wc_teams) -> dict[str, float]:
    """A 48-team fair_probs dict that sums to ~1.0 with no team above 0.2."""
    # Even split: each team ~2.08%, sum to 1.0 exactly
    return {team: 1.0 / len(wc_teams) for team in wc_teams}


# ValidationReport / Issue


def test_validation_report_pass_when_no_issues():
    """PASS verdict when issue list is empty."""
    report = ValidationReport(
        report_type="elo_history", verdict="PASS", issues=[], n_checked=48
    )
    assert report.verdict == "PASS"
    assert not report.has_fail
    assert not report.has_warn


def test_validation_report_warn_aggregates_only_warns():
    """If only WARN issues exist, has_warn=True but has_fail=False."""
    report = ValidationReport(
        report_type="fair_probs",
        verdict="WARN",
        issues=[ValidationIssue(severity="warn", code="x", message="m")],
        n_checked=48,
    )
    assert report.has_warn
    assert not report.has_fail


def test_validation_report_fail_overrides_warn():
    """If any FAIL exists, has_fail=True regardless of WARNs."""
    report = ValidationReport(
        report_type="fair_probs",
        verdict="FAIL",
        issues=[
            ValidationIssue(severity="warn", code="a", message=""),
            ValidationIssue(severity="fail", code="b", message=""),
        ],
        n_checked=48,
    )
    assert report.has_fail
    assert report.has_warn


def test_validation_report_to_dict_includes_counts():
    """to_dict() exposes n_warn and n_fail for monitoring."""
    report = ValidationReport(
        report_type="fair_probs",
        verdict="FAIL",
        issues=[
            ValidationIssue(severity="warn", code="a", message=""),
            ValidationIssue(severity="warn", code="b", message=""),
            ValidationIssue(severity="fail", code="c", message=""),
        ],
        n_checked=10,
    )
    d = report.to_dict()
    assert d["n_warn"] == 2
    assert d["n_fail"] == 1
    assert d["verdict"] == "FAIL"


# validate_elo_history


def test_elo_validator_pass_on_healthy_input(healthy_ratings, wc_groups):
    """Happy path: all 48 teams resolve, all Elos in range → PASS."""
    report = validate_elo_history(healthy_ratings, groups=wc_groups)
    # Note: this may still WARN on staleness depending on elo_history.csv's
    # most-recent date. We only assert no FAIL-severity issues.
    assert not report.has_fail
    assert report.n_checked == 48


def test_elo_validator_fails_on_missing_team(healthy_ratings, wc_groups):
    """If one WC team has no resolvable Elo, FAIL with helpful message."""
    # Drop one team to simulate missing data
    missing_team = next(iter(healthy_ratings))
    incomplete = {k: v for k, v in healthy_ratings.items() if k != missing_team}

    report = validate_elo_history(incomplete, groups=wc_groups)
    assert report.has_fail
    fail_codes = [i.code for i in report.issues if i.severity == "fail"]
    assert "elo_team_missing" in fail_codes


def test_elo_validator_fails_on_nan(healthy_ratings, wc_groups):
    """NaN Elo for a WC team is FAIL (propagator would produce garbage)."""
    team = next(iter(healthy_ratings))
    healthy_ratings[team] = float("nan")

    report = validate_elo_history(healthy_ratings, groups=wc_groups)
    fail_codes = [i.code for i in report.issues if i.severity == "fail"]
    assert "elo_not_finite" in fail_codes


def test_elo_validator_fails_on_inf(healthy_ratings, wc_groups):
    """Infinite Elo is FAIL."""
    team = next(iter(healthy_ratings))
    healthy_ratings[team] = float("inf")

    report = validate_elo_history(healthy_ratings, groups=wc_groups)
    fail_codes = [i.code for i in report.issues if i.severity == "fail"]
    assert "elo_not_finite" in fail_codes


def test_elo_validator_fails_on_negative(healthy_ratings, wc_groups):
    """Negative Elo is FAIL (impossible value)."""
    team = next(iter(healthy_ratings))
    healthy_ratings[team] = -100.0

    report = validate_elo_history(healthy_ratings, groups=wc_groups)
    fail_codes = [i.code for i in report.issues if i.severity == "fail"]
    assert "elo_negative" in fail_codes


def test_elo_validator_warns_on_low_but_plausible(healthy_ratings, wc_groups):
    """Elo at 1100 (below WARN_LOW=1200) is a WARN, not a FAIL.

    1100 is unusual for a WC qualifier but not physically impossible.
    """
    team = next(iter(healthy_ratings))
    healthy_ratings[team] = 1100.0

    report = validate_elo_history(healthy_ratings, groups=wc_groups)
    warn_codes = [i.code for i in report.issues if i.severity == "warn"]
    # Should be in warn range, not fail range
    assert "elo_out_of_range" in warn_codes
    # And the fatal range check should NOT trigger
    fail_codes = [i.code for i in report.issues if i.severity == "fail"]
    assert "elo_out_of_range_fatal" not in fail_codes


def test_elo_validator_fails_on_extreme_low(healthy_ratings, wc_groups):
    """Elo at 500 (below FAIL_LOW=1000) is a FAIL."""
    team = next(iter(healthy_ratings))
    healthy_ratings[team] = 500.0

    report = validate_elo_history(healthy_ratings, groups=wc_groups)
    fail_codes = [i.code for i in report.issues if i.severity == "fail"]
    assert "elo_out_of_range_fatal" in fail_codes


def test_elo_validator_diagnostics_populated(healthy_ratings, wc_groups):
    """Diagnostics dict has min/max/mean Elo info for log readability."""
    # Set distinct Elos so min/max are findable
    teams = list(healthy_ratings)
    healthy_ratings[teams[0]] = 2200.0  # max
    healthy_ratings[teams[1]] = 1400.0  # min

    report = validate_elo_history(healthy_ratings, groups=wc_groups)
    assert "elo_min" in report.diagnostics
    assert "elo_max" in report.diagnostics
    assert "elo_mean" in report.diagnostics
    assert report.diagnostics["elo_min"] == 1400.0
    assert report.diagnostics["elo_max"] == 2200.0


# validate_fair_probs


def test_fair_probs_validator_pass_on_healthy_input(healthy_fair_probs, wc_groups):
    """Happy path: 48 teams, sum ≈ 1, no team above max threshold."""
    report = validate_fair_probs(healthy_fair_probs, groups=wc_groups)
    assert report.verdict == "PASS"
    assert report.n_checked == 48
    assert not report.issues


def test_fair_probs_validator_fails_on_wrong_team_count(wc_groups):
    """API returning != 48 teams is FAIL (WC field is fixed-size)."""
    too_few = {f"Team{i}": 0.02 for i in range(30)}
    report = validate_fair_probs(too_few, groups=wc_groups)
    fail_codes = [i.code for i in report.issues if i.severity == "fail"]
    assert "team_count_wrong" in fail_codes


def test_fair_probs_validator_fails_on_huge_sum(wc_teams, wc_groups):
    """Sum > 1.10 is fatal - API parser malfunction territory."""
    # Each team at 0.05 = sum 2.4
    bad = {t: 0.05 for t in wc_teams}
    report = validate_fair_probs(bad, groups=wc_groups)
    fail_codes = [i.code for i in report.issues if i.severity == "fail"]
    assert "sum_out_of_range_fatal" in fail_codes


def test_fair_probs_validator_warns_on_slight_overround(wc_teams, wc_groups):
    """Sum at 1.025 is a WARN (outside tight [0.99, 1.01] but not fatal)."""
    each = 1.025 / len(wc_teams)
    probs = {t: each for t in wc_teams}
    report = validate_fair_probs(probs, groups=wc_groups)
    # Should warn but not fail
    assert report.has_warn
    assert not report.has_fail
    warn_codes = [i.code for i in report.issues if i.severity == "warn"]
    assert "sum_out_of_range" in warn_codes


def test_fair_probs_validator_warns_on_high_max(healthy_fair_probs, wc_groups):
    """Top team prob > 0.30 is a WARN (no pre-tournament fave has been this high)."""
    team = next(iter(healthy_fair_probs))
    # Boost one team to 0.35, reduce another to compensate
    healthy_fair_probs[team] = 0.35
    teams = list(healthy_fair_probs)
    healthy_fair_probs[teams[1]] = 0.001  # absorb the difference

    report = validate_fair_probs(healthy_fair_probs, groups=wc_groups)
    warn_codes = [i.code for i in report.issues if i.severity == "warn"]
    assert "max_prob_too_high" in warn_codes


def test_fair_probs_validator_fails_on_impossible_max(healthy_fair_probs, wc_groups):
    """Top team prob > 0.50 is FAIL - no team has ever been 50%+ pre-WC."""
    team = next(iter(healthy_fair_probs))
    healthy_fair_probs[team] = 0.60
    teams = list(healthy_fair_probs)
    healthy_fair_probs[teams[1]] = -0.30 + 1.0 / len(healthy_fair_probs)  # rebalance roughly

    report = validate_fair_probs(healthy_fair_probs, groups=wc_groups)
    fail_codes = [i.code for i in report.issues if i.severity == "fail"]
    assert "max_prob_too_high_fatal" in fail_codes


def test_fair_probs_validator_fails_on_exact_zero(healthy_fair_probs, wc_groups):
    """REGRESSION: exact-zero fair_prob means API has no view → FAIL.

    Distinct from small-but-nonzero (which is normal for weakest teams -
    threshold debugging during item 16 discovered Haiti legitimately ~0.07%).
    """
    team = next(iter(healthy_fair_probs))
    healthy_fair_probs[team] = 0.0

    report = validate_fair_probs(healthy_fair_probs, groups=wc_groups)
    fail_codes = [i.code for i in report.issues if i.severity == "fail"]
    assert "fair_prob_zero" in fail_codes


def test_fair_probs_validator_passes_on_tiny_but_nonzero(healthy_fair_probs, wc_groups):
    """REGRESSION: small nonzero values (e.g. Haiti ~0.07%) are NOT flagged.

    The original threshold (FAIR_PROB_MIN_WARN = 0.001) fired spurious WARNs
    on legitimate weakest-team probabilities in a 48-team field. Replaced
    with an exact-zero check; small nonzero values are now PASS.
    """
    team = next(iter(healthy_fair_probs))
    healthy_fair_probs[team] = 0.0007  # like Haiti

    report = validate_fair_probs(healthy_fair_probs, groups=wc_groups)
    # No issues about min prob at all
    codes = [i.code for i in report.issues]
    assert "min_prob_very_low" not in codes  # the old check should not exist
    assert "fair_prob_zero" not in codes  # nonzero, so this shouldn't fire


def test_fair_probs_validator_fails_on_nan(healthy_fair_probs, wc_groups):
    """NaN fair_prob is fatal."""
    team = next(iter(healthy_fair_probs))
    healthy_fair_probs[team] = float("nan")

    report = validate_fair_probs(healthy_fair_probs, groups=wc_groups)
    fail_codes = [i.code for i in report.issues if i.severity == "fail"]
    assert "fair_prob_not_finite" in fail_codes


def test_fair_probs_validator_diagnostics_populated(healthy_fair_probs, wc_groups):
    """Diagnostics show top/bottom team for log readability."""
    teams = list(healthy_fair_probs)
    healthy_fair_probs[teams[0]] = 0.15  # max
    healthy_fair_probs[teams[1]] = 0.0005  # min (just barely above zero)

    report = validate_fair_probs(healthy_fair_probs, groups=wc_groups)
    assert "max_team" in report.diagnostics
    assert "max_prob" in report.diagnostics
    assert "min_team" in report.diagnostics
    assert "min_prob" in report.diagnostics
    assert report.diagnostics["max_team"] == teams[0]
    assert report.diagnostics["min_team"] == teams[1]


# assert_validation_passes


def test_assert_passes_on_clean_report():
    """No issues → no exception."""
    report = ValidationReport(
        report_type="fair_probs", verdict="PASS", issues=[], n_checked=48
    )
    # Should not raise
    assert_validation_passes(report)


def test_assert_passes_on_warn_only_default():
    """allow_warn=True (default): WARN-only report doesn't raise."""
    report = ValidationReport(
        report_type="fair_probs",
        verdict="WARN",
        issues=[ValidationIssue(severity="warn", code="x", message="m")],
        n_checked=48,
    )
    assert_validation_passes(report)  # default allow_warn=True


def test_assert_raises_on_fail():
    """Any FAIL issue triggers InputValidationError, carrying the report."""
    issue = ValidationIssue(severity="fail", code="x", message="bad thing")
    report = ValidationReport(
        report_type="fair_probs",
        verdict="FAIL",
        issues=[issue],
        n_checked=48,
    )
    with pytest.raises(InputValidationError) as exc_info:
        assert_validation_passes(report)
    # The exception carries the report for downstream inspection
    assert exc_info.value.report is report
    # The message includes the issue code and text
    msg = str(exc_info.value)
    assert "x" in msg
    assert "bad thing" in msg


def test_assert_strict_mode_escalates_warn(caplog):
    """allow_warn=False: WARN-only report raises (strict mode)."""
    report = ValidationReport(
        report_type="fair_probs",
        verdict="WARN",
        issues=[ValidationIssue(severity="warn", code="x", message="m")],
        n_checked=48,
    )
    with pytest.raises(InputValidationError):
        assert_validation_passes(report, allow_warn=False)