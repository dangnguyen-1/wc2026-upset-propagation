"""Tests for upset_propagation.health.

Coverage:
  - All 5 verdicts triggered by appropriate input (HEALTHY, STALE,
    FAILURE, MISSING, DEGRADED)
  - HealthVerdict.exit_code mapping
  - REQUIRED_OUTPUT_FILES check (item 10 - stale-output detection)
  - Malformed health.json handled gracefully
  - max_age_hours threshold semantics

Strategy: tmp_path-based health.json fixtures. Each verdict has a
focused test that constructs the minimum health.json needed to
trigger it, then asserts on the report's verdict and exit code.
This mirrors the smoke tests we ran by hand during item 9+10
development - codified as regression guards.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from upset_propagation.health import (
    REQUIRED_OUTPUT_FILES,
    HealthVerdict,
    check_health,
)


# Helpers


def _write_health_json(
    tmp_path: Path,
    age_hours: float = 0.0,
    exit_status: str = "success",
    validation_pass: bool | None = True,
    calibration_max_residual: float | None = 0.0001,
    exit_reason: str | None = None,
) -> None:
    """Write a synthetic health.json into tmp_path/health.json.

    age_hours: how long ago last_run_utc should be.
    """
    last_run = (
        datetime.now(timezone.utc) - timedelta(hours=age_hours)
    ).isoformat()
    payload = {
        "last_run_utc": last_run,
        "duration_sec": 247.5,
        "exit_status": exit_status,
        "n_scenarios": 79,
    }
    if validation_pass is not None:
        payload["validation_pass"] = validation_pass
    if calibration_max_residual is not None:
        payload["calibration_max_residual"] = calibration_max_residual
    if exit_reason is not None:
        payload["exit_reason"] = exit_reason
    (tmp_path / "health.json").write_text(json.dumps(payload))


def _write_required_files(tmp_path: Path) -> None:
    """Write empty placeholder files for each REQUIRED_OUTPUT_FILES entry."""
    for name in REQUIRED_OUTPUT_FILES:
        (tmp_path / name).write_text("{}")


# HealthVerdict exit codes


def test_exit_codes_match_spec():
    """The 5 verdicts map to 5 distinct exit codes for monitoring tools."""
    assert HealthVerdict.HEALTHY.exit_code == 0
    assert HealthVerdict.STALE.exit_code == 1
    assert HealthVerdict.FAILURE.exit_code == 2
    assert HealthVerdict.MISSING.exit_code == 3
    assert HealthVerdict.DEGRADED.exit_code == 4


def test_exit_codes_all_distinct():
    """No two verdicts share an exit code (ambiguous for monitoring)."""
    codes = [v.exit_code for v in HealthVerdict]
    assert len(set(codes)) == len(codes)


# MISSING verdict


def test_missing_when_health_json_absent(tmp_path):
    """No health.json → MISSING (3)."""
    # Don't write any files
    report = check_health(tmp_path)
    assert report.verdict == HealthVerdict.MISSING
    assert "missing" in report.reason.lower()


def test_missing_when_health_json_unparseable(tmp_path):
    """health.json with junk content → MISSING with parse-error reason."""
    (tmp_path / "health.json").write_text("not valid json {")
    report = check_health(tmp_path)
    assert report.verdict == HealthVerdict.MISSING


def test_missing_when_required_output_file_absent(tmp_path):
    """health.json says success but index.json (REQUIRED) is missing → MISSING."""
    _write_health_json(tmp_path)
    # Don't write REQUIRED_OUTPUT_FILES - only health.json
    report = check_health(tmp_path)
    assert report.verdict == HealthVerdict.MISSING
    # The missing file is one of the required set
    assert any(
        name in str(report.details.get("missing_files", []))
        for name in REQUIRED_OUTPUT_FILES
    )


def test_missing_when_last_run_utc_unparseable(tmp_path):
    """Malformed last_run_utc string → MISSING (validator's defensive path)."""
    (tmp_path / "health.json").write_text(json.dumps({
        "exit_status": "success",
        "last_run_utc": "not-a-timestamp",
    }))
    report = check_health(tmp_path)
    assert report.verdict == HealthVerdict.MISSING


# FAILURE verdict


def test_failure_when_exit_status_failure(tmp_path):
    """exit_status=failure → FAILURE (2), exit_reason surfaced."""
    _write_required_files(tmp_path)
    _write_health_json(
        tmp_path,
        exit_status="failure",
        exit_reason="ConnectionError: API timed out",
    )
    report = check_health(tmp_path)
    assert report.verdict == HealthVerdict.FAILURE
    assert "ConnectionError" in report.reason


def test_failure_exit_status_unknown_also_fails(tmp_path):
    """Any non-success exit_status counts as FAILURE."""
    _write_required_files(tmp_path)
    _write_health_json(tmp_path, exit_status="weird_state")
    report = check_health(tmp_path)
    assert report.verdict == HealthVerdict.FAILURE


# STALE verdict


def test_stale_when_last_run_older_than_threshold(tmp_path):
    """last_run_utc 5h ago, threshold 3h → STALE (1)."""
    _write_required_files(tmp_path)
    _write_health_json(tmp_path, age_hours=5.0)
    report = check_health(tmp_path, max_age_hours=3.0)
    assert report.verdict == HealthVerdict.STALE
    assert "5" in report.reason  # the actual age appears


def test_not_stale_within_threshold(tmp_path):
    """last_run_utc 2h ago, threshold 3h → HEALTHY (not stale)."""
    _write_required_files(tmp_path)
    _write_health_json(tmp_path, age_hours=2.0)
    report = check_health(tmp_path, max_age_hours=3.0)
    assert report.verdict == HealthVerdict.HEALTHY


def test_stale_threshold_configurable(tmp_path):
    """Custom max_age_hours respected - same data, different threshold."""
    _write_required_files(tmp_path)
    _write_health_json(tmp_path, age_hours=5.0)

    # Strict (1h) → STALE
    strict = check_health(tmp_path, max_age_hours=1.0)
    assert strict.verdict == HealthVerdict.STALE

    # Lenient (10h) → HEALTHY
    lenient = check_health(tmp_path, max_age_hours=10.0)
    assert lenient.verdict == HealthVerdict.HEALTHY


# DEGRADED verdict


def test_degraded_when_validation_fails(tmp_path):
    """exit_status=success but validation_pass=False → DEGRADED (4)."""
    _write_required_files(tmp_path)
    _write_health_json(tmp_path, validation_pass=False)
    report = check_health(tmp_path)
    assert report.verdict == HealthVerdict.DEGRADED
    assert "validation" in report.reason.lower()


def test_degraded_when_calibration_residual_too_high(tmp_path):
    """exit_status=success but max_residual > tolerance → DEGRADED."""
    _write_required_files(tmp_path)
    _write_health_json(
        tmp_path,
        validation_pass=True,
        calibration_max_residual=0.05,  # well above tolerance (0.005)
    )
    report = check_health(tmp_path)
    assert report.verdict == HealthVerdict.DEGRADED
    assert "residual" in report.reason.lower() or "0.05" in report.reason


def test_residual_at_tolerance_is_ok(tmp_path):
    """Calibration residual at exactly tolerance is acceptable, not DEGRADED."""
    from upset_propagation.config import CALIBRATION_TOLERANCE
    _write_required_files(tmp_path)
    _write_health_json(
        tmp_path,
        calibration_max_residual=CALIBRATION_TOLERANCE,  # at the line
    )
    report = check_health(tmp_path)
    # Strictly: residual > tolerance is degraded; equal is OK
    assert report.verdict == HealthVerdict.HEALTHY


# HEALTHY verdict


def test_healthy_when_all_checks_pass(tmp_path):
    """Fresh successful run, validation pass, residual OK → HEALTHY (0)."""
    _write_required_files(tmp_path)
    _write_health_json(tmp_path)  # all defaults are healthy values
    report = check_health(tmp_path)
    assert report.verdict == HealthVerdict.HEALTHY
    assert report.verdict.exit_code == 0


# Priority ordering


def test_failure_takes_priority_over_stale(tmp_path):
    """If exit_status=failure AND last_run is old, we report FAILURE first.

    The check order is: missing > failure > stale > missing-files > degraded.
    Failure should beat stale because actionable reason is the failure cause,
    not the staleness.
    """
    _write_required_files(tmp_path)
    _write_health_json(
        tmp_path,
        age_hours=10.0,  # would trigger STALE if not for...
        exit_status="failure",
        exit_reason="some failure",
    )
    report = check_health(tmp_path)
    assert report.verdict == HealthVerdict.FAILURE


# HealthReport.to_dict


def test_health_report_to_dict_contains_all_fields(tmp_path):
    """to_dict produces verdict, exit_code, reason, details, checked_at."""
    _write_required_files(tmp_path)
    _write_health_json(tmp_path)
    report = check_health(tmp_path)
    d = report.to_dict()
    assert "verdict" in d
    assert "exit_code" in d
    assert "reason" in d
    assert "details" in d
    assert "checked_at" in d
    assert d["exit_code"] == 0  # HEALTHY