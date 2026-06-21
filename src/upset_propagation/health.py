"""Framework health check.

CLI command: `python -m upset_propagation.health [--max-age-hours N]`

Checks both layers of "is the framework healthy":

  Layer 1 - health.json (item 9): the metadata the cron writes at the
  end of each run. Tells us "did the last run succeed?" and "when?"

  Layer 2 - required output files (item 10): the actual data files
  consumers read. Catches the failure mode where health.json reports
  success but index.json / baseline.json / top_10_ranking.json are
  missing or corrupted.

Both layers in one command - simpler ops story than two.

Exit codes (for monitoring tools):
  0 - healthy
  1 - stale (last run > --max-age-hours ago)
  2 - last run reported failure
  3 - missing/corrupt (required files don't exist)
  4 - degraded (succeeded but validation failed or calibration didn't converge)

External monitoring maps these to alert severities. Operators reading
human output get the specific reason.

Public API:
    HealthVerdict enum (string values match exit codes for clarity)
    HealthReport dataclass
    check_health(output_dir, max_age_hours) → HealthReport
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from upset_propagation.config import (
    CALIBRATION_TOLERANCE,
    OUTPUT_DIR,
)


logger = logging.getLogger(__name__)


# Files that consumers read directly. If any is missing, the framework
# can't fulfill its contract regardless of what health.json claims.
# Doesn't include the 79 individual scenario files (index.json enumerates
# them - if index references a missing file, the matcher catches it).
REQUIRED_OUTPUT_FILES = (
    "index.json",
    "baseline.json",
    "validation_report.json",
)

# Files that should exist after the cron has had a chance to write them.
# Distinct from REQUIRED - these are item-7+ additions, not v2-baseline.
EXPECTED_CRON_FILES = (
    "top_10_ranking.json",
    "health.json",
)


# Verdict / report


class HealthVerdict(str, Enum):
    """Health check outcomes. String values used in JSON output.

    Severity order (most-to-least concerning):
      missing > failure > stale > degraded > healthy
    """
    HEALTHY = "healthy"
    STALE = "stale"
    FAILURE = "failure"
    MISSING = "missing"
    DEGRADED = "degraded"

    @property
    def exit_code(self) -> int:
        return {
            HealthVerdict.HEALTHY: 0,
            HealthVerdict.STALE: 1,
            HealthVerdict.FAILURE: 2,
            HealthVerdict.MISSING: 3,
            HealthVerdict.DEGRADED: 4,
        }[self]


@dataclass
class HealthReport:
    """Full health check result.

    Attributes:
        verdict: overall HealthVerdict (drives exit code)
        reason: short human-readable explanation
        details: free-form structured details (last_run_utc, age_hours,
            missing_files, calibration_max_residual, etc.)
        checked_at: when the health check ran
    """
    verdict: HealthVerdict
    reason: str
    details: dict[str, Any] = field(default_factory=dict)
    checked_at: str = ""

    def __post_init__(self):
        if not self.checked_at:
            self.checked_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "exit_code": self.verdict.exit_code,
            "reason": self.reason,
            "details": self.details,
            "checked_at": self.checked_at,
        }


# Check logic


def _parse_iso_timestamp(s: str) -> Optional[datetime]:
    """Parse an ISO 8601 timestamp; return None on malformed input.

    Defensive - health.json might be hand-edited or corrupted.
    """
    try:
        # fromisoformat handles "2026-06-15T14:30:00+00:00" and similar.
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def check_health(
    output_dir: Optional[Path] = None,
    max_age_hours: float = 3.0,
) -> HealthReport:
    """Run all health checks against `output_dir` and return a report.

    Order of checks (early-exits on first failure):
      1. health.json exists and is parseable. MISSING if not.
      2. exit_status in health.json is "success". FAILURE if not.
      3. last_run_utc is within max_age_hours. STALE if older.
      4. All REQUIRED_OUTPUT_FILES exist. MISSING if any absent.
      5. validation_pass is True (if present). DEGRADED if False.
      6. calibration_max_residual ≤ CALIBRATION_TOLERANCE. DEGRADED if higher.

    If all checks pass → HEALTHY.

    Args:
        output_dir: where to look for health.json + required files
            (default: ./output/)
        max_age_hours: how recent last_run_utc must be (default: 3.0)

    Returns: HealthReport with verdict, reason, and structured details.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    details: dict[str, Any] = {"output_dir": str(output_dir)}

    # Check 1: health.json exists and is parseable
    health_path = output_dir / "health.json"
    if not health_path.exists():
        return HealthReport(
            verdict=HealthVerdict.MISSING,
            reason=f"health.json missing at {health_path}",
            details={
                **details,
                "missing_file": str(health_path),
                "hint": (
                    "Cron has never run with --cron-mode, or output/ was "
                    "deleted. Run `python -m upset_propagation.run --cron-mode` "
                    "to bootstrap."
                ),
            },
        )

    try:
        health = json.loads(health_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return HealthReport(
            verdict=HealthVerdict.MISSING,
            reason=f"health.json is unreadable or malformed: {exc}",
            details={**details, "health_path": str(health_path), "error": str(exc)},
        )

    details["health"] = health  # everything from health.json now inspectable

    # Check 2: exit_status is "success"
    exit_status = health.get("exit_status")
    if exit_status != "success":
        exit_reason = health.get("exit_reason", "(no exit_reason in health.json)")
        return HealthReport(
            verdict=HealthVerdict.FAILURE,
            reason=f"Last run reported failure: {exit_reason}",
            details={
                **details,
                "exit_status": exit_status,
                "exit_reason": exit_reason,
            },
        )

    # Check 3: last_run_utc is fresh
    last_run_str = health.get("last_run_utc")
    if last_run_str:
        last_run = _parse_iso_timestamp(last_run_str)
        if last_run is None:
            return HealthReport(
                verdict=HealthVerdict.MISSING,
                reason=f"last_run_utc is unparseable: {last_run_str!r}",
                details={**details, "last_run_utc_raw": last_run_str},
            )
        now = datetime.now(timezone.utc)
        # Make naive timestamps timezone-aware (assume UTC).
        if last_run.tzinfo is None:
            last_run = last_run.replace(tzinfo=timezone.utc)
        age_seconds = (now - last_run).total_seconds()
        age_hours = age_seconds / 3600.0
        details["last_run_age_hours"] = round(age_hours, 3)
        details["max_age_hours"] = max_age_hours
        if age_hours > max_age_hours:
            return HealthReport(
                verdict=HealthVerdict.STALE,
                reason=(
                    f"Last run was {age_hours:.2f}h ago, exceeds threshold "
                    f"{max_age_hours:.1f}h. Cron may have stopped."
                ),
                details=details,
            )

    # Check 4: all REQUIRED files exist
    missing = [
        name for name in REQUIRED_OUTPUT_FILES
        if not (output_dir / name).exists()
    ]
    if missing:
        return HealthReport(
            verdict=HealthVerdict.MISSING,
            reason=(
                f"{len(missing)} required output file(s) missing: {missing}. "
                f"health.json says success but data files are absent."
            ),
            details={**details, "missing_files": missing},
        )
    # Also note which optional cron-only files are missing - not a fail,
    # just useful info.
    missing_optional = [
        name for name in EXPECTED_CRON_FILES
        if not (output_dir / name).exists() and name != "health.json"
    ]
    if missing_optional:
        details["missing_optional"] = missing_optional

    # Check 5: validation passed (if reported)
    validation_pass = health.get("validation_pass")
    if validation_pass is False:
        return HealthReport(
            verdict=HealthVerdict.DEGRADED,
            reason=(
                "Last run succeeded but validation regressed. "
                "See validation_report.json for the specific failures."
            ),
            details={**details, "validation_pass": False},
        )

    # Check 6: calibration converged
    max_residual = health.get("calibration_max_residual")
    if max_residual is not None and max_residual > CALIBRATION_TOLERANCE:
        return HealthReport(
            verdict=HealthVerdict.DEGRADED,
            reason=(
                f"Calibration max_residual={max_residual:.5f} exceeds "
                f"tolerance {CALIBRATION_TOLERANCE:.4f}. Outputs may be "
                f"unreliable. Try --max-iter 5000."
            ),
            details={
                **details,
                "calibration_max_residual": max_residual,
                "calibration_tolerance": CALIBRATION_TOLERANCE,
            },
        )

    # All checks passed
    return HealthReport(
        verdict=HealthVerdict.HEALTHY,
        reason="All checks passed",
        details=details,
    )


# CLI


def main() -> None:
    """`python -m upset_propagation.health` - health check command.

    Exit codes match HealthVerdict.exit_code:
      0 healthy, 1 stale, 2 failure, 3 missing, 4 degraded
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: ./output/)",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=3.0,
        help=(
            "Maximum acceptable age of the last run in hours. "
            "Default 3.0 (2h cron cadence + 1h slack). Set lower for "
            "tighter monitoring, higher for looser."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of human-readable text",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout; exit code conveys verdict",
    )
    args = parser.parse_args()

    report = check_health(
        output_dir=args.output_dir,
        max_age_hours=args.max_age_hours,
    )

    if args.quiet:
        sys.exit(report.verdict.exit_code)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
        sys.exit(report.verdict.exit_code)

    # Human-readable output
    verdict_emoji = {
        HealthVerdict.HEALTHY: "✓",
        HealthVerdict.STALE: "⚠",
        HealthVerdict.FAILURE: "✗",
        HealthVerdict.MISSING: "✗",
        HealthVerdict.DEGRADED: "⚠",
    }[report.verdict]
    print(f"{verdict_emoji} {report.verdict.value.upper()} - {report.reason}")
    print()

    # Print useful details based on what's available
    d = report.details
    if "last_run_age_hours" in d:
        age = d["last_run_age_hours"]
        print(f"  Last run: {age:.2f}h ago "
              f"(threshold: {d.get('max_age_hours', 3.0):.1f}h)")
    health_info = d.get("health", {})
    if "duration_sec" in health_info:
        print(f"  Last run duration: {health_info['duration_sec']:.1f}s")
    if "n_scenarios" in health_info:
        print(f"  Scenarios produced: {health_info['n_scenarios']}")
    if "calibration_max_residual" in health_info:
        cal = health_info["calibration_max_residual"]
        tol = CALIBRATION_TOLERANCE
        marker = "✓" if cal <= tol else "⚠"
        print(f"  Calibration max_residual: {cal:.5f} (tolerance {tol:.4f}) {marker}")
    if "validation_pass" in health_info:
        v = health_info["validation_pass"]
        marker = "✓" if v else "✗"
        print(f"  Validation: {'PASS' if v else 'FAIL'} {marker}")
    if d.get("missing_optional"):
        print(f"  Optional files missing (not fatal): {d['missing_optional']}")
    if d.get("missing_files"):
        print(f"  Required files missing: {d['missing_files']}")
    if d.get("hint"):
        print(f"\n  Hint: {d['hint']}")

    sys.exit(report.verdict.exit_code)


if __name__ == "__main__":
    main()