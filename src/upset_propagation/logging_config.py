"""Logging configuration.

Two presets:

  - interactive: legacy print()-like output. Goes to stdout. Used by
    direct `python -m upset_propagation.run` invocations and tests.

  - cron: structured timestamped output to a log file under
    output/logs/. Only WARN+ escapes to stderr (which cron typically
    forwards to email - so unexpected stderr output is a signal that
    something needs operator attention).

Both presets configure the root logger so any module's `logger.info(...)`
call is captured. The framework's existing print() calls are converted
to logger.info() in run.py for unified routing.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Log line format for both modes: timestamp + level + logger name + message.
# Logger name lets us see which module produced what (useful when
# diagnosing failures across the framework's many modules).
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"


def configure_interactive_logging(quiet: bool = False) -> None:
    """Set up logging for direct CLI / test invocations.

    All output to stdout. Level is INFO by default, WARNING if quiet.
    No log file written.

    Args:
        quiet: if True, suppress INFO and below; show only WARN+.
            Matches the existing --quiet flag semantics.
    """
    root = logging.getLogger()
    # Wipe any prior handlers (idempotent if called twice)
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_PlainFormatter())
    root.addHandler(handler)
    root.setLevel(logging.WARNING if quiet else logging.INFO)


def configure_cron_logging(
    output_dir: Path,
    timestamp: Optional[str] = None,
) -> Path:
    """Set up logging for unattended cron runs.

    INFO+ goes to a timestamped log file under output/logs/. WARN+ also
    escapes to stderr (so cron's email-on-stderr feature surfaces real
    problems to ops).

    The log file path follows the pattern output/logs/run-{ts}.log where
    {ts} is YYYYMMDD-HHMMSS-UTC. This matches the existing snapshot
    directory convention (output/runs/{ts}/) so timestamps line up
    between logs and outputs for any given run.

    Args:
        output_dir: framework's output directory (logs go to a `logs/`
            subdirectory under it)
        timestamp: optional override (defaults to current UTC timestamp)

    Returns: the path of the active log file (caller may want to log it
        on exit for visibility).
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run-{timestamp}.log"

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    # Detailed format → log file
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    file_handler.setLevel(logging.INFO)
    root.addHandler(file_handler)

    # WARN+ also to stderr for cron-email alerting
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
    stderr_handler.setLevel(logging.WARNING)
    root.addHandler(stderr_handler)

    root.setLevel(logging.INFO)

    return log_path


class _PlainFormatter(logging.Formatter):
    """Interactive mode formatter - passes the message through unchanged.

    Matches the look of the framework's existing print() output (no
    timestamps, no level prefix) so users who ran the framework on v1/v2
    see no visual change in v3.
    """
    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()