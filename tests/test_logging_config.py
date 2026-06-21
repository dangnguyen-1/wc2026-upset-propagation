"""Tests for upset_propagation.logging_config.

Tiny smoke-test suite - three checks that catch the bug class "someone
broke logging setup without noticing":

  1. configure_interactive_logging is callable and configures handlers
  2. configure_cron_logging writes a log file at the expected path
  3. Re-configuring doesn't stack handlers (each log emitted once, not N times)

The third is a real bug class - logging module is notorious for this.
The current implementation wipes prior handlers, so calling
configure_*_logging twice is safe. Test guards against future regression.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from upset_propagation.logging_config import (
    configure_cron_logging,
    configure_interactive_logging,
)


@pytest.fixture(autouse=True)
def reset_root_logger():
    """Restore root logger state after each test - these functions mutate
    the global root logger, so isolating tests is important."""
    yield
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.WARNING)  # Python default


def test_interactive_logging_configures_root_handler():
    """After calling, root logger has at least one handler and INFO level."""
    configure_interactive_logging()
    root = logging.getLogger()
    assert len(root.handlers) >= 1
    assert root.level == logging.INFO


def test_cron_logging_writes_file_at_expected_path(tmp_path):
    """configure_cron_logging creates output/logs/run-{ts}.log and returns its path."""
    log_path = configure_cron_logging(tmp_path, timestamp="20260610-120000")
    expected = tmp_path / "logs" / "run-20260610-120000.log"
    assert log_path == expected
    assert log_path.parent.exists()  # logs/ dir created
    # Emit a log line - file should exist after this since the handler
    # opens it lazily on first write
    logging.getLogger().info("test marker")
    # File handler should have created the file
    assert log_path.exists()


def test_reconfiguring_does_not_stack_handlers():
    """REGRESSION: calling configure_interactive_logging twice should NOT
    accumulate handlers. Each log line must be emitted ONCE, not N times.

    The implementation wipes handlers at the top of each call; this
    test guards against that line being removed/refactored away.
    """
    configure_interactive_logging()
    handlers_after_first = len(logging.getLogger().handlers)

    configure_interactive_logging()
    handlers_after_second = len(logging.getLogger().handlers)

    assert handlers_after_first == handlers_after_second, (
        f"Re-configuring stacked handlers: "
        f"{handlers_after_first} → {handlers_after_second}. "
        f"Each log line would now be emitted twice."
    )