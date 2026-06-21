"""Tests for upset_propagation.cron_utils.

Coverage:
  - lockfile_acquired: acquires, releases, fails on busy, writes PID
  - LockBusyError: carries holding_pid when readable
  - force_unlock: clears stale (dead PID), refuses live PID,
    handles missing/empty/malformed lockfiles
  - atomic_output_dir: stages to .pending/, swaps on success,
    preserves .pending/ on failure (forensics), leaves target untouched
  - write_health: payload structure, success vs failure shape, extras

Strategy: pure tmp_path-based filesystem tests. Lockfile tests use
subprocess to acquire the lock from a separate process (so we can
test the busy case without trying to acquire from within the same
process - fcntl.flock on Linux/macOS is per-process, so a single
process can re-acquire its own lock without blocking).

This module is production-critical (it's the safety layer that prevents
concurrent corruption), so tests prioritize: failure modes, edge
cases, and the "what does the filesystem look like after each call"
question. Not pure happy-path coverage.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from upset_propagation.cron_utils import (
    LOCK_FILENAME,
    HEALTH_FILENAME,
    LockBusyError,
    atomic_output_dir,
    force_unlock,
    lockfile_acquired,
    write_health,
)


# Lockfile tests


def test_lockfile_acquired_releases_on_normal_exit(tmp_path):
    """After exiting the context, the lock can be re-acquired.

    Note: fcntl.flock is per-PID, so within ONE process the same PID can
    re-acquire its own lock. The test here is that the file handle is
    properly released so a re-entry doesn't see leftover state.
    """
    lock_path = tmp_path / "test.lock"
    with lockfile_acquired(lock_path):
        # We're inside the lock; file should exist and contain our PID
        assert lock_path.exists()
        content = lock_path.read_text()
        assert str(os.getpid()) in content
    # After exit, file still exists but is truncated (lock released)
    assert lock_path.exists()
    # Should be safely re-acquirable
    with lockfile_acquired(lock_path):
        pass  # just verify no exception


def test_lockfile_acquired_releases_on_exception(tmp_path):
    """If the with-body raises, the lock is still released."""
    lock_path = tmp_path / "test.lock"
    with pytest.raises(RuntimeError, match="test bang"):
        with lockfile_acquired(lock_path):
            raise RuntimeError("test bang")
    # Re-acquire should still work - lock was released even though body raised
    with lockfile_acquired(lock_path):
        pass


def test_lockfile_writes_pid_for_diagnostics(tmp_path):
    """The lockfile body includes our PID (so force_unlock can check)."""
    lock_path = tmp_path / "test.lock"
    with lockfile_acquired(lock_path):
        content = lock_path.read_text().strip()
        # First whitespace-separated token should be our PID
        first_token = content.split()[0]
        assert int(first_token) == os.getpid()


@pytest.mark.slow
def test_lockfile_blocks_concurrent_process(tmp_path):
    """If another process holds the lock, LockBusyError is raised.

    Uses subprocess to acquire the lock from a different PID (within a
    single process, fcntl.flock would let us re-acquire our own lock).
    """
    lock_path = tmp_path / "test.lock"

    # Helper script: acquire the lock, sleep, then exit cleanly. While the
    # subprocess is sleeping, the lock is held by ITS PID, not ours.
    helper = textwrap.dedent(f"""
        import sys
        import time
        sys.path.insert(0, {str(Path(__file__).resolve().parent.parent / "src")!r})
        from pathlib import Path
        from upset_propagation.cron_utils import lockfile_acquired
        with lockfile_acquired(Path({str(lock_path)!r})):
            # Tell parent we've acquired
            print("acquired", flush=True)
            time.sleep(2.0)
    """)
    proc = subprocess.Popen(
        [sys.executable, "-c", helper],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        # Wait until the subprocess confirms it has the lock
        line = proc.stdout.readline().strip()
        assert line == "acquired", f"helper failed to acquire: {line!r}"

        # Now WE try to acquire from this process - should fail
        with pytest.raises(LockBusyError) as exc_info:
            with lockfile_acquired(lock_path):
                pass  # should never reach here

        # The error should know which PID is holding
        assert exc_info.value.holding_pid is not None
        assert exc_info.value.holding_pid == proc.pid
    finally:
        proc.wait(timeout=5)

    # After the subprocess exits, the lock is released and we can acquire
    with lockfile_acquired(lock_path):
        pass


# force_unlock tests


def test_force_unlock_returns_false_when_no_lock(tmp_path):
    """No lockfile → returns False, no exception."""
    lock_path = tmp_path / "test.lock"
    assert lock_path.exists() is False
    cleared = force_unlock(lock_path)
    assert cleared is False


def test_force_unlock_clears_dead_pid(tmp_path):
    """If lockfile holds a PID that doesn't exist, force_unlock clears it."""
    lock_path = tmp_path / "test.lock"
    # Write a PID that's almost certainly dead (>10000 below typical PIDs)
    # Use a very high invalid PID instead, since os.kill returns ESRCH cleanly.
    fake_pid = 99999
    lock_path.write_text(f"{fake_pid} 2026-01-01T00:00:00+00:00\n")

    # First verify the fake PID really isn't a live process
    try:
        os.kill(fake_pid, 0)
        pytest.skip(f"PID {fake_pid} happens to be alive on this machine; skip")
    except ProcessLookupError:
        pass

    cleared = force_unlock(lock_path, only_if_stale=True)
    assert cleared is True
    assert not lock_path.exists()


def test_force_unlock_refuses_live_pid(tmp_path):
    """If the PID in lockfile is alive, force_unlock refuses to clear."""
    lock_path = tmp_path / "test.lock"
    # Our own PID is definitely alive
    lock_path.write_text(f"{os.getpid()} 2026-01-01T00:00:00+00:00\n")

    cleared = force_unlock(lock_path, only_if_stale=True)
    assert cleared is False
    # File still exists
    assert lock_path.exists()


def test_force_unlock_dangerous_bypasses_pid_check(tmp_path):
    """only_if_stale=False clears even when PID is alive."""
    lock_path = tmp_path / "test.lock"
    lock_path.write_text(f"{os.getpid()} 2026-01-01T00:00:00+00:00\n")

    cleared = force_unlock(lock_path, only_if_stale=False)
    assert cleared is True
    assert not lock_path.exists()


def test_force_unlock_handles_empty_lockfile(tmp_path):
    """Empty lockfile (zero bytes) is treated as stale - safe to clear."""
    lock_path = tmp_path / "test.lock"
    lock_path.write_text("")
    cleared = force_unlock(lock_path, only_if_stale=True)
    assert cleared is True
    assert not lock_path.exists()


def test_force_unlock_refuses_malformed_lockfile(tmp_path):
    """Unparseable lockfile content → refuse to clear (conservative)."""
    lock_path = tmp_path / "test.lock"
    lock_path.write_text("not-a-pid\n")
    cleared = force_unlock(lock_path, only_if_stale=True)
    # Refuses because we can't parse a PID to check
    assert cleared is False
    assert lock_path.exists()


# atomic_output_dir tests


def test_atomic_writes_to_pending_then_swaps(tmp_path):
    """On clean exit, target_dir contains the files we wrote during the context."""
    target = tmp_path / "output"
    target.mkdir()
    # Pre-existing file in target that should get replaced
    (target / "old.json").write_text('{"old": true}')

    with atomic_output_dir(target) as staging:
        # We're writing to staging, NOT to target yet
        assert staging != target
        assert staging.name.endswith(".pending")
        (staging / "new.json").write_text('{"new": true}')
        # target_dir is still untouched
        assert (target / "old.json").exists()
        assert not (target / "new.json").exists()

    # After context exits: target has new file, old file is gone
    assert (target / "new.json").exists()
    assert not (target / "old.json").exists()
    # Pending dir cleaned up
    pending = tmp_path / "output.pending"
    assert not pending.exists()


def test_atomic_preserves_target_on_exception(tmp_path):
    """If body raises, target_dir is untouched and pending stays for forensics."""
    target = tmp_path / "output"
    target.mkdir()
    (target / "important.json").write_text('{"keep": "me"}')

    with pytest.raises(RuntimeError, match="boom"):
        with atomic_output_dir(target) as staging:
            (staging / "partial.json").write_text('{"partial": true}')
            raise RuntimeError("boom")

    # Original file still there
    assert (target / "important.json").exists()
    assert json.loads((target / "important.json").read_text()) == {"keep": "me"}
    # And the partial write IS preserved in pending/ for forensics
    pending = tmp_path / "output.pending"
    assert pending.exists()
    assert (pending / "partial.json").exists()


def test_atomic_overwrites_existing_pending(tmp_path):
    """Leftover output.pending/ from a previous crashed run is wiped clean."""
    target = tmp_path / "output"
    target.mkdir()
    pending = tmp_path / "output.pending"
    pending.mkdir()
    (pending / "old_partial.json").write_text('{"leftover": true}')

    with atomic_output_dir(target) as staging:
        # The old partial file should be gone - staging dir is fresh
        assert not (staging / "old_partial.json").exists()
        (staging / "fresh.json").write_text('{"fresh": true}')

    # After swap: target has only the fresh file
    assert (target / "fresh.json").exists()
    assert not (target / "old_partial.json").exists()


def test_atomic_works_when_target_does_not_exist(tmp_path):
    """First-ever run: target_dir doesn't exist yet."""
    target = tmp_path / "output"  # not created
    assert not target.exists()

    with atomic_output_dir(target) as staging:
        (staging / "first.json").write_text('{"first": true}')

    assert target.exists()
    assert (target / "first.json").exists()


# write_health tests


def test_write_health_success_payload(tmp_path):
    """Success path: all the standard fields are present."""
    path = write_health(
        tmp_path,
        exit_status="success",
        duration_sec=247.5,
        n_scenarios=79,
        calibration_max_residual=0.00018,
        validation_pass=True,
    )
    assert path == tmp_path / HEALTH_FILENAME
    payload = json.loads(path.read_text())
    assert payload["exit_status"] == "success"
    assert payload["duration_sec"] == 247.5
    assert payload["n_scenarios"] == 79
    assert payload["calibration_max_residual"] == 0.000180
    assert payload["validation_pass"] is True
    # last_run_utc auto-populated with a parseable timestamp
    assert "last_run_utc" in payload
    assert "T" in payload["last_run_utc"]  # ISO 8601 marker


def test_write_health_failure_payload(tmp_path):
    """Failure path: exit_reason included, success-only fields omitted."""
    path = write_health(
        tmp_path,
        exit_status="failure",
        duration_sec=12.4,
        exit_reason="ConnectionError: API timed out",
    )
    payload = json.loads(path.read_text())
    assert payload["exit_status"] == "failure"
    assert payload["exit_reason"] == "ConnectionError: API timed out"
    # Success-only fields should be absent (or at least, not populated by
    # default in failure mode)
    assert "n_scenarios" not in payload
    assert "calibration_max_residual" not in payload


def test_write_health_extras_merged(tmp_path):
    """`extra` dict is merged into the payload (used for skipped-run info)."""
    path = write_health(
        tmp_path,
        exit_status="success",
        duration_sec=0.5,
        extra={"skipped": True, "skip_reason": "fresh"},
    )
    payload = json.loads(path.read_text())
    assert payload["skipped"] is True
    assert payload["skip_reason"] == "fresh"


def test_write_health_creates_output_dir(tmp_path):
    """If output_dir doesn't exist, it's created."""
    target = tmp_path / "newly_created" / "output"
    assert not target.exists()

    path = write_health(target, exit_status="success", duration_sec=1.0)
    assert target.exists()
    assert path.exists()