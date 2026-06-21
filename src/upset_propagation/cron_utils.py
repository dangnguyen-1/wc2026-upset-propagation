"""Cron-deployment utilities.

Three concerns for running the framework as a scheduled cron job every
2 hours:

  1. Concurrency safety: prevent overlapping runs that would corrupt
     outputs. Use a POSIX file lock (fcntl.flock) at output/.lock.

  2. Output atomicity: a partial-write during failure leaves consumers
     reading inconsistent data. Use a staging directory (output/pending/)
     and atomic rename on success.

  3. Health observability: external monitoring needs a cheap way to
     check liveness. Write output/health.json at the end of every run.

These are framework-internal mechanics - DEPLOYMENT.md covers the ops-
facing concerns (crontab entries, log rotation, monitoring setup).

Public API:
    LockBusyError exception
    lockfile_acquired(lock_path) context manager
    force_unlock(lock_path, *, only_if_stale) → bool
    atomic_output_dir(target_dir) context manager
    write_health(output_dir, **fields) → None
"""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


logger = logging.getLogger(__name__)


LOCK_FILENAME = ".lock"
HEALTH_FILENAME = "health.json"
PENDING_SUFFIX = ".pending"
OLD_SUFFIX = ".old"


# Lockfile


class LockBusyError(RuntimeError):
    """Another process holds the framework's output lock.

    Carries the PID of the holding process if we can read it from the
    lockfile (best-effort - not guaranteed if the lockfile was written
    partially or by a non-Python user).
    """
    def __init__(self, lock_path: Path, holding_pid: Optional[int] = None):
        self.lock_path = lock_path
        self.holding_pid = holding_pid
        msg = f"Lock at {lock_path} is held"
        if holding_pid is not None:
            msg += f" by PID {holding_pid}"
        msg += ". Another upset_propagation.run is in progress, or a previous run crashed without releasing the lock."
        super().__init__(msg)


@contextmanager
def lockfile_acquired(lock_path: Path):
    """Acquire an exclusive non-blocking lock on `lock_path`.

    Writes our PID to the file body for diagnostic visibility. Releases
    the lock on context exit (success or exception).

    Implementation: fcntl.flock with LOCK_EX | LOCK_NB. Cross-process,
    POSIX-portable (works on macOS dev and Linux production). The kernel
    automatically releases the lock when the file descriptor closes
    (so a crashed process can't keep us locked out forever - the
    holding_pid in LockBusyError is informational only).

    Raises LockBusyError if another process holds the lock.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Open in read-write mode (creating if needed). r+ would fail if file
    # doesn't exist; w would truncate any holder's PID on every open.
    # 'a+' appends-creates, and we can seek+truncate after acquiring.
    fd = open(lock_path, "a+")
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            # Try to read the holder's PID (best-effort)
            holding_pid: Optional[int] = None
            try:
                fd.seek(0)
                content = fd.read().strip()
                if content:
                    holding_pid = int(content.split()[0])
            except (ValueError, OSError):
                pass
            fd.close()
            raise LockBusyError(lock_path, holding_pid) from exc

        # We hold the lock. Write our PID + start time for diagnostics.
        fd.seek(0)
        fd.truncate()
        fd.write(f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}\n")
        fd.flush()
        os.fsync(fd.fileno())

        try:
            yield
        finally:
            # Releasing the flock is implicit on close, but be explicit
            # for clarity. Also truncate the file so the PID doesn't
            # mislead next reader after release.
            try:
                fd.seek(0)
                fd.truncate()
            except OSError:
                pass
            fcntl.flock(fd, fcntl.LOCK_UN)
            fd.close()
    except Exception:
        try:
            fd.close()
        except Exception:
            pass
        raise


def force_unlock(lock_path: Path, *, only_if_stale: bool = True) -> bool:
    """Manually clear a lockfile.

    For operator use when a previous run crashed and the lockfile points
    at a now-dead PID. Returns True if the lock was cleared, False if
    not (lock didn't exist, or `only_if_stale=True` and the PID is alive).

    Args:
        lock_path: path to the lockfile
        only_if_stale: if True, only delete if the holding PID is dead.
            Set False to force-delete regardless (dangerous - will
            silently corrupt an active run if used carelessly).

    Process-liveness check uses os.kill(pid, 0), which raises ProcessLookupError
    if the PID is dead. We catch and treat as stale.
    """
    if not lock_path.exists():
        logger.info(f"No lockfile at {lock_path}; nothing to unlock.")
        return False

    if only_if_stale:
        try:
            with lock_path.open() as f:
                content = f.read().strip()
            if not content:
                # Empty lockfile - safe to delete
                lock_path.unlink()
                logger.info(f"Cleared empty lockfile at {lock_path}.")
                return True
            holding_pid = int(content.split()[0])
        except (ValueError, OSError) as exc:
            logger.warning(
                f"Could not parse lockfile {lock_path} ({exc}); refusing to delete. "
                f"Use only_if_stale=False to force."
            )
            return False

        try:
            os.kill(holding_pid, 0)
            # PID is alive → not stale
            logger.warning(
                f"Lockfile at {lock_path} held by PID {holding_pid}, which is alive. "
                f"Refusing to clear. Use only_if_stale=False to force."
            )
            return False
        except ProcessLookupError:
            # PID is dead → safe to clear
            pass
        except PermissionError:
            # PID exists but we lack permission to signal it. Treat as alive
            # (conservative).
            logger.warning(
                f"PID {holding_pid} exists but we cannot signal it. "
                f"Refusing to clear lockfile."
            )
            return False

    lock_path.unlink()
    logger.info(f"Cleared lockfile at {lock_path}.")
    return True


# Atomic output directory


@contextmanager
def atomic_output_dir(target_dir: Path):
    """Yield a staging directory; atomically swap into target_dir on success.

    Mechanism:
      - Creates target_dir.pending/ (fresh - wipes any leftover staging)
      - Yields it to the caller; caller writes files there
      - On clean exit: renames target_dir.pending/ → target_dir/, after
        moving any existing target_dir/ → target_dir.old.{ts}/ which
        is then removed
      - On exception: leaves target_dir.pending/ in place (forensics),
        target_dir/ untouched; re-raises

    This is "pattern A" from the design discussion - there is a small
    window during the 3 renames where readers see a transient state.
    For our 2-hour cron cadence and the current consumer pattern
    (reads are on-demand, not polling), this is acceptable. A
    symlink-based "pattern B" would close that window but adds
    operational complexity not warranted yet.

    Returns: the staging directory Path.
    """
    target_dir = target_dir.resolve()
    pending_dir = target_dir.with_suffix(target_dir.suffix + PENDING_SUFFIX) \
        if target_dir.suffix else target_dir.parent / (target_dir.name + PENDING_SUFFIX)

    # Wipe any leftover pending from a previous crashed run.
    if pending_dir.exists():
        logger.info(f"Removing leftover staging dir {pending_dir}")
        shutil.rmtree(pending_dir)
    pending_dir.mkdir(parents=True)

    try:
        yield pending_dir
    except Exception:
        # Leave pending in place for forensics; do not touch target_dir.
        logger.error(
            f"Atomic write failed; staging dir preserved at {pending_dir} for forensics. "
            f"Target {target_dir} is untouched."
        )
        raise

    # Successful exit - swap pending into target. Three-step:
    #   1. Move current target_dir aside (if it exists)
    #   2. Rename pending_dir → target_dir
    #   3. Remove the moved-aside copy
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    old_dir = target_dir.with_suffix(
        target_dir.suffix + f"{OLD_SUFFIX}.{timestamp}"
    ) if target_dir.suffix else target_dir.parent / (
        f"{target_dir.name}{OLD_SUFFIX}.{timestamp}"
    )

    if target_dir.exists():
        os.rename(target_dir, old_dir)
    try:
        os.rename(pending_dir, target_dir)
    except Exception:
        # Best-effort rollback: put the old one back.
        if old_dir.exists():
            try:
                os.rename(old_dir, target_dir)
            except OSError:
                pass
        raise

    # Cleanup the .old.{ts} copy. Failure here is non-fatal - leaves a
    # disk-space leak that ops can clean later but doesn't affect
    # correctness.
    if old_dir.exists():
        try:
            shutil.rmtree(old_dir)
        except OSError as exc:
            logger.warning(
                f"Could not clean up {old_dir} after atomic swap: {exc}. "
                f"Safe to remove manually."
            )


# Health JSON


def write_health(
    output_dir: Path,
    *,
    exit_status: str,  # "success" or "failure"
    duration_sec: float,
    n_scenarios: Optional[int] = None,
    calibration_max_residual: Optional[float] = None,
    validation_pass: Optional[bool] = None,
    exit_reason: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    """Write a small JSON file with the latest run's outcome.

    Consumed by external monitoring (item 9 - `python -m upset_propagation.health`
    parses this and exits 0/1 for cron alerting).

    Args:
        output_dir: where to write (typically the framework's output dir,
            so health.json sits alongside the scenarios)
        exit_status: "success" or "failure"
        duration_sec: wall time of the run
        n_scenarios: how many scenario files were produced (success only)
        calibration_max_residual: from CalibrationResult (success only)
        validation_pass: True if validation passed (success only)
        exit_reason: short human-readable reason (failure only)
        extra: any additional fields to merge in

    Returns: path to the written health.json.
    """
    payload: dict[str, Any] = {
        "last_run_utc": datetime.now(timezone.utc).isoformat(),
        "duration_sec": round(duration_sec, 2),
        "exit_status": exit_status,
    }
    if n_scenarios is not None:
        payload["n_scenarios"] = n_scenarios
    if calibration_max_residual is not None:
        payload["calibration_max_residual"] = round(calibration_max_residual, 6)
    if validation_pass is not None:
        payload["validation_pass"] = validation_pass
    if exit_reason is not None:
        payload["exit_reason"] = exit_reason
    if extra:
        payload.update(extra)

    output_dir.mkdir(parents=True, exist_ok=True)
    health_path = output_dir / HEALTH_FILENAME
    health_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Wrote health to {health_path}: {exit_status}")
    return health_path