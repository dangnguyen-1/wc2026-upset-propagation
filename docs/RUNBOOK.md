# Runbook

What to do when something goes wrong. Each section is a symptom →
diagnosis → fix flow. The goal is "your alert just fired and it's 3am,
what command do I run?" not "how does the framework work
conceptually."

For setup and cron configuration, see `DEPLOYMENT.md`. For the data
format consumers read, see `OUTPUT_FORMAT.md`. For the journey of why
v3 looks the way it does, see the v3 working doc.

---

## Quick reference: health check + exit codes

The single command operators run most:

```bash
python -m upset_propagation.health
echo $?
```

Exit codes (each maps to a section below):

| Code | Verdict   | What it means                                    | Action       |
|------|-----------|--------------------------------------------------|--------------|
| 0    | HEALTHY   | All checks passed                                | None         |
| 1    | STALE     | Last run is older than `--max-age-hours`         | See §1       |
| 2    | FAILURE   | Last run reported `exit_status: failure`         | See §2       |
| 3    | MISSING   | `health.json` or required output file is absent  | See §3       |
| 4    | DEGRADED  | Succeeded but validation failed or residual bad  | See §4       |

For cron monitoring, wire the exit code into your alert system:

```bash
*/15 * * * * cd /opt/wc2026-upset-propagation && /opt/wc2026-upset-propagation/.venv/bin/python -m upset_propagation.health --quiet || /usr/local/bin/alert-script
```

---

## §1 - STALE (exit 1): cron hasn't run recently

**Symptom**: `python -m upset_propagation.health` exits 1 with a
message like *"Last run was 5.2h ago, exceeds threshold 3.0h."*

The framework itself is fine; it's the **cron scheduler** that hasn't
fired it.

### Diagnosis checklist

1. **Is the cron daemon running?**
   ```bash
   systemctl status cron       # Linux
   sudo launchctl list | grep cron   # macOS
   ```

2. **Is our entry still in crontab?**
   ```bash
   crontab -l | grep wc2026-upset-propagation
   ```

3. **Was there a recent run attempt?** Check the cron log directory:
   ```bash
   ls -lt /opt/wc2026-upset-propagation/output/logs/ | head -5
   ```
   If the newest log is older than expected, cron tried but exited
   abnormally - read that log first.

4. **Is the lockfile stuck?** See §5 ("stuck lockfile") if so.

5. **Is the host out of disk?**
   ```bash
   df -h /opt/wc2026-upset-propagation
   ```
   Atomic rename + snapshot retention can quietly fill the disk over
   weeks if `--no-snapshot` isn't set.

### Fix

Most common cause is a stuck lockfile from a previous crashed run.
After confirming no live cron process is running:

```bash
python -m upset_propagation.run --force-unlock
# Then re-run manually to verify recovery
python -m upset_propagation.run --cron-mode
```

If the cause was disk-full, free space first. The atomic-rename
machinery leaves `output.old.<ts>/` directories behind if cleanup
itself fails - those can be safely deleted:

```bash
rm -rf /opt/wc2026-upset-propagation/output.old.*
```

---

## §2 - FAILURE (exit 2): last run reported failure

**Symptom**: health command shows `"Last run reported failure:
<exit_reason>"`.

The exit_reason is the actionable detail. Common cases:

### `ConnectionError` / `Timeout` (FairLine API down)

The framework's API client has built-in retry (3 attempts, exponential
backoff). If those all failed, FairLine is either down or unreachable.

```bash
# Verify yourself
curl -s -o /dev/null -w "%{http_code}\n" \
    https://seal-app-yatxw.ondigitalocean.app/api/wc2026/fair_probs
```

If 200, retry the run; the API may have recovered:

```bash
python -m upset_propagation.run --cron-mode
```

If non-200 or timeout, nothing for us to do - wait it out. The next
scheduled cron will retry automatically.

### `InputValidationError`

Pre-calibration input validation rejected the API response. Look at
the exception message:

```bash
# Latest log file has the full traceback
ls -t output/logs/ | head -1 | xargs -I{} tail -100 output/logs/{}
```

Failure codes you might see (from `input_validation.py`):

- `team_count_wrong` - API returned ≠ 48 teams. Possibly an API
  schema change. Stop and investigate; don't blindly retry.
- `sum_out_of_range_fatal` - fair_probs sum > 1.10 or < 0.90.
  Suggests an API parser bug or a malformed upstream response.
- `fair_prob_zero` - some team has exactly 0%. API likely missed a
  team in its bookmaker aggregation. Stop and report to FairLine.
- `max_prob_too_high_fatal` - top team > 50%. Implausible.
  Same as above - investigate, don't proceed.
- `elo_team_missing` - a WC2026 team has no Elo in
  `elo_history.csv`. Most likely cause: stale Elo file. Pull latest:
  ```bash
  git pull --rebase
  ```
- `elo_not_finite` / `elo_negative` - corrupt Elo data. Investigate
  `data/mc_simu/elo_history.csv` for the offending row.

### `CalibrationError` (Nelder-Mead didn't converge)

Rare. Default `--max-iter 2000` is enough for normal data. If it
fails, try more iterations:

```bash
python -m upset_propagation.run --cron-mode --max-iter 5000
```

If that also fails, the input fair_probs are likely degenerate in a
way calibration can't recover from. Read the latest log, check the
final loss reported by the calibrator. Calibration with loss > 10x
typical (typical ~1e-4) means the predictor can't fit the market -
escalate.

### Anything else

Read the full traceback in the latest log file. The framework's
exception handler captures the exception chain - the root cause is
usually the deepest `Caused by:` line.

```bash
less $(ls -t output/logs/*.log | head -1)
```

---

## §3 - MISSING (exit 3): output files missing

**Symptom**: health shows `"health.json missing"` or `"N required
output file(s) missing"`.

### If `health.json` is missing

Either the cron has never run with `--cron-mode`, or `output/` was
manually deleted. Bootstrap:

```bash
python -m upset_propagation.run --cron-mode
```

This takes ~4 minutes (calibration). The initial run writes all
output files.

### If `health.json` is present but consumer files are missing

This is the failure mode item 10 was designed to catch - health says
success but `index.json` / `baseline.json` / `validation_report.json`
isn't there. Most likely cause: manual deletion or partial file
corruption.

Verify, then re-run:

```bash
ls -la output/*.json | head -20
python -m upset_propagation.run --cron-mode
```

If the missing-file pattern persists across runs, something is
deleting them between cron cycles - investigate that, not the
framework.

---

## §4 - DEGRADED (exit 4): succeeded but data quality is degraded

**Symptom**: the run completed and exited "success" but one of:
- `validation_pass: false`, or
- `calibration_max_residual` > tolerance (default 0.005)

This is a soft alert. The framework is producing output the operator may still
want to read, but caveats apply.

### Validation failed

Read `output/validation_report.json` to find the specific check that
fired:

```bash
cat output/validation_report.json | python -m json.tool | less
```

Common failures:

- **Calibration didn't converge to baseline within tolerance.** Try
  `--max-iter 5000` (see §2).
- **Sensitivity check regressed.** A perturbation to top-team Elo
  ratings now produces large swings in `Win` probabilities - the
  framework is more brittle than acceptable. Read the report; this
  may indicate a real change in the input data shape.

### Calibration residual too high

The calibrator completed iterations but the propagator's output
doesn't match the market closely enough. Try:

```bash
python -m upset_propagation.run --cron-mode --max-iter 5000
```

If still degraded with `max_residual` above 0.01 (2x tolerance), the
calibrated predictor is unreliable. The signal in `our_vs_thy.json`
should be treated with extra skepticism until the next clean run.

---

## §5 - Stuck lockfile

**Symptom**: cron runs are failing with `LockBusyError`, or health
shows STALE for many cycles in a row, and the lockfile shows an old
PID.

```bash
# Inspect the lockfile
cat output/.cron.lock
# Format: "<pid> <iso_timestamp>"
```

### If the PID is dead (no live process)

Safe to clear. The framework checks PID liveness before clearing:

```bash
python -m upset_propagation.run --force-unlock
```

The command refuses to clear if the PID is still alive - safety guard.

### If the PID is alive but you're sure it's not us

This is the dangerous case - usually means another script grabbed our
PID number after our process died. Confirm visually first:

```bash
ps -p $(awk '{print $1}' output/.cron.lock)
```

If the process shown is **not** a `python ... upset_propagation.run`
invocation, bypass the safety check:

```bash
python -m upset_propagation.run --force-unlock-dangerous
```

If the process IS another `upset_propagation.run`, you have a real
concurrency problem - investigate cron / systemd before clearing.

---

## §6 - Forensics: investigating a specific past run

Each cron run writes a snapshot to `output/runs/<UTC-timestamp>/`. To
investigate "what happened on June 18 at 14:00 UTC":

```bash
ls output/runs/ | grep 2026-06-18
# Find the right directory, then inspect
ls output/runs/2026-06-18T14-00-00Z/
cat output/runs/2026-06-18T14-00-00Z/validation_report.json
cat output/runs/2026-06-18T14-00-00Z/api_snapshot.json
```

The snapshot contains everything `output/` had at that moment -
useful for "the framework said X yesterday but Y today; what
changed?" type questions.

### Re-running with a historical snapshot

To reproduce a past run's output without hitting the API (e.g. to
debug why the matcher produced a specific ranking):

```bash
python -m upset_propagation.run \
    --from-snapshot output/runs/2026-06-18T14-00-00Z/api_snapshot.json
```

---

## §7 - Atomic rename forensics: `output.pending/` and `output.old.*/`

These directories normally don't exist between runs. If you see them,
something is up.

### `output.pending/` exists

A run crashed mid-write. The atomic-rename machinery preserved the
partial work for forensics. Inspect what got written:

```bash
ls output.pending/
diff -rq output/ output.pending/  # see what differs from current output/
```

When done investigating, remove it (the next clean run will start
fresh):

```bash
rm -rf output.pending/
```

### `output.old.<ts>/` exists

A previous run successfully swapped output, but cleanup of the old
contents itself failed. The framework moved the old output here for
manual cleanup. Safe to delete:

```bash
rm -rf output.old.*
```

---

## §8 - Rolling back

If a deployment broke something and you need to revert quickly:

```bash
cd /opt/wc2026-upset-propagation
git log --oneline -10              # find the last known-good commit
git checkout <known-good-sha>
python -m upset_propagation.run --cron-mode    # bootstrap fresh outputs
```

The framework is stateless across versions (each run produces fresh
output from the API + Elo + groups). Reverting code and re-running is
safe.

For the production deployment to use a different branch:

```bash
git checkout main      # or whichever branch is current production
git pull --rebase
```

---

## §9 - When to wake the operator up

Not every alert is wake-up worthy. Severity guide:

| Situation                              | Wake up? |
|----------------------------------------|----------|
| Single STALE alert                     | No       |
| STALE for >6 hours during tournament   | Yes      |
| FAILURE with ConnectionError           | No (API will recover) |
| FAILURE with InputValidationError      | Yes (something upstream changed) |
| MISSING (manual deletion)              | Probably (someone touched the box) |
| DEGRADED with validation regression    | Yes (data quality is suspect) |
| DEGRADED with calibration residual     | No (rerun + escalate if persists) |
| Disk full / host issues                | Whatever your normal infra escalation is |

The framework is non-critical infrastructure: a few hours of stale
output won't directly cost money. Wake the operator for things that suggest
the *signal itself* is wrong, not for things that suggest the cron
ran a bit late.