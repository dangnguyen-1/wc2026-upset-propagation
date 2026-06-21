# Deployment Guide

How to run `wc2026-upset-propagation` as a production cron job during
the World Cup. Covers install, scheduling, monitoring, troubleshooting,
and updates.

## Prerequisites

- **Python 3.11 or later**
- **~500 MB free disk space** for outputs + 30 days of historical
  snapshots (with `--no-snapshot`, ~5 MB suffices)
- **Network access** to the FairLine API
  (`https://seal-app-yatxw.ondigitalocean.app`)
- **A non-root user account** to run the framework (don't run as root)

Tested platforms: macOS (development), Ubuntu 24.04 (production
template). Should work on any POSIX-ish system; `fcntl.flock` is the
only platform-specific dependency.

## One-time install

On a fresh box:

```bash
# 1. Pick a service directory. Convention: /opt/wc2026-upset-propagation/
sudo mkdir -p /opt/wc2026-upset-propagation
sudo chown $USER:$USER /opt/wc2026-upset-propagation
cd /opt/wc2026-upset-propagation

# 2. Clone the repo and check out the production branch (main).
git clone https://github.com/dangnguyen-1/wc2026-upset-propagation.git .
git checkout main

# 3. Create a virtualenv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# 4. Sanity check - should print 'imports OK' and pass all 101 tests
python -c "from upset_propagation.scenarios import build_all_scenarios; print('imports OK')"
pytest tests/ -q

# 5. Initial calibration (takes ~4 minutes)
python -m upset_propagation.run --cron-mode
```

After the initial run, `output/` should contain ~80 JSON files, an
`index.json`, a `validation_report.json`, and a fresh `health.json`.

## Cron setup

The framework recalibrates against the FairLine model's latest fair_probs every 2 hours
during the tournament. Use this crontab entry:

```
# /etc/cron.d/wc2026-upset-propagation or `crontab -e` as your service user
#
# Every 2 hours: recalibrate + propagate + validate + emit health.
# --skip-if-fresh 1.5 short-circuits if outputs are newer than 90 min
# (catches the case where a previous run took >2h, avoids overlap).
0 */2 * * *   cd /opt/wc2026-upset-propagation && /opt/wc2026-upset-propagation/.venv/bin/python -m upset_propagation.run --cron-mode --skip-if-fresh 1.5
```

**Why `--cron-mode`:**

- Structured log output to `output/logs/run-{ts}.log` instead of stdout
- Only WARN+ escapes to stderr (cron-email picks up real problems)
- Exclusive lockfile prevents overlapping runs from corrupting outputs
- Atomic output swap: a partial-write goes to `output.pending/`, not `output/`
- Writes `output/health.json` for external monitoring

**Why `--skip-if-fresh 1.5`:**

If a previous run is delayed (network issue, API timeout retry), the
next 2-hour cron tick should not pile on top. The 1.5h threshold means
"if outputs are <1.5h old, this scheduled run does nothing." Saves
calibration time and prevents wasted compute.

## Log rotation

The framework writes one log file per run to `output/logs/`. Without
rotation, this grows by ~50 KB per run × 12 runs/day × 30 days = ~18 MB
over a month. Manageable but worth rotating for cleanliness.

Sample `logrotate` config at `/etc/logrotate.d/wc2026-upset-propagation`:

```
/opt/wc2026-upset-propagation/output/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
```

On macOS (no logrotate), a simple cron entry suffices:

```
# Delete logs older than 30 days, run daily at 4am
0 4 * * *   find /opt/wc2026-upset-propagation/output/logs -name "*.log" -mtime +30 -delete
```

## Monitoring

### `output/health.json`

Updated at the end of every cron run. Schema:

```json
{
  "last_run_utc": "2026-06-15T14:30:00+00:00",
  "duration_sec": 247.83,
  "exit_status": "success",
  "n_scenarios": 79,
  "calibration_max_residual": 0.00018,
  "validation_pass": true
}
```

Or, on failure:

```json
{
  "last_run_utc": "2026-06-15T14:31:12+00:00",
  "duration_sec": 12.4,
  "exit_status": "failure",
  "exit_reason": "ConnectionError: HTTPSConnectionPool host='seal-app-yatxw.ondigitalocean.app' Max retries exceeded"
}
```

External monitoring should:

1. Check `last_run_utc` is recent (<3h ago, allowing for the 2h cron cadence + slack)
2. Check `exit_status == "success"`
3. Check `validation_pass == true` (if success)
4. Check `calibration_max_residual < 0.005` (loose tolerance - tightens if the operator adjusts)

The `python -m upset_propagation.health` command will provide a
canned consumer for the above. Until then, parse `health.json` directly.

### Cron email

Stderr from `--cron-mode` triggers cron's email-on-output behavior. The
framework only emits stderr on WARN+ events (validation regressions,
calibration warnings, lock-busy errors, exceptions). A successful run
is silent on stderr. So any cron email is a real signal.

To enable cron email, set the `MAILTO=` env var in your crontab:

```
MAILTO=ops-alerts@example.com
0 */2 * * *   cd /opt/wc2026-upset-propagation && ...
```

## Operator procedures

### Lockfile is stuck

Symptom: cron runs fail with `Lock busy: Lock at output/.lock is held by
PID N. Another upset_propagation.run is in progress, or a previous run
crashed without releasing the lock.`

Diagnostic:

```bash
cd /opt/wc2026-upset-propagation
cat output/.lock          # shows holding PID + start time
ps -p $(awk '{print $1}' output/.lock)  # is the PID alive?
```

If the PID is dead (most likely cause - previous run crashed):

```bash
# Clears stale lock if PID is confirmed dead; refuses if alive
.venv/bin/python -m upset_propagation.run --force-unlock
```

If the PID is alive but the run looks hung (e.g. >30 minutes when typical
is 4):

```bash
# Investigate first - what's the process doing?
sudo cat /proc/$(awk '{print $1}' output/.lock)/status

# If you decide to kill it:
kill $(awk '{print $1}' output/.lock)
# Then verify it's gone, then clear the lock:
.venv/bin/python -m upset_propagation.run --force-unlock
```

**Last-resort, dangerous override** (only if you're certain no run is in
progress):

```bash
.venv/bin/python -m upset_propagation.run --force-unlock-dangerous
```

### Force a fresh run

`--skip-if-fresh` is skipped when omitted. To force-recompute even with
fresh outputs:

```bash
.venv/bin/python -m upset_propagation.run --cron-mode
```

### Inspect the last run's logs

```bash
ls -lt output/logs/ | head -5
tail -100 output/logs/$(ls -t output/logs/ | head -1)
```

### Outputs corrupted / partial state

The atomic-output design prevents this in normal failure modes. If you
ever see it (e.g. one of the scenario JSONs is empty or malformed),
the framework's `output.pending/` should contain the in-progress
attempt. Check it:

```bash
ls output.pending/  # forensic snapshot of the failed run's writes
```

Then clear and re-run:

```bash
rm -rf output.pending/
.venv/bin/python -m upset_propagation.run --cron-mode
```

If the corruption is in `output/` itself (somehow), restore from the
most recent snapshot:

```bash
ls -lt output/runs/ | head -3
# pick a recent one, e.g. 20260615-103022/
cp output/runs/20260615-103022/*.json output/
```

## Updating the vendored vendored code

If the vendored simulator ships an algorithmic change to the MC simulator, follow the
procedure in `docs/VENDORING.md`. **Schedule the re-vendor during a
maintenance window** - it requires:

1. Disabling the cron entry temporarily
2. Running the re-vendor process (~30 min including verification)
3. Confirming `pytest tests/`, `python -m upset_propagation.state_matcher`,
   and a manual `python -m upset_propagation.run` all pass cleanly
4. Re-enabling the cron

Don't do this during a critical match window - calibration drift could
affect numbers consumers are actively reading.

## Updating the framework itself

For framework code changes (new matchers, bug fixes):

```bash
cd /opt/wc2026-upset-propagation

# 1. Pause cron - disable the entry temporarily, or comment it out
# 2. Pull the new version
git pull origin main
.venv/bin/pip install -e .  # re-install if dependencies changed

# 3. Run tests
.venv/bin/pytest tests/ -q

# 4. Smoke test the new code with a forced run
.venv/bin/python -m upset_propagation.run --cron-mode

# 5. Verify outputs look sane
cat output/health.json

# 6. Re-enable cron
```

## Troubleshooting reference

| Symptom | Likely cause | Fix |
|---|---|---|
| Lock busy on every run | Previous crash | `--force-unlock` |
| All runs fail with `ConnectionError` | FairLine API down | Wait; retry; check `https://seal-app-yatxw.ondigitalocean.app/healthz` |
| `max_residual` slowly drifting up | Calibration converging poorly | Increase `--max-iter` (default 3000 → try 5000) |
| Validation fails repeatedly | Real signal regression OR a vendoring drift | Read `validation_report.json` |
| `output.pending/` keeps appearing | Atomic swap consistently failing | Check disk space; check filesystem permissions |
| Stale outputs (`last_run_utc` very old) | Cron not firing OR every run is locked | Check `crontab -l`; check `output/.lock` |
| Disk filling up | Snapshots accumulating | Remove old `output/runs/<ts>/` directories OR add `--no-snapshot` to cron |