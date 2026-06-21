# WC2026 Upset Propagation Engine

A trading-signal framework for the 2026 FIFA World Cup. Produces per-team
tournament-winner probabilities conditioned on the realised group-stage
state, then compares them against prediction-market prices to surface
disagreements as potential trade signals.

The framework runs as a 2-hour cron during the tournament, reading from
sportsbook-derived data and prediction-market data, and writing JSON
files that downstream tooling (dashboards, alerts, manual review) can
consume.

## What problem does this solve?

Prediction markets (Polymarket, Kalshi) and sportsbooks (Pinnacle,
Betfair) often price World Cup outright winners differently. A team can
trade at 16% on a prediction market while sportsbook-derived "true" odds
suggest 17% - that's a 1-point edge if the framework agrees with the
sportsbooks.

But a single-number "consensus" estimate isn't enough. As group-stage
matches play out, the conditional probability shifts: if Spain wins
Group H, every team's bracket path changes, and so does its tournament-
winner probability. A static pre-tournament estimate goes stale fast.

This framework precomputes 79 bracket scenarios (1 unconditioned baseline
+ 12 single-deviation + 66 pairwise compound) covering the most-likely
group-stage outcomes. As results come in, the matcher maps the live
state to the closest precomputed scenarios, blends them, and produces a
per-team implied probability. That implied probability is then compared
against the prediction-market price to surface signal.

## How it works

The pipeline runs in five conceptual steps:

1. **Pull baseline.** Fetch sportsbook-derived devigged probabilities
   from the FairLine API (Shin + Power averaging across Pinnacle,
   Betfair, and other books). This is the calibration anchor.

2. **Calibrate.** Tune per-team Elo offsets so that, when the underlying
   match predictor (the vendored Monte Carlo simulator) runs the
   baseline scenario (everyone follows seed), it reproduces the
   sportsbook-derived probabilities. ~4 minutes via Nelder-Mead.

3. **Propagate 79 scenarios.** Each scenario is a hypothesis about how
   group stage plays out (Spain wins Group H, France's group has a
   3-team logjam, etc.). For each, deterministically walk the bracket
   from group standings → R32 → R16 → QF → SF → Final and record per-
   team survival probabilities at each round.

4. **Match the realised state.** As actual results come in (read from
   `output/match_results.csv`), the ensemble matcher scores all 79
   precomputed scenarios against the live state using two similarity
   signals - Hamming distance on group standings, L1 distance on
   propagation outputs - combined via Borda count averaging.

5. **Implied vs market.** Take a weighted average of the matched
   scenarios' tournament-winner probabilities and compare against
   the prediction-market price (Polymarket, devigged). Disagreement
   above 10% relative is flagged as overpriced or underpriced.

Two probability sources serve different purposes in the pipeline:

- **Sportsbook-derived (FairLine `fair-odds` endpoint)** - used for
  calibration and for the matcher's internal baseline construction.
  These are the more refined input.
- **Prediction-market prices (FairLine `prices` endpoint, Polymarket)** -
  used as the comparison surface in `our_vs_market.json`. The trader
  trades against these prices, so they're what matters for signal.

## Outputs

The cron writes a working set to `output/` and a snapshot to
`output/runs/<timestamp>/` each run. Four files matter most:

- **`our_vs_market.json`** - the primary trade signal. Per-team
  comparison: market price, our implied probability (computed at two
  weight exponents, p=4 primary and p=8 sharper-weighting), delta,
  and direction (`overpriced` / `underpriced` / `fair`).

- **`top_10_ranking.json`** - the ensemble matcher's top-10 best-fit
  scenarios for the current state. Diagnostic confidence in the
  signal: if the top 10 are all tightly clustered around one scenario,
  the matcher is confident; if they're spread out, less so.

- **`market_log.jsonl`** - append-only historical record. One entry
  per cron run. The previous market state plus diff plus full
  edge table. Useful for "what was the framework saying last
  Tuesday?" questions.

- **`health.json`** - single-file status check. Last run timestamp,
  success/failure, scenario count, calibration residual, validation
  pass/fail. Read via `python -m upset_propagation.health`.

Full JSON schema is in `docs/OUTPUT_FORMAT.md`.

## Setup

Requires Python 3.11+ and ~500 MB free disk (for historical snapshots;
`--no-snapshot` reduces this to ~5 MB).

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/dangnguyen-1/wc2026-upset-propagation.git
cd wc2026-upset-propagation
python -m venv .venv
source .venv/bin/activate

# 2. Install
pip install -e .

# 3. Verify
python -c "from upset_propagation.scenarios import build_all_scenarios; print('OK')"
pytest tests/ -q   # ~254 tests should pass
```

## Run

```bash
# Bootstrap: full pipeline, ~4 minutes
python -m upset_propagation.run --cron-mode

# Production cron entry (suggested cadence: every 2 hours)
0 */2 * * * cd /opt/wc2026-upset-propagation && .venv/bin/python -m upset_propagation.run --cron-mode --skip-if-fresh 1

# Check health
python -m upset_propagation.health

# Read the trade signal
python -m upset_propagation.our_vs_market --top 10

# Inspect the matcher's confidence
python -m upset_propagation.top_ranking
```

See `docs/DEPLOYMENT.md` for full production setup (lockfile semantics,
log shipping, alerting).

## Layout

```
src/upset_propagation/
  _vendored/               the vendored MC simulator (read-only snapshot)
  baseline.py              FairLine API client (fair-odds + prices)
  calibrator.py            Nelder-Mead Elo offset tuning
  config.py                Constants and endpoints
  cron_utils.py            Atomic rename + lockfile + health.json
  ensemble_matcher.py      Hamming + L1 → Borda count
  health.py                Monitoring CLI
  implied_probs.py         Weighted average across matched scenarios
  input_validation.py      Pre-calibration input validation
  l1_matcher.py            L1 propagation similarity
  logging_config.py        Structured logging setup
  market_log.py            Append-only historical log + edge computation
  match_results.py         FIFA tiebreaker wrapper for group standings
  our_vs_market.py         Comparison view + CLI
  propagator.py            Bracket walk: group standings → per-round probs
  run.py                   Top-level cron orchestrator
  scenarios.py             The 79-scenario library
  state_matcher.py         Hamming matcher: group-standings similarity
  top_ranking.py           Top-10 matcher diagnostic snapshot
  validation.py            Self-check on calibration + sensitivity

data/                      Reference data (groups, Elo history, FIFA seeding)
output/                    Generated files (gitignored)
  runs/<timestamp>/        Historical snapshots
tests/                     ~254 unit tests
docs/                      Deployment, output format, runbook, vendoring
```

## Limitations

- **Group-stage signal only.** The 79-scenario library is built from
  group-stage outcomes. Once R32 begins (around June 28, 2026), the
  framework's view becomes a pre-knockout snapshot rather than a live
  reflection of knockout outcomes. Knockout-stage trading requires a
  separate model - not yet built.

- **Tournament-Winner column only.** The framework computes per-team
  probabilities for all 6 knockout rounds (R32 / R16 / QF / SF / F /
  Win), but only the Win column is calibrated and surfaced in the
  trade-signal output. Other rounds are unverified and not used.

- **No sizing, no liquidity, no execution.** The framework produces
  signal (disagreement between our model and the market). It does not
  recommend position size, model order-book depth, or generate fillable
  orders. Those are the trader's domain.

- **Polymarket only for the comparison.** Kalshi prices are available
  via the same FairLine API but have ~14 missing teams in their listing.
  Polymarket has full 48-team coverage. If Polymarket goes down for an
  extended period, the cron logs a warning and `our_vs_market.json`
  becomes stale until they recover.

## Acknowledgements

This project builds a scenario-propagation and signal layer on top of two
components contributed by others:

- **Thy Nguyen** - the **FairLine** model, which provides the sportsbook-derived
  fair odds (Shin + Power devigging across Pinnacle, Betfair, and other books)
  used for calibration, plus the prediction-market prices used as the comparison
  surface. Accessed here via its API.
- **Duy Anh Nguyen** - the Monte Carlo match simulator vendored under
  `src/upset_propagation/_vendored/` (see `docs/VENDORING.md`).

All calibration, the 79-scenario library, scenario propagation, the ensemble
matcher, and the implied-vs-market signal layer are my own work.

## Further reading

- **`docs/DEPLOYMENT.md`** - production setup, cron configuration,
  alerting, log management
- **`docs/OUTPUT_FORMAT.md`** - JSON schema for every output file
- **`docs/RUNBOOK.md`** - operational troubleshooting: what to do when
  the health check fires an alert at 3am
- **`docs/VENDORING.md`** - how to update the pinned Monte Carlo simulator dependency