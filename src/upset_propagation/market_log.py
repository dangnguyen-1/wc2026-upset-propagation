"""Market snapshot logging.

Append-only log capturing what the FairLine model's market said at each cron run, with
diff against the previous entry and edge vs our framework's implied
fair_probs. Useful for post-tournament trade research: reconstruct
exactly what we and the market both thought at any past moment.

Output file: output/market_log.jsonl

JSON Lines format (one JSON object per line) - chosen for:
  - Cheap append (no file-rewrite)
  - Stream-processable (tail, jq, pandas.read_json(lines=True))
  - Robust to mid-write corruption (one bad line, not the whole log)

The existing per-run output/runs/<ts>/api_snapshot.json captures the
RAW fair_probs response. market_log.jsonl is COMPLEMENTARY:
  - api_snapshot.json: full response for reproducibility (--from-snapshot)
  - market_log.jsonl: per-run summary + diff + implied-edge for analysis

Public API:
    MarketLogEntry dataclass
    compute_market_log_entry(...) -> MarketLogEntry
    append_market_log(entry, output_dir) -> Path
    read_last_market_log_entry(output_dir) -> Optional[MarketLogEntry]
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from upset_propagation._vendored.simulator import Predictor
from upset_propagation.config import OUTPUT_DIR
from upset_propagation.implied_probs import compute_implied_probs
from upset_propagation.state_matcher import RealisedState


logger = logging.getLogger(__name__)


MARKET_LOG_FILENAME = "market_log.jsonl"

# Top-N teams to track explicitly. Smaller than 48 keeps the log compact;
# big enough to cover all genuine contenders. Sum of top-15 fair_probs is
# usually ~95%+ of the total mass.
TOP_N_TEAMS = 15

# Threshold for including a team in diff_from_previous_run. Teams that
# moved less than this aren't interesting noise - they're inside the
# API's quantization precision. 0.1pp = 0.001 in probability units.
MIN_DIFF_THRESHOLD = 0.001


# Data types


@dataclass
class TeamFairProb:
    """One team's market fair_prob, used in top_market list."""
    team: str
    fair_prob: float


@dataclass
class TeamDiff:
    """One team's pp change from previous market snapshot."""
    team: str
    previous_pp: float
    current_pp: float
    delta_pp: float


@dataclass
class TeamEdge:
    """One team's framework-implied vs market gap (the trade signal),
    computed at two weight exponents (p=4 primary, p=8 sharper view).

    Both exponents are surfaced side by side on every cron run rather
    than picking a single value. p=4 was selected as the primary from
    an empirical sweep (matched scenario gets ~18% of total weight,
    implied tracks within 0.3pp). p=8 is closer to a "single scenario
    lookup" - top match dominates and smoothing across plausible
    alternatives is largely lost. They typically agree during early
    group stage and diverge as the matcher discriminates more sharply.
    """
    team: str
    market_pp: float
    implied_p4_pp: float
    edge_p4_pp: float  # implied_p4 - market; positive = underpriced under p=4
    implied_p8_pp: float
    edge_p8_pp: float  # implied_p8 - market; positive = underpriced under p=8


@dataclass
class MarketLogEntry:
    """One entry in market_log.jsonl. Captures what the market said at one
    cron run, plus diffs and edges if computable.

    Attributes:
        timestamp: ISO 8601 UTC
        n_teams_in_api: total team count in the FairLine response
        fair_probs_sum: sum of all team fair_probs (should be ≈ 1.0)
        top_market: top-N teams by market fair_prob
        diff_from_previous_run: top movers since the last log entry
            (only teams that moved >= MIN_DIFF_THRESHOLD); None if this
            is the first entry or previous entry is unreadable
        implied_vs_market: per-team edge from our framework's implied
            fair_probs vs the market; None if implied couldn't be computed
            (e.g. no calibrated predictor available)
        n_groups_observed: how many of 12 groups have played out (for
            interpreting implied - pre-tournament implied is uniform across
            scenarios so edges are small; mid-tournament edges sharpen)
    """
    timestamp: str
    n_teams_in_api: int
    fair_probs_sum: float
    top_market: list[TeamFairProb] = field(default_factory=list)
    diff_from_previous_run: Optional[list[TeamDiff]] = None
    implied_vs_market: Optional[list[TeamEdge]] = None
    n_groups_observed: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict for serialization."""
        payload: dict[str, Any] = {
            "timestamp": self.timestamp,
            "n_teams_in_api": self.n_teams_in_api,
            "fair_probs_sum": round(self.fair_probs_sum, 6),
            "top_market": [asdict(t) for t in self.top_market],
            "n_groups_observed": self.n_groups_observed,
        }
        if self.diff_from_previous_run is not None:
            payload["diff_from_previous_run"] = [
                asdict(t) for t in self.diff_from_previous_run
            ]
        if self.implied_vs_market is not None:
            payload["implied_vs_market"] = [
                asdict(t) for t in self.implied_vs_market
            ]
        return payload


# Reading the previous entry


def read_last_market_log_entry(
    output_dir: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    """Read the last entry of market_log.jsonl, or None if not present.

    Returns a dict (raw JSON) rather than a MarketLogEntry - we only need
    the fair_probs to compute the diff, and the dict is easier to consume
    without round-tripping through dataclasses. Forwards-compatible with
    schema additions.

    For efficiency, reads the last line directly rather than parsing the
    whole file. The log can grow to thousands of lines over a tournament.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    log_path = output_dir / MARKET_LOG_FILENAME
    if not log_path.exists():
        return None

    # Read the last non-empty line. For typical entries (~1-3KB), seeking
    # to the end and scanning back is overkill - just read the whole file
    # and split. If the log gets enormous we can optimize.
    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning(f"Could not read {log_path}: {exc}")
        return None

    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None

    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        logger.warning(
            f"Last line of {log_path} is malformed ({exc}); "
            f"treating as no previous entry."
        )
        return None


# Computing the new entry


def _top_n_teams(
    fair_probs: dict[str, float], n: int = TOP_N_TEAMS
) -> list[TeamFairProb]:
    """Return the top-n teams by fair_prob descending."""
    items = sorted(fair_probs.items(), key=lambda kv: -kv[1])[:n]
    return [TeamFairProb(team=t, fair_prob=round(p, 6)) for t, p in items]


def _compute_diff(
    current_fair_probs: dict[str, float],
    previous_entry: Optional[dict[str, Any]],
    threshold: float = MIN_DIFF_THRESHOLD,
    top_n: int = TOP_N_TEAMS,
) -> Optional[list[TeamDiff]]:
    """Return top movers between current and previous entries, or None.

    The previous_entry's top_market list contains the teams we tracked
    last run. Any team in EITHER the current or previous top_market is
    a candidate for the diff list.

    Filters out teams that moved less than `threshold` (noise).
    Returns top_n movers by absolute delta.
    """
    if previous_entry is None:
        return None

    prev_top = previous_entry.get("top_market", [])
    if not prev_top:
        return None

    prev_probs = {entry["team"]: entry["fair_prob"] for entry in prev_top}

    # Union of teams we care about - currently top + previously top
    current_top_teams = {
        t for t, p in sorted(current_fair_probs.items(), key=lambda kv: -kv[1])[:top_n]
    }
    teams_of_interest = current_top_teams | set(prev_probs.keys())

    diffs: list[TeamDiff] = []
    for team in teams_of_interest:
        prev_p = prev_probs.get(team, 0.0)
        curr_p = current_fair_probs.get(team, 0.0)
        delta = curr_p - prev_p
        if abs(delta) < threshold:
            continue
        diffs.append(TeamDiff(
            team=team,
            previous_pp=round(prev_p * 100, 4),
            current_pp=round(curr_p * 100, 4),
            delta_pp=round(delta * 100, 4),
        ))

    # Sort by absolute delta descending; cap at top_n
    diffs.sort(key=lambda d: -abs(d.delta_pp))
    return diffs[:top_n]


def _compute_edges(
    market_probs: dict[str, float],
    fairline_probs: dict[str, float],
    state: RealisedState,
    predictor: Optional[Predictor],
    ratings: Optional[dict[str, float]],
    output_dir: Optional[Path] = None,
    top_n: int = TOP_N_TEAMS,
) -> Optional[list[TeamEdge]]:
    """Compute framework-implied vs market edges for the top-n teams at
    BOTH weight exponents (p=4 primary, p=8 sharper).

    Returns None if implied_probs can't be computed (missing predictor or
    ratings). This is the trade signal - for each top team, where does
    our model disagree with the prediction market?

    The edge is `implied - market`. Positive = we think the team is
    UNDERPRICED by the market (they have more chance than the market thinks).
    Negative = OVERPRICED.

    Two separate probability inputs:
      - market_probs: what we compare AGAINST. Currently Polymarket via
        fetch_market_prices() - the raw prediction-market view
        actually traded against.
      - fairline_probs: what the matcher uses internally for baseline
        construction (build_baseline_standings). That's the FairLine model's
        calibrated view (fetch_baseline_fair_probs) and stays as
        the calibration-anchored input.

    Two weight exponents:
      - p=4: empirical sweep winner. Matched scenario gets ~18% of
        total weight; tracking within ~0.3pp. The recommended default.
      - p=8: closer to single-scenario lookup. Matched scenario gets
        ~35% of total weight; smoothing largely lost. Useful as a
        "sharper" view to see how aggressive the matcher's top pick is.

    output_dir must point at the directory containing the 79 scenario
    JSONs - compute_implied_probs needs to load each one's survival table
    to compute the weighted average. In cron-mode, this is the staging
    dir (output.pending/), NOT the live output/ - the atomic swap happens
    after this function returns.
    """
    if predictor is None or ratings is None:
        return None

    try:
        implied_p4 = compute_implied_probs(
            state, predictor, ratings, fairline_probs,
            output_dir=output_dir,
            weight_exponent=4.0,
        )
        implied_p8 = compute_implied_probs(
            state, predictor, ratings, fairline_probs,
            output_dir=output_dir,
            weight_exponent=8.0,
        )
    except Exception as exc:
        logger.warning(f"compute_implied_probs failed for market_log: {exc}")
        return None

    # Top-n by market price - these are the teams where trade signals
    # are most actionable (high-liquidity outcomes one can actually move
    # size on).
    top_market_teams = [
        t for t, p in sorted(market_probs.items(), key=lambda kv: -kv[1])[:top_n]
    ]

    edges: list[TeamEdge] = []
    for team in top_market_teams:
        market_p = market_probs.get(team, 0.0)
        implied_p4_val = implied_p4.probs.get(team, {}).get("Win", 0.0)
        implied_p8_val = implied_p8.probs.get(team, {}).get("Win", 0.0)
        edges.append(TeamEdge(
            team=team,
            market_pp=round(market_p * 100, 4),
            implied_p4_pp=round(implied_p4_val * 100, 4),
            edge_p4_pp=round((implied_p4_val - market_p) * 100, 4),
            implied_p8_pp=round(implied_p8_val * 100, 4),
            edge_p8_pp=round((implied_p8_val - market_p) * 100, 4),
        ))
    return edges


def compute_market_log_entry(
    market_probs: dict[str, float],
    fairline_probs: dict[str, float],
    state: Optional[RealisedState] = None,
    predictor: Optional[Predictor] = None,
    ratings: Optional[dict[str, float]] = None,
    output_dir: Optional[Path] = None,
) -> MarketLogEntry:
    """Build one MarketLogEntry from current inputs.

    Args:
        market_probs: prediction-market probabilities (from
            fetch_market_prices, Polymarket-derived). This is the
            "market view" we compare against - what's logged in
            top_market and used for the edge computation.
        fairline_probs: the FairLine model's calibrated fair-odds view (from
            fetch_baseline_fair_probs). Passed through to the matcher
            for internal baseline construction. NOT logged directly.
        state: realised state for the implied-vs-market edge computation;
            if None, edges are not computed
        predictor: calibrated predictor; if None, edges are not computed
        ratings: team Elo ratings; if None, edges are not computed
        output_dir: where market_log.jsonl lives (for diff lookup)

    Returns: MarketLogEntry, ready to append via append_market_log.

    The JSON field name `fair_probs_sum` is kept for log-format
    backward compatibility but now represents the market_probs sum
    (should be ~1.0 after Polymarket vig devigging).
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    timestamp = datetime.now(timezone.utc).isoformat()
    total = sum(market_probs.values())

    # Previous entry (None on first-ever log write)
    previous = read_last_market_log_entry(output_dir)

    diff = _compute_diff(market_probs, previous)
    edges = _compute_edges(
        market_probs, fairline_probs,
        state or RealisedState(standings={}),
        predictor, ratings, output_dir=output_dir,
    )

    return MarketLogEntry(
        timestamp=timestamp,
        n_teams_in_api=len(market_probs),
        fair_probs_sum=total,
        top_market=_top_n_teams(market_probs),
        diff_from_previous_run=diff,
        implied_vs_market=edges,
        n_groups_observed=len(state.played_groups) if state else 0,
    )


# Appending


def append_market_log(
    entry: MarketLogEntry,
    output_dir: Optional[Path] = None,
) -> Path:
    """Append one entry to output/market_log.jsonl as a single line.

    Returns: path to the log file.

    JSONL semantics: each line is a complete JSON object, separated by
    newlines. We write with a final newline so the next append doesn't
    need to seek and check.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / MARKET_LOG_FILENAME
    line = json.dumps(entry.to_dict(), separators=(",", ":"))
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")
    logger.info(
        f"Appended market_log entry to {path} "
        f"(top: {entry.top_market[0].team if entry.top_market else 'n/a'} "
        f"@ {entry.top_market[0].fair_prob if entry.top_market else 0:.4f})"
    )
    return path


# CLI smoke test


if __name__ == "__main__":
    # Manual smoke test - `python -m upset_propagation.market_log`
    #
    # Builds one log entry from the live API + current outputs and
    # appends to market_log.jsonl. Prints a summary.
    from upset_propagation.baseline import fetch_baseline_fair_probs
    from upset_propagation.l1_matcher import load_calibrated_predictor_from_index
    from upset_propagation.logging_config import configure_interactive_logging
    from upset_propagation.scenarios import load_latest_elo
    from upset_propagation.top_ranking import load_state_for_cron

    configure_interactive_logging()
    output_dir = OUTPUT_DIR

    logger.info("Loading inputs...")
    fair_probs = fetch_baseline_fair_probs()
    ratings = load_latest_elo()
    predictor = load_calibrated_predictor_from_index(output_dir / "index.json")
    state = load_state_for_cron(output_dir, ratings)

    logger.info("Computing market log entry...")
    entry = compute_market_log_entry(
        fair_probs, state, predictor, ratings, output_dir
    )

    print()
    print(f"Market snapshot at {entry.timestamp}")
    print(f"  n_teams={entry.n_teams_in_api}, Σ={entry.fair_probs_sum:.4f}, "
          f"observed_groups={entry.n_groups_observed}/12")
    print(f"  Top 5 by market: " + ", ".join(
        f"{t.team}={t.fair_prob*100:.2f}%" for t in entry.top_market[:5]
    ))
    if entry.diff_from_previous_run is None:
        print(f"  (no previous entry - diff omitted)")
    elif not entry.diff_from_previous_run:
        print(f"  No movers ≥{MIN_DIFF_THRESHOLD*100:.1f}pp since last run "
              f"(API quantization noise only)")
    else:
        print(f"  Top movers vs previous run:")
        for d in entry.diff_from_previous_run[:5]:
            print(f"    {d.team:25s}  {d.previous_pp:+6.2f} → {d.current_pp:+6.2f}  "
                  f"(Δ {d.delta_pp:+6.2f}pp)")
    if entry.implied_vs_market:
        print(f"  Top implied-vs-market edges, p=4 view (top-5 by |edge|):")
        edges_sorted = sorted(entry.implied_vs_market, key=lambda e: -abs(e.edge_p4_pp))
        for e in edges_sorted[:5]:
            print(f"    {e.team:25s}  market={e.market_pp:5.2f}pp  "
                  f"p4_implied={e.implied_p4_pp:5.2f}pp  p4_edge={e.edge_p4_pp:+6.2f}pp  "
                  f"p8_implied={e.implied_p8_pp:5.2f}pp  p8_edge={e.edge_p8_pp:+6.2f}pp")

    append_market_log(entry, output_dir)