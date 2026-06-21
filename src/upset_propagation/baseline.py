"""Fetch baseline fair_probs from the FairLine model's public API.

These are the unconditional sportsbook-devigged tournament-winner probabilities
for each WC 2026 team. They're the "starting point" - the framework computes
how these probabilities shift under each bracket scenario, but never overrides
the unconditional baseline.

The framework's invariant: when run with an empty scenario (no group standings
fixed), the propagator's output must reproduce these baselines. The calibration
step is what makes that true.

For debuggability: this module supports capturing the raw API response to disk
(`fetch_baseline_with_snapshot`) and reading from a saved snapshot
(`load_baseline_from_snapshot`). This lets us reproduce any past run exactly,
even after the live API has drifted.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import requests

from upset_propagation.config import (
    FAIRLINE_FAIR_ODDS_ENDPOINT,
    FAIRLINE_PRICES_ENDPOINT,
)


logger = logging.getLogger(__name__)


# Hard ceiling on HTTP timeout. The API normally responds in <1s; longer means
# something's wrong and we want to fail fast.
REQUEST_TIMEOUT_SEC = 10

# Retry policy for the FairLine API. Item 14 - transient network failures
# (brief 5xx, connection errors, timeouts) shouldn't kill the whole cron
# run. Exponential backoff: 1s, 2s, 4s between attempts.
MAX_RETRIES = 3
RETRY_BASE_DELAY_SEC = 1.0

# HTTP status codes that warrant a retry. 5xx is server-side / transient.
# We don't retry on 4xx (client error - won't fix itself) or 2xx/3xx
# (success).
RETRYABLE_HTTP_STATUS = frozenset({500, 502, 503, 504})


class BaselineFetchError(RuntimeError):
    """Raised when the FairLine API is unreachable or returns malformed data.

    Wraps the underlying exception (network error, JSON parse error, etc.) so
    callers get a single exception type to catch.
    """


def _parse_rows_to_fair_probs(rows: list, source: str) -> dict[str, float]:
    """Parse the API response rows into {team: fair_prob}.

    Shared between live-fetch and snapshot-load paths so they handle malformed
    data identically. `source` is included in error messages for traceability.
    """
    if not isinstance(rows, list):
        raise BaselineFetchError(
            f"Response from {source} was not a list (got {type(rows).__name__})"
        )

    fair_probs: dict[str, float] = {}
    for i, row in enumerate(rows):
        # The API uses `outcome` as the team-name field (one row per team in
        # a `winner` market - the team name appears in `outcome`, not `team`).
        try:
            team = row["outcome"]
            prob = float(row["fair_prob"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BaselineFetchError(
                f"Row {i} from {source} missing or malformed "
                f"'outcome'/'fair_prob': {row!r}"
            ) from exc
        fair_probs[team] = prob

    if not fair_probs:
        raise BaselineFetchError(f"Response from {source} was empty (no teams returned)")

    return fair_probs


def fetch_baseline_fair_probs() -> dict[str, float]:
    """Return {team: fair_prob} from the FairLine API for WC 2026.

    The API returns one row per team with several columns; we only need
    `outcome` (team name) and `fair_prob`. Other columns (power_prob,
    shin_prob, confidence, n_sources) are diagnostic and ignored here.

    Raises:
        BaselineFetchError: on network failure, HTTP error, malformed JSON,
            or missing required fields.
    """
    rows = _fetch_raw_rows()
    return _parse_rows_to_fair_probs(rows, source=FAIRLINE_FAIR_ODDS_ENDPOINT)


def _fetch_raw_rows(url: str = FAIRLINE_FAIR_ODDS_ENDPOINT) -> list:
    """Hit a FairLine API endpoint and return the parsed JSON rows.

    Returns the unprocessed list of row dicts straight from the API - useful
    for snapshotting before our schema parsing strips fields.

    Args:
        url: API endpoint to hit. Defaults to the fair-odds endpoint
            (the FairLine devigged model); the prices endpoint is the other
            common caller.

    Item 14: retries up to MAX_RETRIES times on transient failures
    (connection errors, timeouts, 5xx responses) with exponential backoff.
    Doesn't retry on 4xx (client error - config problem, retrying won't
    help). Each retry attempt logs at WARNING so cron-mode's stderr
    forwarding surfaces "API trouble" alerts even if the run ultimately
    succeeds.
    """
    last_exception: Optional[BaseException] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SEC)
        except (requests.ConnectionError, requests.Timeout) as exc:
            # Transient - connection didn't even establish, or it timed out.
            last_exception = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
                logger.warning(
                    f"FairLine API attempt {attempt}/{MAX_RETRIES} failed "
                    f"({type(exc).__name__}: {exc}); retrying in {delay:.1f}s"
                )
                time.sleep(delay)
                continue
            # Fall through to BaselineFetchError below
            break
        except requests.RequestException as exc:
            # Other request-construction errors (e.g. invalid URL, SSL issues)
            # - these are NOT transient; fail immediately.
            raise BaselineFetchError(
                f"Failed to fetch from {url}: {exc}"
            ) from exc

        # Got a response. Decide whether it's retryable.
        if response.status_code in RETRYABLE_HTTP_STATUS and attempt < MAX_RETRIES:
            delay = RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
            logger.warning(
                f"FairLine API attempt {attempt}/{MAX_RETRIES} returned "
                f"HTTP {response.status_code}; retrying in {delay:.1f}s"
            )
            last_exception = requests.HTTPError(
                f"HTTP {response.status_code}", response=response
            )
            time.sleep(delay)
            continue

        # Either a success (2xx), a non-retryable error (4xx), or our last
        # attempt at a 5xx. Let raise_for_status handle the error path.
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise BaselineFetchError(
                f"Failed to fetch from {url}: {exc}"
            ) from exc

        # Success.
        if attempt > 1:
            logger.info(f"FairLine API succeeded on attempt {attempt}/{MAX_RETRIES}")
        try:
            return response.json()
        except ValueError as exc:
            raise BaselineFetchError(
                f"Baseline response was not valid JSON: {exc}"
            ) from exc

    # All retries exhausted on transient errors.
    raise BaselineFetchError(
        f"Failed to fetch from {url} "
        f"after {MAX_RETRIES} attempts: {last_exception}"
    ) from last_exception


def fetch_baseline_with_snapshot(
    snapshot_path: Path,
) -> dict[str, float]:
    """Fetch from API AND save the raw response to disk.

    Used in production runs so we can reproduce/audit past runs. The snapshot
    file includes the raw API response unchanged - all fields, not just
    outcome/fair_prob - so it can serve future debugging needs we don't
    anticipate today.

    Args:
        snapshot_path: where to write the snapshot JSON (parent dir must exist)

    Returns: {team: fair_prob} same as fetch_baseline_fair_probs().
    """
    rows = _fetch_raw_rows()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return _parse_rows_to_fair_probs(rows, source=FAIRLINE_FAIR_ODDS_ENDPOINT)


def load_baseline_from_snapshot(snapshot_path: Path) -> dict[str, float]:
    """Read a previously-saved API snapshot from disk.

    Used to reproduce a past run exactly: pair this with the same Elo
    history and the same code, and you get the same calibration outputs
    (subject to Nelder-Mead's stochastic drift, but propagated probs are
    stable across runs).

    Raises BaselineFetchError if the file is missing, malformed, or empty.
    """
    if not snapshot_path.exists():
        raise BaselineFetchError(f"Snapshot file does not exist: {snapshot_path}")
    try:
        rows = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineFetchError(
            f"Snapshot at {snapshot_path} is not valid JSON: {exc}"
        ) from exc
    return _parse_rows_to_fair_probs(rows, source=str(snapshot_path))


def summarise(fair_probs: dict[str, float]) -> str:
    """Format a one-line summary - useful for logging and CLI output."""
    n = len(fair_probs)
    total = sum(fair_probs.values())
    top = sorted(fair_probs.items(), key=lambda kv: -kv[1])[:3]
    top_str = ", ".join(f"{t} {p:.3f}" for t, p in top)
    return f"{n} teams, Σ={total:.3f}, top: {top_str}"


# Prediction-market prices (Polymarket)


# Polymarket uses different name spellings than our canonical groups.json
# for 5 of the 48 WC2026 teams. Map Polymarket name → canonical name so
# downstream code (which keys on the canonical name) finds the right team.
# Unmapped names pass through unchanged.
#
# Verified against /api/events/world_cup_2026/prices on 2026-06-11. If
# Polymarket changes a spelling, add the new mapping here. If they add
# a team name we don't recognize, the loader logs a warning and drops
# that row.
MARKET_NAME_ALIASES: dict[str, str] = {
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Congo DR":           "DR Congo",
    "Curaçao":            "Curacao",
    "Turkiye":            "Turkey",
    "USA":                "United States",
}

# Polymarket platform key in the /prices endpoint response. We use
# Polymarket exclusively for the comparison signal: full 48-team
# coverage, lowest vig of the available platforms, and the primary
# venue for WC outright trading.
POLYMARKET_PLATFORM_KEY = "polymarket_live"


def _parse_rows_to_market_probs(
    rows: list,
    source: str,
    canonical_teams: Optional[set[str]] = None,
) -> dict[str, float]:
    """Parse the /prices API response into renormalized {team: prob}.

    Steps:
      1. Filter to Polymarket rows only
      2. Map each row's team name through MARKET_NAME_ALIASES → canonical
      3. Drop any row whose canonical name isn't in canonical_teams
         (if provided) with a WARNING; this catches Polymarket listing
         teams that aren't in our roster (e.g. data drift)
      4. Renormalize all kept mids to sum to 1.0 (devig the platform)

    Args:
        rows: raw API response (list of {team, platform, mid, ...} dicts)
        source: included in error messages for traceability
        canonical_teams: optional set of expected team names; rows for
            teams outside this set are logged + dropped. None disables
            this check.

    Returns: {canonical_team_name: renormalized_probability}, summing
        to ~1.0 (small float imprecision aside).

    Raises:
        BaselineFetchError on malformed structure (not a list, no
        Polymarket rows, all renormalized mids = 0).
    """
    if not isinstance(rows, list):
        raise BaselineFetchError(
            f"Response from {source} was not a list (got {type(rows).__name__})"
        )

    raw_mids: dict[str, float] = {}
    for i, row in enumerate(rows):
        try:
            platform = row["platform"]
        except (KeyError, TypeError) as exc:
            raise BaselineFetchError(
                f"Row {i} from {source} missing 'platform': {row!r}"
            ) from exc

        if platform != POLYMARKET_PLATFORM_KEY:
            continue

        try:
            raw_team = row["team"]
            mid_val = row.get("mid")
        except (KeyError, TypeError) as exc:
            raise BaselineFetchError(
                f"Row {i} from {source} missing 'team': {row!r}"
            ) from exc

        # mid can be None if Polymarket has the team listed but no live
        # quote - skip those rather than crash; this surfaces as a missing-team
        # warning when n_market_teams in the log doesn't equal 48.
        if mid_val is None:
            logger.warning(
                f"Polymarket row {i} for team {raw_team!r} has no mid; "
                f"skipping. Source: {source}"
            )
            continue

        try:
            mid = float(mid_val)
        except (TypeError, ValueError) as exc:
            raise BaselineFetchError(
                f"Row {i} from {source} has non-numeric mid: {row!r}"
            ) from exc

        # Map to canonical team name
        canonical = MARKET_NAME_ALIASES.get(raw_team, raw_team)

        if canonical_teams is not None and canonical not in canonical_teams:
            logger.warning(
                f"Polymarket row {i} for {raw_team!r} (canonical "
                f"{canonical!r}) is not in our roster; dropping. "
                f"Source: {source}"
            )
            continue

        raw_mids[canonical] = mid

    if not raw_mids:
        raise BaselineFetchError(
            f"Response from {source} had no usable Polymarket rows "
            f"(platform key {POLYMARKET_PLATFORM_KEY!r})"
        )

    # Renormalize to sum=1 (devig Polymarket's ~4-5% overround)
    total = sum(raw_mids.values())
    if total <= 0:
        raise BaselineFetchError(
            f"Sum of Polymarket mids from {source} is {total} "
            f"(expected ~1.04); can't renormalize"
        )

    return {team: mid / total for team, mid in raw_mids.items()}


def fetch_market_prices(
    canonical_teams: Optional[set[str]] = None,
) -> dict[str, float]:
    """Return {team: renormalized_probability} from Polymarket via FairLine.

    Unlike fetch_baseline_fair_probs (which returns the FairLine model's calibrated
    sportsbook-devigged view), this returns the raw prediction-market
    view normalized to a probability distribution.

    The framework calibrates against the FairLine model's view (fetch_baseline_fair_probs)
    but compares its implied probabilities against this market view -
    that comparison is the trade signal.

    Args:
        canonical_teams: optional set of expected team names (typically
            the 48 from groups.json). Polymarket rows for teams outside
            this set are logged and dropped. If None, all Polymarket
            rows are kept after alias mapping.

    Returns: {canonical_team_name: probability}, summing to ~1.0.

    Raises:
        BaselineFetchError: on network failure, HTTP error, malformed
            JSON, no Polymarket rows, or all-zero mids.
    """
    rows = _fetch_raw_rows(url=FAIRLINE_PRICES_ENDPOINT)
    return _parse_rows_to_market_probs(
        rows,
        source=FAIRLINE_PRICES_ENDPOINT,
        canonical_teams=canonical_teams,
    )


if __name__ == "__main__":
    # Manual smoke test - `python -m upset_propagation.baseline`
    probs = fetch_baseline_fair_probs()
    print(summarise(probs))