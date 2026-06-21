"""Our-vs-Market comparison view.

Surfaces the per-team edge between our framework's implied probability
and the prediction-market view (Polymarket via /api/.../prices). Where
they disagree is where the trade signal is.

Two surfaces:

  - CLI: `python -m upset_propagation.our_vs_market` - primary surface.
    Prints a readable table. Default sort by absolute edge (biggest
    trade signal first); --sort flag for alternatives.

  - JSON snapshot: output/our_vs_market.json, written by the cron each
    run. For any UI consumer that prefers structured data over parsing
    market_log.jsonl.

Implementation: read the last entry of market_log.jsonl and surface
its `implied_vs_market` field. No new computation - items 8 + 12 share
the math; this module is a presentation layer.

If market_log.jsonl is missing or its latest entry has no
implied_vs_market field, the view explains why (typically: pre-tournament,
or the cron hasn't run yet).

Public API:
    OurVsMarketRow / OurVsMarketSnapshot dataclasses
    build_snapshot(output_dir) -> OurVsMarketSnapshot
    write_snapshot_file(snapshot, output_dir) -> Path
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from upset_propagation.config import OUTPUT_DIR
from upset_propagation.market_log import (
    MARKET_LOG_FILENAME,
    read_last_market_log_entry,
)


logger = logging.getLogger(__name__)


OUR_VS_MARKET_FILENAME = "our_vs_market.json"


# Data types


@dataclass
class OurVsMarketRow:
    """One team's market vs framework comparison, with both p=4 (primary)
    and p=8 (sharper) views surfaced side by side.

    delta_p4_pp = our_implied_p4_pp - market_pp. Same for p8. Sign convention:
    positive = we think team has more chance than market (underpriced);
    negative = we think team has less chance (overpriced).

    edge_direction_p4 and edge_direction_p8 are independent classifications
    based on the relative-threshold rule. They can disagree - when they
    do, it's a signal that the matcher's top scenario is very different
    from the smoothed-across-plausible-alternatives view, which is itself
    information.
    """
    team: str
    market_pp: float
    our_implied_p4_pp: float
    delta_p4_pp: float
    edge_direction_p4: str   # "overpriced" | "underpriced" | "fair"
    our_implied_p8_pp: float
    delta_p8_pp: float
    edge_direction_p8: str


@dataclass
class OurVsMarketSnapshot:
    """Full snapshot for output/our_vs_market.json."""
    computed_at: str
    source_market_log_ts: Optional[str]  # the market_log entry we sourced from
    n_groups_observed: int
    rows: list[OurVsMarketRow] = field(default_factory=list)
    note: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "computed_at": self.computed_at,
            "source_market_log_ts": self.source_market_log_ts,
            "n_groups_observed": self.n_groups_observed,
            "rows": [asdict(r) for r in self.rows],
        }
        if self.note is not None:
            payload["note"] = self.note
        return payload


# Build the snapshot from market_log


# A team's edge is called "fair" if its relative magnitude
# |delta_pp / market_pp| is below this threshold; otherwise overpriced
# or underpriced. 10% relative (i.e. 0.10) is "meaningful disagreement
# vs the market for this team specifically." This catches genuine
# underdog moves (Haiti 0.001 → 0.002 = 100% relative move) that an
# absolute threshold would hide, while treating favorite noise (France
# 16.0 → 16.2 = 1.25% relative move) as fair.
EDGE_DIRECTION_THRESHOLD_RELATIVE = 0.10


def _classify_edge(delta_pp: float, market_pp: float) -> str:
    """Categorize a delta into overpriced / underpriced / fair using a
    relative threshold.

    Convention: delta = our - market.
      - delta > 0  → we think team has more chance than market → underpriced
      - delta < 0  → we think team has less chance than market → overpriced
      - |delta / market| < threshold → fair (no actionable disagreement)

    Special case: market_pp == 0. A team at exactly 0% market price is
    a degenerate case (Polymarket likely doesn't list them). Any non-
    zero implied is mathematically infinite-percent off, but practically
    not actionable (can't trade at 0% odds). Return "fair" to avoid
    surfacing meaningless signal.
    """
    if market_pp == 0:
        return "fair"
    if abs(delta_pp / market_pp) < EDGE_DIRECTION_THRESHOLD_RELATIVE:
        return "fair"
    return "underpriced" if delta_pp > 0 else "overpriced"


def build_snapshot(output_dir: Optional[Path] = None) -> OurVsMarketSnapshot:
    """Read the latest market_log entry and project the implied_vs_market
    field into an OurVsMarketSnapshot.

    Pre-tournament / no entry / no implied_vs_market → returns a snapshot
    with empty rows and an explanatory note.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    computed_at = datetime.now(timezone.utc).isoformat()
    last = read_last_market_log_entry(output_dir)

    if last is None:
        return OurVsMarketSnapshot(
            computed_at=computed_at,
            source_market_log_ts=None,
            n_groups_observed=0,
            note=(
                f"No {MARKET_LOG_FILENAME} found at {output_dir}. "
                f"Run the framework with --cron-mode to bootstrap, or "
                f"`python -m upset_propagation.market_log` for a one-off entry."
            ),
        )

    source_ts = last.get("timestamp")
    n_observed = last.get("n_groups_observed", 0)
    edges = last.get("implied_vs_market")

    if not edges:
        return OurVsMarketSnapshot(
            computed_at=computed_at,
            source_market_log_ts=source_ts,
            n_groups_observed=n_observed,
            note=(
                "Latest market_log entry has no implied_vs_market data. "
                "Typically means the calibrated predictor wasn't available "
                "when that entry was logged."
            ),
        )

    rows = [
        OurVsMarketRow(
            team=e["team"],
            market_pp=e["market_pp"],
            our_implied_p4_pp=e["implied_p4_pp"],
            delta_p4_pp=e["edge_p4_pp"],
            edge_direction_p4=_classify_edge(e["edge_p4_pp"], e["market_pp"]),
            our_implied_p8_pp=e["implied_p8_pp"],
            delta_p8_pp=e["edge_p8_pp"],
            edge_direction_p8=_classify_edge(e["edge_p8_pp"], e["market_pp"]),
        )
        for e in edges
    ]
    return OurVsMarketSnapshot(
        computed_at=computed_at,
        source_market_log_ts=source_ts,
        n_groups_observed=n_observed,
        rows=rows,
    )


def write_snapshot_file(
    snapshot: OurVsMarketSnapshot,
    output_dir: Optional[Path] = None,
) -> Path:
    """Write the snapshot to output/our_vs_market.json."""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / OUR_VS_MARKET_FILENAME
    path.write_text(
        json.dumps(snapshot.to_dict(), indent=2),
        encoding="utf-8",
    )
    logger.info(
        f"Wrote our-vs-market snapshot to {path} "
        f"({len(snapshot.rows)} rows, source_ts={snapshot.source_market_log_ts})"
    )
    return path


# CLI


SortKey = Literal["edge", "team", "market", "implied"]


def _sort_rows(
    rows: list[OurVsMarketRow], sort: SortKey
) -> list[OurVsMarketRow]:
    """Sort rows for display. Default sort is by |delta_p4| descending -
    p=4 is the primary view, so its edges drive ranking. The p=8 column
    is informational alongside, not the basis for sort order.
    """
    if sort == "edge":
        return sorted(rows, key=lambda r: -abs(r.delta_p4_pp))
    elif sort == "team":
        return sorted(rows, key=lambda r: r.team)
    elif sort == "market":
        return sorted(rows, key=lambda r: -r.market_pp)
    elif sort == "implied":
        return sorted(rows, key=lambda r: -r.our_implied_p4_pp)
    return rows


def _print_table(snapshot: OurVsMarketSnapshot, sort: SortKey, top: Optional[int]) -> None:
    """Print the comparison as a readable table to stdout.

    8-column layout: Team | Market | p=4 Implied | p=4 Δ | p=4 Dir |
    p=8 Implied | p=8 Δ | p=8 Dir. Wide; needs ~110-char terminal.
    """
    print()
    print(f"Our vs Market - {snapshot.n_groups_observed}/12 groups observed")
    print(f"  Source: market_log entry at {snapshot.source_market_log_ts}")
    print(f"  Sort: {sort}" + (f" (top {top})" if top else ""))
    print(f"  p=4 is primary view; p=8 is the sharper-weighting comparison")
    print()
    if snapshot.note:
        print(f"  {snapshot.note}")
        return
    if not snapshot.rows:
        print("  (no rows - implied_vs_market field was empty)")
        return

    rows = _sort_rows(snapshot.rows, sort)
    if top is not None:
        rows = rows[:top]

    # Direction → short marker for the table
    def _marker(direction: str) -> str:
        return {
            "overpriced":  "↓ over",
            "underpriced": "↑ under",
            "fair":        "  fair",
        }[direction]

    # Header
    print(
        f"  {'Team':22s}  {'Market':>8s}  "
        f"{'p4 Impl':>9s}  {'p4 Δ':>7s}  {'p4 Dir':>9s}  "
        f"{'p8 Impl':>9s}  {'p8 Δ':>7s}  {'p8 Dir':>9s}"
    )
    print(
        f"  {'-' * 22}  {'-' * 8}  "
        f"{'-' * 9}  {'-' * 7}  {'-' * 9}  "
        f"{'-' * 9}  {'-' * 7}  {'-' * 9}"
    )
    for r in rows:
        print(
            f"  {r.team:22s}  {r.market_pp:7.2f}%  "
            f"{r.our_implied_p4_pp:8.2f}%  {r.delta_p4_pp:+7.2f}  "
            f"{_marker(r.edge_direction_p4):>9s}  "
            f"{r.our_implied_p8_pp:8.2f}%  {r.delta_p8_pp:+7.2f}  "
            f"{_marker(r.edge_direction_p8):>9s}"
        )


def main() -> None:
    """`python -m upset_propagation.our_vs_market` - print + optionally write snapshot."""
    from upset_propagation.logging_config import configure_interactive_logging

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: ./output/)",
    )
    parser.add_argument(
        "--sort",
        choices=["edge", "team", "market", "implied"],
        default="edge",
        help="Sort order (default: edge - biggest trade signal first)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Show only top N rows (default: all)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of table",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO-level logging",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Don't write output/our_vs_market.json (table-only mode)",
    )
    args = parser.parse_args()

    configure_interactive_logging(quiet=args.quiet)
    snapshot = build_snapshot(output_dir=args.output_dir)

    if not args.no_write:
        write_snapshot_file(snapshot, output_dir=args.output_dir)

    if args.json:
        print(json.dumps(snapshot.to_dict(), indent=2))
    else:
        _print_table(snapshot, sort=args.sort, top=args.top)


if __name__ == "__main__":
    main()