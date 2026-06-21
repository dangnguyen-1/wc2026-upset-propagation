"""Tests for upset_propagation.our_vs_market.

Coverage:
  - build_snapshot: no market_log (returns note), empty edges (returns note),
    happy path (returns rows)
  - _classify_edge: overpriced / underpriced / fair threshold logic
  - write_snapshot_file: persistence + JSON shape
  - Sort logic for the CLI table

Strategy: synthetic market_log.jsonl files in tmp_path. We exercise
the snapshot reader and the edge classifier as pure functions.
This is the primary trade signal surface, so tests prioritize what
a reader sees in the JSON (presence/absence of note, row
shape, direction labels).
"""

from __future__ import annotations

import json

import pytest

from upset_propagation.market_log import MARKET_LOG_FILENAME
from upset_propagation.our_vs_market import (
    EDGE_DIRECTION_THRESHOLD_RELATIVE,
    OUR_VS_MARKET_FILENAME,
    OurVsMarketRow,
    OurVsMarketSnapshot,
    _classify_edge,
    _sort_rows,
    build_snapshot,
    write_snapshot_file,
)


# Helpers


def _write_market_log_with_edges(
    tmp_path,
    timestamp: str = "2026-06-10T12:00:00+00:00",
    edges: list[dict] | None = None,
    n_groups_observed: int = 12,
) -> None:
    """Write a market_log.jsonl with one entry that has implied_vs_market."""
    entry = {
        "timestamp": timestamp,
        "n_teams_in_api": 48,
        "fair_probs_sum": 1.0,
        "top_market": [],
        "n_groups_observed": n_groups_observed,
    }
    if edges is not None:
        entry["implied_vs_market"] = edges
    (tmp_path / MARKET_LOG_FILENAME).write_text(json.dumps(entry) + "\n")


# _classify_edge: relative-threshold semantics


def test_classify_overpriced():
    """|delta/market| > threshold AND delta < 0 → overpriced.

    Market at 10pp, our at 8pp → delta -2pp, relative -20% → overpriced
    (we think team has less chance than market).
    """
    assert _classify_edge(delta_pp=-2.0, market_pp=10.0) == "overpriced"
    assert _classify_edge(delta_pp=-5.0, market_pp=10.0) == "overpriced"


def test_classify_underpriced():
    """|delta/market| > threshold AND delta > 0 → underpriced.

    Market at 10pp, our at 12pp → delta +2pp, relative +20% → underpriced
    (we think team has more chance than market).
    """
    assert _classify_edge(delta_pp=+2.0, market_pp=10.0) == "underpriced"
    assert _classify_edge(delta_pp=+5.0, market_pp=10.0) == "underpriced"


def test_classify_fair_inside_relative_threshold():
    """|delta/market| < threshold → fair.

    Market at 16pp, our at 16.2pp → relative 1.25% → fair (favorite noise
    that the old absolute threshold would also have classified fair).
    """
    assert _classify_edge(delta_pp=+0.2, market_pp=16.0) == "fair"
    assert _classify_edge(delta_pp=-0.5, market_pp=16.0) == "fair"
    assert _classify_edge(delta_pp=0.0, market_pp=10.0) == "fair"


def test_classify_underdog_signal_surfaces_correctly():
    """REGRESSION: relative threshold catches underdog moves that absolute
    threshold would miss.

    Haiti at 0.001 (0.1pp market), our at 0.002 (0.2pp implied) → delta
    +0.1pp absolute, +100% relative → underpriced. Under the old 0.5pp
    absolute threshold this would have been classified 'fair' and the
    signal would have been buried.
    """
    assert _classify_edge(delta_pp=+0.1, market_pp=0.1) == "underpriced"


def test_classify_favorite_noise_stays_fair():
    """REGRESSION: relative threshold treats small favorite moves as noise.

    France at 16.0pp market, our at 16.5pp implied → delta +0.5pp absolute,
    +3.1% relative → fair. Under the old 0.5pp absolute threshold this
    would have been at the boundary, surfacing noise as actionable.
    """
    assert _classify_edge(delta_pp=+0.5, market_pp=16.0) == "fair"


def test_classify_market_pp_zero_is_fair():
    """Defensive: team at exactly 0% market with any implied → fair.

    Division-by-zero protection. A team Polymarket doesn't list (or
    refuses to price) can't be traded; signal there is meaningless.
    """
    assert _classify_edge(delta_pp=+0.5, market_pp=0.0) == "fair"
    assert _classify_edge(delta_pp=-1.0, market_pp=0.0) == "fair"


def test_classify_edge_threshold_boundary():
    """At exactly the threshold (10% relative), classified as fair.

    abs(delta/market) < threshold means strict less-than → equality
    falls outside.
    """
    # Market 10pp, delta = market * threshold = exactly at threshold
    # → underpriced (NOT fair, because the strict-less-than check
    # excludes the boundary)
    boundary_delta = 10.0 * EDGE_DIRECTION_THRESHOLD_RELATIVE
    assert _classify_edge(delta_pp=+boundary_delta, market_pp=10.0) == "underpriced"
    assert _classify_edge(delta_pp=-boundary_delta, market_pp=10.0) == "overpriced"
    # Just under the threshold
    assert _classify_edge(
        delta_pp=+boundary_delta - 0.0001, market_pp=10.0,
    ) == "fair"


# build_snapshot: empty cases


def test_build_snapshot_returns_note_when_no_market_log(tmp_path):
    """No market_log.jsonl → empty snapshot with explanatory note."""
    snap = build_snapshot(tmp_path)
    assert snap.rows == []
    assert snap.note is not None
    assert "market_log.jsonl" in snap.note.lower() or MARKET_LOG_FILENAME in snap.note


def test_build_snapshot_returns_note_when_no_edges(tmp_path):
    """market_log entry exists but no implied_vs_market field → note explains why."""
    # Write an entry without implied_vs_market
    _write_market_log_with_edges(tmp_path, edges=None)
    snap = build_snapshot(tmp_path)
    assert snap.rows == []
    assert snap.note is not None
    assert "implied_vs_market" in snap.note


def test_build_snapshot_returns_note_when_edges_empty_list(tmp_path):
    """implied_vs_market: [] is still empty → note explains."""
    _write_market_log_with_edges(tmp_path, edges=[])
    snap = build_snapshot(tmp_path)
    assert snap.rows == []
    assert snap.note is not None


# build_snapshot: happy path


def test_build_snapshot_populates_rows(tmp_path):
    """With edges in market_log, build_snapshot translates into row dataclasses.

    Picks numbers that genuinely demonstrate each direction label under
    the relative-threshold classifier. Both p=4 and p=8 are populated
    independently; in this test we use identical p4/p8 values for
    simplicity. Real cron output will show divergence between them.
    """
    edges = [
        {
            "team": "Spain", "market_pp": 16.26,
            "implied_p4_pp": 13.18, "edge_p4_pp": -3.08,
            "implied_p8_pp": 13.18, "edge_p8_pp": -3.08,
        },
        {
            "team": "France", "market_pp": 15.20,
            "implied_p4_pp": 17.50, "edge_p4_pp": +2.30,
            "implied_p8_pp": 17.50, "edge_p8_pp": +2.30,
        },
        {
            "team": "Brazil", "market_pp": 8.53,
            "implied_p4_pp": 8.72, "edge_p4_pp": +0.19,
            "implied_p8_pp": 8.72, "edge_p8_pp": +0.19,
        },
    ]
    _write_market_log_with_edges(tmp_path, edges=edges)
    snap = build_snapshot(tmp_path)
    assert snap.note is None
    assert len(snap.rows) == 3

    spain = next(r for r in snap.rows if r.team == "Spain")
    assert spain.market_pp == 16.26
    assert spain.our_implied_p4_pp == 13.18
    assert spain.delta_p4_pp == -3.08
    assert spain.edge_direction_p4 == "overpriced"
    # p8 same as p4 in this test
    assert spain.our_implied_p8_pp == 13.18
    assert spain.edge_direction_p8 == "overpriced"

    france = next(r for r in snap.rows if r.team == "France")
    assert france.edge_direction_p4 == "underpriced"

    brazil = next(r for r in snap.rows if r.team == "Brazil")
    # delta/market = 0.19/8.53 = 2.2% < 10% → fair
    assert brazil.edge_direction_p4 == "fair"


def test_build_snapshot_p4_p8_can_disagree(tmp_path):
    """REGRESSION: when p=4 and p=8 produce different implied values,
    they can classify into different directions. The two columns
    surface this divergence - which is itself information for the reader.

    Example: market 10pp, p=4 implied 10.5pp (smoothed, +5% rel → fair),
    p=8 implied 12pp (sharper, +20% rel → underpriced). Same team,
    two views; the disagreement signals that the matcher's top scenario
    is bullish on the team but smoothing dilutes it.
    """
    edges = [{
        "team": "TeamX", "market_pp": 10.0,
        "implied_p4_pp": 10.5, "edge_p4_pp": +0.5,
        "implied_p8_pp": 12.0, "edge_p8_pp": +2.0,
    }]
    _write_market_log_with_edges(tmp_path, edges=edges)
    snap = build_snapshot(tmp_path)
    row = snap.rows[0]
    assert row.edge_direction_p4 == "fair"        # +5% rel
    assert row.edge_direction_p8 == "underpriced"  # +20% rel


def test_build_snapshot_propagates_n_groups_observed(tmp_path):
    """n_groups_observed comes through from the market_log entry."""
    _write_market_log_with_edges(
        tmp_path,
        edges=[{
            "team": "Spain", "market_pp": 16.0,
            "implied_p4_pp": 13.0, "edge_p4_pp": -3.0,
            "implied_p8_pp": 13.0, "edge_p8_pp": -3.0,
        }],
        n_groups_observed=7,
    )
    snap = build_snapshot(tmp_path)
    assert snap.n_groups_observed == 7


# write_snapshot_file


def test_write_snapshot_serializes_rows(tmp_path):
    """JSON output preserves all row fields including both p=4 and p=8."""
    snap = OurVsMarketSnapshot(
        computed_at="2026-06-10T12:00:00+00:00",
        source_market_log_ts="2026-06-10T11:00:00+00:00",
        n_groups_observed=12,
        rows=[
            OurVsMarketRow(
                team="Spain",
                market_pp=16.26,
                our_implied_p4_pp=13.18,
                delta_p4_pp=-3.08,
                edge_direction_p4="overpriced",
                our_implied_p8_pp=12.50,
                delta_p8_pp=-3.76,
                edge_direction_p8="overpriced",
            )
        ],
    )
    path = write_snapshot_file(snap, tmp_path)
    assert path == tmp_path / OUR_VS_MARKET_FILENAME
    parsed = json.loads(path.read_text())
    assert parsed["n_groups_observed"] == 12
    assert len(parsed["rows"]) == 1
    assert parsed["rows"][0]["team"] == "Spain"
    assert parsed["rows"][0]["edge_direction_p4"] == "overpriced"
    assert parsed["rows"][0]["edge_direction_p8"] == "overpriced"
    assert parsed["rows"][0]["our_implied_p8_pp"] == 12.50


def test_write_snapshot_includes_note_when_present(tmp_path):
    """Snapshots with a note include it in the JSON."""
    snap = OurVsMarketSnapshot(
        computed_at="2026-06-10T12:00:00+00:00",
        source_market_log_ts=None,
        n_groups_observed=0,
        rows=[],
        note="No market_log found",
    )
    write_snapshot_file(snap, tmp_path)
    parsed = json.loads((tmp_path / OUR_VS_MARKET_FILENAME).read_text())
    assert parsed["note"] == "No market_log found"


def test_write_snapshot_omits_note_when_absent(tmp_path):
    """When note is None, the field is omitted from JSON (clean schema)."""
    snap = OurVsMarketSnapshot(
        computed_at="2026-06-10T12:00:00+00:00",
        source_market_log_ts="2026-06-10T11:00:00+00:00",
        n_groups_observed=12,
        rows=[],
        note=None,
    )
    write_snapshot_file(snap, tmp_path)
    parsed = json.loads((tmp_path / OUR_VS_MARKET_FILENAME).read_text())
    assert "note" not in parsed


# _sort_rows


def _make_row(team: str, market: float, implied: float) -> OurVsMarketRow:
    """Test helper. Uses same implied for both p4 and p8 - fine for sort
    tests which only care about the magnitude ordering, not p4/p8 divergence.
    """
    delta = implied - market
    direction = _classify_edge(delta, market)
    return OurVsMarketRow(
        team=team,
        market_pp=market,
        our_implied_p4_pp=implied,
        delta_p4_pp=delta,
        edge_direction_p4=direction,
        our_implied_p8_pp=implied,
        delta_p8_pp=delta,
        edge_direction_p8=direction,
    )


def test_sort_by_edge_descending_absolute():
    """Default sort: biggest |edge| first (most actionable trade signal)."""
    rows = [
        _make_row("A", 5.0, 5.5),    # +0.5pp
        _make_row("B", 10.0, 6.0),   # -4.0pp
        _make_row("C", 8.0, 8.2),    # +0.2pp
        _make_row("D", 12.0, 14.5),  # +2.5pp
    ]
    sorted_rows = _sort_rows(rows, "edge")
    assert [r.team for r in sorted_rows] == ["B", "D", "A", "C"]


def test_sort_by_team_alphabetical():
    """team sort: A-Z, useful for looking up specific team."""
    rows = [
        _make_row("Brazil", 8.0, 8.0),
        _make_row("Argentina", 5.0, 5.0),
        _make_row("France", 15.0, 15.0),
    ]
    sorted_rows = _sort_rows(rows, "team")
    assert [r.team for r in sorted_rows] == ["Argentina", "Brazil", "France"]


def test_sort_by_market_descending():
    """market sort: top contenders by market view first."""
    rows = [
        _make_row("Brazil", 8.0, 8.0),
        _make_row("Spain", 16.0, 13.0),
        _make_row("Argentina", 5.0, 5.0),
    ]
    sorted_rows = _sort_rows(rows, "market")
    assert [r.team for r in sorted_rows] == ["Spain", "Brazil", "Argentina"]