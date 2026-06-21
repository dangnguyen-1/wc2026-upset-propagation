"""WC2026 tournament orchestrator — Phase 3 §4.1 Step 3.5.

End-to-end one-iteration sim:

    72 group matches → 12 group standings (rank_group)
    → 12 winners + 12 runners-up + 12 third-place teams
    → top 8 thirds (rank_best_thirds)
    → R32 seeding via r32_seeding_table.json + static runner-up bracket
    → R32 (16 matches) → R16 (8) → QF (4) → SF (2) → Final (1)
    → return (champion, {group_letter: winner_name})

Bracket structure extracted from Wikipedia "2026 FIFA World Cup knockout stage"
(HTML cached in data/mc_simu/cache/wc2026_knockout.html). 8 winner-vs-3rd
matches are resolved dynamically from r32_seeding_table.json (Phase 0 parse,
validated against 8 FIFA Annex C constraints in test_mc_standings.py). The
other 8 R32 matches are static bracket positions per FIFA's draw.

Match numbering follows FIFA fixture IDs 73-104 (R32 = 73-88, R16 = 89-96,
QF = 97-100, SF = 101-102, 3rd-place = 103 [skipped v1], Final = 104).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .._common import banner  # noqa: E402,F401 — UTF-8 stdout side-effect
from ..simulator import (  # noqa: E402
    HostInfo,
    Predictor,
    build_group_samplers,
    build_ko_advance_table,
    make_elo_predictor,
    sample_knockout_winner,
    sample_scores_batch,
)
from ..single_game import ModelParams  # noqa: E402
from ..standings import (  # noqa: E402
    GroupMatch,
    TeamStats,
    compute_stats,
    rank_best_thirds,
    rank_group,
)


# ── Bracket constants ─────────────────────────────────────────────────────────


GROUP_LETTERS = list("ABCDEFGHIJKL")  # 12 groups
WC2026_HOST_COUNTRIES = ["United States", "Mexico", "Canada"]
WC2026_HOST_CONFEDERATION = "CONCACAF"

# Name aliases — WC2026 groups.json uses modern FIFA names; elo_history.csv
# (built from 1872-2026 Kaggle data) uses historical canonical names.
ELO_HISTORY_NAME_ALIASES: dict[str, str] = {
    "Czechia":  "Czech Republic",  # FIFA renamed 2022; pre-2022 history is "Czech Republic"
    "Curacao":  "Curaçao",    # cedilla in canonical Kaggle name
}


# R32 matches 73-88. Each: (match_id, left_source, right_source).
# Source tuples:
#   ("W",  letter)         — winner of group `letter`
#   ("RU", letter)         — runner-up of group `letter`
#   ("T3", winner_letter)  — best-third assigned to winner `winner_letter` via
#                            r32_seeding_table.json lookup (key = sorted 8-letter
#                            string of advancing thirds' group origins)
#
# 8 of 16 are static (RU-RU or W-RU pairings). The other 8 are W-vs-T3 and
# depend on which 8 thirds advance (495 possible keys).
R32_BRACKET: list[tuple[int, tuple[str, str], tuple[str, str]]] = [
    (73, ("RU", "A"), ("RU", "B")),
    (74, ("W",  "E"), ("T3", "E")),
    (75, ("W",  "F"), ("RU", "C")),
    (76, ("W",  "C"), ("RU", "F")),
    (77, ("W",  "I"), ("T3", "I")),
    (78, ("RU", "E"), ("RU", "I")),
    (79, ("W",  "A"), ("T3", "A")),
    (80, ("W",  "L"), ("T3", "L")),
    (81, ("W",  "D"), ("T3", "D")),
    (82, ("W",  "G"), ("T3", "G")),
    (83, ("RU", "K"), ("RU", "L")),
    (84, ("W",  "H"), ("RU", "J")),
    (85, ("W",  "B"), ("T3", "B")),
    (86, ("W",  "J"), ("RU", "H")),
    (87, ("W",  "K"), ("T3", "K")),
    (88, ("RU", "D"), ("RU", "G")),
]

# R16-Final: (match_id, prev_left_match_id, prev_right_match_id). Both teams =
# winner of named previous match.
LATER_ROUNDS: list[tuple[int, int, int]] = [
    # R16 (M89-96)
    (89, 73, 75), (90, 74, 77), (91, 76, 78), (92, 79, 80),
    (93, 83, 84), (94, 81, 82), (95, 86, 88), (96, 85, 87),
    # QF (M97-100)
    (97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96),
    # SF (M101-102)
    (101, 97, 98), (102, 99, 100),
    # Final (M104) — M103 (3rd-place play-off) is intentionally skipped in v1
    (104, 101, 102),
]


# ── Data loading ──────────────────────────────────────────────────────────────


@dataclass
class WC2026Bundle:
    """All static data needed for one MC run, loaded once."""

    fixtures_df: pd.DataFrame                 # 104 rows from wc2026_fixtures.csv
    group_membership: dict[str, list[str]]    # {"A": [4 teams], ...}
    r32_table: dict[str, dict[str, str]]      # {"ABCDEFGH": {"1E": "3C", ...}}
    ratings: dict[str, float]                 # frozen Elo snapshot at sim time
    host: HostInfo
    params: ModelParams


def load_wc2026_bundle(
    ratings: dict[str, float],
    params: ModelParams | None = None,
    data_dir: Path | None = None,
) -> WC2026Bundle:
    """Load fixtures + group membership + R32 seeding + tournament metadata.

    Args:
        ratings: pre-tournament Elo snapshot. Keyed by the elo_history name
            (e.g., "Czech Republic" not "Czechia"). load_wc2026_bundle resolves
            WC2026 names via ELO_HISTORY_NAME_ALIASES so callers can pass either.
        params: ModelParams (defaults to Phase 1 calibrated values).
        data_dir: override for data/mc_simu/ path (default = PROJECT_ROOT/data/mc_simu).
    """
    if params is None:
        params = ModelParams()
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data" / "mc_simu"

    # Resolve aliases: build ratings dict keyed by WC2026 group names.
    ratings_resolved: dict[str, float] = {}
    for wc_name, elo_name in ELO_HISTORY_NAME_ALIASES.items():
        if wc_name not in ratings and elo_name in ratings:
            ratings_resolved[wc_name] = ratings[elo_name]
    ratings_resolved.update(ratings)
    ratings = ratings_resolved

    fixtures_df = pd.read_csv(data_dir / "wc2026_fixtures.csv")

    with (data_dir / "wc2026_groups.json").open() as f:
        groups_data = json.load(f)
    group_membership = {g: list(teams) for g, teams in groups_data["groups"].items()}

    with (data_dir / "r32_seeding_table.json").open() as f:
        r32_table = json.load(f)

    host = HostInfo(
        host_countries=list(WC2026_HOST_COUNTRIES),
        host_confederation=WC2026_HOST_CONFEDERATION,
    )

    return WC2026Bundle(
        fixtures_df=fixtures_df,
        group_membership=group_membership,
        r32_table=r32_table,
        ratings=ratings,
        host=host,
        params=params,
    )


# ── Venue enrichment ──────────────────────────────────────────────────────────


def enrich_group_venues(fixtures_df: pd.DataFrame, host_countries: list[str]) -> pd.DataFrame:
    """Fill TBD venues for matches involving a host team.

    Phase 0 fixture builder only set venue for the opening match (Mexico-SA).
    Per FIFA policy, each host plays all 3 group matches in its own country.
    For group matches involving a host team, set venue_country = that host.
    Non-host group matches stay TBD (model uses no α for those).
    """
    df = fixtures_df.copy()
    host_set = set(host_countries)
    group_mask = df["stage"] == "group"
    tbd_mask = df["venue_country"] == "TBD"
    needs_enrichment = group_mask & tbd_mask
    for idx in df.index[needs_enrichment]:
        home, away = df.at[idx, "home_team"], df.at[idx, "away_team"]
        host_in_match = (host_set & {home, away})
        if host_in_match:
            df.at[idx, "venue_country"] = next(iter(host_in_match))
    return df


# ── Single-tournament simulator ───────────────────────────────────────────────


def simulate_tournament(
    group_cdfs_per_group: dict[str, np.ndarray],
    group_team_pairs: dict[str, list[tuple[str, str]]],
    r32_table: dict[str, dict[str, str]],
    ko_advance: dict[tuple[str, str], float],
    fifa_ranking: dict[str, float],
    rng: np.random.Generator,
) -> tuple[str, dict[str, str]]:
    """One MC iteration. Returns (champion, group_winners_by_letter).

    Args:
        group_cdfs_per_group: {group_letter: ndarray shape (6, 81)} — pre-stacked
            CDFs for each group's 6 fixtures, for vectorized batch sampling.
        group_team_pairs: {group_letter: [(home_team, away_team) × 6]} — parallel
            to group_cdfs_per_group rows, supplies team names for GroupMatch.
        r32_table: 495-combo seeding map
        ko_advance: precomputed {(team_a, team_b): p(a advances)} for all 48×47 pairs
        fifa_ranking: {team: score} used as final-final tiebreaker in standings
        rng: np.random.default_rng(seed)
    """
    # --- 1. Sim 72 group matches; assemble per-group match list ---
    matches_by_group: dict[str, list[GroupMatch]] = {}
    for group_letter in GROUP_LETTERS:
        scores = sample_scores_batch(group_cdfs_per_group[group_letter], rng)  # (6, 2)
        pairs = group_team_pairs[group_letter]
        matches_by_group[group_letter] = [
            GroupMatch(
                home_team=h,
                away_team=a,
                home_goals=int(scores[i, 0]),
                away_goals=int(scores[i, 1]),
            )
            for i, (h, a) in enumerate(pairs)
        ]

    # --- 2. Rank each group; capture winner/runner-up/third + per-team stats ---
    winners: dict[str, str] = {}
    runners_up: dict[str, str] = {}
    thirds: list[tuple[str, str]] = []  # (team, group_letter)
    all_team_stats: dict[str, TeamStats] = {}

    for group_letter in GROUP_LETTERS:
        matches = matches_by_group[group_letter]
        group_stats = compute_stats(matches)
        ranked = rank_group(matches, final_tiebreaker=fifa_ranking, stats=group_stats)
        winners[group_letter] = ranked[0]
        runners_up[group_letter] = ranked[1]
        thirds.append((ranked[2], group_letter))
        all_team_stats.update(group_stats)

    # --- 3. Pick top 8 third-place teams (no H2H — different groups) ---
    ranked_thirds = rank_best_thirds(thirds, all_team_stats, fifa_ranking)
    top8_thirds = ranked_thirds[:8]  # list of (team, group_letter)
    third_origin_letters = sorted(g for _team, g in top8_thirds)
    r32_key = "".join(third_origin_letters)
    seeding = r32_table[r32_key]  # {"1A": "3X", "1E": "3Y", ...}
    third_by_group: dict[str, str] = {g: t for t, g in top8_thirds}

    # --- 4. Resolve R32 bracket: each match → (team_a, team_b) ---
    # Bracket sources: ("W", g) | ("RU", g) | ("T3", winner_letter)
    def resolve_source(source: tuple[str, str]) -> str:
        kind, key = source
        if kind == "W":
            return winners[key]
        if kind == "RU":
            return runners_up[key]
        if kind == "T3":
            # Look up which third advances vs this winner
            third_slot = seeding[f"1{key}"]  # e.g., "3C"
            third_group = third_slot[1]
            return third_by_group[third_group]
        raise ValueError(f"Unknown source kind: {kind}")

    match_winners: dict[int, str] = {}

    # R32 matches
    for match_id, left_src, right_src in R32_BRACKET:
        team_a = resolve_source(left_src)
        team_b = resolve_source(right_src)
        p_advance_a = ko_advance[(team_a, team_b)]
        match_winners[match_id] = sample_knockout_winner(team_a, team_b, p_advance_a, rng)

    # R16 / QF / SF / Final
    for match_id, prev_left, prev_right in LATER_ROUNDS:
        team_a = match_winners[prev_left]
        team_b = match_winners[prev_right]
        p_advance_a = ko_advance[(team_a, team_b)]
        match_winners[match_id] = sample_knockout_winner(team_a, team_b, p_advance_a, rng)

    champion = match_winners[104]
    return champion, winners


# ── Pre-build everything needed for the iteration loop ────────────────────────


@dataclass
class WC2026SimContext:
    """Pre-computed artifacts for run_monte_carlo. Built once, reused N times.

    group_cdfs_per_group/group_team_pairs are arranged for vectorized batch
    sampling in the hot loop (one numpy call per group, 6 fixtures at a time).
    """

    group_cdfs_per_group: dict[str, np.ndarray]            # letter → (6, 81)
    group_team_pairs: dict[str, list[tuple[str, str]]]     # letter → 6×(home, away)
    r32_table: dict[str, dict[str, str]]
    ko_advance: dict[tuple[str, str], float]
    fifa_ranking: dict[str, float]
    teams: list[str]


def build_sim_context(
    bundle: WC2026Bundle,
    predictor: Predictor | None = None,
) -> WC2026SimContext:
    """Pre-compute group CDFs + KO advance table + bracket indexing.

    Cost: 72 + 48×47 = 2328 predictor calls (~600ms with default Elo). Run
    once per MC invocation; iterations then only do cheap vectorized
    searchsorts + Bernoulli.

    Args:
        bundle:    WC2026 data bundle (fixtures + ratings + groups + R32 table)
        predictor: optional model `(elo_a, elo_b, ctx) → MatchPrediction`.
                   Defaults to `make_elo_predictor(bundle.params)` (Phase 1
                   Elo+Poisson+HFA). Pass `baseline.predict_match_40_40_20`
                   for ignorance baseline, or any other model implementing
                   the Predictor protocol.
    """
    if predictor is None:
        predictor = make_elo_predictor(bundle.params)

    fixtures_df = enrich_group_venues(bundle.fixtures_df, bundle.host.host_countries)

    group_df = fixtures_df[fixtures_df["stage"] == "group"].copy()
    group_fixtures = group_df.to_dict("records")
    for fx in group_fixtures:
        fx["tournament_type"] = "world_cup_final"

    samplers_list = build_group_samplers(
        fixtures=group_fixtures,
        ratings=bundle.ratings,
        host=bundle.host,
        predictor=predictor,
    )

    # Stack CDFs into per-group (6, 81) arrays for batch sampling.
    group_cdfs_per_group: dict[str, list[np.ndarray]] = {g: [] for g in GROUP_LETTERS}
    group_team_pairs: dict[str, list[tuple[str, str]]] = {g: [] for g in GROUP_LETTERS}
    for fx, sampler in zip(group_fixtures, samplers_list):
        g = fx["group"]
        group_cdfs_per_group[g].append(sampler.cdf_flat)
        group_team_pairs[g].append((sampler.home_team, sampler.away_team))
    group_cdfs_stacked = {g: np.stack(cdfs) for g, cdfs in group_cdfs_per_group.items()}

    teams: list[str] = []
    for group_letter in GROUP_LETTERS:
        teams.extend(bundle.group_membership[group_letter])

    ko_advance = build_ko_advance_table(
        teams=teams,
        ratings=bundle.ratings,
        host=bundle.host,
        predictor=predictor,
    )

    # FIFA ranking proxy = Elo (higher Elo → better rank in our convention)
    fifa_ranking = dict(bundle.ratings)

    return WC2026SimContext(
        group_cdfs_per_group=group_cdfs_stacked,
        group_team_pairs=group_team_pairs,
        r32_table=bundle.r32_table,
        ko_advance=ko_advance,
        fifa_ranking=fifa_ranking,
        teams=teams,
    )


# ── Monte Carlo loop ──────────────────────────────────────────────────────────


def run_monte_carlo(
    bundle: WC2026Bundle,
    n_iterations: int = 100_000,
    seed: int = 42,
    progress: bool = True,
    predictor: Predictor | None = None,
) -> dict[str, Any]:
    """Run N tournament simulations. Aggregate champion + group-winner frequencies.

    Returns dict with two keys:
        'champion':       {team: {'mc_fair_prob': float, 'mc_se': float, 'n_iterations': int}}
        'group_winners':  {group_letter: {team: {'mc_fair_prob', 'mc_se', 'n_iterations'}}}

    Args:
        predictor: optional model — see build_sim_context. Default Elo+Poisson+HFA.
    """
    sim = build_sim_context(bundle, predictor=predictor)
    rng = np.random.default_rng(seed)

    champion_counter: dict[str, int] = defaultdict(int)
    group_winner_counter: dict[str, dict[str, int]] = {
        g: defaultdict(int) for g in GROUP_LETTERS
    }

    iterator = range(n_iterations)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"MC ({n_iterations:,} iter)")
        except ImportError:
            pass

    for _ in iterator:
        champion, group_winners = simulate_tournament(
            group_cdfs_per_group=sim.group_cdfs_per_group,
            group_team_pairs=sim.group_team_pairs,
            r32_table=sim.r32_table,
            ko_advance=sim.ko_advance,
            fifa_ranking=sim.fifa_ranking,
            rng=rng,
        )
        champion_counter[champion] += 1
        for g, w in group_winners.items():
            group_winner_counter[g][w] += 1

    def to_results(counter: dict[str, int], n: int) -> dict[str, dict[str, float | int]]:
        out: dict[str, dict[str, float | int]] = {}
        for team, count in counter.items():
            p = count / n
            out[team] = {
                "mc_fair_prob": p,
                "mc_se": float(np.sqrt(p * (1 - p) / n)),
                "n_iterations": n,
            }
        return out

    return {
        "champion": to_results(champion_counter, n_iterations),
        "group_winners": {
            g: to_results(group_winner_counter[g], n_iterations) for g in GROUP_LETTERS
        },
    }
