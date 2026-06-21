"""Elo ratings engine — match-Elo only (Phase 1, no roster overlay per §0.8).

Implements:
    elo_expected            — logistic expected outcome for rating UPDATE path
    K_FACTORS               — eloratings.net 5-bucket spec (decision #9)
    mov_multiplier          — margin-of-victory boost
    elo_update              — incremental Elo step
    build_rating_history    — chronological full-history processor
    get_rating_as_of        — strict less-than as-of lookup (no lookahead)

The Elo-space HFA (`hfa_elo=100`) here is used in `elo_expected` for rating
UPDATES only. The log-goal-space HFA (α/β) used in match PREDICTION lives in
single_game.hfa_log_goals — see plan §"Decisions adopted" #9 and module docstring
there for the rationale of the dual-HFA convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from ._common import infer_tournament_type  # noqa: E402


# ── K-factor table per eloratings.net 5-bucket spec (Decision #9) ─────────────
# 6 logical buckets (Phase 0 hotfix split friendly into bilateral=20 + multi-team=30).
K_FACTORS: dict[str, int] = {
    "world_cup_final":   60,
    "continental_final": 50,
    "qualifier":         40,
    "nations_league":    40,
    "other_tournament":  30,
    "friendly":          20,
}


# ── Step 1.1 — expected outcome ───────────────────────────────────────────────


def elo_expected(rating_home: float, rating_away: float,
                 hfa_elo: float = 100.0, is_neutral: bool = False) -> float:
    """Return P(home wins / advances) via logistic from Elo difference.

    hfa_elo: home-field advantage in Elo points, added to rating_home iff NOT neutral.
    """
    rh = rating_home + (0.0 if is_neutral else hfa_elo)
    return 1.0 / (1.0 + 10 ** (-(rh - rating_away) / 400.0))


# ── Step 1.2 — margin of victory + update ─────────────────────────────────────


def mov_multiplier(goal_diff: int) -> float:
    """eloratings.net sqrt-based MoV bonus."""
    gd = abs(int(goal_diff))
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    if gd == 3:
        return 1.75
    return 1.75 + (gd - 3) / 8.0


def elo_update(rating: float, expected: float, actual: float,
               K: float, mov_mult: float = 1.0) -> float:
    """Single-side Elo step. actual ∈ {0, 0.5, 1}."""
    return rating + K * mov_mult * (actual - expected)


# ── Step 1.3 — history builder ────────────────────────────────────────────────


def _actual_score(home_score: float, away_score: float) -> tuple[float, float]:
    if home_score > away_score:
        return 1.0, 0.0
    if home_score == away_score:
        return 0.5, 0.5
    return 0.0, 1.0


def build_rating_history(
    matches_df: pd.DataFrame,
    *,
    initial_rating: float = 1500.0,
    cutoff_date: pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """Process matches chronologically, return {team: DataFrame(date, rating_after, ...)}.

    Input contract (Kaggle results.csv / matches_1998_2026.csv):
        required cols: date, home_team, away_team, home_score, away_score,
                       tournament, neutral
        optional cols: tournament_type (recomputed defensively from `tournament`)

    Filtering:
        - drop NaN home_score or away_score (e.g., 72 future-WC2026 rows)
        - drop rows where date > cutoff_date (default: today)

    Tournament type is re-derived via infer_tournament_type(row.tournament) at
    runtime so the function works whether or not the CSV column is fresh.

    Same-day matches: processed strictly chronologically with intra-day chaining
    (eloratings.net standard). Tournament's same-day group games therefore update
    ratings in order. The validation gate's get_rating_as_of uses STRICT less-than
    semantics to prevent same-day intra-tournament lookahead at prediction time.

    No inter-match decay applied (spec §0.5 "Time decay halflife 18 months | Fix v1"
    — kept fixed in v1; future-reserved for v2).

    Returns: dict[team_name → DataFrame] with columns:
        date           (pd.Timestamp), rating_after (float),
        opponent (str), tournament_type (str), K (int), mov (float),
        gs_self (int), gs_opp (int)
    """
    if cutoff_date is None:
        cutoff_date = pd.Timestamp.now()

    df = matches_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["home_score"].notna() & df["away_score"].notna()]
    df = df[df["date"] <= cutoff_date]
    df = df.sort_values("date", kind="stable").reset_index(drop=True)

    # Normalize neutral column to bool — Kaggle CSV stores TRUE/FALSE as strings/bools.
    if df["neutral"].dtype == object:
        df["neutral_bool"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    else:
        df["neutral_bool"] = df["neutral"].astype(bool)

    ratings: dict[str, float] = {}
    history: dict[str, list[dict]] = {}

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        rh = ratings.get(home, initial_rating)
        ra = ratings.get(away, initial_rating)

        exp_h = elo_expected(rh, ra, hfa_elo=100.0, is_neutral=row.neutral_bool)
        exp_a = 1.0 - exp_h
        act_h, act_a = _actual_score(row.home_score, row.away_score)

        ttype = infer_tournament_type(row.tournament)
        K = K_FACTORS.get(ttype, 20)  # safety: unknown → friendly K
        mov = mov_multiplier(int(row.home_score) - int(row.away_score))

        new_rh = elo_update(rh, exp_h, act_h, K, mov)
        new_ra = elo_update(ra, exp_a, act_a, K, mov)
        ratings[home], ratings[away] = new_rh, new_ra

        history.setdefault(home, []).append({
            "date": row.date, "rating_after": new_rh,
            "opponent": away, "tournament_type": ttype,
            "K": K, "mov": mov,
            "gs_self": int(row.home_score), "gs_opp": int(row.away_score),
        })
        history.setdefault(away, []).append({
            "date": row.date, "rating_after": new_ra,
            "opponent": home, "tournament_type": ttype,
            "K": K, "mov": mov,
            "gs_self": int(row.away_score), "gs_opp": int(row.home_score),
        })

    return {team: pd.DataFrame(rows) for team, rows in history.items()}


# ── Step 1.3 — as-of lookup ───────────────────────────────────────────────────


def get_rating_as_of(history: dict[str, pd.DataFrame],
                     team: str, date: pd.Timestamp,
                     initial_rating: float = 1500.0) -> float:
    """Strict less-than as-of lookup. Returns rating after last match where match.date < date.

    Returns initial_rating (1500) if team unseen before `date` (new FIFA member +
    warm-up edge). pd.Timestamp comparison is safe across naive/aware in this dataset.
    """
    df = history.get(team)
    if df is None or df.empty:
        return initial_rating
    date = pd.Timestamp(date)
    # `date_after` column is sorted by build_rating_history's stable sort.
    mask = df["date"] < date
    if not mask.any():
        return initial_rating
    return float(df.loc[mask, "rating_after"].iloc[-1])


def latest_ratings(history: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Return {team → latest rating_after}. Convenience for sanity checks."""
    return {team: float(df["rating_after"].iloc[-1]) for team, df in history.items() if not df.empty}


def history_to_long_df(history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Flatten {team: DataFrame} → long-format DataFrame with `team` column.

    Schema: team, date, rating_after, opponent, tournament_type, K, mov, gs_self, gs_opp
    """
    parts: list[pd.DataFrame] = []
    for team, df in history.items():
        if df.empty:
            continue
        d = df.copy()
        d.insert(0, "team", team)
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["team", "date", "rating_after", "opponent",
                                      "tournament_type", "K", "mov", "gs_self", "gs_opp"])
    return pd.concat(parts, ignore_index=True)


__all__ = [
    "K_FACTORS",
    "elo_expected",
    "mov_multiplier",
    "elo_update",
    "build_rating_history",
    "get_rating_as_of",
    "latest_ratings",
    "history_to_long_df",
]
