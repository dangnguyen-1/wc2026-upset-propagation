"""Match-level Monte Carlo sampling primitives.

Implements Phase 3 §4.1 Step 3.4. Two sample paths:

  Group match  — sample (home_goals, away_goals) from goal_grid (9×9 = 81 cells)
  Knockout     — Bernoulli on P(team_a advances) = p_home + 0.5 × p_draw
                 (50/50 split of regulation-time draw mass between extra time +
                 penalty shootout; v1 simplification per spec §0.5)

Performance design — all predictions for a given MC run are PRE-COMPUTED ONCE:

  * 72 group fixtures (fixed teams + venue) → 72 CDF arrays
  * 48 × 47 = 2256 directed (team_a, team_b) KO pairings → scalar p_advance lookup

Each MC iteration then only does cheap searchsorted / Bernoulli draws. This is
the difference between ~1 minute for 100k iterations and ~80 minutes if we
re-call predict_match per sample.

KO context simplification: WC2026 knockout venues are TBD in wc2026_fixtures.csv
(spec §1.2). v1 treats all KO matches as venue-unknown:

  * α (own-country) never applies in KO — even USA/Mexico/Canada (host venue
    might or might not match the team's country; we don't gamble)
  * β (same-confed-as-host) still applies for CONCACAF non-host teams
    (Costa Rica, Jamaica, etc.) because the tournament IS in CONCACAF region

This is asymmetric vs Phase 1 group-stage HFA which uses actual venue, but the
asymmetry is small (β=0.09 log-goals) and applies to a few teams.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402

from ._common import banner  # noqa: E402,F401 — UTF-8 stdout side-effect
from .confederations import get_confederation  # noqa: E402
from .single_game import (  # noqa: E402
    MatchContext,
    MatchPrediction,
    ModelParams,
    predict_match,
)


MAX_GOALS = 8  # matches single_game.goal_distribution default → 9×9 grid

# Swappable predictor signature: (elo_a, elo_b, ctx) → MatchPrediction.
# Per spec §3.4: callers pre-bind their model's params via functools.partial,
# yielding a 3-argument callable. The bracket adapter / simulator pipeline does
# NOT know which model is plugged in — that's the harness/predictor separation.
Predictor = Callable[[float, float, MatchContext], MatchPrediction]


def make_elo_predictor(params: ModelParams | None = None) -> Predictor:
    """Default predictor: predict_match (Phase 1 Elo+Poisson+HFA) with frozen params.

    Used by `tournaments/*.py` when no explicit predictor is supplied. Alternative
    predictors (baseline.py, future market-consensus, etc.) bind their own params
    the same way.
    """
    if params is None:
        params = ModelParams()
    return partial(predict_match, params=params)


# ── Data records ──────────────────────────────────────────────────────────────


@dataclass
class GroupSampler:
    """Pre-computed CDF for sampling (home_goals, away_goals) from one fixture's goal_grid."""

    home_team: str
    away_team: str
    cdf_flat: np.ndarray  # shape (81,), cumulative sum of grid.flatten()


@dataclass
class HostInfo:
    """Tournament-level host metadata for HFA computation."""

    host_countries: list[str]      # e.g., ["United States", "Mexico", "Canada"]
    host_confederation: str         # e.g., "CONCACAF"


# ── Match-level samplers ──────────────────────────────────────────────────────


def sample_group_score(sampler: GroupSampler, rng: np.random.Generator) -> tuple[int, int]:
    """Sample (home_goals, away_goals) from the pre-computed flat CDF.

    Inverse-CDF via searchsorted on an 81-element array (microseconds per call).
    Used for one-off sampling and tests; the MC hot path uses
    sample_scores_batch for vectorized sampling across all 72 fixtures.
    """
    u = rng.random()
    idx = int(np.searchsorted(sampler.cdf_flat, u, side="right"))
    # Defensive: cdf might end at 1.0 - epsilon; clip to last cell.
    if idx >= 81:
        idx = 80
    return idx // (MAX_GOALS + 1), idx % (MAX_GOALS + 1)


def sample_scores_batch(
    cdfs_2d: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized inverse-CDF sample across N fixtures simultaneously.

    Args:
        cdfs_2d: shape (N, 81) — each row is a fixture's CDF (sums to 1, last cell = 1.0)
        rng:     np.random.Generator

    Returns: int array shape (N, 2) — (home_goals, away_goals) per fixture.

    Mechanism: u_array = rng.random(N). For each row find first index where
    cdf > u via row-wise argmax (boolean → int). ~3-5× faster than per-fixture
    sample_group_score loop because numpy's broadcast comparison is C-level.
    """
    n = cdfs_2d.shape[0]
    u = rng.random(n)
    # cdfs_2d > u[:, None] gives (N, 81) bool array — first True per row is our index.
    idx = np.argmax(cdfs_2d > u[:, None], axis=1)  # shape (N,)
    out = np.empty((n, 2), dtype=np.int64)
    out[:, 0] = idx // (MAX_GOALS + 1)  # home goals
    out[:, 1] = idx % (MAX_GOALS + 1)   # away goals
    return out


def sample_knockout_winner(
    team_a: str,
    team_b: str,
    p_advance: float,
    rng: np.random.Generator,
) -> str:
    """Bernoulli: team_a wins with probability p_advance else team_b.

    p_advance is computed as predict_match(team_a vs team_b).p_home + 0.5 ·
    p_draw — see build_ko_advance_table.
    """
    return team_a if rng.random() < p_advance else team_b


# ── Pre-compute builders ──────────────────────────────────────────────────────


def _build_group_match_context(
    home_team: str,
    away_team: str,
    venue_country: str,
    tournament_type: str,
    host: HostInfo,
    attendance_pct: float = 1.0,
) -> MatchContext:
    return MatchContext(
        is_neutral=(home_team != venue_country and away_team != venue_country),
        tournament_type=tournament_type,
        home_country=home_team,
        away_country=away_team,
        venue_country=venue_country,
        venue_confederation=get_confederation(venue_country) if venue_country else "UNKNOWN",
        home_confederation=get_confederation(home_team),
        away_confederation=get_confederation(away_team),
        host_countries=host.host_countries,
        host_confederation=host.host_confederation,
        attendance_pct=attendance_pct,
    )


def _build_ko_match_context(home_team: str, away_team: str, host: HostInfo) -> MatchContext:
    """Knockout context — venue TBD, so no α; β still applies for CONCACAF non-host."""
    return MatchContext(
        is_neutral=True,
        tournament_type="world_cup_final",
        home_country=home_team,
        away_country=away_team,
        venue_country="",                          # TBD → α disabled
        venue_confederation=host.host_confederation,
        home_confederation=get_confederation(home_team),
        away_confederation=get_confederation(away_team),
        host_countries=host.host_countries,
        host_confederation=host.host_confederation,
        attendance_pct=1.0,
    )


def build_group_samplers(
    fixtures: list[dict],
    ratings: dict[str, float],
    host: HostInfo,
    predictor: Predictor,
) -> list[GroupSampler]:
    """Pre-compute CDFs for the group-stage fixtures.

    Args:
        fixtures:  list of dicts with keys home_team, away_team, venue_country,
                   tournament_type, attendance_pct. Each = one group match.
        ratings:   pre-tournament Elo snapshot {team: rating} — frozen for the
                   entire MC run.
        host:      HostInfo for the tournament.
        predictor: callable `(elo_a, elo_b, ctx) → MatchPrediction`. Built once
                   per MC run via `make_elo_predictor(params)` or by any other
                   model (e.g., `baseline.predict_match_40_40_20`).

    Returns one GroupSampler per fixture in input order. Caller indexes by
    fixture position.
    """
    samplers: list[GroupSampler] = []
    for fx in fixtures:
        home, away = fx["home_team"], fx["away_team"]
        ctx = _build_group_match_context(
            home_team=home,
            away_team=away,
            venue_country=fx.get("venue_country", ""),
            tournament_type=fx.get("tournament_type", "world_cup_final"),
            host=host,
            attendance_pct=fx.get("attendance_pct", 1.0),
        )
        pred = predictor(ratings[home], ratings[away], ctx)
        cdf = np.cumsum(pred.goal_grid.flatten())
        # Force last cell to exactly 1.0 to guarantee searchsorted hits a valid index.
        cdf[-1] = 1.0
        samplers.append(GroupSampler(home_team=home, away_team=away, cdf_flat=cdf))
    return samplers


def build_ko_advance_table(
    teams: list[str],
    ratings: dict[str, float],
    host: HostInfo,
    predictor: Predictor,
) -> dict[tuple[str, str], float]:
    """Pre-compute P(team_a advances) for every ordered KO pairing (a, b), a ≠ b.

    Returns {(team_a, team_b): p_advance_a} where
        p_advance_a = pred.p_home + 0.5 · pred.p_draw
    and pred is predictor(elo_a, elo_b, ctx_with_team_a_as_home).

    Asymmetric when one team is CONCACAF non-host and other isn't — but for any
    given (A, B, A→B) pairing, p[(A,B)] + p[(B,A)] = 1.0 holds because swapping
    home/away in predict_match transposes the goal grid (verified by audit
    `audit_phase3_wiring.py::audit_ko_advance_symmetry`).
    """
    table: dict[tuple[str, str], float] = {}
    for a in teams:
        ea = ratings[a]
        for b in teams:
            if a == b:
                continue
            ctx = _build_ko_match_context(a, b, host)
            pred = predictor(ea, ratings[b], ctx)
            table[(a, b)] = pred.p_home + 0.5 * pred.p_draw
    return table
