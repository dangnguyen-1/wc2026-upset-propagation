"""Single-game prediction model — Elo + Poisson grid + HFA (additive log-goals).

Implements Phase 1 §2.1 Steps 1.4–1.6. Contract:

    predict_match(elo_home, elo_away, ctx, params) → MatchPrediction

The log-goal-space HFA (α/β) used here is intentionally separate from the Elo-space
HFA (+100) used in elo.elo_expected for rating UPDATES. Different physical convention,
both per spec (§2.1 Step 1.1 vs Step 1.6).

ModelParams.blend_match_weight defaults to 1.0 (match-Elo only) per §0.8 — overrides
spec's published default 0.75 because Phase 2 (roster overlay) is SKIPPED for v1.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
from scipy.stats import poisson  # noqa: E402


# ── Context + parameter records ───────────────────────────────────────────────


@dataclass
class MatchContext:
    """Per-match context used by predict_match.

    Construction at validation time:
        is_neutral          = bool(row.neutral)
        tournament_type     = infer_tournament_type(row.tournament)
        home_country        = row.home_team
        away_country        = row.away_team
        venue_country       = row.country
        venue_confederation = get_confederation(row.country)
        home_confederation  = get_confederation(row.home_team)
        away_confederation  = get_confederation(row.away_team)
        host_countries      = tournament_meta[label].host_countries  # may be []
        host_confederation  = tournament_meta[label].host_confederation  # may be None
        attendance_pct      = row.attendance_pct if pd.notna(row.attendance_pct) else 1.0
    """

    is_neutral: bool
    tournament_type: str
    home_country: str
    away_country: str
    venue_country: str
    venue_confederation: str
    home_confederation: str
    away_confederation: str
    host_countries: list[str] = field(default_factory=list)
    host_confederation: str | None = None
    attendance_pct: float = 1.0


@dataclass
class ModelParams:
    """Tunable structural parameters for predict_match.

    Default HFA values calibrated from TWO empirical sources after unit conversion
    (additive goals/game → log-goals via ln(λ_host / λ_base) at baseline λ=1.35):
      α=0.27 = average of 538 SPI WC2018 (+0.40 additive → 0.260 log) and
               Bilalić et al. 2021 (+0.44 additive → 0.282 log).
      β=0.09 = 538 SPI's same-confed-as-host (+0.13 additive → 0.09 log).

    Phase 1 plan originally used 0.40 and 0.13 (interpreting 538's published
    additive values directly as log-goals coefficients — wrong unit). Phase 1
    deep audit caught the scale issue; both empirical sources unit-converted
    point to ~0.27 not 0.40. Phase 4 LOTO-CV tune grid covers both interpretations.

    diagonal_inflation=0.20 calibrated empirically for international tournaments
    (Phase 1 Brier sweep on validation set). 538 SPI uses ~0.09 for club football.

    blend_match_weight=1.0 fixed per §0.8 (Phase 2 SKIPPED).
    """

    alpha: float = 0.27                  # own-country HFA, log-goals (538 + Bilalić unit-conv avg)
    beta: float = 0.09                   # same-confed-as-host (non-host) HFA (538 SPI additive → log)
    diagonal_inflation: float = 0.20     # draw mass uplift; empirical on int'l tournaments
    blend_match_weight: float = 1.0      # OVERRIDE spec default 0.75 — Phase 2 SKIPPED


@dataclass
class MatchPrediction:
    p_home: float
    p_draw: float
    p_away: float
    goal_grid: np.ndarray   # shape (max_goals+1, max_goals+1); sums to 1


# ── Step 1.4 — Elo diff → expected goals ──────────────────────────────────────


ELO_GOALS_DENOMINATOR = 1400.0  # spec correction #4 — see docstring


def elo_to_lambda(elo_team: float, elo_opp: float,
                  hfa_log_goals: float = 0.0,
                  league_avg_goals: float = 2.7) -> float:
    """Map Elo difference + additive log-goal HFA to expected goals (λ).

    base: split league_avg_goals half-and-half, scaled by 10^(ΔElo / D).
    HFA bonus is multiplicative on log-goal scale: λ → λ · exp(hfa_log_goals).

    SPEC CORRECTION (audit_report Phase 1 findings #4 + #6):

    Iteration 1 (Phase 1 initial): spec §2.1 Step 1.4 uses `D=400`, but Elo's
    400-denominator calibrates WIN ODDS RATIO not goal-count ratio. At ΔElo=400
    the spec formula gives 10× goal ratio, predict_match returned (1, 0, 0) for
    moderate gaps. Initially fixed to D=800 (10^(400/800)=3.16) calibrated to
    Elo's "+400 Elo → 0.91 win" rule.

    Iteration 2 (Phase 1 deep audit, finding #6): D=800 was still over-calibrated
    because Elo's 91% is the BINARY head-to-head win prob; in a 3-way W/D/L
    distribution under Poisson convolution, draws absorb mass, so real-data
    P_home at ΔElo=400 ≈ 0.74-0.78 (literature, also our validation set).
    Brier-minimization sweep on validation set 230 matches: global minimum at
    D=1400-1500, diag=0.15-0.30. Set D=1400.

    Phase 4 LOTO-CV (12 tournaments) may further tune D — Phase 1 calibration
    on 4-tournament subset is acknowledged minor overfit; structural choice not
    free-parameter.
    """
    half_avg = league_avg_goals / 2.0
    base_lambda = half_avg * 10.0 ** ((elo_team - elo_opp) / ELO_GOALS_DENOMINATOR)
    return base_lambda * np.exp(hfa_log_goals)


# ── Step 1.5 — Poisson grid with diagonal inflation ───────────────────────────


def goal_distribution(lambda_home: float, lambda_away: float,
                      diagonal_inflation: float = 0.20,
                      max_goals: int = 8) -> np.ndarray:
    """Build (max_goals+1)² grid where grid[i, j] = P(home=i, away=j).

    SIMPLIFIED diagonal-inflation heuristic — NOT full bivariate Poisson. Mechanism:
    independent Poisson product grid, multiply diagonal cells by (1 + inflation),
    renormalize so total mass = 1. With diagonal_inflation=0, result is a normalized
    truncated independent Poisson product (tail mass past max_goals is folded back
    uniformly via the final renormalization).

    Reference: this heuristic is INSPIRED by Karlis & Ntzoufras 2003 (J Royal Stat
    Soc Ser D 52(3):381-393), which formulates the full diagonal-inflated bivariate
    Poisson model using an explicit correlation parameter λ_3 in the joint distribution.
    Our impl is a simplified post-hoc diagonal boost — does not capture the goal
    correlation structure of true bivariate Poisson. 538 SPI club football uses
    similar simplified heuristic with ~0.09 inflation; we calibrated to 0.20
    empirically for international tournaments where draw rate is higher.

    Phase 4 plan decision #11: if Brier improvement justifies, switch to full
    Karlis-Ntzoufras via footBayes R package (rpy2 wrapper) or PyStan port.

    Underflow guard: at very high λ (ΔElo > ~6000 under D=1400), all P(0..max_goals)
    values fall below float64 precision (≈1e-300) and the grid sums to literal 0.
    In that pathological case we return a degenerate grid placing all mass on the
    corner most consistent with the λ ordering: λ_home >> λ_away → (max_goals, 0)
    (home blowout), λ_away >> λ_home → (0, max_goals), equal → (0, 0) draw.
    """
    i = np.arange(max_goals + 1)
    pmf_h = poisson.pmf(i, lambda_home)            # shape (n,)
    pmf_a = poisson.pmf(i, lambda_away)            # shape (n,)
    grid = np.outer(pmf_h, pmf_a)                  # (n, n), grid[i,j] = P(H=i)·P(A=j)
    np.fill_diagonal(grid, grid.diagonal() * (1.0 + diagonal_inflation))
    total = grid.sum()
    if total <= 0.0:
        # Underflow path — place all mass to preserve the W/D/L ordering implied by λ.
        grid = np.zeros_like(grid)
        if lambda_home > lambda_away:
            grid[max_goals, 0] = 1.0           # home blowout
        elif lambda_away > lambda_home:
            grid[0, max_goals] = 1.0           # away blowout
        else:
            grid[0, 0] = 1.0                    # 0-0 draw (parity at impossible scoreline)
        return grid
    return grid / total


# ── Step 1.6 — HFA in log-goal space + predict_match ──────────────────────────


def hfa_log_goals(team_country: str, team_confederation: str,
                  ctx: MatchContext, params: ModelParams) -> float:
    """Additive HFA bonus in log-goal space (spec §0.5 HFA matrix).

    bonus = α·I(team_country == venue_country) + β·I(same_confed_as_host AND not host)

    COVID attendance scaling (refined Phase 1 audit post-Wunderlich/Bilalić 2023):
        <25% capacity (empty/near-empty)  → 0.55× (Bilalić 2021: ~41% HA reduction empty)
        25-50% capacity                    → 0.65× (Euro 2020 1/3-cap: ~50% reduction vs qualif)
        50-75% capacity                    → 0.85× (partial — gradient interpolation)
        ≥75% capacity                      → 1.00× (no scaling)

    Spec original {0.67, 0.80, 1.00} 3-bin step was too gentle; magnitudes
    recalibrated against COVID-era empirical findings.

    UNKNOWN team_confederation never matches host_confederation (β=0 safe fallback).
    host_confederation=None (qualifier/friendly matches) → β=0 for everyone.
    """
    own_country = team_country == ctx.venue_country
    is_host = team_country in ctx.host_countries
    same_confed_as_host = (
        ctx.host_confederation is not None
        and team_confederation == ctx.host_confederation
        and not is_host
        and team_confederation != "UNKNOWN"
    )

    bonus = (params.alpha if own_country else 0.0) + \
            (params.beta if same_confed_as_host else 0.0)

    if ctx.attendance_pct < 0.25:
        bonus *= 0.55
    elif ctx.attendance_pct < 0.50:
        bonus *= 0.65
    elif ctx.attendance_pct < 0.75:
        bonus *= 0.85
    # else: 1.00× (no scaling)

    return bonus


def predict_match(elo_home: float, elo_away: float,
                  ctx: MatchContext, params: ModelParams) -> MatchPrediction:
    """Predict 3-way W/D/L probabilities + full goal grid.

    Spec convention: grid[i, j] = P(home=i, away=j). tril (i>j) = home wins,
    diag (i=j) = draw, triu (i<j) = away wins.
    """
    hfa_h = hfa_log_goals(ctx.home_country, ctx.home_confederation, ctx, params)
    hfa_a = hfa_log_goals(ctx.away_country, ctx.away_confederation, ctx, params)

    lambda_h = elo_to_lambda(elo_home, elo_away, hfa_log_goals=hfa_h)
    lambda_a = elo_to_lambda(elo_away, elo_home, hfa_log_goals=hfa_a)

    grid = goal_distribution(lambda_h, lambda_a, params.diagonal_inflation)

    p_home = float(np.sum(np.tril(grid, k=-1)))
    p_draw = float(np.sum(np.diag(grid)))
    p_away = float(np.sum(np.triu(grid, k=1)))

    return MatchPrediction(p_home=p_home, p_draw=p_draw, p_away=p_away, goal_grid=grid)


__all__ = [
    "MatchContext", "ModelParams", "MatchPrediction",
    "elo_to_lambda", "goal_distribution",
    "hfa_log_goals", "predict_match",
]
