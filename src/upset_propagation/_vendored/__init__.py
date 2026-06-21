"""MC Simulator (Stream 3) — standalone tournament outcome predictor.

Vendored Monte Carlo simulator by Duy Anh Nguyen.
See docs/VENDORING.md for details.

Imports rewritten from absolute (`from mc_simu.x import y`) to relative
(`from ._common import ...`) to work inside this package layout. The actual
logic — every other file in this directory — is byte-identical to upstream.
"""
from ._common import HARD_STOP_LOG, banner, hard_stop, infer_tournament_type
from .confederations import get_confederation
from .elo import (
    K_FACTORS,
    build_rating_history,
    elo_expected,
    elo_update,
    get_rating_as_of,
    mov_multiplier,
)
from .single_game import (
    MatchContext,
    MatchPrediction,
    ModelParams,
    elo_to_lambda,
    goal_distribution,
    hfa_log_goals,
    predict_match,
)

__all__ = [
    "HARD_STOP_LOG", "banner", "hard_stop", "infer_tournament_type",
    "get_confederation",
    "K_FACTORS", "elo_expected", "mov_multiplier", "elo_update",
    "build_rating_history", "get_rating_as_of",
    "MatchContext", "ModelParams", "MatchPrediction",
    "elo_to_lambda", "goal_distribution", "hfa_log_goals", "predict_match",
]
