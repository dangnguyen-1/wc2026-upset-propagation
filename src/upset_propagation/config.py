"""Central configuration - paths, constants, the FairLine API base URL.

Keep this file boring. Anything that might need to change between environments
(API URLs, default scenario count, output format) lives here so the rest of
the code stays focused on logic.
"""

from __future__ import annotations

from pathlib import Path

# Paths

# upset_propagation is at src/upset_propagation/config.py
# → parents[0] = upset_propagation/
# → parents[1] = src/
# → parents[2] = repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data lives at data/mc_simu/ rather than data/ to match the layout the
# vendored code expects (PROJECT_ROOT / "data" / "mc_simu" / ...). Keeps
# vendored snapshot byte-identical to upstream - see docs/VENDORING.md.
DATA_DIR = PROJECT_ROOT / "data" / "mc_simu"
OUTPUT_DIR = PROJECT_ROOT / "output"

R32_SEEDING_TABLE = DATA_DIR / "r32_seeding_table.json"
WC2026_GROUPS = DATA_DIR / "wc2026_groups.json"
WC2026_FIXTURES = DATA_DIR / "wc2026_fixtures.csv"
ELO_HISTORY = DATA_DIR / "elo_history.csv"


# Files that live inside output/ but are NOT scenario propagation tables.
# Centralized here so all matchers / iterators / future-readers share one
# source of truth. When adding a new output artifact (e.g. health.json,
# top_10_ranking.json, embedding cache, etc.), add its filename here
# rather than touching every matcher's skip list. Pattern emerged from
# repeated KeyError('scenario_id') bugs as later versions added non-scenario files.
NON_SCENARIO_FILENAMES = frozenset({
    "index.json",
    "validation_report.json",
    "top_10_ranking.json",
    "health.json",
    "our_vs_market.json",
})


# FairLine API

FAIRLINE_API_BASE = "https://seal-app-yatxw.ondigitalocean.app/api"

# The FairLine model: calibrated, devigged sportsbook-derived odds. Used as the
# calibration target - the framework's propagator output is fit to match
# this. Higher signal-to-noise than raw market prices because devigging
# removes sportsbook overrounds.
FAIRLINE_FAIR_ODDS_ENDPOINT = f"{FAIRLINE_API_BASE}/events/world_cup_2026/fair-odds"

# Raw prediction-market prices (Polymarket, Kalshi). Used for the
# comparison surface - what our framework's implied probabilities
# disagree with, that's the trade signal. Vig included; callers must
# renormalize per-platform.
FAIRLINE_PRICES_ENDPOINT = f"{FAIRLINE_API_BASE}/events/world_cup_2026/prices"


# WC 2026 reference data

# 12 groups, fixed by the December 5, 2025 draw.
GROUP_LETTERS = list("ABCDEFGHIJKL")

HOST_COUNTRIES = ["United States", "Mexico", "Canada"]
HOST_CONFEDERATION = "CONCACAF"

# Round names in order - matches the output table schema.
KNOCKOUT_ROUNDS = ["R32", "R16", "QF", "SF", "F", "Win"]

# Tournament-level event key (matches FairLine's `event` field).
EVENT_KEY = "world_cup_2026"


# Numerical tolerances

# Probabilities must sum to 1 ± SUM_TOLERANCE. Catches arithmetic bugs but
# allows floating-point noise from ~10k arithmetic ops.
SUM_TOLERANCE = 1e-9

# After calibration, the framework's empty-scenario tournament-winner prob
# for each team should match the FairLine model's baseline within this tolerance (probability,
# not bps - so 0.005 = 50 bps).
CALIBRATION_TOLERANCE = 0.005