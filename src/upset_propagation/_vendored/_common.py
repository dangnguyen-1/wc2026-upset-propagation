"""Shared utilities for MC Simu Phase 0+ — HARD STOP logic, banners, mappings."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows (default cp1252 cannot encode banners).
# Side-effect on import — all mc_simu modules importing _common inherit this.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover — best-effort
            pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HARD_STOP_LOG = PROJECT_ROOT / "logs" / "mc_simu_hard_stop.log"
OVERRIDE_ENV = "MC_SIMU_OVERRIDE_HARD_STOP"


def hard_stop(check_name: str, actual: object, expected: object, *, exit_code: int = 2) -> None:
    """Spec §0.7. Write failure summary to logs/mc_simu_hard_stop.log and exit 2.

    Override via env var MC_SIMU_OVERRIDE_HARD_STOP=1 (must document reason in
    audit_overrides.md).
    """
    if os.environ.get(OVERRIDE_ENV) == "1":
        print(f"WARN: {check_name} would HARD STOP but {OVERRIDE_ENV}=1 — continuing")
        print("      Document reason in audit_overrides.md")
        return

    HARD_STOP_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with HARD_STOP_LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {check_name}\n  actual:   {actual}\n  expected: {expected}\n\n")

    print(f"HARD STOP: {check_name}")
    print(f"  actual:   {actual}")
    print(f"  expected: {expected}")
    sys.exit(exit_code)


def banner(text: str, width: int = 65) -> None:
    """Print a === bordered banner — matches FairLine logging style per spec §8."""
    print("=" * width)
    print(text)
    print("=" * width)


# Substring lists for tournament classification heuristic. First match wins; ordering matters.
_CONTINENTAL_FINAL_NAMES = (
    "uefa euro",
    "copa américa", "copa america",
    "african cup of nations", "africa cup of nations",
    "afc asian cup", "asian cup",
    "gold cup", "concacaf championship",
    "oceania nations cup", "ofc nations cup",
    "confederations cup",
)

_OTHER_TOURNAMENT_NAMES = (
    "cecafa", "cosafa", "aff championship", "gulf cup", "saff",
    "eaff", "cfu caribbean", "uncaf", "waff", "arab cup",
    "kirin cup", "king's cup", "kings cup", "china cup",
    "cyprus international", "olympic", "afc challenge",
    "african nations championship",
    "conifa",
    "island games",
)


def infer_tournament_type(tournament: str) -> str:
    """Map Kaggle `tournament` string to spec category (§1.3 Check 2).

    Categories (matches K_FACTORS keys in elo.py, per eloratings.net 5-bucket spec):
      world_cup_final   — K=60 (FIFA WC finals only)
      continental_final — K=50 (UEFA Euro, Copa América, AFCON, AFC Asian Cup,
                                Gold Cup, OFC Nations Cup, Confederations Cup)
      qualifier         — K=40 (all confederation qualifiers)
      nations_league    — K=40 (UEFA NL, CONCACAF NL)
      other_tournament  — K=30 (sub-confederation cups, Kirin/King's/China Cup,
                                Olympic, CHAN, Island Games, CONIFA)
      friendly          — K=20 (bilateral friendlies + fallback)
    """
    t = tournament.lower()
    # Priority 1: FIFA WC finals
    if "fifa world cup" in t and "qualif" not in t:
        return "world_cup_final"
    # Priority 2: continental finals (must not be qualifier)
    if any(n in t for n in _CONTINENTAL_FINAL_NAMES) and "qualif" not in t:
        return "continental_final"
    # Priority 3: any qualifier
    if "qualif" in t:
        return "qualifier"
    # Priority 4: Nations Leagues
    if "nations league" in t:
        return "nations_league"
    # Priority 5: other competitive tournaments (sub-confed cups + invitationals)
    if any(n in t for n in _OTHER_TOURNAMENT_NAMES):
        return "other_tournament"
    # Priority 6: fallback — bilateral friendlies + obscure unmapped
    return "friendly"
