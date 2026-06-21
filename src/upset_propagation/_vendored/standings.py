"""Group-stage standings + tiebreakers per FIFA 2026 Competition Regulations.

Implements Phase 3 §4.1 Steps 3.1 + 3.2. Contract:

    rank_group(matches, final_tiebreaker)         → list[str] of 4 teams 1st→4th
    rank_best_thirds(third_entries, final_tiebreaker)  → list[tuple[team, group]]

FIFA 2026 chain (corrected per claude.ai cross-verification 2026-05-20; FIFA
Regulations Article 13, p.26):

    1. Points (overall)
    2. H2H Points (within tied subset)
    3. H2H Goal Difference
    4. H2H Goals Scored
    5. (Recurse on still-tied subset — re-apply 2-4 with H2H recomputed)
    6. Overall Goal Difference
    7. Overall Goals Scored
    8. Team Conduct (fair-play score)
    9. FIFA World Ranking (final tiebreaker — no drawing of lots in 2026)

Earlier spec §4.1 Step 3.1 had H2H AFTER Overall GD/GF — that ordering was the
PRE-2018 FIFA chain, not 2026. Phase 3 decision #22 corrects this.

For best-thirds (rank_best_thirds), step 2-5 (H2H block) is skipped because the
tied teams come from different groups → no H2H matches exist.

Fair-play (team conduct) scoring per FIFA 2026 Article 13:
    -1 per yellow card
    -3 per indirect red (2nd yellow same match)
    -4 per direct red
    -5 per yellow + direct red (single player, single match)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ._common import banner  # noqa: E402,F401 — UTF-8 stdout side-effect


# ── Fair-play (team conduct) constants ────────────────────────────────────────

YELLOW_CARD = -1
INDIRECT_RED = -3
DIRECT_RED = -4
YELLOW_PLUS_DIRECT_RED = -5  # single player yellow + later direct red same match


# ── Data records ──────────────────────────────────────────────────────────────


@dataclass
class GroupMatch:
    """One round-robin group match.

    Card counts default to 0; only populated for unit tests against real
    tournaments. In MC simulation we don't sample cards, so fair-play
    contributes nothing to the tiebreaker (effective skip to FIFA ranking).
    """

    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    home_yellows: int = 0
    home_indirect_reds: int = 0
    home_direct_reds: int = 0
    home_yellow_plus_reds: int = 0
    away_yellows: int = 0
    away_indirect_reds: int = 0
    away_direct_reds: int = 0
    away_yellow_plus_reds: int = 0


@dataclass
class TeamStats:
    """Aggregate stats per team across a set of matches."""

    team: str
    played: int = 0
    points: int = 0
    goals_for: int = 0
    goals_against: int = 0
    fair_play: int = 0  # negative; higher (less negative) is better

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


# ── Fair-play scorer ──────────────────────────────────────────────────────────


def fair_play_score(
    yellows: int = 0,
    indirect_reds: int = 0,
    direct_reds: int = 0,
    yellow_plus_reds: int = 0,
) -> int:
    """Team conduct score per FIFA 2026 Article 13. Higher (less negative) is better.

    Only one deduction per player per match — caller is responsible for not
    double-counting (e.g., a player who got a yellow then a direct red is
    counted ONCE in yellow_plus_reds, NOT separately in yellows + direct_reds).
    """
    return (
        yellows * YELLOW_CARD
        + indirect_reds * INDIRECT_RED
        + direct_reds * DIRECT_RED
        + yellow_plus_reds * YELLOW_PLUS_DIRECT_RED
    )


# ── Stats aggregation ─────────────────────────────────────────────────────────


def compute_stats(matches: list[GroupMatch]) -> dict[str, TeamStats]:
    """Aggregate points / goals / fair-play across the given matches.

    Standard 3-1-0 scoring. Caller passes either the full group's 6 matches
    (for overall stats) or only the matches between a tied subset (for H2H).
    """
    teams: set[str] = set()
    for m in matches:
        teams.add(m.home_team)
        teams.add(m.away_team)

    stats = {t: TeamStats(team=t) for t in teams}

    for m in matches:
        h, a = stats[m.home_team], stats[m.away_team]
        h.played += 1
        a.played += 1
        h.goals_for += m.home_goals
        h.goals_against += m.away_goals
        a.goals_for += m.away_goals
        a.goals_against += m.home_goals
        if m.home_goals > m.away_goals:
            h.points += 3
        elif m.home_goals < m.away_goals:
            a.points += 3
        else:
            h.points += 1
            a.points += 1
        h.fair_play += fair_play_score(
            m.home_yellows,
            m.home_indirect_reds,
            m.home_direct_reds,
            m.home_yellow_plus_reds,
        )
        a.fair_play += fair_play_score(
            m.away_yellows,
            m.away_indirect_reds,
            m.away_direct_reds,
            m.away_yellow_plus_reds,
        )

    return stats


def _h2h_stats(tied: list[str], all_matches: list[GroupMatch]) -> dict[str, TeamStats]:
    """Recompute stats restricted to matches between tied teams only."""
    subset = set(tied)
    relevant = [m for m in all_matches if m.home_team in subset and m.away_team in subset]
    return compute_stats(relevant)


# ── Ranking ───────────────────────────────────────────────────────────────────


def rank_group(
    matches: list[GroupMatch],
    final_tiebreaker: dict[str, float],
    *,
    stats: dict[str, TeamStats] | None = None,
) -> list[str]:
    """Return 4 team names ranked 1st → 4th per FIFA 2026 chain.

    Args:
        matches: round-robin matches in the group (6 for 4 teams)
        final_tiebreaker: {team: score} for ranking criterion #9 (FIFA World
            Ranking). Higher score = better rank. In MC sim we pass current
            Elo ratings as proxy for FIFA ranking. Must include every team
            appearing in `matches`.
        stats: optional pre-computed stats dict (skips internal compute_stats
            call — useful when caller already aggregated for diagnostics).

    Tiebreaker chain: points → H2H pts → H2H GD → H2H GF → (recurse) →
    Overall GD → Overall GF → Fair-play → final_tiebreaker.
    """
    if stats is None:
        stats = compute_stats(matches)
    teams = sorted(stats.keys(), key=lambda t: -stats[t].points)

    result: list[str] = []
    i = 0
    while i < len(teams):
        j = i
        while j < len(teams) and stats[teams[j]].points == stats[teams[i]].points:
            j += 1
        tied = teams[i:j]
        if len(tied) == 1:
            result.append(tied[0])
        else:
            result.extend(_resolve_h2h(tied, matches, stats, final_tiebreaker))
        i = j

    return result


def _resolve_h2h(
    tied: list[str],
    all_matches: list[GroupMatch],
    overall_stats: dict[str, TeamStats],
    final_tiebreaker: dict[str, float],
) -> list[str]:
    """Apply H2H pts → GD → GF on `tied` subset, recursing on still-tied groups.

    If H2H criteria fail to separate ANY teams (i.e. the entire subset shares
    identical H2H pts/GD/GF), fall through to overall criteria.
    """
    if len(tied) == 1:
        return list(tied)

    h2h = _h2h_stats(tied, all_matches)

    def h2h_key(t: str) -> tuple[int, int, int]:
        s = h2h[t]
        return (-s.points, -s.goal_difference, -s.goals_for)

    tied_sorted = sorted(tied, key=h2h_key)

    result: list[str] = []
    i = 0
    while i < len(tied_sorted):
        j = i
        while j < len(tied_sorted) and h2h_key(tied_sorted[j]) == h2h_key(tied_sorted[i]):
            j += 1
        sub = tied_sorted[i:j]
        if len(sub) == 1:
            result.append(sub[0])
        elif len(sub) == len(tied):
            # H2H didn't separate any team — fall to overall criteria
            result.extend(_resolve_overall(sub, overall_stats, final_tiebreaker))
        else:
            # Smaller still-tied subset — recurse with H2H recomputed
            result.extend(_resolve_h2h(sub, all_matches, overall_stats, final_tiebreaker))
        i = j

    return result


def _resolve_overall(
    tied: list[str],
    stats: dict[str, TeamStats],
    final_tiebreaker: dict[str, float],
) -> list[str]:
    """Overall GD → Overall GF → Fair-play → FIFA ranking.

    final_tiebreaker score is sorted descending (higher = better rank).
    """

    def key(t: str) -> tuple[int, int, int, float]:
        s = stats[t]
        return (
            -s.goal_difference,
            -s.goals_for,
            -s.fair_play,  # higher (less negative) is better
            -final_tiebreaker.get(t, 0.0),
        )

    return sorted(tied, key=key)


# ── Best-thirds ranking (cross-group) ─────────────────────────────────────────


def rank_best_thirds(
    third_entries: list[tuple[str, str]],
    stats_by_team: dict[str, TeamStats],
    final_tiebreaker: dict[str, float],
) -> list[tuple[str, str]]:
    """Rank 12 third-place finishers across groups (no H2H — different groups).

    Args:
        third_entries: list of (team_name, group_letter) tuples
        stats_by_team: full-group TeamStats keyed by team name
        final_tiebreaker: {team: score} (FIFA ranking / Elo proxy)

    Chain: Points → Overall GD → Overall GF → Fair-play → FIFA ranking.

    Returns same list sorted best→worst. Caller takes first 8 to advance.
    """

    def key(entry: tuple[str, str]) -> tuple[int, int, int, int, float]:
        team, _group = entry
        s = stats_by_team[team]
        return (
            -s.points,
            -s.goal_difference,
            -s.goals_for,
            -s.fair_play,
            -final_tiebreaker.get(team, 0.0),
        )

    return sorted(third_entries, key=key)


def collect_thirds(
    group_rankings: dict[str, list[str]],
) -> list[tuple[str, str]]:
    """Extract (team, group_letter) for the 3rd-place finisher in each group.

    Convenience helper for the WC2026 pipeline. Expects a dict {group_letter:
    [1st, 2nd, 3rd, 4th]} as produced by rank_group called per group.
    """
    return [(ranked[2], group) for group, ranked in group_rankings.items() if len(ranked) >= 3]
