"""Convert played-match results into a RealisedState.

At tournament time, the operator feeds the framework match results, not group
standings. This module bridges the gap, applying the FIFA 2026 tiebreaker
chain via the vendored vendored `standings.rank_group()` (which we don't
reimplement - his version is already meticulous, including a documented
correction for the pre-2018-vs-2026 chain ordering ambiguity).

Public API:
    MatchResult dataclass
    state_from_matches(matches, ratings, groups) -> RealisedState
    state_from_matches_csv(path, ratings, groups) -> RealisedState

Behaviour:
  - Groups with all 6 matches played → full standings in the result
  - Groups with 0 matches → absent from the result (matcher fills with
    seeded baseline downstream)
  - Groups with 1-5 matches (in progress) → skipped silently. We don't
    compute mid-group standings; FIFA tiebreakers assume completed groups.
  - Unknown teams → loud KeyError. Catches typos before they corrupt the
    downstream pipeline.
  - Inter-group match (home and away from different groups) → loud
    ValueError. Indicates real data corruption.

Cards (yellows/reds) are accepted as optional with defaults of 0. In
practice the operator won't have card data live, and fair-play (FIFA tiebreaker
step 8) is exceedingly rare anyway - step 1-7 usually resolve it. If
cards matter for a specific group, they can be passed explicitly.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from upset_propagation._vendored.standings import GroupMatch, rank_group
from upset_propagation.scenarios import (
    load_groups,
    resolve_elo_for_wc_team,
)
from upset_propagation.state_matcher import RealisedState


# Number of matches per group when complete (4 teams × 3 games / 2)
MATCHES_PER_FULL_GROUP = 6


# Input data type


@dataclass
class MatchResult:
    """One played match. Card counts are optional and default to 0.

    All team names use wc2026_groups.json spelling (modern FIFA names:
    "Czechia", "Curacao", "United States", "Bosnia and Herzegovina").
    The framework's alias handling (resolve_elo_for_wc_team) takes care
    of the elo_history.csv name differences internally.

    Card field semantics (matching the vendored GroupMatch):
      - yellows: standalone yellow cards (count separately from those
        that became part of a 2-yellow red)
      - indirect_reds: red cards via 2-yellow accumulation
      - direct_reds: red cards from a single direct dismissal
      - yellow_plus_reds: the FIRST yellow of a 2-yellow red (kept
        separate from `yellows` to avoid double-counting)
    """
    home: str
    away: str
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


# Conversion to the vendored GroupMatch


def _to_group_match(m: MatchResult) -> GroupMatch:
    """Convert our MatchResult to the vendored GroupMatch (1:1 field map)."""
    return GroupMatch(
        home_team=m.home,
        away_team=m.away,
        home_goals=m.home_goals,
        away_goals=m.away_goals,
        home_yellows=m.home_yellows,
        home_indirect_reds=m.home_indirect_reds,
        home_direct_reds=m.home_direct_reds,
        home_yellow_plus_reds=m.home_yellow_plus_reds,
        away_yellows=m.away_yellows,
        away_indirect_reds=m.away_indirect_reds,
        away_direct_reds=m.away_direct_reds,
        away_yellow_plus_reds=m.away_yellow_plus_reds,
    )


# Team-to-group lookup


def _build_team_to_group(groups: dict[str, list[str]]) -> dict[str, str]:
    """{team_name: group_letter} reverse mapping from wc2026_groups.json."""
    team_to_group: dict[str, str] = {}
    for letter, teams in groups.items():
        for team in teams:
            team_to_group[team] = letter
    return team_to_group


# Final-tiebreaker dict (Elo as FIFA-ranking proxy)


def _build_elo_tiebreaker(
    teams: set[str],
    ratings: dict[str, float],
) -> dict[str, float]:
    """Build the {team: elo} dict to pass as rank_group's final_tiebreaker.

    Uses resolve_elo_for_wc_team for each team to handle the spelling
    differences between wc2026_groups.json (modern FIFA names) and
    elo_history.csv (Kaggle canonical names).
    """
    result: dict[str, float] = {}
    for team in teams:
        result[team] = resolve_elo_for_wc_team(team, ratings)
    return result


# Main entry point


def state_from_matches(
    matches: list[MatchResult],
    ratings: dict[str, float],
    groups: Optional[dict[str, list[str]]] = None,
) -> RealisedState:
    """Compute group standings from match results using FIFA 2026 tiebreakers.

    Only groups with all 6 matches played appear in the result. Partial
    groups (1-5 matches) are skipped silently - the framework's matcher
    fills unobserved groups with the seeded baseline downstream.

    Args:
        matches: list of MatchResult, in any order, across any subset of
            the 12 groups
        ratings: {team: Elo} from scenarios.load_latest_elo(); used as
            final_tiebreaker (FIFA ranking proxy, step 9 of the chain)
        groups: optional override; defaults to load_groups() from disk

    Returns: RealisedState with standings for each fully-played group
        and the corresponding played_groups set.

    Raises:
        KeyError: unknown team name (not in any of the 12 groups)
        ValueError: a match's home and away are in different groups
    """
    if groups is None:
        groups = load_groups()
    team_to_group = _build_team_to_group(groups)

    # Step 1: group matches by their group letter, validating both teams
    # belong to the same group along the way.
    matches_by_group: dict[str, list[MatchResult]] = {}
    for m in matches:
        if m.home not in team_to_group:
            raise KeyError(
                f"Unknown home team {m.home!r}. "
                f"Expected one of the 48 WC2026 teams using groups.json spelling."
            )
        if m.away not in team_to_group:
            raise KeyError(
                f"Unknown away team {m.away!r}. "
                f"Expected one of the 48 WC2026 teams using groups.json spelling."
            )
        g_home = team_to_group[m.home]
        g_away = team_to_group[m.away]
        if g_home != g_away:
            raise ValueError(
                f"Match {m.home} vs {m.away}: teams are in different groups "
                f"({g_home} vs {g_away}). Group-stage matches are within-group only."
            )
        matches_by_group.setdefault(g_home, []).append(m)

    # Step 2: for each group with the full match count, run the FIFA chain.
    standings: dict[str, list[str]] = {}
    for letter, group_matches in matches_by_group.items():
        if len(group_matches) < MATCHES_PER_FULL_GROUP:
            # Partial group - skip silently. Caller's RealisedState will
            # treat this group as unobserved; the matcher fills with seeded
            # baseline.
            continue
        if len(group_matches) > MATCHES_PER_FULL_GROUP:
            # More than 6 matches for one group - probably a duplicate
            # entry. Loud failure.
            raise ValueError(
                f"Group {letter} has {len(group_matches)} matches, expected "
                f"{MATCHES_PER_FULL_GROUP}. Check for duplicates."
            )

        # Convert to vendored type
        vendored_matches = [_to_group_match(m) for m in group_matches]

        # Build the Elo tiebreaker dict for just this group's 4 teams
        group_teams = set(groups[letter])
        tiebreaker = _build_elo_tiebreaker(group_teams, ratings)

        # Call the vendored FIFA-chain implementation
        ranked = rank_group(vendored_matches, final_tiebreaker=tiebreaker)
        # rank_group returns [1st, 2nd, 3rd, 4th] team names
        standings[letter] = list(ranked)

    return RealisedState(standings=standings)


# CSV input convenience


def load_matches_from_csv(path: Path) -> list[MatchResult]:
    """Load MatchResult records from a CSV file.

    Required columns: home, away, home_goals, away_goals
    Optional columns: home_yellows, home_indirect_reds, home_direct_reds,
        home_yellow_plus_reds, away_yellows, away_indirect_reds,
        away_direct_reds, away_yellow_plus_reds

    Missing optional columns default to 0. Goals are parsed as ints.

    Useful for the cron pipeline - the operator can drop a CSV in a
    known location and the framework picks it up.
    """
    matches: list[MatchResult] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        required = {"home", "away", "home_goals", "away_goals"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV at {path} missing required columns: {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            matches.append(MatchResult(
                home=row["home"].strip(),
                away=row["away"].strip(),
                home_goals=int(row["home_goals"]),
                away_goals=int(row["away_goals"]),
                home_yellows=int(row.get("home_yellows", 0) or 0),
                home_indirect_reds=int(row.get("home_indirect_reds", 0) or 0),
                home_direct_reds=int(row.get("home_direct_reds", 0) or 0),
                home_yellow_plus_reds=int(row.get("home_yellow_plus_reds", 0) or 0),
                away_yellows=int(row.get("away_yellows", 0) or 0),
                away_indirect_reds=int(row.get("away_indirect_reds", 0) or 0),
                away_direct_reds=int(row.get("away_direct_reds", 0) or 0),
                away_yellow_plus_reds=int(row.get("away_yellow_plus_reds", 0) or 0),
            ))
    return matches


def state_from_matches_csv(
    path: Path,
    ratings: dict[str, float],
    groups: Optional[dict[str, list[str]]] = None,
) -> RealisedState:
    """Convenience: load matches from CSV, then compute the state."""
    matches = load_matches_from_csv(path)
    return state_from_matches(matches, ratings, groups)


# CLI smoke test


def _construct_seeded_baseline_matches(
    groups: dict[str, list[str]],
    fair_probs: dict[str, float],
) -> list[MatchResult]:
    """Construct a synthetic match list where every group resolves to
    the framework's baseline ordering.

    Uses build_baseline_standings() to derive the seeded [1st, 2nd, 3rd, 4th]
    ordering per group (which uses fair_prob-descending - NOT Elo-descending).
    Then constructs the 6 round-robin matches where 1st beats everyone 2-0,
    2nd beats 3rd and 4th 2-0, and 3rd beats 4th 2-0.

    Points: 1st=9, 2nd=6, 3rd=3, 4th=0. No ties → standings match seeded order.

    Used in the smoke test and downstream integration testing. Earlier
    version sorted by Elo as a proxy for 'favourite' but this disagreed with
    build_baseline_standings() in 7 of 12 groups (market fair_probs and pure
    Elo don't always agree). Fixed to use the framework's actual baseline.
    """
    from upset_propagation.scenarios import build_baseline_standings

    baseline_standings = build_baseline_standings(groups, fair_probs)
    matches: list[MatchResult] = []
    for letter, sorted_teams in baseline_standings.items():
        a, b, c, d = sorted_teams[0], sorted_teams[1], sorted_teams[2], sorted_teams[3]
        # A wins all 3 games 2-0
        matches.append(MatchResult(home=a, away=b, home_goals=2, away_goals=0))
        matches.append(MatchResult(home=a, away=c, home_goals=2, away_goals=0))
        matches.append(MatchResult(home=a, away=d, home_goals=2, away_goals=0))
        # B beats C and D 2-0
        matches.append(MatchResult(home=b, away=c, home_goals=2, away_goals=0))
        matches.append(MatchResult(home=b, away=d, home_goals=2, away_goals=0))
        # C beats D 2-0
        matches.append(MatchResult(home=c, away=d, home_goals=2, away_goals=0))
    return matches


if __name__ == "__main__":
    # Manual smoke test - `python -m upset_propagation.match_results`
    from upset_propagation.baseline import fetch_baseline_fair_probs
    from upset_propagation.scenarios import build_baseline_standings, load_latest_elo

    fair_probs = fetch_baseline_fair_probs()
    ratings = load_latest_elo()
    groups = load_groups()
    baseline_standings = build_baseline_standings(groups, fair_probs)

    print("Test 1: 12 groups, every favourite wins all 3 matches 2-0")
    matches = _construct_seeded_baseline_matches(groups, fair_probs)
    print(f"  Built {len(matches)} matches across 12 groups")
    state = state_from_matches(matches, ratings)
    print(f"  Resulting state: {len(state.standings)}/12 groups, "
          f"is_complete={state.is_complete}")
    print(f"  Sample standings (first 3 groups):")
    for letter in sorted(state.standings.keys())[:3]:
        print(f"    Group {letter}: {state.standings[letter]}")
    # Verify state matches baseline exactly (the whole point of the fix)
    n_matching = sum(
        1 for letter in state.standings
        if state.standings[letter] == baseline_standings[letter]
    )
    print(f"  Equivalence to baseline: {n_matching}/12 groups match exactly")
    print()

    print("Test 2: Spain-H upset (Spain loses to Uruguay, finishes 2nd)")
    # Use baseline_standings to identify Group H's seeded 1st and 2nd
    h_seeded = baseline_standings["H"]
    spain = h_seeded[0]  # 1st by baseline = favourite
    uruguay = h_seeded[1]  # 2nd by baseline
    print(f"  Group H baseline-seeded: {h_seeded}")
    matches = _construct_seeded_baseline_matches(groups, fair_probs)
    # Remove the Spain-Uruguay match (it was Spain-2-0-Uruguay because
    # Spain is A in the helper's output). Replace with Uruguay 1-0 Spain.
    matches = [
        m for m in matches
        if not {m.home, m.away} == {spain, uruguay}
    ]
    matches.append(MatchResult(home=uruguay, away=spain, home_goals=1, away_goals=0))
    # Points: Uruguay 9 (3W), Spain 6 (2W+1L), 3rd/4th unchanged.
    state = state_from_matches(matches, ratings)
    print(f"  Group H standings: {state.standings['H']}")
    if state.standings["H"][0] == uruguay and state.standings["H"][1] == spain:
        print(f"  ✓ {uruguay} 1st, {spain} 2nd as expected")
    else:
        print(f"  ✗ Expected [{uruguay}, {spain}, ...], got {state.standings['H']}")
    print()

    print("Test 3: Partial state - only first 2 matches of Group A played")
    a_seeded = baseline_standings["A"]
    matches = [
        MatchResult(home=a_seeded[0], away=a_seeded[1], home_goals=2, away_goals=0),
        MatchResult(home=a_seeded[0], away=a_seeded[2], home_goals=2, away_goals=0),
    ]
    state = state_from_matches(matches, ratings)
    print(f"  played_groups: {state.played_groups}")
    print(f"  is_complete: {state.is_complete}")
    if not state.played_groups:
        print(f"  ✓ Partial group correctly skipped from standings")
    else:
        print(f"  ✗ Expected empty played_groups, got {state.played_groups}")
    print()

    print("Test 4: Unknown team - should error loudly")
    bad_matches = [
        MatchResult(home="Atlantis", away=a_seeded[1], home_goals=1, away_goals=0),
    ]
    try:
        state_from_matches(bad_matches, ratings)
        print(f"  ✗ Expected KeyError but no exception raised")
    except KeyError as e:
        print(f"  ✓ KeyError raised: {e}")
    print()

    print("Test 5: Inter-group match - should error loudly")
    cross_match = [
        MatchResult(
            home=groups["A"][0], away=groups["H"][0],
            home_goals=1, away_goals=0,
        ),
    ]
    try:
        state_from_matches(cross_match, ratings)
        print(f"  ✗ Expected ValueError but no exception raised")
    except ValueError as e:
        print(f"  ✓ ValueError raised: {e}")