"""The 12 bracket scenarios for WC 2026.

Each scenario is a single deviation from the seeded baseline: in one group,
the highest-`fair_prob` team finishes 2nd instead of 1st, and the
second-highest-`fair_prob` team wins the group. All other groups resolve with
their `fair_prob`-favourite winning.

Why fair_prob and not Elo? Raw Elo ignores host-field advantage and recent
market information. Mexico's Elo may be below South Korea's, but at WC 2026 -
hosted in Mexico - Mexico is the actual market favourite in Group A. Using
fair_prob to pick the deviation gets this right automatically because the FairLine model's
pipeline already bakes HFA and recent form into the devigged sportsbook
consensus.

The full set of 12 scenarios - one per group A through L - is built
programmatically here from `wc2026_groups.json` + the current fair_probs
fetched from the FairLine API. The team names are not hardcoded; if the
market re-prices, the scenarios update automatically on the next run.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from upset_propagation.config import (
    ELO_HISTORY,
    GROUP_LETTERS,
    WC2026_GROUPS,
)
from upset_propagation._vendored.tournaments.wc2026 import ELO_HISTORY_NAME_ALIASES


# Data structures


# A group's final standings: 4 teams in order [1st, 2nd, 3rd, 4th].
GroupStanding = list[str]

# All 12 groups' standings keyed by letter.
GroupStandings = dict[str, GroupStanding]


@dataclass(frozen=True)
class Scenario:
    """One bracket scenario.

    Supports both single-group and pairwise deviations through a
    pair of "singular vs plural" field sets. The singular fields are
    populated for backward compatibility; newer code should prefer the tuple
    fields which generalize to N deviations.

    Attributes:
        scenario_id: short identifier, e.g. "spain_runner_up_H" or
            "argentina_spain_runner_up_HJ"
        description: human-readable summary
        deviating_group: (v1, singular) the deviating group letter; empty
            string for baseline. For pairwise scenarios, this holds the
            first group of the pair - prefer `deviating_groups` for v2.
        favourite: (v1, singular) the favourite of `deviating_group`.
            Empty for baseline. For pairwise, first favourite.
        upset_winner: (v1, singular) the upset winner of `deviating_group`.
            Empty for baseline. For pairwise, first upset winner.
        deviating_groups: (v2, plural) tuple of all deviating group letters.
            Length 0 for baseline, 1 for v1 single, 2 for v2 pairwise.
        favourites: (v2, plural) tuple of all favourites (parallel to
            deviating_groups).
        upset_winners: (v2, plural) tuple of all upset winners (parallel
            to deviating_groups).
        standings: full group standings for all 12 groups
    """

    scenario_id: str
    description: str
    deviating_group: str
    favourite: str
    upset_winner: str
    standings: GroupStandings = field(default_factory=dict)
    deviating_groups: tuple[str, ...] = field(default_factory=tuple)
    favourites: tuple[str, ...] = field(default_factory=tuple)
    upset_winners: tuple[str, ...] = field(default_factory=tuple)


# Elo loading (kept here for the calibrator + propagator)


def load_latest_elo() -> dict[str, float]:
    """Return {team: latest_elo_rating} from data/mc_simu/elo_history.csv.

    Reads the full chronological history and takes the last `rating_after`
    per team. Matches the vendored `run_mc_simu.load_latest_ratings` exactly so
    the calibrator and propagator see the same ratings he does.

    Not used by scenario building itself (which ranks by fair_prob) - kept in
    this module because it's the natural place for reference-data loaders.
    """
    df = pd.read_csv(ELO_HISTORY)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    latest = df.groupby("team").tail(1).set_index("team")["rating_after"]
    return latest.to_dict()


def resolve_elo_for_wc_team(team: str, ratings: dict[str, float]) -> float:
    """Look up Elo for a WC2026 team name, applying name aliases.

    the vendored `elo_history.csv` uses canonical Kaggle dataset names
    ("Czech Republic", "Curaçao") which differ from the modern FIFA names
    in `wc2026_groups.json` ("Czechia", "Curacao"). We reuse his alias map
    (`ELO_HISTORY_NAME_ALIASES`) directly so we stay consistent.
    """
    if team in ratings:
        return ratings[team]
    elo_name = ELO_HISTORY_NAME_ALIASES.get(team)
    if elo_name and elo_name in ratings:
        return ratings[elo_name]
    raise KeyError(
        f"No Elo rating found for {team!r} (also tried alias {elo_name!r})"
    )


# Group loading


def load_groups() -> dict[str, list[str]]:
    """Return {group_letter: [4 teams]} from wc2026_groups.json."""
    with WC2026_GROUPS.open() as f:
        data = json.load(f)
    return {letter: list(teams) for letter, teams in data["groups"].items()}


# Name alignment: API names ↔ groups.json names


# The FairLine API uses some name spellings that differ from the canonical
# names in `wc2026_groups.json`. We map FROM the groups.json name TO whatever
# variant the API uses, so when ranking by fair_prob we can find each group
# team in the baseline.
#
# Differences observed in production output of baseline.fetch_baseline_fair_probs:
#   groups.json                  → API spelling
#   "Czechia"                    → "Czech Republic"
#   "Bosnia and Herzegovina"     → "Bosnia & Herzegovina"
#   "United States"              → "USA"
#   "Curacao"                    → "Curaçao"
GROUPS_TO_API_NAME: dict[str, str] = {
    "Czechia": "Czech Republic",
    "Bosnia and Herzegovina": "Bosnia & Herzegovina",
    "United States": "USA",
    "Curacao": "Curaçao",
}


def resolve_fair_prob_for_group_team(
    team: str, fair_probs: dict[str, float]
) -> float:
    """Look up `team` (in groups.json spelling) in the API fair_prob dict.

    Tries the direct name first, then the API alias. Raises if neither hits -
    that would indicate a real coverage gap in the FairLine model's pipeline for one of the
    48 qualified teams, which we want to fail loudly on.
    """
    if team in fair_probs:
        return fair_probs[team]
    api_name = GROUPS_TO_API_NAME.get(team)
    if api_name and api_name in fair_probs:
        return fair_probs[api_name]
    raise KeyError(
        f"No fair_prob found for {team!r} (also tried alias {api_name!r}); "
        f"available teams: {sorted(fair_probs.keys())[:5]}..."
    )


# Scenario builder


def rank_group_by_fair_prob(
    teams: list[str], fair_probs: dict[str, float]
) -> list[str]:
    """Order teams in a group by fair_prob descending.

    Returns a list of 4 team names: [strongest, 2nd, 3rd, weakest], where
    'strongest' = highest fair_prob.
    """
    return sorted(
        teams,
        key=lambda t: -resolve_fair_prob_for_group_team(t, fair_probs),
    )


def build_baseline_standings(
    groups: dict[str, list[str]], fair_probs: dict[str, float]
) -> GroupStandings:
    """The market-implied baseline: every group resolves with the fair_prob
    favourite winning.

    Returns {letter: [1st, 2nd, 3rd, 4th]} for all 12 groups, ordered by
    fair_prob descending within each group. This is the reference state -
    scenarios deviate from this by swapping 1st and 2nd in exactly one group.
    """
    return {
        letter: rank_group_by_fair_prob(teams, fair_probs)
        for letter, teams in groups.items()
    }


def build_scenario_for_group(
    deviating_group: str,
    groups: dict[str, list[str]],
    fair_probs: dict[str, float],
) -> Scenario:
    """Construct the scenario where the fair_prob favourite in `deviating_group`
    slips to 2nd.

    Concretely:
      - All other groups: seeded order [1st, 2nd, 3rd, 4th] by fair_prob desc.
      - The deviating group: positions 1 and 2 are SWAPPED; positions 3 and 4
        keep their seeded order.
    """
    if deviating_group not in groups:
        raise ValueError(
            f"Unknown group {deviating_group!r}; expected one of {GROUP_LETTERS}"
        )

    standings = build_baseline_standings(groups, fair_probs)
    seeded = standings[deviating_group]
    favourite = seeded[0]
    upset_winner = seeded[1]
    standings[deviating_group] = [upset_winner, favourite, seeded[2], seeded[3]]

    return Scenario(
        scenario_id=f"{_slug(favourite)}_runner_up_{deviating_group}",
        description=(
            f"{favourite} finishes 2nd in Group {deviating_group}; "
            f"{upset_winner} wins the group. All other groups resolve as seeded."
        ),
        deviating_group=deviating_group,
        favourite=favourite,
        upset_winner=upset_winner,
        standings=standings,
        # v2 tuple fields - populated as 1-tuples for v1 single-deviation
        # scenarios so v2 code can iterate uniformly across v1 and v2.
        deviating_groups=(deviating_group,),
        favourites=(favourite,),
        upset_winners=(upset_winner,),
    )


def build_all_scenarios(
    fair_probs: dict[str, float],
    groups: Optional[dict[str, list[str]]] = None,
) -> list[Scenario]:
    """The 12 scenarios - one per group A through L.

    Args:
        fair_probs: {team: fair_prob} from `baseline.fetch_baseline_fair_probs()`.
            Required - scenarios depend on the current market snapshot.
        groups: optional override for the 12-groups dict (default: load from disk).
    """
    if groups is None:
        groups = load_groups()
    return [
        build_scenario_for_group(letter, groups, fair_probs)
        for letter in GROUP_LETTERS
    ]


# v2: pairwise compound scenarios


def build_pairwise_scenario(
    deviating_group_1: str,
    deviating_group_2: str,
    groups: dict[str, list[str]],
    fair_probs: dict[str, float],
) -> Scenario:
    """Construct the scenario where the favourites in TWO distinct groups
    both slip to 2nd.

    Concretely:
      - The two deviating groups: positions 1 and 2 are SWAPPED in each.
      - All other 10 groups: seeded order by fair_prob descending.

    The scenario_id is built from the two favourites sorted alphabetically
    and the two group letters sorted, so build_pairwise_scenario("H", "J")
    and build_pairwise_scenario("J", "H") produce the same scenario_id.
    This means the same scenario file is written regardless of argument
    order, avoiding accidental duplicate scenarios.

    Args:
        deviating_group_1, deviating_group_2: two distinct group letters
            (one of A-L). Same-group pairs raise ValueError per the scoping
            decision (no single-group multi-deviations).
        groups: 12-group dict
        fair_probs: market view, used to identify each group's favourite
            and upset winner

    Raises: ValueError if the two groups are equal or unknown.
    """
    if deviating_group_1 == deviating_group_2:
        raise ValueError(
            f"Pairwise scenario requires two distinct groups; got "
            f"{deviating_group_1!r} twice. Single-group multi-deviations are "
            f"out of scope for v2."
        )
    for g in (deviating_group_1, deviating_group_2):
        if g not in groups:
            raise ValueError(
                f"Unknown group {g!r}; expected one of {GROUP_LETTERS}"
            )

    standings = build_baseline_standings(groups, fair_probs)

    # Apply both swaps. Each is independent; the order doesn't matter
    # because the two groups are distinct.
    favourites_by_group: dict[str, str] = {}
    upsets_by_group: dict[str, str] = {}
    for g in (deviating_group_1, deviating_group_2):
        seeded = standings[g]
        fav = seeded[0]
        upset = seeded[1]
        standings[g] = [upset, fav, seeded[2], seeded[3]]
        favourites_by_group[g] = fav
        upsets_by_group[g] = upset

    # Canonicalize ordering: sort group letters so scenario_id is
    # invariant under argument-order swap. Favourites and upset_winners
    # follow the same canonical order.
    canonical_groups = tuple(sorted([deviating_group_1, deviating_group_2]))
    canonical_favourites = tuple(favourites_by_group[g] for g in canonical_groups)
    canonical_upsets = tuple(upsets_by_group[g] for g in canonical_groups)

    fav_slugs_sorted = sorted([_slug(f) for f in canonical_favourites])
    scenario_id = (
        f"{fav_slugs_sorted[0]}_{fav_slugs_sorted[1]}_runner_up_"
        f"{canonical_groups[0]}{canonical_groups[1]}"
    )
    description = (
        f"{canonical_favourites[0]} finishes 2nd in Group {canonical_groups[0]} "
        f"({canonical_upsets[0]} wins) AND "
        f"{canonical_favourites[1]} finishes 2nd in Group {canonical_groups[1]} "
        f"({canonical_upsets[1]} wins). All other groups resolve as seeded."
    )

    return Scenario(
        scenario_id=scenario_id,
        description=description,
        # Singular fields: populate with the FIRST deviating group's values
        # for backward compatibility with v1 code that reads them. v2 code
        # should use the tuple fields below.
        deviating_group=canonical_groups[0],
        favourite=canonical_favourites[0],
        upset_winner=canonical_upsets[0],
        standings=standings,
        deviating_groups=canonical_groups,
        favourites=canonical_favourites,
        upset_winners=canonical_upsets,
    )


def build_all_pairwise_scenarios(
    fair_probs: dict[str, float],
    groups: Optional[dict[str, list[str]]] = None,
) -> list[Scenario]:
    """All 12-choose-2 = 66 pairwise compound scenarios.

    Iterates over all unordered pairs of distinct group letters. Result
    order: lexicographic by (group_1, group_2), so the first scenario is
    the AB pair, the last is KL.

    Args:
        fair_probs: market view from `baseline.fetch_baseline_fair_probs()`
        groups: optional override; default loads from disk.
    """
    from itertools import combinations

    if groups is None:
        groups = load_groups()
    return [
        build_pairwise_scenario(g1, g2, groups, fair_probs)
        for g1, g2 in combinations(GROUP_LETTERS, 2)
    ]


# Helpers


def _slug(team_name: str) -> str:
    """Convert a team name into a safe identifier fragment.

    'Bosnia and Herzegovina' -> 'bosnia_and_herzegovina'
    'South Korea' -> 'south_korea'
    'Curaçao' -> 'curacao'
    """
    normalised = unicodedata.normalize("NFKD", team_name)
    ascii_only = normalised.encode("ascii", errors="ignore").decode("ascii")
    return ascii_only.lower().replace(" ", "_").replace("&", "and")


if __name__ == "__main__":
    # Manual smoke test - `python -m upset_propagation.scenarios`
    from upset_propagation.baseline import fetch_baseline_fair_probs

    fair_probs = fetch_baseline_fair_probs()
    scenarios = build_all_scenarios(fair_probs)
    print(f"Built {len(scenarios)} single-deviation scenarios:\n")
    for s in scenarios:
        print(f"  [{s.deviating_group}] {s.scenario_id}")
        print(f"      {s.description}")
        print(
            f"      Group {s.deviating_group} standing: "
            f"{' > '.join(s.standings[s.deviating_group])}"
        )
        print()

    pairwise = build_all_pairwise_scenarios(fair_probs)
    print(f"\nBuilt {len(pairwise)} pairwise compound scenarios. First 5:\n")
    for s in pairwise[:5]:
        groups_str = "".join(s.deviating_groups)
        print(f"  [{groups_str}] {s.scenario_id}")
        print(f"      {s.description}")
        print()
    print(f"  ... and {len(pairwise) - 5} more.")