"""Match a realised tournament state to the closest precomputed scenario.

During the actual tournament, group games play out and the realised state
won't generally match any single precomputed scenario exactly. This module
ranks the precomputed scenarios (v1 single-deviation + v2 pairwise = 79
total) by how close each one is to the realised state, so the trader can
pick the right propagation table for pricing.

Distance metric (per the spec - Hamming-on-top-2 with lexicographic-by-
severity tie-breaker):

  Primary:    Hamming distance over groups - count of groups where the
              realised top-2 disagrees with the scenario's top-2 (as an
              ordered pair).

  Tie-break:  When two scenarios have equal Hamming distance, classify
              each disagreement by SEVERITY (SWAP < PARTIAL < DISJOINT)
              and compare the sorted-descending severity vectors
              lexicographically. Lower (less severe) wins.

  Partial-state-handling: groups not present in the realised state are
              ignored - only groups in both state and scenario are scored.

Public API:
    parse_state_from_dict(d) -> RealisedState
    find_best_scenarios(state, output_dir, k=5) -> list[ScenarioMatch]
    load_survival_table(scenario_match) -> dict
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Optional

from upset_propagation.config import GROUP_LETTERS, NON_SCENARIO_FILENAMES, OUTPUT_DIR


# Severity tier classification


class Severity(IntEnum):
    """How badly does this group disagree between state and scenario?

    Ordered intentionally so IntEnum comparison works naturally:
        EXACT < SWAP < PARTIAL < DISJOINT
    """
    EXACT = 0      # state[g][:2] == scenario[g][:2] - perfect
    SWAP = 1       # same teams, different order (e.g., (Spain, Uruguay) vs (Uruguay, Spain))
    PARTIAL = 2    # one team shared, one different
    DISJOINT = 3   # both teams in top-2 are different


def classify_group(
    state_top2: tuple[str, str],
    scenario_top2: tuple[str, str],
) -> Severity:
    """Classify how the two pairs differ. See Severity for tier definitions."""
    if state_top2 == scenario_top2:
        return Severity.EXACT
    state_set = set(state_top2)
    scen_set = set(scenario_top2)
    if state_set == scen_set:
        return Severity.SWAP
    overlap = len(state_set & scen_set)
    if overlap == 1:
        return Severity.PARTIAL
    return Severity.DISJOINT


# Realised state


@dataclass
class RealisedState:
    """The realised tournament state at a point in time.

    Attributes:
        standings: {group_letter: [1st, 2nd, 3rd, 4th]} for each group
            that has played out. Groups not yet decided are absent.
        played_groups: convenience - the set of group letters in `standings`.
        observed_at: ISO timestamp for when this state was observed.
    """
    standings: dict[str, list[str]]
    observed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def played_groups(self) -> set[str]:
        return set(self.standings.keys())

    @property
    def is_complete(self) -> bool:
        """True if all 12 groups have been decided."""
        return self.played_groups == set(GROUP_LETTERS)


def parse_state_from_dict(d: dict[str, list[str]]) -> RealisedState:
    """Build a RealisedState from a plain {group: [4 teams]} dict.

    Validates that each group has exactly 4 teams. Groups not present in
    `d` are treated as not-yet-played.
    """
    for letter, teams in d.items():
        if letter not in GROUP_LETTERS:
            raise ValueError(f"Unknown group letter {letter!r}; expected one of {GROUP_LETTERS}")
        if not isinstance(teams, list) or len(teams) != 4:
            raise ValueError(
                f"Group {letter} standing must be a list of 4 teams, got {teams!r}"
            )
    return RealisedState(standings=dict(d))


# Scenario match result


@dataclass
class ScenarioMatch:
    """Result of matching one scenario against a realised state.

    Attributes:
        scenario_id: the matched scenario's id (e.g., "spain_runner_up_H")
        scenario_path: path to the full output JSON for the scenario
        hamming_distance: number of groups where state and scenario
            disagree (any tier > EXACT)
        severities: per-group severity tiers, in canonical group-letter
            order ({letter: Severity}). Includes EXACT for matching groups.
        disagreement_vector: severities of disagreeing groups only, sorted
            descending. Used for the lexicographic tie-breaker.
    """
    scenario_id: str
    scenario_path: Path
    hamming_distance: int
    severities: dict[str, Severity]
    disagreement_vector: tuple[Severity, ...]

    @property
    def sort_key(self) -> tuple[int, tuple[Severity, ...]]:
        """Composite key for ranking - lower is better.

        Primary: hamming_distance (fewer disagreements is better).
        Secondary: disagreement_vector lex-compared (less severe is better).
        """
        return (self.hamming_distance, self.disagreement_vector)


# Distance computation


def compute_distance(
    state: RealisedState,
    scenario_id: str,
    scenario_standings: dict[str, list[str]],
    scenario_path: Path,
) -> ScenarioMatch:
    """Compute the distance between a realised state and one scenario."""
    severities: dict[str, Severity] = {}
    for letter in GROUP_LETTERS:
        if letter not in state.standings:
            # Group not yet played - skip, don't score.
            continue
        state_top2 = (state.standings[letter][0], state.standings[letter][1])
        scen_top2 = (scenario_standings[letter][0], scenario_standings[letter][1])
        severities[letter] = classify_group(state_top2, scen_top2)

    hamming = sum(1 for sev in severities.values() if sev != Severity.EXACT)
    disagreement_vector = tuple(
        sorted(
            (sev for sev in severities.values() if sev != Severity.EXACT),
            reverse=True,  # descending: worst severity first for lex comparison
        )
    )
    return ScenarioMatch(
        scenario_id=scenario_id,
        scenario_path=scenario_path,
        hamming_distance=hamming,
        severities=severities,
        disagreement_vector=disagreement_vector,
    )


# Loading scenario metadata


def load_scenarios_metadata(
    output_dir: Optional[Path] = None,
) -> list[tuple[str, dict[str, list[str]], Path]]:
    """Load just the (scenario_id, standings, path) for every scenario file.

    Avoids loading the full survival table for ranking - we only need the
    standings to compute distances. Saves memory and I/O for hot paths
    (e.g. a UI calling find_best_scenarios on every group-game tick).

    Includes baseline.json - it's a valid candidate match (the "empty
    scenario" where every group's favourite wins).

    Skips:
        - index.json (metadata-of-metadata, not a scenario)
        - validation_report.json (auto-validation output, not a scenario)
        - top_10_ranking.json (real-time ranking artifact, not a scenario)
        - health.json (cron monitoring artifact, not a scenario)
        - non-.json files (.gitkeep etc.)

    The full list is centralized in config.NON_SCENARIO_FILENAMES so
    adding a new output artifact only requires updating one place.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    metadata: list[tuple[str, dict[str, list[str]], Path]] = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name in NON_SCENARIO_FILENAMES:
            continue
        with path.open() as f:
            d = json.load(f)
        metadata.append((d["scenario_id"], d["standings"], path))
    return metadata


# Main entry point


def find_best_scenarios(
    state: RealisedState,
    output_dir: Optional[Path] = None,
    k: int = 5,
) -> list[ScenarioMatch]:
    """Rank precomputed scenarios by distance to the realised state.

    Args:
        state: the realised state, possibly partial
        output_dir: where the scenario JSONs live (default: ./output/)
        k: how many top matches to return

    Returns: the top-k scenarios sorted by (hamming_distance,
        disagreement_vector). The first entry is the best match.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    metadata = load_scenarios_metadata(output_dir)
    matches = [
        compute_distance(state, scen_id, standings, path)
        for scen_id, standings, path in metadata
    ]
    matches.sort(key=lambda m: m.sort_key)
    return matches[:k]


def load_survival_table(match: ScenarioMatch) -> dict:
    """Load the full survival table for a matched scenario.

    Use this lazily after find_best_scenarios - we don't load the table
    during ranking because we don't need it then, and there are 78 of them.
    """
    with match.scenario_path.open() as f:
        return json.load(f)


# High-level trading API


@dataclass
class MatchedScenarioForState:
    """End-to-end result of matching a state and loading its scenario table.

    The trader-facing return type. Carries the matched scenario's survival
    table for direct pricing use, plus diagnostics for the caller to decide
    whether to trust the match or fall back to a different policy.

    Attributes:
        best_match: the top ScenarioMatch (lowest hamming, best severity)
        survival_table: the full JSON payload (survival + deltas)
        is_exact_match: True iff hamming_distance == 0
        is_ambiguous: True iff multiple scenarios tied at the same
            (hamming_distance, disagreement_vector) - caller should
            consider all of them rather than blindly trusting the first
        ambiguous_alternatives: the OTHER scenarios tied with the best.
            Empty list if not ambiguous.
        ambiguity_reason: human-readable explanation of why the match is
            ambiguous (or empty string if not)
    """
    best_match: ScenarioMatch
    survival_table: dict
    is_exact_match: bool
    is_ambiguous: bool
    ambiguous_alternatives: list[ScenarioMatch]
    ambiguity_reason: str


def get_scenario_table_for_state(
    state: RealisedState,
    output_dir: Optional[Path] = None,
) -> MatchedScenarioForState:
    """Trading-API entry point: state → matched scenario's survival table.

    Composes find_best_scenarios + load_survival_table with ambiguity
    detection. This is what a trader calls during the tournament: "given
    today's group standings, what's the right propagation table to price
    against?"

    Args:
        state: the realised tournament state (full or partial)
        output_dir: where the scenario JSONs live (default: ./output/)

    Returns: MatchedScenarioForState with the survival table + diagnostics.

    Ambiguity detection:
      If multiple scenarios tie at the same (hamming_distance,
      disagreement_vector), is_ambiguous=True and ambiguous_alternatives
      lists the others. Common case: early in the tournament when only a
      few groups are decided, many scenarios are equally compatible - the
      caller should know this rather than silently trust the first
      lexicographically-sorted match.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    # Get enough candidates that ties below the cutoff are visible
    all_matches = find_best_scenarios(state, output_dir, k=80)
    best = all_matches[0]

    # Find all scenarios tied with the best on both Hamming AND severity
    tied = [
        m for m in all_matches
        if m.hamming_distance == best.hamming_distance
        and m.disagreement_vector == best.disagreement_vector
    ]
    alternatives = tied[1:]  # exclude `best` itself

    is_exact_match = best.hamming_distance == 0
    is_ambiguous = len(tied) > 1

    if is_ambiguous:
        n_unobserved = 12 - len(state.played_groups)
        if n_unobserved > 0:
            ambiguity_reason = (
                f"{len(tied)} scenarios tied at distance {best.hamming_distance}; "
                f"caller has only observed {len(state.played_groups)}/12 groups, "
                f"so scenarios differing only in the {n_unobserved} unobserved "
                f"groups appear identical to the matcher"
            )
        else:
            ambiguity_reason = (
                f"{len(tied)} scenarios tied at distance {best.hamming_distance} "
                f"with identical severity profiles {best.disagreement_vector}; "
                f"none of the deviating-group identities distinguish them"
            )
    else:
        ambiguity_reason = ""

    survival_table = load_survival_table(best)

    return MatchedScenarioForState(
        best_match=best,
        survival_table=survival_table,
        is_exact_match=is_exact_match,
        is_ambiguous=is_ambiguous,
        ambiguous_alternatives=alternatives,
        ambiguity_reason=ambiguity_reason,
    )


# Pretty-printing for CLI / diagnostic use


def format_match(match: ScenarioMatch) -> str:
    """One-line summary of a match for logs / smoke tests."""
    sev_summary = (
        "+".join(s.name for s in match.disagreement_vector)
        if match.disagreement_vector
        else "EXACT_MATCH"
    )
    return (
        f"  d={match.hamming_distance}  "
        f"{match.scenario_id:40s}  "
        f"disagreements=[{sev_summary}]"
    )


# CLI smoke test


if __name__ == "__main__":
    # Manual smoke test - exercises a few realistic scenarios.
    # Run: `python -m upset_propagation.state_matcher`
    #
    # We construct a few hand-crafted realised states and find their
    # top matches, just to confirm the matcher is doing what we expect.
    from upset_propagation.baseline import fetch_baseline_fair_probs
    from upset_propagation.scenarios import build_baseline_standings, load_groups

    fair_probs = fetch_baseline_fair_probs()
    groups = load_groups()
    baseline_standings = build_baseline_standings(groups, fair_probs)

    # Test 1: state == baseline (every group's favourite won)
    print("Test 1: realised state == baseline (all favourites win)")
    state1 = parse_state_from_dict(baseline_standings)
    top = find_best_scenarios(state1, k=3)
    for m in top:
        print(format_match(m))
    print()

    # Test 2: state has Spain finishing 2nd in H (matches the H single)
    print("Test 2: Spain finishes 2nd in H - should match spain_runner_up_H exactly")
    state2_standings = {g: list(s) for g, s in baseline_standings.items()}
    # Swap 1st and 2nd in Group H
    h = state2_standings["H"]
    state2_standings["H"] = [h[1], h[0], h[2], h[3]]
    state2 = parse_state_from_dict(state2_standings)
    top = find_best_scenarios(state2, k=3)
    for m in top:
        print(format_match(m))
    print()

    # Test 3: state has BOTH Spain-H and Argentina-J slip
    print("Test 3: Spain-H AND Argentina-J both slip - should match HJ pairwise")
    state3_standings = {g: list(s) for g, s in baseline_standings.items()}
    h = state3_standings["H"]
    state3_standings["H"] = [h[1], h[0], h[2], h[3]]
    j = state3_standings["J"]
    state3_standings["J"] = [j[1], j[0], j[2], j[3]]
    state3 = parse_state_from_dict(state3_standings)
    top = find_best_scenarios(state3, k=3)
    for m in top:
        print(format_match(m))
    print()

    # Test 4: partial state - only Group H played, Spain slipped
    print("Test 4: PARTIAL state - only Group H played (Spain finished 2nd)")
    state4 = parse_state_from_dict({"H": state2_standings["H"]})
    print(f"  played_groups={sorted(state4.played_groups)}, is_complete={state4.is_complete}")
    top = find_best_scenarios(state4, k=5)
    for m in top:
        print(format_match(m))
    print()

    # Test 5: consumer helper end-to-end
    print("Test 5: get_scenario_table_for_state() - full trading-API call")
    print()
    print("  5a. Exact-match case (Spain runner-up in H, all other groups seeded):")
    result5a = get_scenario_table_for_state(state2)
    print(f"     matched: {result5a.best_match.scenario_id}")
    print(f"     is_exact_match: {result5a.is_exact_match}")
    print(f"     is_ambiguous: {result5a.is_ambiguous}")
    print(f"     ambiguous_alternatives: {len(result5a.ambiguous_alternatives)}")
    print()
    print("  5b. Ambiguous-match case (only Group H observed):")
    result5b = get_scenario_table_for_state(state4)
    print(f"     matched: {result5b.best_match.scenario_id}")
    print(f"     is_exact_match: {result5b.is_exact_match}")
    print(f"     is_ambiguous: {result5b.is_ambiguous}")
    print(f"     ambiguous_alternatives: {len(result5b.ambiguous_alternatives)}")
    print(f"     ambiguity_reason: {result5b.ambiguity_reason}")