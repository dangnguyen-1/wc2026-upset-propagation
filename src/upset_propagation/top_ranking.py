"""Real-time top-10 scenario ranking.

Two surfaces:

  1. Cron-generated artifact: output/top_10_ranking.json, written each
     run alongside the scenarios. A dashboard polls this; no recomputation
     needed for read access.

  2. On-demand function + CLI: compute the top-10 fresh against any
     RealisedState. Used for ad-hoc operator queries when the
     2-hour cron snapshot might be stale.

The cron's input state comes from output/match_results.csv if present
(via item 5's state_from_matches_csv); otherwise an empty state, which
yields an empty ranking with a clear "no groups played yet" note.

Public API:
    compute_top_ranking(state, predictor, ratings, fair_probs, ...) -> TopRanking
    write_top_ranking_file(top_ranking, output_dir) -> Path
    load_state_for_cron(output_dir, ratings) -> RealisedState
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from upset_propagation._vendored.simulator import Predictor
from upset_propagation.config import OUTPUT_DIR
from upset_propagation.ensemble_matcher import (
    EnsembleMatch,
    find_best_scenarios_ensemble,
)
from upset_propagation.state_matcher import RealisedState


logger = logging.getLogger(__name__)


# File written under output/ by the cron each run
TOP_RANKING_FILENAME = "top_10_ranking.json"

# Default location the cron looks for match results to feed in; a CSV is dropped
# here as group games play out.
MATCH_RESULTS_CSV_FILENAME = "match_results.csv"

# Default top-k for the cron file. The "10" in top_10_ranking is a
# convention; can be overridden when calling compute_top_ranking() directly.
DEFAULT_K = 10


# Result types


@dataclass
class TopRankingEntry:
    """One scenario's entry in the top-10 ranking.

    A flattened, JSON-friendly view of EnsembleMatch with explicit rank
    and explicit scenario_path (relative to output_dir for portability).
    """
    rank: int
    scenario_id: str
    score: float
    borda_sum: float
    per_matcher_ranks: dict[str, float]
    per_matcher_distances: dict[str, float]
    scenario_filename: str  # e.g. "spain_runner_up_H.json" - relative to output/

    @classmethod
    def from_ensemble_match(
        cls, rank: int, match: EnsembleMatch
    ) -> TopRankingEntry:
        return cls(
            rank=rank,
            scenario_id=match.scenario_id,
            score=round(match.score, 4),
            borda_sum=match.borda_sum,
            per_matcher_ranks={
                k: round(v, 2) for k, v in match.per_matcher_ranks.items()
            },
            per_matcher_distances={
                k: round(v, 6) for k, v in match.per_matcher_distances.items()
            },
            scenario_filename=match.scenario_path.name,
        )


@dataclass
class TopRanking:
    """The full top-k ranking payload, written as top_10_ranking.json.

    Attributes:
        computed_at: ISO 8601 UTC timestamp of when the ranking was produced
        realised_state: serialized RealisedState.standings (only observed
            groups; unobserved are absent)
        n_groups_observed: 0-12
        is_complete: True iff all 12 groups observed
        top_scenarios: list of TopRankingEntry, sorted by rank ascending
        note: optional human-readable status (e.g. "no groups played yet")
    """
    computed_at: str
    realised_state: dict[str, list[str]]
    n_groups_observed: int
    is_complete: bool
    top_scenarios: list[TopRankingEntry] = field(default_factory=list)
    note: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation for writing to disk."""
        payload: dict[str, Any] = {
            "computed_at": self.computed_at,
            "realised_state": self.realised_state,
            "n_groups_observed": self.n_groups_observed,
            "is_complete": self.is_complete,
            "top_scenarios": [asdict(e) for e in self.top_scenarios],
        }
        if self.note is not None:
            payload["note"] = self.note
        return payload


# Compute the ranking


def compute_top_ranking(
    state: RealisedState,
    predictor: Predictor,
    ratings: dict[str, float],
    fair_probs: dict[str, float],
    output_dir: Optional[Path] = None,
    k: int = DEFAULT_K,
) -> TopRanking:
    """Compute the top-k scenarios for a realised state via the ensemble matcher.

    Pre-tournament behavior: when state is empty (no groups observed),
    we skip the matcher call and return an empty ranking with a clear
    note. The matcher would otherwise return all 79 scenarios essentially
    tied, which isn't meaningfully a "ranking" - and we'd waste ~1s of
    compute on a non-result.

    Args:
        state: RealisedState (full or partial; empty allowed)
        predictor: calibrated predictor (from
            l1_matcher.load_calibrated_predictor_from_index)
        ratings: team Elo ratings
        fair_probs: current API fair_probs
        output_dir: where the scenario JSONs live (default: ./output/)
        k: how many top scenarios to include

    Returns: TopRanking, ready to serialize.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    computed_at = datetime.now(timezone.utc).isoformat()
    n_observed = len(state.played_groups)
    realised_dict = {g: list(state.standings[g]) for g in sorted(state.standings)}

    if n_observed == 0:
        # No groups observed - return empty ranking with explanatory note.
        # We could compute it (all scenarios would tie at score ≈ uniform-
        # avg), but the result isn't useful and we don't want consumers
        # showing a meaningless top-10.
        return TopRanking(
            computed_at=computed_at,
            realised_state=realised_dict,
            n_groups_observed=0,
            is_complete=False,
            top_scenarios=[],
            note=(
                "No groups observed yet. Top-10 ranking becomes meaningful "
                "once at least one group has completed. Drop a match results "
                f"CSV at output/{MATCH_RESULTS_CSV_FILENAME} to populate the "
                "state."
            ),
        )

    # Run the ensemble matcher
    matches = find_best_scenarios_ensemble(
        state, predictor, ratings, fair_probs,
        output_dir=output_dir, k=k,
    )

    entries = [
        TopRankingEntry.from_ensemble_match(rank=i + 1, match=m)
        for i, m in enumerate(matches)
    ]

    return TopRanking(
        computed_at=computed_at,
        realised_state=realised_dict,
        n_groups_observed=n_observed,
        is_complete=state.is_complete,
        top_scenarios=entries,
    )


# Persist to disk


def write_top_ranking_file(
    top_ranking: TopRanking,
    output_dir: Optional[Path] = None,
) -> Path:
    """Write the ranking as JSON to output/top_10_ranking.json.

    Returns: the written file path.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / TOP_RANKING_FILENAME
    path.write_text(
        json.dumps(top_ranking.to_dict(), indent=2),
        encoding="utf-8",
    )
    logger.info(
        f"Wrote top-{len(top_ranking.top_scenarios)} ranking to {path} "
        f"({top_ranking.n_groups_observed}/12 groups observed)"
    )
    return path


# Cron-side state loader


def load_state_for_cron(
    output_dir: Path,
    ratings: dict[str, float],
) -> RealisedState:
    """Load the realised state for the cron's top-10 ranking call.

    Looks for output/match_results.csv. If present, reads via item 5's
    state_from_matches_csv (which applies FIFA tiebreakers). If absent,
    returns an empty RealisedState - the cron writes an empty top-10
    file with a "no groups played yet" note.

    Centralized here (rather than in run.py) so the same logic is used
    by both the cron path and the CLI's auto-detect default.

    Args:
        output_dir: where to look for the CSV
        ratings: team Elo ratings (passed to FIFA tiebreaker via item 5)

    Returns: RealisedState (possibly empty).
    """
    # Defer the import to keep module load light when match_results isn't
    # involved (e.g. tests importing top_ranking without the FIFA logic).
    from upset_propagation.match_results import state_from_matches_csv

    csv_path = output_dir / MATCH_RESULTS_CSV_FILENAME
    if not csv_path.exists():
        logger.info(
            f"No {MATCH_RESULTS_CSV_FILENAME} at {csv_path}; "
            f"using empty state for top-10 ranking."
        )
        return RealisedState(standings={})

    logger.info(f"Reading match results from {csv_path}")
    return state_from_matches_csv(csv_path, ratings)


# CLI


def main() -> None:
    """`python -m upset_propagation.top_ranking` - ad-hoc operator query.

    Computes the current top-10 ranking for a state read from disk
    (output/match_results.csv if present, else empty). Writes the result
    to output/top_10_ranking.json and prints a summary to stdout.

    Useful for operator queries when the cron's last snapshot is stale
    or when testing a new set of match results before committing to the
    next cron cycle.
    """
    import argparse
    from upset_propagation.baseline import fetch_baseline_fair_probs
    from upset_propagation.l1_matcher import (
        load_calibrated_predictor_from_index,
    )
    from upset_propagation.logging_config import configure_interactive_logging
    from upset_propagation.scenarios import load_latest_elo

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: ./output/)",
    )
    parser.add_argument(
        "--state-csv",
        type=Path,
        default=None,
        help=(
            "Path to match results CSV (default: output/match_results.csv "
            "if present, else empty state)."
        ),
    )
    parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_K,
        help=f"Number of top scenarios (default: {DEFAULT_K})",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO-level logging",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help=(
            "Print to stdout only; don't overwrite output/top_10_ranking.json. "
            "Use this for what-if queries that shouldn't update the cron file."
        ),
    )
    args = parser.parse_args()

    configure_interactive_logging(quiet=args.quiet)
    output_dir = args.output_dir or OUTPUT_DIR

    # Load inputs
    logger.info("Loading inputs...")
    fair_probs = fetch_baseline_fair_probs()
    ratings = load_latest_elo()
    predictor = load_calibrated_predictor_from_index(output_dir / "index.json")

    # Resolve state
    if args.state_csv is not None:
        from upset_propagation.match_results import state_from_matches_csv
        logger.info(f"Reading match results from {args.state_csv}")
        state = state_from_matches_csv(args.state_csv, ratings)
    else:
        state = load_state_for_cron(output_dir, ratings)

    # Compute and present
    top = compute_top_ranking(
        state, predictor, ratings, fair_probs,
        output_dir=output_dir, k=args.k,
    )

    if not args.no_write:
        write_top_ranking_file(top, output_dir=output_dir)

    # Print summary
    print()
    print(f"Top-{args.k} ranking ({top.n_groups_observed}/12 groups observed):")
    if top.note:
        print(f"  Note: {top.note}")
    if not top.top_scenarios:
        print("  (no scenarios - see note above)")
    else:
        for e in top.top_scenarios:
            ranks_str = ", ".join(
                f"{m}={r:.1f}" for m, r in e.per_matcher_ranks.items()
            )
            print(
                f"  #{e.rank:2d}  score={e.score:.4f}  "
                f"borda={e.borda_sum:6.1f}  {e.scenario_id:42s}  "
                f"ranks=[{ranks_str}]"
            )


if __name__ == "__main__":
    main()