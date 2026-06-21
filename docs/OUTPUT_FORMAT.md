# Output Format Reference

Quick reference for the JSON files written to `output/` by
`python -m upset_propagation.run`. For the conceptual background see the README.

## File inventory

After a successful run, `output/` contains **80 JSON files**:

| File | Purpose |
|------|---------|
| `index.json` | Calibration diagnostics + scenario file directory |
| `baseline.json` | Empty-scenario propagation table (reference point) |
| 12× `<favourite>_runner_up_<group>.json` | v1 single-deviation scenarios |
| 66× `<fav1>_<fav2>_runner_up_<g1g2>.json` | v2 pairwise compound scenarios |

The 79 non-index files are all "scenario" files with the same structure.

---

## `index.json` - diagnostics + directory

```json
{
  "computed_at": "2026-06-08T05:20:12+00:00",
  "calibration": {
    "final_loss": 5.22e-05,
    "max_residual": 0.0036,
    "n_iterations": 3711,
    "elapsed_sec": 229.34,
    "within_tolerance": true,
    "tolerance": 0.005,
    "offsets": {"Spain": -21.58, "France": 63.67, ...}
  },
  "baseline_file": "baseline.json",
  "scenario_files": ["mexico_runner_up_A.json", ...]
}
```

**Key fields to inspect after each run:**

- `calibration.within_tolerance` - bool, should be `true`. If `false`, the
  calibration didn't fully converge; treat the scenario tables with care.
- `calibration.max_residual` - max |propagated_Win − target_Win| in
  probability. Should be < 0.005 (= 0.5pp).
- `computed_at` - ISO timestamp. Important when correlating against
  fair_probs snapshots used to compute the run.

---

## Scenario file structure

Every scenario file (including `baseline.json`) has this top-level shape:

```json
{
  "scenario_id": "spain_runner_up_H",
  "description": "Spain finishes 2nd in Group H; Uruguay wins the group.",
  "deviating_group": "H",
  "favourite": "Spain",
  "upset_winner": "Uruguay",
  "computed_at": "2026-06-08T05:20:12+00:00",
  "standings": { ... },
  "survival": { ... },
  "delta_from_baseline": { ... }
}
```

**Note on v1 vs v2 scenarios:**

v1 (single-deviation) scenarios populate `deviating_group`, `favourite`, and
`upset_winner` as single strings. For v2 (pairwise) scenarios, these fields
still exist but hold only the FIRST of the two deviating groups -
backward-compatible with v1 consumers. v2 scenarios additionally carry the
deviation in tuple form via `deviating_groups`, `favourites`, and
`upset_winners` (these aren't currently written to the JSON; if you need
them, ask).

`baseline.json` has empty strings for `deviating_group`, `favourite`,
`upset_winner` (no deviation), and has no `delta_from_baseline` block.

---

## `standings` - input to the scenario

```json
"standings": {
  "A": ["Mexico", "South Korea", "Czechia", "South Africa"],
  "B": ["Canada", "Switzerland", "Bosnia and Herzegovina", "Qatar"],
  ...
  "H": ["Uruguay", "Spain", "Cape Verde", "Saudi Arabia"],
  ...
}
```

For each of the 12 groups, the final standing as `[1st, 2nd, 3rd, 4th]`.
The 1st-place team wins the group; 2nd advances as runner-up; 3rd may
advance via the best-8-of-12 wildcard rule; 4th is eliminated.

---

## `survival` - what to price against

```json
"survival": {
  "France": {"R32": 1.0, "R16": 0.8030, "QF": 0.5366, "SF": 0.3958, "F": 0.2981, "Win": 0.1750},
  "Spain":  {"R32": 1.0, "R16": 0.5775, "QF": 0.4577, "SF": 0.3136, "F": 0.1907, "Win": 0.1238},
  ...
}
```

For each team that reached R32 under this scenario, the probability of
reaching each later round, conditional on this scenario being realised.

**Reading guide:**

- Teams are listed in descending P(Win) order - the favourite of each
  scenario is first.
- `R32: 1.0` for every team in this dict (they all reached R32 by
  construction).
- `R16` through `Win` monotonically decrease per team.
- `Σ P(Win) across all teams = 1.0` (exactly one team wins).
- Only the 32 knockout participants appear here; group-stage-eliminated
  teams are omitted.

---

## `delta_from_baseline` - the trading signal

```json
"delta_from_baseline": {
  "Spain":     {"R32": 0.0, "R16": -0.1809, "QF": -0.0966, "SF": -0.1140, "F": -0.0720, "Win": -0.0369},
  "Argentina": {"R32": 0.0, "R16": -0.2599, "QF": -0.1896, "SF": -0.1165, "F": -0.0622, "Win": -0.0273},
  "France":    {"R32": 0.0, "R16":  0.0000, "QF":  0.0000, "SF":  0.0000, "F":  0.0426, "Win": +0.0183},
  ...
}
```

Per team, `delta = scenario.survival[team][round] − baseline.survival[team][round]`.

**Reading guide:**

- Positive ΔWin = team benefits from this scenario being realised
- Negative ΔWin = team is hurt by this scenario being realised
- Teams are sorted by `|ΔWin|` descending - biggest movers first
- Group-stage-eliminated teams under EITHER scenario contribute 0
- `Σ Δ_round across all teams = 0` for each round (probability is conserved)
- `baseline.json` has no `delta_from_baseline` field

**Use:** if the market reaction to a deviation event is more extreme than
the framework's Δ predicts, this is a trade signal (fade the market). If
less extreme, the market may be under-reacting (lean into the move).

---

## Programmatic access

To use the outputs from Python:

```python
import json
from pathlib import Path

# Direct file load
with open("output/spain_runner_up_H.json") as f:
    scenario = json.load(f)
print(f"Spain ΔWin under H: {scenario['delta_from_baseline']['Spain']['Win']:+.4f}")
```

For state-based scenario lookup (the trading entry point), see
`upset_propagation.state_matcher`:

```python
from upset_propagation.state_matcher import (
    get_scenario_table_for_state, parse_state_from_dict,
)

# As the tournament plays out, build the current state and look up
# the closest precomputed scenario.
state = parse_state_from_dict({
    "H": ["Uruguay", "Spain", "Cape Verde", "Saudi Arabia"],  # Spain slipped
    # ... other groups as they're decided ...
})
result = get_scenario_table_for_state(state)
print(f"Matched: {result.best_match.scenario_id}")
print(f"Exact match: {result.is_exact_match}")
if result.is_ambiguous:
    print(f"⚠️  {result.ambiguity_reason}")
    print(f"  Other equally-good matches: {len(result.ambiguous_alternatives)}")
# result.survival_table is the full scenario JSON
```

---

## How outputs change between runs

Calibration is stochastic (Nelder-Mead's loss surface has flat directions),
so two runs against the same fair_probs will produce slightly different
calibration offsets. But the propagated probabilities and deltas are
stable - typically within ~0.1pp across runs.

Fair_probs change as sportsbook prices move. A run made 24 hours apart from
another will have different inputs and may produce noticeably different
outputs. Use `index.json:computed_at` to know how stale a snapshot is.

To avoid wasteful re-runs:

```bash
# Only re-run if outputs are >6 hours old:
python -m upset_propagation.run --skip-if-fresh 6
```