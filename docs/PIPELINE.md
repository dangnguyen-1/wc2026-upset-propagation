# Pipeline - End-to-End Walkthrough

This document explains every step of the framework's pipeline in depth,
with math, code references, and a running worked example. It's longer
than the README on purpose - the goal is "I can pick this up in six
months and rebuild my understanding."

Read order: linear top to bottom. Each section builds on the previous.

For setup and deployment, see `DEPLOYMENT.md`. For operational
troubleshooting, see `RUNBOOK.md`. For JSON schema, see
`OUTPUT_FORMAT.md`. This document is the conceptual deep-dive.


## 1. Introduction and Running Example

The framework answers one question: **for each team, what is the
probability that they win the 2026 FIFA World Cup, conditioned on
everything we've observed in group stage so far, and does that
probability disagree with what the prediction market is currently
pricing?**

The answer is a number per team, updated every two hours by the cron,
written to `output/our_vs_market.json`.

To make the rest of this doc concrete, we'll follow one team -
**Norway** - through every step. Norway is interesting because in
our current pre-tournament snapshot:

- Polymarket prices them at **2.26%** to win the tournament
- Sportsbook-derived devigged probability is **2.46%** (slightly higher)
- Our framework's implied probability is **2.66%** at p=4 (higher still)
- The 2.66% vs 2.26% market = **+0.41pp**, which is **+18% relative**
- Classified as `underpriced` (our model thinks Norway is more likely
  than the market does)

This is the kind of edge the framework is designed to surface. Read
the rest of this doc to understand how every number above is produced.


## 2. The Two Probability Sources

The framework consumes two different probability streams from the
same upstream API (`https://seal-app-yatxw.ondigitalocean.app`), and
the distinction matters.

### 2.1 Sportsbook-derived (calibration anchor)

Endpoint: `GET /api/events/world_cup_2026/fair-odds`

This returns, for each of the 48 WC2026 teams, a `fair_prob` field -
the **Shin + Power devigged tournament-winner probability** derived
from a panel of sportsbooks (Pinnacle, Betfair, and others). Devigging
removes the bookmaker's overround (the ~3-7% mathematical edge baked
into raw odds) and averages across two devigging methods (Shin and
Power) to produce a single "fair" probability.

Critical property: **these probabilities sum to ~1.0** across all 48
teams (small numerical drift aside). They've already been normalized.

Code: `upset_propagation.baseline.fetch_baseline_fair_probs()` calls
`/fair-odds` and parses out `(team, fair_prob)` pairs.

This stream's role in the pipeline: **it's the calibration target.**
The framework tunes itself so that its unconditioned baseline
prediction (everyone wins their group as the favorite) reproduces
these numbers. After calibration, when the framework propagates the
"no upsets" scenario, it should produce probabilities very close to
the `fair_prob` field for each team.

### 2.2 Prediction-market prices (comparison surface)

Endpoint: `GET /api/events/world_cup_2026/prices`

This returns one row per team per platform (Polymarket and Kalshi).
Each row has `mid` (mid-price), `bid`, `ask`. We filter to
`platform = "polymarket_live"` for 48-team coverage.

Critical property: **mids do NOT sum to 1.0** across all teams. The
prediction market itself has a vig (around 4-6% overround on
Polymarket). To turn raw mids into proper probabilities, we devig by
dividing each mid by the sum of all mids:

```
P_market(team) = mid(team) / Σ_team mid(team)
```

After this renormalization, the resulting probabilities sum to exactly
1.0.

Code: `upset_propagation.baseline.fetch_market_prices()` calls
`/prices`, filters to Polymarket, devigs, returns
`{team: probability}`.

This stream's role: **it's what we compare against.** The framework's
implied probabilities (computed downstream) are subtracted from these
market probabilities to produce the edge signal in
`output/our_vs_market.json`.

### 2.3 Why the distinction matters

For most of development, we conflated these. The framework
calibrated against `fair_prob` AND compared its implied output against
`fair_prob`. By construction, the comparison was nearly tautological
(model vs model, where the model was fit to that exact reference) and
the edges were near-zero.

The post-meeting refactor split the two: calibration still targets
`fair_prob` (because the FairLine model's devigging produces a more refined,
less-noisy reference), but the comparison surface is now Polymarket
prices. The edge becomes genuinely "model vs market" - what our
sportsbook-anchored framework thinks vs what the prediction market is
actually charging.

### 2.4 Worked example for Norway

For our running example, the two sources give:

| Source | Norway probability |
|---|---|
| `fair_prob` from `/fair-odds` (the FairLine model's devigged sportsbook) | 0.0246 (2.46%) |
| `mid` from `/prices` (Polymarket raw) | ~0.0235 (raw, includes vig) |
| `mid / Σ mid` (Polymarket devigged) | 0.0226 (2.26%) |

The framework will use `0.0246` for calibration anchoring, and `0.0226`
as Norway's "market price" for the edge computation.

### 2.5 Name aliases

Polymarket spells five teams differently from our canonical roster
(which matches `data/groups.json`). We map them at parse time:

| Polymarket spelling | Canonical name |
|---|---|
| `Bosnia-Herzegovina` | `Bosnia and Herzegovina` |
| `Congo DR` | `DR Congo` |
| `Curaçao` | `Curacao` |
| `Turkiye` | `Turkey` |
| `USA` | `United States` |

Code: `upset_propagation.baseline.MARKET_NAME_ALIASES`. Unmapped names
pass through unchanged. Teams Polymarket lists that aren't in our
canonical 48 are logged and dropped.


## 3. Input Validation

Before the calibrator burns four minutes, we validate inputs. Bad
data caught early is bad data not propagated through the whole
pipeline.

### 3.1 Elo validation

Code: `upset_propagation.input_validation.validate_elo_history(elo)`

Returns a `ValidationReport` with FAILs and WARNs.

**Hard FAILs** (halt the run):
- `elo_team_missing` - a WC2026 team has no Elo entry in
  `data/mc_simu/elo_history.csv`. Most common cause: stale Elo file.
- `elo_not_finite` - some team's Elo is NaN or ±inf.
- `elo_negative` - some team has negative Elo (Elo is bounded below
  by 0 in our scale).
- `elo_out_of_range_fatal` - Elo below 1000 or above 2600. Real WC
  teams cluster between 1400 and 2100; anything outside the wider
  1000-2600 envelope is structurally wrong.

**WARNs** (logged but don't halt):
- `elo_out_of_range` - Elo between 1000-1200 or 2400-2600. Plausible
  outliers, but flag for review.
- Staleness - the file's most-recent date is more than 30 days old.

### 3.2 Fair-probs validation

Code: `upset_propagation.input_validation.validate_fair_probs(probs)`

**Hard FAILs**:
- `team_count_wrong` - API returned ≠ 48 teams. Likely a schema
  change upstream.
- `fair_prob_zero` - a team's probability is exactly 0. Means the
  upstream aggregator missed that team in one or more of the
  sportsbooks. We never see 0% in legitimate data.
- `fair_prob_not_finite` - NaN or ±inf in the response.
- `sum_out_of_range_fatal` - sum < 0.90 or > 1.10. Devigged probs
  should sum to within ±1% of 1.0.
- `max_prob_too_high_fatal` - top team > 50%. Implausible for a
  48-team tournament where the most likely winner is typically
  10-20%.

**WARNs**:
- `max_prob_too_high` - top team > 30%.
- `sum_out_of_range` - sum drifts more than 1% from 1.0 but stays
  within ±10%.

### 3.3 The Haiti threshold

An interesting design lesson. The first cut of `validate_fair_probs`
flagged any team below 0.001 (0.1%) as a WARN ("API may have lost
this team"). On a real API response, this fired on Haiti at 0.07%.

But Haiti's 0.07% is legitimate. In a 48-team field with two genuine
giants and a long tail, the weakest teams have probabilities around
0.05-0.15%. Flagging them all was noise.

The check was inverted: **exactly zero = FAIL** (the failure mode we
actually care about, where the aggregator dropped a team). Small but
non-zero stays unflagged.

The lesson: threshold tuning should match the failure mode you're
trying to detect, not your intuition about what "looks small."


## 4. Calibration

The framework's most computationally expensive step. Roughly four
minutes per cron cycle.

### 4.1 What calibration produces

A dict of per-team **Elo offsets**: `{team: Δelo}`. Adding
`offset[team]` to `elo[team]` produces the "calibrated rating" that
makes the underlying match predictor agree with the market on
tournament-winner probabilities.

Code: `upset_propagation.calibrator.calibrate()` returns a
`CalibrationResult` carrying the offsets, the final loss, the
iteration count, and the calibrated predictor.

### 4.2 What's being optimized

Let:
- `P_market(team)` = the sportsbook-derived `fair_prob` for the team
- `P_propagated(team, offsets)` = the framework's predicted
  tournament-winner probability for the team, when the propagator
  runs the **baseline scenario** (everyone wins their group as the
  favorite, best 3rd-placers picked by raw Elo) with each team's
  Elo shifted by `offsets[team]`

The loss function is sum of squared errors across all 48 teams:

```
loss(offsets) = Σ_team (P_propagated(team, offsets) - P_market(team))²
```

The optimizer searches for `offsets*` that minimizes this loss.

### 4.3 Why this works mathematically

Match outcomes in the underlying predictor are driven by Elo
differences. If two teams have Elos 1900 and 1750, the predictor
gives the higher-rated team a specific win probability based on the
150-point gap, transformed through a logistic function.

Tournament-winner probability is a product of round-by-round survival
probabilities. Each survival probability is a function of all
opponents the team might face, weighted by the probability of facing
each. Increasing one team's Elo offset raises its survival in every
round, which raises its tournament-winner probability.

By choosing offsets per team, we have 48 degrees of freedom to fit
48 target probabilities. The system is exactly-determined in principle,
under-determined in practice (the propagation function isn't
invertible analytically), which is why we need an iterative optimizer.

### 4.4 The optimizer: Nelder-Mead

We use `scipy.optimize.minimize(method="Nelder-Mead")`. Why this
choice:

- **No gradient required.** The propagation function is built from
  many discrete bracket steps; the gradient with respect to Elo
  offsets is well-defined in theory but expensive to compute. Nelder-
  Mead uses only function evaluations.
- **Robust to local minima.** Nelder-Mead is a "simplex search" that
  explores by reflecting and contracting a 48-vertex simplex in the
  48-dimensional offset space. It's not guaranteed to find the global
  optimum, but it consistently converges to a near-zero loss on this
  problem.
- **Reasonable convergence speed.** Typical convergence is 2000-4000
  iterations; each iteration is one full propagation of all 48 teams
  through the baseline scenario, taking ~50ms. Total wall time: 2-5
  minutes.

We pass `options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-8}` to
limit iterations and convergence tolerance.

### 4.5 What "converged" looks like

After calibration succeeds:
- `final_loss` is typically 1e-5 to 1e-4 (mean squared error per
  team of around 0.1 to 1 basis point)
- `max_residual` is the maximum |P_propagated - P_market| across all
  48 teams; this is the metric `health.json` reports

The `--max-iter` CLI flag (default 2000) can be raised if calibration
doesn't converge - see `RUNBOOK.md` §2 for the diagnostic flow.

### 4.6 Worked example: Norway's calibration offset

In our running snapshot, the calibration result includes:

```
offsets = {
    "Norway":        +28,
    "Spain":         -7,
    "France":        +12,
    ...  (48 entries)
}
```

A +28 offset means: to make the baseline scenario produce Norway's
target tournament probability of 2.46% (matching `/fair-odds`), we
need to shift Norway's raw Elo up by 28 points. Why up? Because
Norway's raw Elo (let's say 1875) is enough to give them ~1.9% in
the baseline propagation, but the market thinks they should be at
2.46%. Shifting their Elo up by 28 increases their match-by-match
win probabilities, which compounds across rounds to lift their
tournament-winner probability to the target.

This calibrated predictor (raw Elo + offsets) is then used for the
79 scenario propagations downstream.


## 5. The 79 Scenarios

The combinatorial library that gives this framework its name.

### 5.1 What a scenario is

A `Scenario` is a hypothesis about how group stage plays out - a
specific assignment of winners and runners-up for each of the 12
groups. The full type:

```python
@dataclass
class Scenario:
    scenario_id: str
    description: str
    deviating_group: str       # e.g., "H" for single-deviation
    favourite: str             # the team market expects to win Group H
    upset_winner: str          # the team that wins instead
    standings: dict[str, list[str]]  # {"A": [1st, 2nd, 3rd, 4th], ...}
    # For pairwise scenarios:
    deviating_groups: tuple    # e.g., ("H", "G")
    favourites: tuple
    upset_winners: tuple
```

Code: `upset_propagation.scenarios`.

The library has exactly 79 scenarios:

- **1 baseline** - every group: favorite 1st, 2nd-favorite 2nd, etc.
  (`baseline.json`)
- **12 single-deviation** - for each group G, the 2nd-favorite wins G
  and the favorite drops to 2nd. Other 11 groups stay at baseline.
  Filename pattern: `{upset_winner}_runner_up_{G}.json`.
- **66 pairwise compound** - for each pair of groups (G1, G2) where
  G1 ≠ G2, deviate in both. Filename pattern:
  `{upset1}_{upset2}_runner_up_{G1}{G2}.json`. 66 = C(12,2).

Total: 1 + 12 + 66 = **79 scenarios.**

### 5.2 Why these 79 specifically

Two design choices:

**Why only winner-vs-runner-up upsets?** The combinatorial space of
"every possible group stage outcome" is enormous: each group has 4!
= 24 orderings, and 12 groups give 24^12 ≈ 1.3 × 10^16 total. We
restrict to the cases most likely to matter: who tops each group
(affects R32 seeding), and the "1-vs-2 swap" since the 2nd-favorite
is the most plausible upset winner.

**Why single + pairwise compound only?** We could extend to triple
combinations (220 of them) or higher. We don't because:
1. Each new scenario costs ~30 seconds of propagation
2. The matcher already has 79 to choose from; triples add diminishing
   returns
3. If two upsets happen in groups G1 and G2 plus a third in G3, the
   pairwise (G1, G2) scenario is usually a close enough fit, and the
   matcher's ensemble scoring naturally identifies it

The 79 are pre-computed once per cron cycle and stored as separate
JSON files in `output/` (one file per scenario).

### 5.3 Scenario file format

Each scenario JSON contains:

```json
{
  "scenario_id": "spain_runner_up_H",
  "description": "Spain finishes 2nd in Group H; Germany 1st",
  "deviating_group": "H",
  "favourite": "Germany",
  "upset_winner": "Spain",
  "standings": {"A": ["Mexico", "Norway", ...], "B": [...], ...},
  "survival": {
    "Mexico":   {"R16": 0.85, "QF": 0.62, "SF": 0.31, "F": 0.18, "Win": 0.10},
    "Norway":   {"R16": 0.78, "QF": 0.48, "SF": 0.22, "F": 0.11, "Win": 0.04},
    ...  (32 teams that qualify for R32 in this scenario)
  }
}
```

The `survival` field is the propagator's output - see Section 6.

Code: `upset_propagation.scenarios.build_all_scenarios()` constructs
the 79 scenarios; `run.py` orchestrates propagating each and writing
the JSONs.


## 6. Propagation

Given a scenario (a specific group standings hypothesis), produce the
per-team survival probabilities at each round (R16, QF, SF, F, Win).

This is where the bracket geometry meets the match predictor.

### 6.1 Inputs and outputs

Code: `upset_propagation.propagator.propagate(scenario, ratings,
predictor)`.

Inputs:
- `scenario` - group standings + the calibrated `Scenario` dataclass
- `ratings` - Elo dictionary
- `predictor` - calibrated match predictor (from Section 4)

Output: `PropagationResult` with:
- `teams` - the 32 teams that qualify for R32 in this scenario
- `survival` - `{team: {"R16": p, "QF": p, "SF": p, "F": p, "Win": p}}`

### 6.2 The bracket walk

For each scenario, we walk the bracket round by round:

**Step 1: Determine R32 lineup.** From the scenario's `standings`,
take the 12 group winners and 12 runners-up. For 3rd-placers, take the
top 8 ranked by raw Elo (this matches FIFA's "best 8 thirds"
methodology in spirit; the exact ranking is determined by group
performance during the actual tournament, but we use Elo as a stable
proxy since we don't have match-by-match data when scenarios are
pre-computed).

Total: 32 teams.

**Step 2: Build R32 pairings.** Using the FIFA bracket structure
encoded in `_vendored/tournaments/wc2026.py`:
- `R32_BRACKET` constant lists the 16 R32 matchups by source slot
  (e.g., "Group A winner plays best 3rd-placer from B/E/F")
- `r32_seeding_table.json` resolves which 3rd-placer is assigned to
  which winner, based on which 8 thirds advanced

Output: 16 specific R32 matchups.

**Step 3: Propagate forward.** For each R32 match, the calibrated
predictor gives us `P(team_a beats team_b)`. The R16 survival
probability of team_a is just `P(team_a beats team_b)` from R32.

For later rounds, it's a sum over all possible opponents:

```
P(team reaches R16) = P(team wins their R32 match)
P(team reaches QF)  = Σ_o P(team in R16) × P(opponent o in R16) × P(team beats o)
P(team reaches SF)  = Σ_o P(team in QF) × P(opponent o in QF) × P(team beats o)
P(team reaches F)   = ...
P(team wins)        = ...
```

Where `o` ranges over all teams that could be team_a's opponent at
that round, weighted by each opponent's own probability of having
made it that far.

Code: `propagator.propagate` walks `R32_BRACKET` and `LATER_ROUNDS`
constants from the vendored module.

### 6.3 Why no Monte Carlo

Note that we don't sample matches. The propagation is **fully
deterministic**: at each branch, we compute the exact probability
distribution over which team is at that bracket position. For a
single-elimination tournament with N=32 entrants, this is tractable
(O(N²) per round across all match positions, ~12,000 ops total).

This means two consecutive propagations of the same scenario produce
**identical** results to floating-point precision. No sampling noise.
The framework's only stochasticity is in the calibration step (Nelder-
Mead's reflection/contraction moves depend on the initial simplex,
which is deterministic given a fixed seed but slightly different from
run to run if we tweak hyperparameters).

### 6.4 Worked example: Norway in the baseline scenario

For Norway in the baseline scenario (everyone wins their group as
seed), Norway is the favorite of Group B. They win Group B, head into
R32. Their R32 opponent is the 3rd-placer assigned to Group B's
winner, which after Elo-ranking the 12 third-placers turns out to be
(say) Honduras.

The calibrated predictor gives `P(Norway beats Honduras) = 0.82`. So
`P(Norway reaches R16) = 0.82` in the baseline scenario.

R16 opponent: the winner of the R32 match in the adjacent bracket
position. Multiple teams could be there with different probabilities;
the propagator sums:

```
P(Norway reaches QF) = 0.82 × [
  P(opp1 in R16) × P(Norway beats opp1) +
  P(opp2 in R16) × P(Norway beats opp2) + ...
]
```

The full survival vector for Norway in the baseline:

```json
"Norway": {
  "R16": 0.82,
  "QF":  0.48,
  "SF":  0.22,
  "F":   0.11,
  "Win": 0.04
}
```

This is stored in `output/baseline.json`. Norway's Win column is 4%
in the baseline.

This vector is what the framework will weight-average across all 79
scenarios to produce Norway's implied probability.


## 7. The Realised State

Once tournament matches actually start playing, the framework
incorporates real results.

### 7.1 The data source

Code: `upset_propagation.match_results.state_from_matches_csv()`.

The cron reads `output/match_results.csv` if it exists. Schema:

```csv
match_id,group,home,away,home_score,away_score
1,A,Mexico,Algeria,3,0
2,A,USA,Curacao,1,1
...
```

Each row is a played match. With all 24 of a group's matches recorded,
the group's final standings are determined by FIFA's tiebreaker rules
(points → goal difference → goals scored → head-to-head → fair play →
drawing lots). This logic lives in
`_vendored/tournaments/wc2026.rank_group()`.

### 7.2 The RealisedState dataclass

```python
@dataclass
class RealisedState:
    standings: dict[str, list[str]]  # {"A": [1st, 2nd, 3rd, 4th], ...}
                                     # Only includes groups with complete results
    is_complete: bool                # True if all 12 groups have standings
    played_groups: set[str]          # The groups with complete data
```

If `match_results.csv` is missing or has 0 rows, `state.standings` is
empty, `played_groups` is empty. This is the pre-tournament case.

If only Group A's matches are done, `state.standings = {"A": [...]}`
and `played_groups = {"A"}`.

If all 12 groups have completed matches, `is_complete = True` and
all 12 groups are present.

### 7.3 What the state is used for

Two consumers:

1. **The ensemble matcher** (Section 8) uses `state.standings` to
   score the 79 precomputed scenarios.
2. **`compute_implied_probs`** uses `state.played_groups` to fill in
   any unobserved groups with baseline standings during partial-state
   matching.

### 7.4 The Elo-vs-fair_prob proxy bug (a lesson)

An earlier implementation of `state_from_matches_csv` used Elo-sorted
favorites to construct the synthetic "all favorites win" baseline.
This disagreed with `build_baseline_standings` (the canonical baseline
used everywhere else), because Elo-favorite and fair_prob-favorite are
not the same team in groups where market information (HFA, recent
form) shifts the order.

The fix: `state_from_matches_csv` now uses
`build_baseline_standings()` directly. The regression test
`test_state_from_matches_matches_baseline_standings` guards against
re-introducing the proxy.

**The general lesson:** when two parts of the framework need "the
favorite of group G," they must use the same definition. Otherwise
silent disagreement produces subtly wrong outputs.


## 8. The Ensemble Matcher

Given the realised state, score all 79 precomputed scenarios. This is
the framework's most interesting algorithmic step.

### 8.1 The two scoring components

Code: `upset_propagation.ensemble_matcher.find_best_scenarios_ensemble`.

**Hamming distance (group standings)**:

For each group G in `state.played_groups`:
- Look at the realised standing's top-2 (positions 1, 2).
- Look at the scenario's top-2 for the same group.
- Score = number of positions that differ (0 if perfect match, up to 2
  if both swapped).

Total Hamming = sum across all played groups.

Code: `upset_propagation.state_matcher.hamming_distance`.

**L1 distance (propagation outputs)**:

- Build the realised state's propagation table by running the
  propagator on the realised standings (with baseline filling in
  unobserved groups).
- Compare this propagation table to each scenario's stored propagation
  table.
- L1 = Σ_team Σ_round |P_realised(team, round) - P_scenario(team, round)|.

Code: `upset_propagation.l1_matcher.l1_distance`.

Hamming captures **structural similarity** (did the scenario predict
the right top-2 in each played group?). L1 captures **numerical
similarity** (does the scenario produce a propagation table close to
what we'd get from the realised state?).

### 8.2 Why use both

Tried alone:
- **Hamming alone**: too coarse for partial states. With 1 group
  observed, 12+ scenarios tie at distance 0 (every scenario that
  doesn't deviate in the observed group). No discrimination.
- **L1 alone**: can pick a scenario that's numerically close but
  structurally wrong (e.g., wrong group's favorite slipped).

Combining them via Borda count gives:
- Hamming prunes structurally-wrong scenarios
- L1 ranks among the structurally-equivalent ones

### 8.3 Borda count: combining two rankings

For each of the 79 scenarios:
1. Compute its Hamming distance and its L1 distance.
2. Rank all 79 scenarios by Hamming (lower = better, so rank 1 has the
   lowest distance).
3. Rank all 79 by L1 (lower = better).
4. Borda sum = `rank_Hamming + rank_L1`.
5. Score = `1 - (borda_sum - min_borda) / (max_borda - min_borda)`,
   normalized to [0, 1] where 1.0 = unanimous best.

Code: `upset_propagation.ensemble_matcher._assign_average_ranks`.

### 8.4 Fractional ranking for ties

A subtle but important detail. When multiple scenarios have the same
Hamming distance (very common with partial states), we use
**fractional ranking** for ties: each tied scenario gets the average
of the positions it would otherwise occupy.

Example: if 5 scenarios all have Hamming = 0, they would have occupied
positions 1, 2, 3, 4, 5. All 5 get rank `(1+2+3+4+5)/5 = 3.0`.

Without fractional ranking, ties would be broken arbitrarily by sort
order, making the matcher non-deterministic when distances are exactly
equal. Fractional ranking restores determinism.

### 8.5 The output

```python
@dataclass
class EnsembleMatch:
    scenario_id: str
    score: float           # 0.0 to 1.0
    borda_sum: float       # raw sum (lower = better)
    per_matcher_ranks: dict[str, float]   # {"hamming": 3.5, "l1": 7.0}
    per_matcher_distances: dict[str, float]  # {"hamming": 0, "l1": 0.234}
```

`find_best_scenarios_ensemble` returns the top k matches (typically
top-10 for `top_10_ranking.json`).

### 8.6 Worked example: Norway pre-tournament

In the pre-tournament case (`played_groups = ∅`), every scenario has
Hamming distance 0 (nothing to mismatch on). L1 distances also collapse
to near-identical values because the propagation tables of all 79
scenarios are weighted variants of the same baseline.

Result: every scenario gets roughly the same score (close to 1.0),
the matcher has no opinion, and the implied probability ends up close
to the baseline (which matches the calibration target, which is close
to Polymarket... close, but not equal - that's where the edge signal
comes from).

In a fully-played group stage with one upset (say, Spain wins Group H),
the matcher would assign Hamming = 0 to the `spain_runner_up_H`
scenario and Hamming > 0 to every other. L1 would rank within the
spain_runner_up_H family. The score for `spain_runner_up_H` would
approach 1.0 while others drop.


## 9. Implied Probabilities

Take the matcher's scenario scores and produce a per-team probability
distribution.

### 9.1 The weighted average formula

Code: `upset_propagation.implied_probs.compute_implied_probs`.

For each team and each round (R16, QF, SF, F, Win):

```
implied_P(team, round) = Σ_i [score_i^p × P_i(team, round)] / Σ_i score_i^p
```

Where:
- `i` indexes scenarios (1 to 79)
- `score_i` is the ensemble score from the matcher (0.0 to 1.0)
- `P_i(team, round)` is the propagation probability for this team
  at this round in scenario i
- `p` is the **weight exponent**

The denominator normalizes so the result is a valid probability.

### 9.2 The weight exponent p

The framework computes implied probabilities at **two values of p**:
p=4 (primary) and p=8 (sharper).

**Effect of p on the weighting:**
- p=1: linear weighting. The 79 scenarios contribute proportional to
  their raw scores. The matched scenario has only a slight pull (~2.5%
  of total weight even when it's the clear winner).
- p=4: matched scenario gets ~18% of total weight. Strong pull but
  significant smoothing across alternatives.
- p=8: matched scenario gets ~35% of total weight. Closer to
  "single-scenario lookup" with light smoothing.

The empirical sweep we ran (on a controlled test where the realised
state matched exactly to a precomputed scenario):

| p | Matched scenario weight share | Implied tracking error |
|---|---|---|
| 1 | ~2.5% | ~1.0 pp per team |
| 2 | ~6%   | ~0.7 pp per team |
| 4 | ~18%  | ~0.3 pp per team |
| 8 | ~35%  | ~0.1 pp per team |

p=4 was chosen as the primary based on a "tracks the matched scenario
within a fraction of a percentage point while still smoothing across
plausible alternatives" criterion.

p=8 was added (per the post-meeting feedback) as a sharper view
because reasonable people can disagree on the right exponent. Both
views are surfaced side-by-side; they agree pre-tournament and diverge
as the matcher discriminates more sharply during group stage.

### 9.3 The trade-off

Lower p (more uniform weighting):
- Smoother across scenarios
- Implied lags the matched scenario even when the match is clear
- More robust to over-confidence in the matcher's top pick

Higher p (more peaked weighting):
- Tracks the matched scenario aggressively
- Loses the smoothing benefit (reduces to "single scenario lookup")
- Vulnerable to matcher being subtly wrong

We surface both rather than pick a definitive answer.

### 9.4 Worked example: Norway's implied probability

In the pre-tournament case, all 79 scenarios contribute roughly
equally. The weighted average for Norway's Win column:

```
implied_P(Norway, Win) = Σ_i [score_i^4 × P_i(Norway, Win)] / Σ_i score_i^4
```

Numerically:
- 79 scenarios each contribute roughly equally (score ≈ 1.0 for all)
- Norway's Win probability varies slightly across scenarios (e.g.,
  0.038 to 0.045) because each scenario implies a slightly different
  bracket structure for Norway
- The weighted average produces 0.0266 (2.66%) - slightly above
  baseline's 0.0246 because the matcher's weights aren't perfectly
  uniform

This is the value that ends up in our_vs_market.json.


## 10. Market Comparison and Edge Classification

Combine the implied probability with the market price to produce the
trade signal.

### 10.1 The edge formula

For each team:

```
edge_pp = (implied_p4 - market_p) × 100
```

Where `market_p` is the devigged Polymarket probability (Section 2.2).

Sign convention:
- `edge > 0` → we think team has more chance than market → underpriced
- `edge < 0` → we think team has less chance than market → overpriced

Same for `edge_p8 = (implied_p8 - market_p) × 100`.

Code: `upset_propagation.market_log._compute_edges`.

### 10.2 The relative threshold

A small edge in absolute pp can be huge in relative terms for an
underdog. Haiti at 0.07% market with our model saying 0.14% is a
+100% relative move - meaningful - but only +0.07pp absolute.

Conversely, France at 16.0% market with our model saying 16.5% is
+3.1% relative - close to noise - but +0.5pp absolute.

Earlier versions used an absolute threshold (0.5pp), which both buried
real underdog signals and surfaced favorite noise. The current version
uses a **relative threshold**:

```
classify(edge_pp, market_pp):
    if market_pp == 0:
        return "fair"           # defensive against division by zero
    if |edge_pp / market_pp| < 0.10:
        return "fair"           # |relative| < 10%
    if edge_pp > 0:
        return "underpriced"
    return "overpriced"
```

Code: `upset_propagation.our_vs_market._classify_edge`.

The 10% threshold was chosen after design review. It can be
tuned later by editing `EDGE_DIRECTION_THRESHOLD_RELATIVE` in
`our_vs_market.py`.

### 10.3 Why classifications can differ between p=4 and p=8

For the same team, p=4 and p=8 produce different implied
probabilities, so different relative edges, so possibly different
classifications. Example:

```
Team X:
  market = 10.0%
  implied_p4 = 10.5%   delta = +0.5pp, relative = +5%  → fair
  implied_p8 = 12.0%   delta = +2.0pp, relative = +20% → underpriced
```

When this happens, it's informative: the matcher's top scenario is
bullish on Team X (driving p=8 sharply), but smoothing across
plausible alternatives at p=4 dilutes that view. You can decide
whether to trust the sharper or smoothed view.

### 10.4 The output: our_vs_market.json

Each row in the JSON has:

```json
{
  "team": "Norway",
  "market_pp": 2.26,
  "our_implied_p4_pp": 2.66,
  "delta_p4_pp": +0.41,
  "edge_direction_p4": "underpriced",
  "our_implied_p8_pp": 2.57,
  "delta_p8_pp": +0.32,
  "edge_direction_p8": "underpriced"
}
```

For Norway, both p=4 and p=8 classify as underpriced - the signal is
consistent across smoothing levels, which strengthens confidence.

Code: `upset_propagation.our_vs_market.build_snapshot`.


## 11. Production Infrastructure

The non-math machinery that makes the cron robust.

### 11.1 Atomic output via staging directory

Code: `upset_propagation.cron_utils.atomic_output_dir`.

A context manager that yields a staging directory `output.pending/`.
The cron writes all 80+ files into the staging directory during the
context. On clean exit, the contents are atomically renamed into
`output/`. On exception, `output/` is untouched and `output.pending/`
remains for forensics.

The atomic rename uses `os.rename()` (atomic on POSIX filesystems),
which means consumers reading `output/` concurrently never see a
half-written state. They always see either the previous complete
output or the new complete output.

Two artifacts can appear on disk:
- `output.pending/` - between a crashed run and the next clean one.
  Inspect for forensics, then delete.
- `output.old.<timestamp>/` - the previous output, moved aside during
  the swap and normally deleted on success. Persists only if cleanup
  itself fails.

### 11.2 Lockfile via fcntl.flock

Code: `upset_propagation.cron_utils.lockfile_acquired`.

Wraps `fcntl.flock(LOCK_EX | LOCK_NB)`. Non-blocking - concurrent
invocations fail loudly with `LockBusyError` carrying the holding
PID, rather than queueing.

The lockfile (`output/.cron.lock`) is a flat text file containing:
```
<pid> <iso_timestamp>
```

Operator-side escape: `force_unlock(lock_path, only_if_stale=True)`
clears the lock only if the holding PID is dead (verified via
`os.kill(pid, 0)`). The `--force-unlock-dangerous` CLI flag bypasses
the liveness check for situations where the framework is confused
about ownership.

### 11.3 Structured logging

Code: `upset_propagation.logging_config`.

Two presets:
- `configure_cron_logging(output_dir, timestamp)` - logs to
  `output/logs/run-<ts>.log` at INFO+, also escapes WARN+ to stderr
  (so cron's email-on-stderr surfaces real problems).
- `configure_interactive_logging(quiet=False)` - stdout only, INFO
  level (or WARN+ if quiet=True).

Both wipe prior handlers on entry. This prevents the
calling-configure-twice-stacks-handlers bug that plagued earlier
versions.

### 11.4 The --cron-mode orchestrator

Code: `upset_propagation.run`.

`python -m upset_propagation.run --cron-mode` chains everything:

```
1. Acquire lockfile
2. Open atomic staging dir
3. Configure cron logging
4. Validate Elo + fair_probs (fail-fast if bad)
5. Run pipeline: calibrate → propagate 79 scenarios → validate
6. Write all scenario JSONs + index + validation_report
7. Compute and write top_10_ranking.json
8a. Fetch market_probs, compute market_log entry, append to JSONL
8b. Build our_vs_market.json from latest market_log entry
8c. (out of scope for v3)
9. Snapshot output to runs/<timestamp>/
10. Write health.json
11. Atomic swap output.pending/ → output/
12. Release lockfile
```

If anything in steps 4-10 fails, the staging directory remains for
forensics and `output/` retains the previous successful run. The operator never
sees a half-broken state.

### 11.5 Health check verdicts

Code: `upset_propagation.health`.

The single command operators run:

```bash
python -m upset_propagation.health
```

Returns one of five verdicts as exit code:

| Code | Verdict | Trigger |
|---|---|---|
| 0 | HEALTHY | All checks passed |
| 1 | STALE | last_run_utc > max_age_hours ago (default 3h) |
| 2 | FAILURE | last run reported exit_status: failure |
| 3 | MISSING | health.json or required output file absent |
| 4 | DEGRADED | success but validation_pass=false OR calibration_max_residual > tolerance |

Six checks run in priority order (STALE, FAILURE, MISSING, DEGRADED,
HEALTHY). First failure short-circuits.

The full alert → diagnosis → fix tree is in `RUNBOOK.md`.

### 11.6 The validation report

Each run produces `output/validation_report.json` with:
- `summary.overall_pass` - single bool, true iff all checks passed
- `directional_sanity` - verifies that the calibrator's output matches
  the calibration target (small residuals, no sign flips)
- `sensitivity` - perturbs top-team Elos by small amounts, verifies
  outputs change smoothly (no chaotic dependence on small input
  changes)

If `overall_pass` is false, the framework's outputs may still be
written but health verdict is DEGRADED.


## 12. Limitations and Why

Honest accounting of what the framework doesn't do, and why.

### 12.1 Group-stage signal only

The 79 scenarios are built from group-stage outcomes. Calibration
fits the Win column against the market's tournament-winner price.
Both anchors assume group-stage discrimination is the primary source
of uncertainty.

Once R32 begins (June 28, 2026), the framework's view stops being a
live reflection of the tournament - the scenarios were precomputed
assuming various group standings, but they don't model knockout
outcomes. After R32, you'd need a different model (the framework's
output becomes a "pre-knockout view" rather than current state).

This is by design: building a knockout-stage model is a separate
project with different inputs, different calibration target, and
different bracket dynamics.

### 12.2 Tournament-Winner column only

The framework computes survival probabilities for all 6 rounds (R32,
R16, QF, SF, F, Win). Only the Win column is calibrated to the
market. The other columns are mechanical outputs of the propagator -
they reflect the calibration's bracket-walking logic, but their
accuracy against an external reference is unverified.

If we ever want to surface "edge on Brazil reaching the SF" as a
trade signal, we'd need to calibrate against a market price for that
column. Which exists on prediction markets (round-reaching markets)
but the framework doesn't currently consume those data feeds.

### 12.3 No sizing, no liquidity, no execution

The framework produces signal (probability disagreement). It does
NOT:
- Recommend position size
- Model order-book depth
- Generate fillable orders
- Account for slippage

Those are the trader's domain.

### 12.4 Polymarket-only comparison

Kalshi prices are available via the same FairLine API but Kalshi has
gaps (~14 missing teams). Polymarket has full 48-team coverage.

Trade-off: a Polymarket outage means our_vs_market.json becomes stale
until they recover. We trade single-platform risk for coverage and
simplicity. A multi-platform aggregator with fallback logic is
possible but not built.

### 12.5 Calibration sensitivity to Elo data

The framework's behavior depends on Elo ratings from
`data/mc_simu/elo_history.csv`. If this file is stale or wrong, the
calibrator will fit offsets that compensate (every team's offset
shifts to make the propagation match the market), but the resulting
predictor may be systematically biased on individual matches.

We catch the most egregious cases (zero/negative/non-finite Elo,
missing teams) in validation. We don't catch "Elo was correct 6
months ago but is now slightly stale across many teams." That would
require a separate quality-assurance process on the Elo source.


## 13. Putting It All Together: One Full Cron Cycle

Final running example. We follow a complete cron run end-to-end, with
Norway as the protagonist.

### 13.1 The setup

It's 13:00 UTC, 2026-06-11. The cron fires:

```bash
python -m upset_propagation.run --cron-mode
```

Pre-tournament. `output/match_results.csv` does not exist.

### 13.2 The pipeline executes

**13:00:01** - Lockfile acquired. Staging dir created.

**13:00:02** - `fetch_baseline_fair_probs()` hits `/fair-odds`.
Returns 48 teams summing to ~1.000. Norway: `0.0246` (2.46%).

**13:00:03** - `validate_fair_probs()` runs. PASSES - sum is 0.997,
no zeros, no implausible values. Haiti at 0.0007 (0.07%) - flagged
as low but not zero, so no FAIL.

**13:00:03** - `load_latest_elo()` loads ratings. 48 teams, all
in [1400, 2100]. validate_elo_history PASSES.

**13:00:04** - `calibrate()` begins. Nelder-Mead with 48-dimensional
simplex.

**13:04:00** - Calibration converges after ~3500 iterations. Final
loss: 1.8e-5. Max residual: 0.003 (i.e., the worst-fit team's
propagated probability differs from its market target by 0.3pp).
Norway's offset: +28.

**13:04:01** - `build_all_scenarios()` constructs the 79 scenarios
in memory.

**13:04:05** - Propagation begins. For each of the 79 scenarios:
1. Determine R32 lineup (24 group winners/runners-up + 8 best
   thirds)
2. Build R32 pairings via `R32_BRACKET` + `r32_seeding_table.json`
3. Walk bracket round by round using calibrated predictor
4. Write `{scenario_id}.json` to `output.pending/`

Norway's survival vector in the baseline scenario:
```json
{"R16": 0.78, "QF": 0.45, "SF": 0.20, "F": 0.10, "Win": 0.024}
```

**13:04:35** - All 79 scenarios written. Total: 79 scenario files +
`baseline.json` + `index.json` + `validation_report.json`.

**13:04:36** - `state_from_matches_csv()` finds no match_results.csv.
Returns empty RealisedState (`played_groups = ∅`).

**13:04:37** - `find_best_scenarios_ensemble(state=empty, ...)` runs.
Since no groups are observed, Hamming distance is 0 for all 79
scenarios. L1 distance is also near-zero (all scenarios are slight
variations on the same baseline). Ranks are nearly tied. Scores all
collapse to near-1.0.

**13:04:37** - Top-10 written to `output/top_10_ranking.json`. All
10 scenarios have score ≈ 1.0, indicating "no meaningful matcher
discrimination yet."

**13:04:38** - `fetch_market_prices()` hits `/prices`. 48 Polymarket
rows. After devig (renormalize to sum=1.0), Norway's market_prob =
0.0226 (2.26%).

**13:04:38** - `compute_implied_probs(state, predictor, ratings,
fair_probs, p=4)` runs. Weighted average across 79 scenarios.
Norway's implied_p4 = 0.0266 (2.66%).

Same call with p=8: Norway's implied_p8 = 0.0257 (2.57%).

**13:04:39** - Edge computation for Norway:
- market_pp = 2.26
- implied_p4_pp = 2.66, edge_p4_pp = +0.41
- implied_p8_pp = 2.57, edge_p8_pp = +0.32

Classification:
- p=4: |0.41/2.26| = 18.1% > 10% threshold → `underpriced`
- p=8: |0.32/2.26| = 14.2% > 10% → `underpriced`

Both views agree: Norway is underpriced.

**13:04:39** - `market_log.jsonl` appended. `our_vs_market.json`
written.

**13:04:40** - `health.json` written. exit_status: success.
n_scenarios: 79. calibration_max_residual: 0.003. validation_pass:
true.

**13:04:41** - Atomic swap: `output.pending/` → `output/`. Lockfile
released.

**13:04:41** - Cron exits 0. Total runtime: 235 seconds.

### 13.3 What the operator sees

```bash
$ python -m upset_propagation.our_vs_market --top 5
```

```
Our vs Market - 0/12 groups observed
  Source: market_log entry at 2026-06-11T13:00:35+00:00
  Sort: edge (top 5)
  p=4 is primary view; p=8 is the sharper-weighting comparison

  Team                Market    p4 Impl    p4 Δ    p4 Dir    p8 Impl    p8 Δ    p8 Dir
  ----------          ------    -------    -----   -------   -------    -----   -------
  England             10.42%    11.22%    +0.80    fair      11.22%    +0.81   fair
  Spain               16.27%    15.54%    -0.73    fair      15.82%    -0.45   fair
  France              15.41%    16.00%    +0.59    fair      16.08%    +0.67   fair
  Norway               2.26%     2.66%    +0.41   ↑ under     2.57%    +0.32  ↑ under
  Portugal            10.42%    10.10%    -0.32    fair      10.13%    -0.29   fair
```

Norway is the actionable signal. The framework's model anchored to
sportsbook-derived odds thinks Norway should be at 2.66%; Polymarket
prices them at 2.26%. That's a 18% relative gap, consistent across
both p views.

That's the entire pipeline. Every cron cycle, every two hours, the
framework recomputes this view as match results come in and as the
market moves.


## 14. References

For deeper reading on specific topics:

- **`docs/DEPLOYMENT.md`** - production cron setup, log shipping,
  alerting.
- **`docs/OUTPUT_FORMAT.md`** - exact JSON schemas.
- **`docs/RUNBOOK.md`** - operational troubleshooting.
- **`docs/VENDORING.md`** - how to update the vendored MC simulator.

For the actual code:
- `src/upset_propagation/` - every module discussed here.
- `src/upset_propagation/_vendored/` - the vendored MC simulator
  (vendored, not modified).
- `tests/` - ~254 unit tests covering every public function.
