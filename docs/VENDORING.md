# Vendored Code

`src/upset_propagation/_vendored/` contains a snapshot of the Monte Carlo tournament simulator written by Duy Anh Nguyen.

## Current pinned version

- **Branch:** `snguyen_mc_simu`
- **Commit SHA:** `40e8f107d453023aa45c767cbd6afc286011cb79`
- **Commit message:** `mc_simu: add MV/star-presence blend, market-comparison + sensitivity tooling, scope1-3 audits + reliability figures`
- **Vendored on:** 2026-06-10

## Vendoring history

| Date | SHA | Notes |
|------|-----|-------|
| 2026-06-02 | (initial v1 vendor) | First snapshot |
| 2026-06-10 | `40e8f107` | Re-vendor. Algorithmic content identical to the initial snapshot - the vendored MV/star-presence/sensitivity work is in separate modules we don't import. Only diff vs v1 is import style (`from mc_simu.x` upstream → `from .x` in this copy). |

## What's vendored

From the upstream `src/mc_simu/`, copied to `src/upset_propagation/_vendored/`:
- `_common.py`
- `confederations.py`
- `elo.py`
- `single_game.py` - defines `predict_match`, `MatchContext`, `ModelParams`
- `simulator.py` - defines `build_ko_advance_table`, `make_elo_predictor`, `HostInfo`, `Predictor`
- `standings.py` - defines `rank_best_thirds`
- `__init__.py`
- `tournaments/__init__.py`
- `tournaments/wc2026.py` - defines `load_wc2026_bundle`, `R32_BRACKET`, `LATER_ROUNDS`, `ELO_HISTORY_NAME_ALIASES`

From `data/mc_simu/` in his repo, copied to `data/`:
- `r32_seeding_table.json` - 495 FIFA bracket combinations
- `wc2026_groups.json` - 12 groups × 4 teams
- `wc2026_fixtures.csv` - 104 matches
- `elo_history.csv` - full Elo history (we extract latest snapshot from it)

## What we did NOT vendor

Everything else from the vendored repo. The MC orchestrator (`run_mc_simu.py`),
the Euro adapters, MV/star-presence blending (`mv_blend.py`,
`star_presence.py`), market-comparison tooling (`wc2026_vs_*.py`,
`tune_to_market.py`), validation harness - we don't need any of it for our
deterministic propagation use case. Specifically verified during the
2026-06-10 re-vendor: none of his new modules are imported by the core
simulator files we vendor, so we can ignore them safely.

## Import-style convention

the vendored repo uses absolute imports (`from mc_simu._common import banner`).
After copying his files into our `_vendored/` directory, those become
broken (no `mc_simu` package at our path). So we systematically rewrite
to relative form:

- Files at `_vendored/` level use one dot: `from ._common import banner`
- Files at `_vendored/tournaments/` level use two dots: `from .._common import banner`

This rewrite is the ONLY change we make to his code - algorithmic content
is preserved byte-for-byte.

## Re-sync procedure

If the vendored simulator ships meaningful algorithmic changes (not just refactors), to
re-sync:

1. Clone or pull his repo:
   ```bash
   cd ~/Desktop  # or wherever you keep external repos
   git clone <upstream-simulator-repo>   # Duy Anh Nguyen's Monte Carlo simulator
   # (or `git pull` if you already have it)
   cd Prediction-Market-Project
   git checkout snguyen_mc_simu
   git log -1 --format="%H %ci %s"   # record this SHA
   ```

2. Inspect what changed in the modules we vendor:
   ```bash
   TARGET=~/Desktop/wc2026-upset-propagation/src/upset_propagation/_vendored
   for f in _common.py confederations.py elo.py single_game.py simulator.py standings.py; do
     echo "=== $f ==="
     diff -u $TARGET/$f src/mc_simu/$f | head -50
   done
   diff -u $TARGET/tournaments/wc2026.py src/mc_simu/tournaments/wc2026.py | head -50
   ```

   If the diffs are *only* `from mc_simu.x` ↔ `from .x` flips, this is a
   no-op for behavior. If the diffs include algorithmic changes, expect
   propagation outputs to shift - re-validate calibration and downstream
   numbers.

3. Verify his new modules aren't required:
   ```bash
   grep -n "mv_blend\|star_presence\|<any new module>" src/mc_simu/simulator.py src/mc_simu/single_game.py
   ```

   If grep returns nothing, our existing vendor list is complete. If it
   returns hits, vendor those new modules too.

4. Copy the files into `_vendored/`, overwriting:
   ```bash
   for f in _common.py confederations.py elo.py single_game.py simulator.py standings.py; do
     cp src/mc_simu/$f $TARGET/
   done
   cp src/mc_simu/tournaments/wc2026.py $TARGET/tournaments/
   ```

5. Convert his absolute imports to our relative form. Each `from mc_simu.x`
   becomes either `from .x` (in `_vendored/`) or `from ..x` (in
   `_vendored/tournaments/`). Use sed:
   ```bash
   cd ~/Desktop/wc2026-upset-propagation
   # Files at _vendored/ level (one dot):
   for f in src/upset_propagation/_vendored/{elo,simulator,standings}.py; do
     sed -i '' 's|^from mc_simu\.|from .|' $f
   done
   # Files at tournaments/ level (two dots):
   sed -i '' 's|^from mc_simu\.|from ..|' src/upset_propagation/_vendored/tournaments/wc2026.py
   ```

6. Verify no stray absolute imports remain:
   ```bash
   grep -rn "from mc_simu" src/upset_propagation/_vendored/
   # Should return only the comment in __init__.py, no actual imports
   ```

7. Run the verification suite:
   ```bash
   python -c "from upset_propagation.scenarios import build_all_scenarios; print('imports OK')"
   pytest tests/ -q                       # must still pass all 101
   python -m upset_propagation.state_matcher  # must pass all 5 smoke tests
   ```

8. If behavior is expected to change (algorithmic update from the vendored simulator),
   re-run the full pipeline and check max_residual + validation pass:
   ```bash
   python -m upset_propagation.run --skip-if-fresh 0
   ```

9. Update this document with the new SHA, date, and a row in the
   Vendoring History table noting what changed.

## Why vendor instead of import?

We vendor (copy his code into our repo) rather than pip-install or
git-submodule for three reasons:

1. **Speed of delivery.** No package publishing dance; copy-paste ships.
2. **Deliberate-update semantics.** His repo can change without affecting
   us until we explicitly re-vendor. Removes the "did upstream just
   break us?" debugging path during the tournament.
3. **Self-containment for downstream integration into FairLine.** The
   framework is one git clone away from being run anywhere; it doesn't
   need to manage a second dependency.

The cost is exactly what this doc handles: a deliberate, documented
re-sync procedure when his code does change.