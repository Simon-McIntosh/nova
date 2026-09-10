# cca-unit-amplitude-current-census — unit-amplitude per-cell current census

Measurement node on `nova:uniform-cell-clip-and-coupling`. One `all_debug` CPU
job evaluates, on the weak-rotation-reactor-static −110 certificate seed and
the committed chord terminal state, the per-cell zeroth current moment at unit
amplitude under `set_support_clip_mode('chord')` and
`set_support_clip_mode('exact')`, and attributes where the two modes differ.

Source revision: `28a49bcb` (worktree HEAD). Receipt:
`docs/figures/uniform-cell-clip-and-coupling/amplitude-census/receipt.json`.
Figures: `difference-contours.png/svg`, `difference-ranked.png/svg` (same dir).
Run: one `all_debug` CPU job (SLURM job 1268821, `98dci4-clu-4074`,
`JAX_PLATFORMS=cpu`, 8 cores), `AMPLITUDE_CENSUS_EXIT=0`, source span
`benchmarks/unit_amplitude_current_census.py`, launcher
`scripts/run_amplitude_census.sh`.

## What was measured

- **States** — the production cold seed of the committed chord row (regenerated
  through `certificate._production_seed` / `profile.cold_seed_portfolio`, `solver.seed`
  digest `7712eda1…` in `…/discriminator/parts/control/weak-rotation-reactor-static-production-route-reduced.json`)
  and the same row's `render_data.terminal_flux_wb` (the committed chord terminal).
- **Quantity** — `operator.cell_current_moments(state).cell_current` at unit
  amplitude (no declared-current scaling), per mode. Unit-amplitude total × the
  requisite amplitude = target current (`target_current_a` = 16,314,773.31 A).
- **Per-cell records** — chord current, exact current, difference, the analytic
  exact current over the cell's analytic plasma intersection (the shared adaptive
  oracle `_exact_cell_integral`), analytic surface class (interior/cut/exterior by
  exact plasma-area fraction), `profile_participation`, `vertex_participation`,
  and each mode's clipped-support vertex count.

## Results

| state | mode | unit-amplitude total [A] | amplitude | vs banked |
|---|---|---|---|---|
| seed | chord | 17,546,965 | 0.9297775241954396 | **exact** (banked 0.9297775241954396) |
| seed | exact | 7,413,774 | 2.200602887 | **within 1.8e-13** (banked 2.200602886684049) |
| terminal | chord | 15,540,897 | 1.0497961485612264 | **exact** (banked 1.0497961485612264) |
| terminal | exact | 15,148,862 | 1.076963617 | no banked cross-check on this state |

Chord-exact total ratio: **2.3668 at the seed** (exact carries 42 % of the chord
current), **1.0259 at the committed chord terminal** (the modes nearly agree).

The seed and terminal-chord unit-amplitude totals reproduce the committed row's
banked amplitude history to machine precision — the strong identity statement
that both modes were evaluated on the same states the row persisted. The
terminal-exact banked amplitude (5.561…) is **not** a reference here: it was
recorded on the c-r100 arm's own terminal state (a different flux state, up to
143 Wb different), not on the committed chord terminal this census evaluates.

## Which cell class carries the missing current

The seed-state gap (`chord − exact = 10.13 MA`) is carried overwhelmingly by
**cells the exact clip drops entirely** — 76 cells (of 135) get an exact
clipped-support vertex count of zero, i.e. the curved clip finds no intersection
with the cell at the seed. Split by the analytic separatrix:

| analytic class | cells | chord−exact [A] | exact = 0 | exact-clip verts = 0 |
|---|---|---|---|---|
| interior | 78 | 5,919,388 | 25 | 25 |
| cut | 43 | 3,985,699 | 37 | 37 |
| exterior | 14 | 228,103 | 14 | 14 |

So at the seed the missing current sits 58 % in analytic-**interior** cells
(25 of them dropped outright — cells wholly inside the true plasma), 39 % in
analytic-**cut** cells, 2 % in analytic-**exterior** cells. The exact clip zeroes
every one of the 76 dropped cells (zero-vertex support), never a partial
reduction: a cell either keeps its clipped closed region or carries nothing.

At the committed chord terminal the same split nearly vanishes: only 25 cells
are dropped (1 interior, 10 cut, 14 exterior) and the total difference is
0.39 MA (2.5 % of the chord total).

## Mechanism and responsible code

The seed-state discrepancy is a **cold-seed property, not an equilibrium
property**: the seed is a uniform current-disc cold start whose own flux
encloses far less than the analytic plasma, so the exact clip (which clips each
cell against the *state's own* curved boundary) drops cells the analytic oracle
places inside the plasma, while the chord mode keeps the profile-participation
cells' full atomic current. On the committed chord terminal (where the state's
boundary approaches the equilibrium boundary) the two modes carry almost the
same current (ratio 1.03).

Responsible lines (the exact path, `nova/equilibrium/forward_operator.py`):

- `:2177` — `participation = masks.profile_participation | vertex_participation`
  (the exact-mode participation is limited to this union; on a cold seed the
  curved boundary excludes most cells from it).
- `:2178-2183` — `traced_support = atomic_mesh.traced_clip(inside_boundary, …,
  participating_cell=participation)` then `exact_support =
  traced_support.qualify(participation)`: every non-intersecting cell keeps
  `vertex_count == 0` and therefore carries zero current.
- `nova/equilibrium/stencil_mesh.py:428` and `:468`
  (`_direct_profile_current_moments`, the `count >= 3` gate) convert the
  zero-vertex support into zero current.

The chord path's contrast is `:2091-2092`
(`_moment_support_masks` — the chord mode returns the profile-partition masks
unchanged, i.e. full atomic cells by label).

## Verification and provenance findings

1. **CPU cannot reproduce the committed seed digest.** The committed row's seed
   digest was produced by the GPU certificate lane (`betelgeuse`,
   `jax_default_backend: gpu`); the seed flux is a JAX reduction whose sum order
   differs CPU-vs-GPU, so the byte digest is not CPU-reproducible. Every
   structural seed field regenerates identically
   (`seed.structural_verification.all_structural_fields_identical = true`,
   worst relative difference 0.0), and the seed's unit-amplitude totals
   reproduce both banked *seed* amplitudes exactly (chord and exact), so the
   regenerated seed **is** the identity state. This is recorded as a finding:
   the plan's "verified against the banked state digest" check is only
   satisfiable on the GPU lane that produced it.
2. **The plan's reduced "exact carries less than half" statement is seed-state
   specific.** The banked amplitude history shows the factor ~2.4 at the seed;
   on the *committed chord terminal*, evaluating the exact clip on that same
   state gives ratio 1.026, not 0.5. The banked terminal-exact amplitude
   (5.561) cannot be used as a same-state reference and should not be cited as
   one.
3. **Curved-level sign check: 0 disagreements at the seed, 8 at the chord
   terminal.** At every cell centroid the spline and local quadratic levels and
   the `inside_boundary` test agree in sign at the seed. At the committed chord
   terminal, the *spline* level disagrees with both the local quadratic and the
   inside test at exactly 8 cell centroids — `cell 3, 38, 52, 76, 89, 114,
   120, 125` — every one an analytic **cut** (boundary-crossing) cell, while
   the local quadratic agrees with `inside_boundary` everywhere. Because
   production selects the spline when the fit executes, those 8 boundary cells'
   vertices are evaluated on the spline's side of the boundary at the terminal
   state. The terminal state is the non-converged chord result (residual
   7.4e-3), so an imperfect spline there is unsurprising, but the spline/local
   disagreement is a measured property of the exact clip near equilibrium and
   is recorded rather than smoothed over.

## Figures

- `difference-contours.png/svg` — whole-domain line-contour panels of per-cell
  chord−exact current (seed, terminal), cut cells outlined in gold, wall and
  magnetic axis drawn (imas-ink through `nova.media`). Positive contour family
  solid red, negative dashed navy, levels stated per panel.
- `difference-ranked.png/svg` — 1-D ranked bar charts of chord−exact over all
  cells (cut cells gold), with the ±1e-6-plasma-current threshold band marked.

## Follow-ons

- The seed-state exact-clip drop of 25 analytic-*interior* cells is the
  concrete face of the cold-start mismatch between the two clip modes; whether
  the solve should be seeded so the exact clip's own boundary contains the
  analytic plasma, or whether chord vs exact should agree at a cold state at
  all, is a design question for the plan, outside this node's measurement scope.
- The committed row's GPU-only seed digest means no CPU measurement can cite the
  byte digest as verification; a CPU-validated seed-digest reference would need
  a CPU-lane certificate regeneration (out of scope, touches certificate
  artifacts).
