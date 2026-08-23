# Parity tree hygiene audit

Status: **complete; every classified debris path was removed and both required
suites are green.**

The exact 36-commit set named by the execution brief contains 74 unique added or
modified paths. All 74 are classified below: 23 live, 47 deliberately banked
evidence, and 4 debris. The four debris paths were removed after an explicit
scope extension; all cited receipts, reports, reproducers, and protected
artifacts remain.

## Path-by-path classification

| Path | Classification | One-line reason |
|---|---|---|
| `benchmarks/efit_forward_parity_slice.py` | live | Current frozen-six and constrained scorecard entry point, imported by later controls; its 20 definition-only historical routines are retained reproducers for digest-protected banked receipts. |
| `benchmarks/efit_parity_boundary_volume.py` | live | Reconciles the two contour volumes and exports the shared protected-artifact verifier and polygon quadrature. |
| `benchmarks/efit_parity_field_instrument.py` | live | Implements the same-support field-energy control consumed by its tests and the tared control. |
| `benchmarks/efit_parity_inductance_partition.py` | live | Implements the disjoint current partition and boundary-moment calculation consumed by its tests and the tared control. |
| `benchmarks/efit_parity_moment_definitions.py` | live | Re-scores both moment definitions and exports the relative-error helper used by the tared control. |
| `benchmarks/efit_parity_root_geometry.py` | live | Supplies the closed-axis contour selection and geometry attribution imported by downstream controls. |
| `benchmarks/efit_parity_tared_external_field.py` | live | Current declared-support tare, analytic null, six-reference control, and mesh-sensitivity entry point. |
| `benchmarks/efit_parity_warm_neighbour.py` | live | Current warm-neighbour, Newton replay, and moment-seeded measurement entry points. |
| `docs/figures/dina-profile-routes/anchor_offsets.png` | deliberately-banked-evidence | Evidence figure for the declared-anchor versus map-anchor offset linked by the landed record. |
| `docs/figures/dina-profile-routes/forcing_routes.png` | deliberately-banked-evidence | Evidence figure comparing the declared and extracted forcing routes. |
| `docs/figures/dina-profile-routes/primitive_integrals.png` | deliberately-banked-evidence | Evidence figure for the stored primitive-integral comparison. |
| `docs/figures/dina-profile-routes/profile_routes.png` | deliberately-banked-evidence | Evidence figure for the two flux-function routes. |
| `docs/figures/dina-profile-routes/route_controls.png` | deliberately-banked-evidence | Addendum control figure cited by the landed record. |
| `docs/figures/dina-profile-routes/route_deviation.png` | deliberately-banked-evidence | Embedded landed-record figure for the route disagreement. |
| `docs/figures/efit-forward-parity/boundary-enclosed-volume-reconciliation.json` | deliberately-banked-evidence | Receipt for the matched contour-volume reconciliation. |
| `docs/figures/efit-forward-parity/boundary-enclosed-volume-reconciliation.png` | deliberately-banked-evidence | Figure paired with the contour-volume receipt. |
| `docs/figures/efit-forward-parity/boundary-imbalance-attribution.json` | deliberately-banked-evidence | Protected receipt attributing the passive-inclusive boundary imbalance. |
| `docs/figures/efit-forward-parity/boundary-imbalance-source-fields.png` | deliberately-banked-evidence | Protected field figure paired with the boundary attribution. |
| `docs/figures/efit-forward-parity/converged-root-geometry-attribution.json` | deliberately-banked-evidence | Integrity source for the 23 protected digests and receipt for the corrected root geometry. |
| `docs/figures/efit-forward-parity/converged-root-geometry-attribution.png` | deliberately-banked-evidence | Root-geometry evidence figure linked by the plan record. |
| `docs/figures/efit-forward-parity/field-energy-instrument-control.json` | deliberately-banked-evidence | Same-instrument field-energy receipt supporting the row retraction. |
| `docs/figures/efit-forward-parity/field-energy-instrument-control.png` | deliberately-banked-evidence | Figure paired with the field-energy control. |
| `docs/figures/efit-forward-parity/free-anchor-arm.json` | deliberately-banked-evidence | Protected negative showing the free-anchor arm did not close. |
| `docs/figures/efit-forward-parity/free-anchor-residual-trajectory.png` | deliberately-banked-evidence | Protected trajectory for the free-anchor negative. |
| `docs/figures/efit-forward-parity/inductance-deficit-partition.json` | deliberately-banked-evidence | Receipt partitioning core, enclosed non-core, and exterior current. |
| `docs/figures/efit-forward-parity/inductance-deficit-partition.png` | deliberately-banked-evidence | Figure paired with the current-partition receipt. |
| `docs/figures/efit-forward-parity/long-budget-plasma-route.json` | deliberately-banked-evidence | Protected negative showing the longer promotion budget still collapsed. |
| `docs/figures/efit-forward-parity/long-budget-residual-trajectories.png` | deliberately-banked-evidence | Protected trajectory figure for the long-budget negative. |
| `docs/figures/efit-forward-parity/mast-dina-composition-diff.json` | deliberately-banked-evidence | Protected one-application MAST/DINA composition comparison. |
| `docs/figures/efit-forward-parity/mast-dina-composition-update-fields.png` | deliberately-banked-evidence | Protected field figure for the composition comparison. |
| `docs/figures/efit-forward-parity/moment-definition-rescore.json` | deliberately-banked-evidence | Receipt showing the common-definition moment rescore. |
| `docs/figures/efit-forward-parity/passive-inclusive-convergence.json` | deliberately-banked-evidence | Protected convergence receipt after the passive-inclusive boundary repair. |
| `docs/figures/efit-forward-parity/passive-inclusive-convergence.png` | deliberately-banked-evidence | Protected trajectory paired with the passive-inclusive convergence receipt. |
| `docs/figures/efit-forward-parity/passive-inclusive-frozen-six-scorecard.json` | deliberately-banked-evidence | Protected frozen-six scorecard carrying the qualified parity verdict. |
| `docs/figures/efit-forward-parity/passive-inclusive-frozen-six-trajectories.png` | deliberately-banked-evidence | Protected frozen-six trajectory figure. |
| `docs/figures/efit-forward-parity/passive-inclusive-parity-slice.json` | deliberately-banked-evidence | Protected passive-inclusive single-slice receipt. |
| `docs/figures/efit-forward-parity/passive-inclusive-parity-slice.png` | deliberately-banked-evidence | Protected figure paired with the passive-inclusive slice. |
| `docs/figures/efit-forward-parity/passive-inclusive-stationary-polish.json` | deliberately-banked-evidence | Protected negative showing stationary polish escaped the plasma basin. |
| `docs/figures/efit-forward-parity/passive-inclusive-stationary-polish.png` | deliberately-banked-evidence | Protected trajectory for the stationary-polish negative. |
| `docs/figures/efit-forward-parity/pinned-parity-slice.json` | deliberately-banked-evidence | Protected pinned-portfolio negative. |
| `docs/figures/efit-forward-parity/pinned-route-survey.json` | deliberately-banked-evidence | Protected multi-route survey supporting the basin diagnosis. |
| `docs/figures/efit-forward-parity/reference-seeded-forward-slice.json` | deliberately-banked-evidence | Protected first-slice receipt recording convergence onto the vacuum root. |
| `docs/figures/efit-forward-parity/reference-seeded-forward-slice.png` | deliberately-banked-evidence | Protected figure paired with the first-slice receipt. |
| `docs/figures/efit-forward-parity/tared-external-field-solve.json` | deliberately-banked-evidence | **Retained VOID tare control:** it records why an all-valid-cell plasma image is wrong; deleting it would invite the same attribution error again. |
| `docs/figures/efit-forward-parity/tared-external-field-solve.png` | deliberately-banked-evidence | Figure paired with the retained void-tare control. |
| `docs/figures/efit-forward-parity/tared-plasma-support-solve.json` | deliberately-banked-evidence | Corrected declared-support tare receipt carrying the external-field attribution. |
| `docs/figures/efit-forward-parity/tared-plasma-support-solve.png` | deliberately-banked-evidence | Figure paired with the corrected declared-support tare. |
| `docs/figures/efit-forward-parity/two-phase-polish-trajectories.png` | deliberately-banked-evidence | Protected trajectory showing the two-phase polish did not close. |
| `docs/figures/efit-forward-parity/two-phase-polish.json` | deliberately-banked-evidence | Protected negative for the two-phase polish route. |
| `docs/figures/efit-forward-parity/vacuum-branch-diagnosis.png` | deliberately-banked-evidence | Protected diagnosis figure for the vacuum-root collapse. |
| `docs/figures/efit-forward-parity/warm-neighbour-stall-lift.json` | deliberately-banked-evidence | Receipt showing warm neighbours lifted none of the bounded stalls. |
| `docs/figures/efit-forward-parity/warm-neighbour-stall-lift.png` | deliberately-banked-evidence | Figure paired with the warm-neighbour negative. |
| `nova/equilibrium/__init__.py` | live | Exports the prescribed-current forward-field API used by package callers. |
| `nova/equilibrium/forward_operator.py` | live | Production forward operator implements the prescribed field and current-moment paths covered by the forward-solve suite. |
| `scripts/constrained_parity_reuse/reuse-report.md` | live | Cited reuse map consumed by the root-geometry receipt and later plan work. |
| `scripts/dina_profile_routes/addendum.json` | deliberately-banked-evidence | Cited addendum receipt for the route-control measurements. |
| `scripts/dina_profile_routes/measure.log` | debris | Removed: no import, test, plan, evidence, or other documentation reference named this stale console capture. |
| `scripts/dina_profile_routes/measure.py` | live | Cited re-runnable producer for the dual-route evidence receipt and figures. |
| `scripts/dina_profile_routes/measure_controls.log` | debris | Removed: no import, test, plan, evidence, or other documentation reference named this stale control capture. |
| `scripts/dina_profile_routes/measure_controls.py` | live | Cited re-runnable producer for the route-control addendum. |
| `scripts/dina_profile_routes/receipt.json` | deliberately-banked-evidence | Cited serialized dual-route receipt. |
| `scripts/dina_profile_routes/report.html` | deliberately-banked-evidence | Cited rendered dual-route evidence report. |
| `scripts/dina_profile_routes/reuse-report.md` | live | Plan-cited reuse map for the DINA profile-route work. |
| `scripts/dina_profile_routes/verify.log` | debris | Removed: no import, test, plan, evidence, or other documentation reference named this stale verification capture. |
| `scripts/dina_profile_routes/verify.py` | live | Cited re-runnable verifier for the dual-route receipt. |
| `scripts/dina_profile_routes/verify_addendum.log` | debris | Removed: no import, test, plan, evidence, or other documentation reference named this stale addendum verification capture. |
| `tests/test_efit_parity_boundary_volume.py` | live | Regression tests for the contour-volume receipt and protected evidence. |
| `tests/test_efit_parity_field_instrument.py` | live | Regression tests for the same-support field-energy control. |
| `tests/test_efit_parity_inductance_partition.py` | live | Regression tests for the disjoint current partition. |
| `tests/test_efit_parity_moment_definitions.py` | live | Regression tests for the common moment definitions. |
| `tests/test_efit_parity_root_geometry.py` | live | Regression tests for closed-axis branch selection and root attribution. |
| `tests/test_efit_parity_tared_external_field.py` | live | Regression tests for the corrected tare, analytic null, protected digests, and mesh receipt. |
| `tests/test_efit_parity_warm_neighbour.py` | live | Regression tests for warm-source selection and outcome classification. |
| `tests/test_equilibrium_forward_solve.py` | live | Production forward-solve regression suite covering the prescribed field and current-moment changes. |

Classification total: **74/74 paths** = **23 live + 47 deliberately-banked
evidence + 4 debris removed**.

## Protected evidence integrity

`docs/figures/efit-forward-parity/converged-root-geometry-attribution.json`
declares 23 protected banked files. A fresh SHA-256 comparison verified
**23/23 digests matching, 0 mismatches**. No protected file was written. The
retained void-tare receipt itself is unchanged at SHA-256
`e5ad919bd25fbf655f6892202440650b114f23bc16fc22044883a789a767cc76`.

## Search-based code audit

The commit-set Python paths were parsed only to enumerate module-level
functions/classes, then every symbol was searched by exact word across
`nova/`, `benchmarks/`, `tests/`, and `scripts/`. The inventory contains 314
top-level symbols, 209 outside test modules, and 21 non-test symbols with only
their definition hit.

Twenty definition-only symbols are in the out-of-scope
`benchmarks/efit_forward_parity_slice.py`:

`solve_arm`, `solve_free_anchor_arm`, `solve_pinned_arm`,
`survey_pinned_routes`, `survey_long_budget_routes`, `_long_budget_figure`,
`run_two_phase_polish`, `_two_phase_figure`, `_control_baseline`,
`_diagnosis_figure`, `_free_anchor_figure`, `_mast_composition_case`,
`_dina_composition_case`, `_attribution_figure`,
`_largest_structural_difference`, `_composition_figure`,
`_passive_inclusive_figure`, `_parity_metric_qualification`,
`_extended_passive_figure`, and `_passive_polish_figure`.

The file and all 20 routines are retained. They reproduce banked receipts in a
shipped constrained-arm benchmark whose 23 artifacts are digest-protected; a
routine that reproduces cited banked evidence is evidence infrastructure even
when no current entry point calls it. Deleting it would make the receipts
unreproducible, while retention has no runtime cost. The remaining
definition-only symbol, `adjudicate_banked_receipt` in
`efit_parity_tared_external_field.py`, is retained for the same reason: it
adjudicates the deliberately retained void receipt without re-running a solve
and is not an all-cell tare implementation.

The known all-cell failure was checked directly by data-flow search. The only
current plasma-image construction is `_implied_current`, which forms
`declared_valid = declared & valid` and zeroes every other cell before
`current_moment_image`; its receipt asserts
`plasma_image_uses_declared_support_only = True`. No parallel all-valid-cell
plasma-image construction survives. The banked void receipt is evidence of the
removed behavior, not an executable alternate path.

No constant `if True`/`if False` branches were found. The
`use_linear_moments` branch in the production operator is exercised with both
values across benchmarks and tests, so it is live rather than an always-one
flag.

## Consolidated helpers

Exact normalized-AST comparison across the six separately authored parity
modules found and removed two copied helper families:

- Relative difference now has one definition, `_relative_error`, in
  `efit_parity_moment_definitions.py`. That module is the natural owner because
  it defines the scoring semantics and was already the helper source consumed
  by the tare. Boundary volume, inductance partition, and field instrument
  import it as `_relative_deviation`, preserving mechanism-appropriate local
  call sites without copied arithmetic.
- Source-receipt digests now have one definition, `_source_digests`, in
  `efit_parity_inductance_partition.py`. That module is the natural owner
  because it first binds the source receipts whose digests the downstream
  field-instrument control consumes; the field module already imports its
  partition sources and now imports their digest helper beside them.

The post-change normalized-AST comparison reports **0 exact duplicate groups**
across the six audited modules. No similarly named copy in
`efit_forward_parity_slice.py` was found or changed.

## Removed debris

Commit `2f0adb98` removed exactly four uncited console captures:

- `scripts/dina_profile_routes/measure.log`
- `scripts/dina_profile_routes/measure_controls.log`
- `scripts/dina_profile_routes/verify.log`
- `scripts/dina_profile_routes/verify_addendum.log`

The cited `reuse-report.md`, `receipt.json`, `addendum.json`, evidence report,
measurement scripts, and verification script remain. Commit `b7645363`
contains only the three-module helper consolidation.

## Naming sweep

The mandatory path, case-insensitive label-word, case-sensitive bare-id, and
changelog-prose sweeps ran over the three modified benchmark modules and the
audit report. The only source hits were hexadecimal plot colours `#24527a` and
`#202020`, plus Ruff import-order code `E402`; these are colour data and tool
codes, both explicit non-label cases. There were **0 naming violations and 0
changelog-prose violations**, so no semantic name or comment rewrite was
needed.

## Post-removal validation

- Seven-file parity suite:
  `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" JAX_PLATFORMS=cpu uv run --no-sync pytest tests/test_efit_parity_root_geometry.py tests/test_efit_parity_warm_neighbour.py tests/test_efit_parity_moment_definitions.py tests/test_efit_parity_boundary_volume.py tests/test_efit_parity_inductance_partition.py tests/test_efit_parity_field_instrument.py tests/test_efit_parity_tared_external_field.py`
  — **55 passed in 162.12 s**, exit 0. Full log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T080558417226-parity-tree-hygiene-audit/parity-suite.log`.
- Forward-solve suite:
  `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" JAX_PLATFORMS=cpu uv run --no-sync pytest -m "slow or not slow" tests/test_equilibrium_forward_solve.py`
  — **28 passed, 1 warning in 54.00 s**, exit 0. The warning is SciPy's
  existing `invalid value encountered in scalar divide` warning in
  `test_the_host_root_find_holds_the_equilibrium_it_is_seeded_on`. Full log:
  `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T080558417226-parity-tree-hygiene-audit/forward-solve-suite-selected.log`.

The first forward-solve preflight omitted the repository-documented marker
override and therefore selected **0 of 28** tests (`28 deselected`, exit 5);
no test executed in that preflight. Its preserved log is
`forward-solve-suite.log`. The corrected command above is the one actual
forward-solve suite execution.

After both suites, a fresh integrity check again verified **23/23 protected
digests matching** and the retained void-tare SHA-256 unchanged. No failed
deletion needed restoration or reclassification.

## Follow-on discriminator

No node work remains. A future owner may re-test each of the 20 retained
definition-only routines against one discriminator: whether it produces an
artifact cited by a plan or evidence record. A cited producer stays evidence
infrastructure; an uncited producer may then be reconsidered as debris. That
future provenance audit is not part of this node and does not qualify the
complete result above.
