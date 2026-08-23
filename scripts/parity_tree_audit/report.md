# Parity tree hygiene audit

Status: **blocked by the exclusive write fence; no deletion has been made.**

The exact 36-commit set named by the execution brief contains 74 unique added or
modified paths. All 74 are classified below: 23 live, 47 deliberately banked
evidence, and 4 debris. The four debris paths are outside this node's exclusive
write paths, so the done-when condition cannot be met without a scope extension.

## Path-by-path classification

| Path | Classification | One-line reason |
|---|---|---|
| `benchmarks/efit_forward_parity_slice.py` | live | Current frozen-six and constrained scorecard entry point; imported by later parity controls, although its definition-only historical routines need the scope ruling recorded below. |
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
| `scripts/dina_profile_routes/measure.log` | debris | No import, test, plan, evidence, or other documentation reference names this run log. |
| `scripts/dina_profile_routes/measure.py` | live | Cited re-runnable producer for the dual-route evidence receipt and figures. |
| `scripts/dina_profile_routes/measure_controls.log` | debris | No import, test, plan, evidence, or other documentation reference names this control run log. |
| `scripts/dina_profile_routes/measure_controls.py` | live | Cited re-runnable producer for the route-control addendum. |
| `scripts/dina_profile_routes/receipt.json` | deliberately-banked-evidence | Cited serialized dual-route receipt. |
| `scripts/dina_profile_routes/report.html` | deliberately-banked-evidence | Cited rendered dual-route evidence report. |
| `scripts/dina_profile_routes/reuse-report.md` | live | Plan-cited reuse map for the DINA profile-route work. |
| `scripts/dina_profile_routes/verify.log` | debris | No import, test, plan, evidence, or other documentation reference names this verification log. |
| `scripts/dina_profile_routes/verify.py` | live | Cited re-runnable verifier for the dual-route receipt. |
| `scripts/dina_profile_routes/verify_addendum.log` | debris | No import, test, plan, evidence, or other documentation reference names this addendum verification log. |
| `tests/test_efit_parity_boundary_volume.py` | live | Regression tests for the contour-volume receipt and protected evidence. |
| `tests/test_efit_parity_field_instrument.py` | live | Regression tests for the same-support field-energy control. |
| `tests/test_efit_parity_inductance_partition.py` | live | Regression tests for the disjoint current partition. |
| `tests/test_efit_parity_moment_definitions.py` | live | Regression tests for the common moment definitions. |
| `tests/test_efit_parity_root_geometry.py` | live | Regression tests for closed-axis branch selection and root attribution. |
| `tests/test_efit_parity_tared_external_field.py` | live | Regression tests for the corrected tare, analytic null, protected digests, and mesh receipt. |
| `tests/test_efit_parity_warm_neighbour.py` | live | Regression tests for warm-source selection and outcome classification. |
| `tests/test_equilibrium_forward_solve.py` | live | Production forward-solve regression suite covering the prescribed field and current-moment changes. |

Classification total: **74/74 paths** = **23 live + 47 deliberately-banked
evidence + 4 debris**.

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

The file itself remains classified live because its current `run`/`main` path
and many imported helpers are active. The isolated historical routines need
either removal or an explicit banked-reproducer ruling, but this node has no
write authority over that file. The remaining definition-only symbol is
`adjudicate_banked_receipt` in `efit_parity_tared_external_field.py`; it
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

## Duplicated helpers

Exact normalized-AST comparison across the six separately authored parity
modules found two copied helper families:

- Relative difference is defined four times: `_relative_error` at
  `efit_parity_moment_definitions.py:23`, and `_relative_deviation` at
  `efit_parity_boundary_volume.py:44`,
  `efit_parity_inductance_partition.py:59`, and
  `efit_parity_field_instrument.py:56`. Call sites occur at moment lines
  205-206, volume lines 91 and 229, partition lines 212-213, field lines
  156, 159, and 326, plus the tared module's imported helper provenance at
  line 956. The intended consolidation is one definition in
  `efit_parity_moment_definitions.py`, imported under the mechanism-appropriate
  local name by the other three modules.
- `_source_digests` is byte-for-byte duplicated at
  `efit_parity_inductance_partition.py:63` and
  `efit_parity_field_instrument.py:60`, called at partition line 346 and field
  line 348. The intended consolidation is the partition definition imported
  by the field-instrument module.

No consolidation was made because discovery of the out-of-scope debris is a
binding stop condition and the complete post-deletion suites could not yet be
produced.

## Scope blocker and required continuation

The debris paths that must be removed are:

- `scripts/dina_profile_routes/measure.log`
- `scripts/dina_profile_routes/measure_controls.log`
- `scripts/dina_profile_routes/verify.log`
- `scripts/dina_profile_routes/verify_addendum.log`

None is in the exclusive write fence. The node therefore cannot remove every
classified debris path, cannot truthfully run the required post-removal suites,
and cannot satisfy the quantitative done-when condition. Extend the fence to
those four logs and, if the definition-only historical routines are ruled
debris rather than retained reproducers, to
`benchmarks/efit_forward_parity_slice.py`; then remove/consolidate, run the
seven-file parity suite and the cited 28-test forward-solve suite once each,
and append their exact green counts here.
