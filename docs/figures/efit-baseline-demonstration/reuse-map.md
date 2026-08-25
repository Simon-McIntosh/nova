# Spline topology extraction capability reuse map

## Outcome

The spline foundation and most low-level topology machinery already exist in
Nova, but the complete requested extraction does not. The shortest sound route
is to compose and lightly factor existing Nova capabilities:

1. `nova.linalg.tensor_spline.fit_tensor_spline` and
   `TensorBSpline.evaluate` are the global, not-a-knot, C2 surface authority.
2. `nova.equilibrium.flux_surface_connectivity.traced_spline_contour` is the
   only existing contour primitive that takes its edge roots and endpoint
   tangents from that same global surface while retaining fixed JAX shapes.
3. The fixed-iteration Hessian Newton loop embedded in
   `traced_spline_contour` is the best donor for a reusable stationary-point
   polish. The current public null candidate path,
   `nova.equilibrium.stencil_nulls.xpoint_candidates`, should continue to
   provide fixed-capacity candidate discovery, but its local quadratic fit
   must not remain the final position authority.
4. Nova already resolves four-crossing saddle cells twice: contour edge
   pairing in `traced_spline_contour`, and two-lobe support partitioning in
   `nova.equilibrium.separatrix_clip._traced_clip`. Neither assembles the
   cell-local arcs into one closed boundary plus retained open legs. Ambix's
   `lcfs_contour(..., clip_legs=True)` proves the desired closed-lobe semantics,
   but discards the open legs and uses host `contourpy` over the sampled grid.
5. The comparison layer should reuse Nova's MAST referee and the frozen-arm
   corroboration metrics, plus the DIII-D labelled-map harness. Their boundary
   metrics need one common closed-boundary implementation: symmetric sup and
   RMS nearest-polyline distances. Ambix contributes a useful firewalled EFIT
   read/judge contract, but its eight-radius RMS is too sparse for the final
   geometric gate.

There is therefore **one implementation gap**: a fixed-capacity graph/branch
assembly over the per-cell global-spline arcs, rooted at the polished saddle,
that returns the axis-enclosing closed branch and the open branches separately.
No candidate in either repository currently supplies that end-to-end contract.

## Search basis and coverage

- Nova source revision surveyed: `8564b05079e6d279ac959b2a9137d5d3c90e203b`.
- imas-ambix tracked source revision surveyed:
  `d05e27b6288a0b93d171a09fbfe518739e2486ca`.
- Search covered `nova/`, `tests/`, `benchmarks/`, relevant evidence/figure
  drivers, and imas-ambix `imas_ambix/`, `tests/`, and relevant docs.
- The map below records **25 candidates across four required capability
  classes: 18 Nova candidates and 7 imas-ambix candidates**. Every row names
  the module, symbols, test evidence (or the explicit absence of a focused
  test), and a one-line fitness verdict.

Fitness words are deliberate: **reuse** means the symbol is suitable as-is;
**factor** means the mechanism is suitable but is trapped inside a larger
function; **qualified** means only part of the contract transfers; **reject**
means using the candidate would preserve the failure this work is meant to
remove.

## Contour tracing from the global tensor spline

| Repository | Module and symbols | Existing test evidence | Fitness verdict |
|---|---|---|---|
| Nova | `nova/linalg/tensor_spline.py`: `fit_tensor_spline`, `TensorBSpline.cell_coefficients`, `TensorBSpline.evaluate`, `TensorSplineEvaluation` | `tests/test_tensor_spline.py::test_values_gradient_and_hessian_match_scipy`; `::test_cell_coefficients_are_c2_across_every_interior_boundary`; `::test_jit_eager_and_vmap_keep_fixed_shapes` | **Reuse as surface authority.** It fits the complete map globally, exposes value/gradient/full Hessian, and supplies fixed `(nz-1, nr-1, 4, 4)` Bernstein blocks; it does not itself trace or connect a contour. |
| Nova | `nova/equilibrium/flux_surface_connectivity.py`: `traced_spline_contour` and its fixed bisection, tangent, saddle-pairing, and cubic-control construction | `benchmarks/higher_order_contour.py` (`measure`, `check`); receipt `docs/figures/coefficient-space-newton/higher-order-contour.json`; transport characterisation targets named in `CHARACTERISATION_TESTS` | **Reuse as the contour primitive.** This is the only fixed-shape path whose crossings and tangents come from the global spline; add branch assembly rather than another tracer. |
| Nova | `nova/equilibrium/flux_surface_extraction.py`: `_surface_clips`, `_bicubic_edge_crossings`, `_solve_bicubic_ordinate`, `_bicubic_arc_moment_correction`, `extract_flux_surface_geometry` | `tests/test_flux_surface_extraction.py::test_even_edge_root_pair_is_resolved_on_iter_inner_surface`; `::test_short_corner_arc_rejects_a_disconnected_stationary_split`; `::test_real_equilibrium_reference_gates` | **Qualified reuse for spline arcs and integration.** `_surface_clips` already feeds global spline blocks into robust roots and quadrature, but returns aggregate surface geometry rather than an ordered LCFS/separatrix polyline. |
| Nova | `nova/biot/contour.py`: `Contour.levelset`, `Contour.closedlevelset`, `Surface.closed`; `nova/biot/levelset.py`: `LevelSet.__call__` | No focused contour-accuracy or saddle-branch test was found; use is exercised only through legacy consumers | **Reject as curve authority.** This is dynamic host `contourpy` geometry on sampled grid values, selects the first closed ring, and provides neither the global C2 surface nor fixed-capacity branch identity. |
| imas-ambix | `imas_ambix/latent/topology.py`: `_axis_enclosing_ring`, `lcfs_contour`, `LcfsContour` | `tests/latent/test_topology.py::test_lcfs_contour_limited_ring_on_wall_xslots_empty`; `::test_lcfs_contour_diverted_ring_pinches_at_xpoint`; `tests/latent/test_lcfs_legclip.py::test_clip_legs_reaches_xpoint_and_excludes_legs` | **Qualified behavioral reuse only.** The outermost axis-enclosing-ring rule is correct, but the implementation is host `contourpy` on the grid surface and cannot be the spline-precise production tracer. |
| imas-ambix | `imas_ambix/latent/connectivity_boundary.py`: `boundary_read_jax`, `_ray_radii`, `ConnectivityBoundary` | `tests/latent/test_connectivity_boundary.py::test_jit_vmap_grad_safe_and_fixed_shape`; `::test_reproduces_host_lcfs_diverted`; `::test_clip_legs_radii_stay_on_lobe` | **Qualified fixed-shape referee.** It supplies stable lobe radii without contour extraction, but it returns sampled rays rather than a spline polyline and cannot retain open legs. |

### Contour decision

Use `traced_spline_contour` as the sole curve generator. Preserve its canonical
zero padding, two-segments-per-cell capacity, and JIT/vmap shape contract.
Build ordering and branch identity over `segment_controls_rz` and
`segment_endpoints_rz`; do not route through `Contour`, `LevelSet`, or Ambix's
host contour generator.

## Stationary-point Newton polish on a C2 surface

| Repository | Module and symbols | Existing test evidence | Fitness verdict |
|---|---|---|---|
| Nova | `nova/linalg/tensor_spline.py`: `TensorBSpline.evaluate` returning both first derivatives and all three independent Hessian entries | `tests/test_tensor_spline.py::test_values_gradient_and_hessian_match_scipy`; `::test_point_gradient_with_respect_to_map_matches_finite_difference` | **Reuse directly.** This is the derivative authority required by Newton on the same C2 surface used for contour roots. |
| Nova | `nova/equilibrium/flux_surface_connectivity.py`: the `locate_saddle` fixed-iteration loop inside `traced_spline_contour`, plus returned `saddle_stationary`, `saddle_rz`, and `saddle_value` | `benchmarks/higher_order_contour.py::_saddle_case` exercises an offset saddle level and an exact tied saddle level; the receipt requires every ambiguous cell to be resolved or explicitly tie-broken | **Factor into the production polish.** The Hessian Newton mechanism is correct and global-spline based, but it is private to ambiguous contour cells and accepts no external candidate seeds. |
| Nova | `nova/equilibrium/flux_surface_extraction.py`: `_axis_expansion_shape` fixed eight-step gradient/Hessian Newton over the four cells adjoining a seed | `tests/test_flux_surface_extraction.py::test_analytic_shape_gradient_jit_and_batch_contract`; real-surface gates cover its downstream shape columns | **Qualified scaffold.** It demonstrates multi-cell seed handling and fail-closed Hessian qualification on the global spline, but it accepts only positive-definite axis extrema and is not a saddle API. |
| Nova | `nova/equilibrium/stencil_nulls.py`: `xpoint_candidates`, `critical_point_candidates_batch`, `_refine_selected_vertices` | `tests/test_stencil_nulls.py::test_classifier_finds_o_and_x`; `::test_xpoint_subgrid_matches_symmetry`; `::test_batch_order_and_scalar_adapter_agree`; device parity test in the same module | **Reuse discovery, replace final polish.** Fixed-capacity candidate scoring and fit-state metadata are production-ready, but `_refine_selected_vertices` fits a local quadratic patch and therefore cannot remain position authority. |
| Nova | `nova/equilibrium/flux_surface_extraction.py`: `_bicubic_stationary_point` | `tests/test_flux_surface_extraction.py::test_bicubic_iterations_preserve_carry_dtype_and_float64_values`; extremum convergence in `::test_extremum_radial_position_converges_faster_than_first_order` | **Reject for X-point polish.** It solves a coordinate extremum constrained to a level set, not `grad(psi)=0`; keep it for surface-shape extrema only. |
| imas-ambix | `imas_ambix/latent/topology.py`: `find_critical_points`, `_gradient_hessian`, `_bilerp`; and `imas_ambix/latent/stencil_nulls.py`: `xpoint_candidates`, `_refine_at`, `subnull` | `tests/latent/test_topology.py::test_saddle_is_an_x_point`; `::test_double_well_two_o_points_bracket_one_x_point`; `tests/test_stencil_nulls.py::test_xpoint_subgrid_matches_symmetry` | **Reject as precision authority; retain semantics as a referee.** Both routes Newton-polish finite-difference or local biquadratic derivatives rather than the global C2 spline, and the host route has dynamic candidate loops. |

### Newton decision

Keep `xpoint_candidates` as the candidate census and scoring boundary. Factor a
small fixed-trip `polish_stationary_points(spline, seed_rz, valid)` kernel from
the Newton algebra already present in `traced_spline_contour`, and return
polished position, value, gradient norm, Hessian determinant/type, convergence,
and in-domain validity in the same fixed slots. This removes the local-fit
position authority without rebuilding discovery or device contracts.

## Separatrix splitting and clipping at a saddle

| Repository | Module and symbols | Existing test evidence | Fitness verdict |
|---|---|---|---|
| Nova | `nova/equilibrium/flux_surface_connectivity.py`: `traced_spline_contour` fields `ambiguous_saddle`, `ambiguous_resolved`, `ambiguous_tie_broken`, `segment_edge_indices`, `segment_controls_rz` | `benchmarks/higher_order_contour.py::_saddle_case`; receipt checks that both resolved and exact tie-broken cases occur | **Reuse for cell-local saddle connectivity.** It chooses the two arc pairings from the global-spline stationary value, but does not connect arcs between cells or label closed versus open branches. |
| Nova | `nova/equilibrium/separatrix_clip.py`: `_traced_clip`, `AtomicCellMesh.traced_clip`, `TracedClippedSupports.branch_support_vertices`, `branch_vertex_count`, `saddle_vertex` | `tests/test_saddle_partition.py::test_four_crossing_saddle_partitions_cell_with_explicit_branch_vertex`; `::test_branch_profiles_remove_the_audited_single_chord_current_error`; `::test_traced_support_contract_is_additive_and_keeps_existing_shapes` | **Reuse the saddle split primitive.** It inserts an explicit saddle and preserves two fixed-capacity lobes exactly, but these are per-cell support polygons, not global closed/open separatrix branches. |
| Nova | `nova/equilibrium/flux_surface_extraction.py`: `_surface_clips` topology qualification through axis-connected flood fill and spline arc corrections | `tests/test_flux_surface_extraction.py::test_boundary_band_is_fixed_shape_and_overflow_fails_closed`; `::test_short_corner_arc_rejects_a_disconnected_stationary_split` | **Reuse qualification and capacity discipline.** It supplies axis-connected participation and fail-closed overflow, but intentionally does not expose a branch polyline. |
| Nova | `docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py`: `_binding_contour` and `_axis_enclosing_closed_contour` | Banked `efit-topology-corroboration.json` and twelve-panel PNG; no focused unit test for the branch selector was found | **Qualified measurement-only donor.** It already selects the axis-enclosing closed contour from figure-time contour lines, but is dynamic host analysis and does not retain legs. |
| imas-ambix | `imas_ambix/latent/topology.py`: `lcfs_contour(..., clip_legs=True)`, `_axis_enclosing_ring`, `emergent_xpoints` | `tests/latent/test_lcfs_legclip.py::test_clip_legs_reaches_xpoint_and_excludes_legs`; `::test_clip_legs_limited_case_matches_plain`; diverted pinch test in `tests/latent/test_topology.py` | **Reuse the semantic oracle, not the implementation.** Closed axis-enclosing selection correctly excludes open legs and reaches the X-point corner, but the open legs are discarded and the curve is grid/contourpy based. |
| imas-ambix | `imas_ambix/latent/connectivity_boundary.py`: `boundary_read_jax` lobe clamp and returned X-set | `tests/latent/test_connectivity_boundary.py::test_clip_legs_radii_stay_on_lobe`; `::test_emergent_xset_holds_both_nulls_of_a_double_null` | **Qualified device-side cross-check.** It prevents ray run-out along a leg and retains fixed X slots, but emits only angular radii and cannot represent separate open branches. |

### Split decision and missing contract

Assemble a fixed-capacity graph whose nodes are canonical shared edge crossings
plus the polished saddle and whose edges are the valid cubic segments. Starting
from the axis side, identify exactly one closed component for LCFS metrics; emit
the remaining saddle-connected components as separate open legs. Reuse
`traced_spline_contour` for segment geometry and saddle pairing, and reuse the
padding/overflow conventions of `TracedClippedSupports`. Do not infer the
closed boundary by longest-polyline selection: that can select a long open leg.

The missing tests are correspondingly explicit:

- a manufactured diverted global C2 field with one closed lobe and two open
  legs, asserting saddle-exact endpoints and branch identity;
- a limited field, asserting one closed branch and zero legs;
- a double-null field, asserting deterministic fixed slots and no leg merge;
- JIT/eager/vmap parity, exact-zero inactive padding, and fail-closed capacity;
- closed-branch-only distance metrics that are unchanged when leg length is
  extended outside the X-point.

## EFIT-baseline comparison harnesses

| Repository | Module and symbols | Existing test evidence | Fitness verdict |
|---|---|---|---|
| Nova | `nova/imas/mast_efit_referee.py`: `read_efit_referee`, `compare_reference_geometry`, `_boundary_distance`, `ReferenceGeometryScores`, `score_with_efit_referee` | `tests/test_efit_referee.py::test_reference_geometry_replaces_every_nan_scorecard_field`; `::test_absent_x_point_is_scored_as_an_explicit_topology_failure`; frozen-shot catalogue test | **Reuse MAST data/time alignment and fail-closed class scoring.** Replace its single symmetric-mean LCFS reduction for this demonstration with shared closed-boundary sup and RMS metrics. |
| Nova | `docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py`: `_boundary_distances`, `_binding_contour`, `measure` | Banked JSON/PNG cover twelve frozen MAST arms; no focused pytest for `_boundary_distances` was found | **Reuse the closest existing MAST demonstration harness.** It already reports symmetric boundary sup/RMS, achieved-class agreement, and selected-saddle-to-nearest-EFIT-X distance; switch its input from grid contour lines to the new closed spline branch. |
| Nova | `benchmarks/efit_topology_boundary_score.py`: `_boundary_scores`, `_x_point_scores`, `_score_shot`, `build_report` | `tests/test_bound_classification.py` and `tests/test_efit_parity_criterion_provenance.py` pin registered bounds/provenance; no direct unit test of all geometry helpers was found | **Qualified MAST stored-map referee.** It supplies signed/symmetric distances, unordered X matching, per-shot class agreement, and a frozen cohort, but uses ray-sampled connectivity geometry and mean rather than RMS. |
| Nova | `benchmarks/diiid_forward_gs_match.py`: `_separatrix`, `contour_separation`, `solve_frame`, `MatchMetrics`, `summarize` | `tests/imas/test_diiid_forward_gs_match.py::test_contour_separation_is_symmetric_and_reported_in_millimetres`; `::test_summary_keeps_nonconvergence_visible_and_fail_closed`; preregistration tests | **Reuse DIII-D selection, solve receipts, preregistration, and label loading.** Replace `_separatrix`'s longest `contourpy` line and KD-tree mean/max metric with the shared spline branch and closed-boundary sup/RMS; preserve nonconvergence as failure. |
| Nova | `benchmarks/diiid_state_of_play_figures.py`: `_topology_overlay`, `_boundary_separation`, `boundary_gradient_minimum`, `TopologyOverlay` | `tests/imas/test_diiid_state_of_play_figures.py::test_boundary_gradient_minimum_uses_shipped_curve_and_map`; `::test_symmetric_boundary_separation_is_in_physical_metres`; receipt provenance test | **Reuse as a DIII-D visual referee only.** It has clear labelled-versus-extracted overlays and source receipts, but its X marker is the minimum gradient on the labelled polygon and its metric is symmetric mean only. |
| imas-ambix | `imas_ambix/eval/efit_referee.py`: `evaluator_context`, `read_efit_geometry`, `judge_geometry`, `GeometryVerdict` | `tests/eval/test_efit_referee.py::test_judge_offset_gives_expected_error`; `::test_judge_masked_reference_component_is_unavailable_not_zero`; firewall tests | **Reuse the evaluator-only firewall and missing-data semantics.** Its eight-angle `boundary_rms` and single primary-X metric are too sparse for the final boundary gate and it has no topology-class score. |
| imas-ambix | `imas_ambix/worldmodel/equilibrium_labels.py`: `load_equilibrium_geometry`, `resample_lcfs_radii`, `xpoint_null_set`, `EquilibriumGeometry` | `tests/worldmodel/test_equilibrium_labels.py::test_resample_lcfs_radii_circle_is_constant`; `::test_xpoint_null_set_is_invariant_as_an_unordered_set`; geometry masking tests | **Qualified adapter reuse.** It provides stable EFIT label loading, masking, and unordered null slots for Ambix evaluation, but radius compression loses the dense polyline needed for sup-distance scoring. |

### Comparison decision

Use one metric function for both machine banks, evaluated on the **closed branch
only**:

- topology class agreement from the achieved saddle-aware read, never a
  requested or legacy flag;
- symmetric Hausdorff/sup distance in metres;
- symmetric RMS of the two directed nearest-polyline distance populations in
  metres;
- nearest polished Nova saddle to the finite unordered EFIT X-point set in
  metres;
- explicit unavailable/failure causes when a boundary, class, or X-point is
  missing, with nonconverged solves retained as failed rows.

For MAST, retain the frozen twelve-arm selection and the existing EFIT `efm`
reader. For DIII-D, retain the preregistered frame/cohort selection and corpus
convention conversion from `diiid_forward_gs_match.py`. Replace only the
geometric extractor and metric implementation. EFIT remains an independent
magnetics-fitted reconstruction, not truth and not a solve input.

## Implementation handoff

The recommended dependency order is:

1. factor the global-spline stationary polish and test it against analytic
   saddles plus the current local-fit candidates;
2. assemble closed/open branches over `traced_spline_contour` outputs and pin
   fixed-shape/capacity behavior;
3. wire the closed branch into the MAST and DIII-D comparison harnesses with one
   shared sup/RMS/X metric implementation;
4. retain Ambix's host lobe clip and firewalled judge as independent behavioral
   referees, not production authorities.

This composition reuses the sanctioned global surface, the current candidate
census, both existing saddle-cell mechanisms, and both EFIT banks. It adds no
second spline representation and does not inherit any grid-resolution contour
as final geometry authority.
