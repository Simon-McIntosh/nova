# Continuous topology margin and DIII-D wall reuse map

Repository snapshot: `5acde3e08b61e1176000953a63b80593e949375a`.

## Executive verdict

The shortest implementation is composition, not a new topology algorithm. Nova
already has a continuous signed topology coordinate,
`class_margin = u_wall - u_xpoint`, whose sign is the hard connectivity
classification, plus a differentiable soft hand-off that publishes
`p_diverted`. Those kernels accept an arbitrary material mask and sampled wall
ring and are already used by the reconstruction profile route. They are not,
however, wired into the `ForwardFluxOperator` seam used by the DIII-D
score-blind solve: that seam still classifies through `Topology.read()` and its
candidate guards consume the Boolean `TopologyState.diverted`.

The DIII-D forward driver has one pseudo-wall constructor. It builds a clockwise
rectangle from the released EFIT-grid bounds, pads each axis by 2% at baseline
and 5% in the one-frame sensitivity arm, evaluates conductor and plasma Green
operators on that rectangle, appends its flux values to every solve state, and
passes an all-true material mask. A separate DIII-D machine-description IDS
already carries the validity-repaired physical limiter ring. The most complete
runtime replacement is therefore: resolve the governed artifact at its pinned
Data Dictionary version, read the wall outline through imas-python, build the
real `inside_limiter` mask and wall targets from that same ring, and expose the
connectivity margin beside the existing Boolean terminal qualification.

Quantitatively, this audit found:

- **3 classification layers**: solver callback, forward-operator topology read,
  and boundary-selector comparison;
- **8 named `remains_diverted` closures** plus one diagnostic candidate observer,
  all reducing to `profile.operator.read(candidate).diverted` where they use real
  topology;
- **6 relevant margin or clearance computations**—two direct normalized-flux
  topology routes, two host-only geometric scaffolds and two evidence/test
  diagnostics—plus one existing production wiring precedent;
- **1 DIII-D pseudo-wall constructor**, used by one coupling builder and one
  seed/profile builder;
- **4 usable wall-ring access routes**, with only the artifact-resolved
  imas-python route carrying the full runtime provenance contract.

No numerical solve was rerun for this inventory. The fitness verdicts below are
about implementation reuse, not proof that the existing continuous margin has
already reproduced the legacy DIII-D classifier on the five held-out frames.

## Candidate-topology classification entry points

| Entry point | What classifies a candidate | Fitness verdict |
| --- | --- | --- |
| `nova/equilibrium/fixed_point.py:1060` — `kink_aware_newton_krylov(..., admissibility_fn=...)` | The nonmonotone ladder maps the caller predicate over every trial at lines 1265-1284 and promotes only finite, admitted candidates inside the residual envelope. | **Direct solver-shell reuse.** Replace the Boolean-only callback input with a margin-aware admission/penalty policy while retaining fixed-shape trial accounting and fail-closed finiteness checks. |
| `nova/equilibrium/fixed_point.py:870` — `newton_krylov(..., admissibility_fn=..., previous_admitted_state=...)` | The manifold route evaluates the caller predicate in relaxed warm-up and again on each predictor/corrector candidate at lines 563-574 and 673-698. | **Direct solver-shell reuse.** It already owns the continuation seam and achieved-advance bookkeeping, but it still receives only a Boolean rather than the continuous margin. |
| `nova/equilibrium/forward_operator.py:373` — `ForwardFluxOperator.read()` | Slices the candidate to physical grid-plus-wall nodes and calls `_fixed_design_topology.read(...)`; DIII-D `remains_diverted` closures use this public seam. | **Direct predicate seam.** Extend the returned topology payload or add a sibling margin read; do not duplicate candidate-state slicing or topology geometry. |
| `nova/equilibrium/topology.py:328` — `Topology.read_with_connectivity()` and `Topology.read()` | Reads axis, X-point and wall extrema; without a requested class it sets `diverted` when the emergent binding flux equals the selected X-point flux. | **Legacy Boolean reference.** Preserve it as the exact comparator for sign-parity tests, but do not derive a differentiable penalty from equality of two selected extrema. |
| `nova/equilibrium/topology.py:215` — `Topology.boundary()` | Selects the X-point saddle or wall extremum by signed flux ordering, with a vertical private-flux-shadow rule that can remove a wall contact from consideration. | **Semantic rule to carry forward.** Any replacement margin must reproduce this reachable-wall/shadow behavior, not compare an X point to every wall vertex indiscriminately. |
| `nova/equilibrium/forward.py:1219` — `ForwardProfile._branch_receipt()` | Re-reads the terminal state without class pinning, compares achieved `topology.diverted` with `requested_class`, and folds consistency into convergence. | **Direct terminal gate reuse.** A smooth intermediate penalty must not weaken this exact terminal requested-class qualification. |
| `nova/equilibrium/branch_selection.py:227` — `select_forward_branch()` with `BranchAdmissibility` | Selects only convergence-qualified, topology-consistent branches allowed by the declared limited/diverted availability. | **Direct portfolio selection reuse, not a step predicate.** Keep it downstream of the continuous solver treatment; it decides among terminal branches rather than grading Newton trials. |
| DIII-D closures at `benchmarks/conditioned_convergence_observables.py:118`, `benchmarks/diiid_admissible_step_control.py:161`, `benchmarks/diiid_manifold_advance.py:117`, `benchmarks/diiid_repaired_solve_remeasure.py:201`, `benchmarks/topology_qualified_mesh_convergence.py:160`, `tests/test_conditioning_shrinkage.py:146`, and `tests/test_factor_ladder_extension.py:121` | Each real closure returns `all(isfinite(candidate)) & profile.operator.read(candidate).diverted`; `tests/test_topology_qualified_admission.py:41` supplies a scalar stand-in for the same callback contract. | **Consolidate at the operator seam.** These repeated local closures are measurement entry points, not independent classifiers; a shared margin callback should eliminate drift while the existing tests retain Boolean regression coverage. |
| `benchmarks/diiid_admission_rate_diagnosis.py:100` — `_candidate_observations()` | Replays candidate map finiteness, `profile.operator.read(candidate).diverted`, and the combined caller verdict without promoting the candidate. | **Direct evidence harness reuse.** Add the signed margin and soft class weight to these non-promoting observations so the five-frame remeasurement can explain how far each refusal lies from the boundary. |

The current score-blind portfolio in
`benchmarks/diiid_forward_gs_match.py:749` is slightly different: it solves a
limited and diverted pair through requested-class-pinned maps, then applies the
terminal branch receipt and `select_forward_branch()`. It does not pass an
`admissibility_fn` during the solve. Reusing a continuous margin there therefore
requires an explicit solver-policy change; merely adding the margin to the
receipt would measure the transition without changing the admitted-advance
count.

## Existing topology margins and material clearances

This table contains every repository hit that actually compares a critical
flux landmark or surface with a wall/material surface, plus the closest
diagnostic implementations whose limitation matters to reuse. Generic numerical
margins, LCFS-to-LCFS reconstruction errors, distances between material units,
and distances to computational-cell edges were screened out.

| Entry point | Quantity and sign | Fitness verdict |
| --- | --- | --- |
| `nova/equilibrium/connectivity_boundary.py:539` — `traced_boundary_read()` | Computes reachable normalized wall-tangency level `u_wall_c` and in-wall X-point saddle level `u_x_c`; lines 613-618 define `class_margin = u_wall_c - u_x_c`, with positive diverted, negative limited and zero marginal. `is_diverted` uses the corresponding `u_x_c <= u_wall_c` comparison. | **Best direct reuse.** This is already the requested continuous topology coordinate and its sign is definitionally paired with the hard connectivity classifier; qualify parity against the older `Topology.read()` before replacing its predicate. |
| `nova/equilibrium/connectivity_boundary.py:717` — `traced_smooth_boundary_read()` and `host_boundary_read_smooth()` | Softmin-blends the same `u_wall_c` and `u_x_c`; returns both operands and the X-candidate softmax weight `p_diverted`. The temperature is in normalized-flux span units and the core boundary uses a sigmoid weight. | **Best differentiable reuse.** Use `u_wall - u_xpoint` as the signed score and `p_diverted` or a monotone penalty as the smooth policy; retain the hard read for exact terminal qualification. |
| `nova/equilibrium/profile.py:413` — `ReconstructProfile._profile_basis()` | Calls `traced_smooth_boundary_read()` with real `inside_limiter`, `wall_r`, `wall_z`, and a declared topology temperature, then uses its smooth core weight in current-profile basis construction. | **Direct wiring precedent, not a forward-map drop-in.** It proves the kernel is usable inside a differentiated production map, but this reconstruction operator has different state packing and current closure from `ForwardProfile`. |
| `nova/geometry/plasmapoints.py:341` — `PlasmaPoints.minimum_gap()` and `point_gap()` | Chooses the nearest sampled first-wall panel midpoint to each separatrix control point and projects point-minus-midpoint onto that panel's normal; the minimum is a signed wall-normal gap. | **Qualified geometric scaffold.** It provides the right physical units and orientation concept, but is host NumPy, uses a legacy global `Wall`, scores selected control points rather than the whole critical surface, and nearest-midpoint normals are not a robust signed distance near corners. |
| `benchmarks/efit_lcfs_outward_offset.py:182` — `_angle_rows()` | Along each fixed LCFS ray, subtracts the stored LCFS polygon radius from the limiter intersection radius, yielding positive outward clearance in metres at the sampled angle. | **Qualified benchmark reuse.** It is the clearest existing material clearance, but only for a star-shaped surface and wall about one centre and only at the fixed angular samples; it is neither JAX-traced nor a global minimum signed distance. |
| `scripts/dual_basin_fixtures/measure.py:68` and `scripts/dual_basin_fixtures/build_diverted_fixture.py:201` | Report boundary-to-wall-contact point distance, boundary-minus-wall flux, and X-point-to-nearest-sampled-wall distance for banked roots. | **Evidence-only reuse.** These diagnostics distinguish a wall-bound limited state from a saddle-bound diverted state, but the distances are unsigned point/vertex approximations and do not define a differentiable surface margin. |
| `benchmarks/limiter_topology_receipts.py:121` — `_synthetic_limited_round_trip()` | Measures Euclidean distance between prescribed and recovered wall-contact points and normalizes it by grid-cell size. | **Test-oracle reuse only.** Keep it to pin the zero/contact case of any new margin; it scores locator accuracy, not signed separation of a free critical surface from material. |

Supporting material geometry is already reusable:

- `nova/equilibrium/wall_mask.py:65` supplies the polygon-inside test;
- `nova/equilibrium/wall_mask.py:242` builds `inside_limiter` from vessel and
  discrete material units;
- `nova/equilibrium/wall_mask.py:337` densifies all wall units for sub-grid
  tangency reads;
- `nova/equilibrium/connectivity_boundary.py:240` localizes the reachable wall
  and X-point binding candidates while preserving axis connectivity.

**No candidate — exact signed geometric distance from an arbitrary critical
flux surface to an arbitrary material ring.** No production symbol computes a
global, oriented LCFS-to-material signed Euclidean distance with stable corner
behavior under JAX. `PlasmaPoints.point_gap()` is a control-point normal
projection, and `_angle_rows()` is a ray-wise clearance; neither can be renamed
into the missing general quantity.

**No candidate — legacy forward-topology margin payload.** `TopologyState`
publishes axis, X-point, wall point, their fluxes and a Boolean `diverted`, but
no signed margin. The continuous connectivity read is a second topology route,
so sign-parity and zero-set agreement must be demonstrated before it becomes
the authority behind the old callback.

Near misses deliberately excluded from the six-candidate count:

- `benchmarks/efit_topology_boundary_score.py:193` computes signed distance
  between an extracted LCFS and a stored LCFS, not between flux and material;
- `nova/equilibrium/wall_mask.py:326` computes a gap between two material-unit
  vertex sets, not a plasma-surface clearance;
- `nova/biot/plasmagap.py:53` builds flux probes along prescribed gap rays but
  does not locate a critical surface or return its distance to the wall;
- `nova/equilibrium/stencil_nulls.py:1128` names an eigenvalue-confidence score
  `class_margin`; it qualifies a fitted null's Hessian class and has no wall
  operand.

## DIII-D pseudo-wall construction and consumption

| Entry point | Current behavior | Fitness verdict |
| --- | --- | --- |
| `benchmarks/diiid_forward_gs_match.py:465` — `pseudo_wall()` | Builds a clockwise, four-sided sampled rectangle. Padding is `expansion * ptp(axis)` independently in R and Z; negative expansion and non-positive inner radius fail. | **Retain only as explicit fallback/control.** It is deterministic and useful for the registered rectangle sweep, but its geometry is unrelated to the DIII-D material surface. |
| `benchmarks/diiid_forward_gs_match.py:612` — `_couplings()` | Keys the geometry cache by physical conductor digest plus expansion, creates the rectangle, and builds source-to-wall and plasma-to-wall Green matrices on its nodes. | **Adapt for a selected wall identity.** Cache identity must include the real ring's governed physical digest or coordinate digest, not an expansion scalar that is meaningless for the physical wall. |
| `benchmarks/diiid_forward_gs_match.py:641` — `build_profile()` | Passes rectangle coordinates as `wall_coordinate`, seeds their flux by spline evaluation, and sets `inside_material` to all true grid nodes. | **Replace as one coherent geometry input.** Real-wall coordinates, real inside-material mask, wall Green rows and seed wall flux must change together; swapping only `wall_coordinate` would mix incompatible operators and state shapes. |
| `benchmarks/diiid_forward_gs_match.py:1263` — `run()` | Runs all five selected frames at expansion 0.02 and only the first frame at 0.05; receipts the pseudo-wall statement and sensitivity. | **Preserve as comparator provenance.** The real-wall arm should be a distinct one-input change and must not be described as another rectangle expansion. |

The physical-wall replacement must also change the material mask. The existing
all-true mask says every grid cell is available to plasma, even if a real wall
cuts through the released rectangular grid. `wall_mask.inside_polygon()` is the
minimum direct supplier for a single closed limiter ring; `build_wall_mask()` is
the more general multi-unit route.

## Machine-description wall-ring accessors

| Entry point | Returned wall data and provenance | Fitness verdict |
| --- | --- | --- |
| `nova/imas/machine_artifact.py:1316` — `resolve_machine_artifact()` plus imas-python `DBEntry(..., dd_version=manifest.dd_version)` and `wall.description_2d[0].limiter.unit[0].outline.{r,z}` | Verifies the content-addressed object, manifest identity and payload before the IDS is opened at its pinned dictionary version. | **Best runtime route.** Use this for a governed DIII-D forward arm and carry artifact digest, physical digest, DD version, occurrence and outline path in the receipt. |
| `nova/imas/diiid_machine_ids.py:639` — `build_diiid_machine_ids()` then `bundle.ids["wall"].description_2d[0].limiter.unit[0].outline` | Rebuilds the DIII-D IDS set from its declared source and calls `bundle.validate()` before returning; validation checks simple-polygon validity, repair digest and vertex count. | **Direct local/source route.** Suitable for construction tests and a controlled local arm; less appropriate than the resolved artifact for a repeatable runtime consumer because it re-enters source authoring. |
| `benchmarks/diiid_vessel_hex_mesh.py:59` — `read_limiter_contour()` | Opens the checked-in netCDF with imas-python and a stated DD version, returns an `N x 2` finite R-Z ring, and validates minimum shape. | **Strong adapter prototype.** Reuse its array conversion and validation, but replace its default path and hard-coded DD pin with the verified artifact manifest. |
| `nova/imas/machine.py:385` and `nova/imas/machine.py:513` — `MachineContour` and `StaticMachineDescription.contour` | Provide immutable pure-Python contour coordinates once an IDS extraction record has been routed through the machine dataclasses. | **Direct downstream type reuse.** Good boundary between IMAS and geometry code, but it does not itself resolve, open or provenance-check the DIII-D artifact. |

Additional audited accessors are evidence-oriented rather than runtime choices:
`DiiidMachineIds.validate()` reads the raw outline at
`nova/imas/diiid_machine_ids.py:158`; `_read_source_description()` converts and
repairs the source outline before authoring at lines 380-416; and
`machine_ids_snapshot()` exports the stable indexed outline leaves at lines
706-740. They are useful validation and receipt suppliers, not alternative wall
authorities.

**No candidate — one public DIII-D helper that resolves the configured artifact
and returns a typed `MachineContour`.** The verified resolver, IDS outline and
pure contour dataclass exist, but the join is currently consumer composition.
The forward arm should either add that narrow adapter in an authorized source
scope or perform the composition locally with all provenance explicit; it must
not read `dataset_machine_description(...).machine.contour`, which is
deliberately `None` for competition-row scope.

## Recommended reuse boundary

1. Keep `Topology.read()` and `_branch_receipt()` as the exact legacy and
   terminal requested-class references.
2. Reuse `traced_boundary_read()` for the signed score and
   `traced_smooth_boundary_read()` for the differentiable penalty. Define the
   score as `u_wall - u_xpoint`; do not substitute physical distance without a
   separately validated geometry definition.
3. Resolve the governed DIII-D machine artifact, read its limiter outline at the
   manifest DD pin, and derive wall nodes, grid material mask, Green rows and
   seed wall flux from that one ring and one identity.
4. Retain `pseudo_wall()` only under an explicit fallback/control selection and
   keep its 0.02/0.05 expansion receipts separate from the physical-wall arm.
5. First demonstrate sign parity and a zero crossing against the legacy Boolean
   on all classified fixtures and the five score-blind frames. Only then route
   the smooth margin into proposal grading, while retaining exact diverted class
   at the terminal state.

This composition leaves two genuinely new pieces: a narrow, provenance-carrying
DIII-D wall adapter if the consumer should not compose it locally, and the
policy mapping the continuous margin to proposal cost/admission. The topology
coordinate, smooth kernel, material-mask machinery, physical ring and terminal
branch gate already exist.
