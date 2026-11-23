# Equilibrium forward-model guidance

This guidance applies when constructing equilibrium machines, running forward
maps, or measuring their fixed points. Repository-wide development and git
rules remain in the root `AGENTS.md`.

## Run a forward solve through the request-receipt seam

The only supported way to run a forward solve with production defaults and a
receipt is a typed request through the public seam. A bare call to a private
fixed-point kernel, or a solver-control keyword threaded past the seam, is a
routing-around: it resolves outside the declared defaults, so its verdict is
not comparable with any production receipt.

- Build a `ForwardSolveRequest` and pass it to `ForwardProfile.solve`; the
  return is a `ForwardSolveReceipt` carrying the terminal state, its
  qualification and termination reason, the per-trip residual and mask
  histories, the compilation-cache hit, wall seconds, and the resolved
  defaults that actually ran. The request, receipt, policy, seed, and
  resolved-defaults types — `ForwardSolveRequest`, `ForwardSolveReceipt`,
  `ForwardSolvePolicy`, `ExplicitSolveSeed`, `ResolvedForwardSolveDefaults` —
  live in `nova/equilibrium/solve_request.py` and are re-exported under
  `nova.equilibrium`. Passing a request together with any solve keyword is
  rejected, so the defaults cannot be half-overridden through the seam.
- The declared-defaults table is `FORWARD_SOLVE_DEFAULTS` in the same module,
  keyed by the installed Nova package version. Read it through
  `declared_forward_solve_policy()` and build a request from those defaults
  with `ForwardSolveRequest.from_defaults(...)`. A deliberate deviation (a
  bank row pinned to a different budget, a stricter qualification floor) is
  expressed through `policy_overrides` or `resolve_forward_solve_policy(...)`
  and is recorded on the receipt under `resolved_defaults.deviations` — never
  passed to the seam as a raw solver keyword.
- Production routes and benchmarks construct one typed request per solve,
  leave every default that is not a deliberate deviation at its declared
  value, and write the resolved-defaults block into any JSON they emit. After
  touching a route, run the default-wiring tests:
  `tests/test_route_default_wiring.py` (each production route receipt resolves
  every declared default and no production route calls a private solver
  kernel) and `tests/test_default_wiring.py` (every production entry point
  resolves the defaults on and its launcher leaves them to the public seam).

## The measured forward and inverse routes

Two production routes are measured, not read. `scripts/prototypes/hot_path_map/job.sh`
drives each in a fresh process under `coverage` on CPU, in one `all_debug` job,
and `map.py` composes a per-module line map. The map this section describes was
measured at revision 8d8b526ae1d315945ed603b8f15477382559cb27 (job 1339132) and is stored with its line numbers
in that job's output; the comparison basis this section is checked against is
`docs/evidence/fragments/code-reuse-and-hot-path/hot-path-written-down/route-modules.json`,
the stripped copy of the same map; the measurement's own record is
`docs/evidence/fragments/code-reuse-and-hot-path/hot-path-written-down.html`.
The earlier line-map record is
`docs/evidence/fragments/code-reuse-and-hot-path/hot-path-map.html`; its map was
measured at revision 09ef90e4 and the tree has moved since, so read the record
for how the route was measured rather than for the module lists.

The two routes:

- **Forward** — `nova.equilibrium.forward.ForwardProfile.solve` driven by a
  `ForwardSolveRequest` on a stored MAST case
  (`scripts/prototypes/hot_path_map/forward_route.py`). This is the one public
  forward route; every other forward solve, the fit's included, enters the map
  through it.
- **Inverse** — `tests/test_equilibrium_magnetics_fit_known_states.py::test_known_state_recovery[mast]`,
  which fits flux-function coefficients inside the equilibrium iteration
  (`MagneticsFit` in `nova/equilibrium/magnetics_fit.py`) and calls the same
  public entry on every proposal it evaluates.

At this revision the forward route executed 182 modules and the inverse route
210. The forward set is a strict subset of the inverse set: 182 modules
are executed by both, none by the forward route alone, and 28 only by
the inverse route, so every module the public forward solve executes is also
executed by the fit inside the iteration. A module is listed when the route
executed at least one of its lines, and that includes a line executed when the
module was imported, so a module carrying single-digit lines was reached
through an import chain rather than by the solve's own calls. The full lists are
the three blocks at the end of this section; the check below holds them to the
measurement in both directions, so a module cannot be added to a list without
being measured on the route it is filed under.

`nova/equilibrium/forward.py` — the entry both routes reach — reads
504/1043 on the forward route and 506/1043 on the inverse route. The
modules the forward route spends its execution on are
`fixed_point.py` (1167 lines), `forward_operator.py` (999),
`connectivity_boundary.py` (617), `stencil_nulls.py` (594), `topology.py`
(518), `forward.py` (504 of 1043), `flux_surface_connectivity.py` (440) and
`constraint.py` (322), with the Biot-Savart kernels beneath them in
`nova/biot/`.

### The forward route, phase by phase

1. **Seam** — `nova/equilibrium/solve_request.py` builds the typed request,
   resolves the declared defaults and assembles the receipt.
2. **Seed and source** — `cold_seed.py`, `source.py` and `profile.py` produce
   the initial flux and the declared profile; `stencil_mesh.py`,
   `cell_partition.py`, `domain.py` and `wall_mask.py` lay out the cells.
3. **Operator** — `forward_operator.py` assembles the map from the Biot-Savart
   operands: `nova/biot/greens.py`, `polygonanalytic.py`, `elliptic.py`,
   `completeelliptic.py` and the per-geometry kernels (`arc.py`, `bow.py`,
   `polybow.py`, `polygon.py`, `polysection.py`, `cylinder.py`, `line.py`,
   `beam.py`, `target.py` and their siblings), over the containers in
   `nova/frame/`, `nova/biot/biotframe.py` and `nova/biot/matrix.py`.
4. **Solve** — `solve.py` factors the Jacobian and runs one correction;
   `fixed_point.py` supplies `newton_krylov`, `anderson` and `picard`, and
   `reduced_newton.py` the reduced routes and their active sets.
5. **Topology and qualification** — `topology.py` reads the null census on the
   flux map; `stencil_nulls.py` and `connectivity_boundary.py` classify the
   cells around it; `fixed_points.py`, `flux_surface_connectivity.py`,
   `separatrix_clip.py`, `separatrix_branches.py`, `clip_quadrature.py` and
   `branch_selection.py` trace and qualify the separatrix.
6. **Observation and moments** — `observation.py`, `observation_kernels.py`,
   `moment.py`, `map_extraction.py`, `measurement.py`, `conservation.py`,
   `current.py`, `conductor_current.py` and `constraint.py`.
7. **Machine, corrections and evidence** — the MAST machine's own chain
   (`nova/imas/mast_*.py`, `machine_*.py`, `ids_*.py`, `dataset.py`,
   `database.py`), which reaches `nova/transport/` through
   `nova/imas/mast_chain_factory.py` and the measurement corrections in
   `nova/calibrate/` through `nova/imas/mast_corrections.py` and
   `mast_block_scale.py`; plotting in `nova/equilibrium/display_flux.py`,
   `nova/media/` and `nova/graphics/`; arrays and splines in `nova/linalg/`,
   `nova/geometry/`, `nova/frame/` and `nova/database/`; and
   `nova/jax/config.py` for the dtype and compilation-cache policy.

### The inverse route

The inverse route runs the public forward entry on every fit proposal, so it
executes every module the forward route does and 28 besides. Those
28 are the fit itself (`nova/equilibrium/magnetics_fit.py`), the frame,
coil and geometry containers its own machine construction reaches
(`nova/frame/coil.py`, `coilset.py`, `polyplot.py`, `vtkplot.py` and their
siblings, `nova/geometry/polygeom.py`), the Biot-Savart entry points it reaches directly
(`nova/biot/biot.py`, `reduce.py`, `loop.py`, `point.py`, `shape.py`,
`space.py`, `crosssection.py`, `flux.py`), its reconstruction store
(`nova/database/netcdf.py`), `nova/control/`, and
`nova/imas/test_utilities.py`. The block below lists them exactly.

### Keeping this section true

`scripts/prototypes/hot_path_map/section_route.py` reads the three blocks below
out of their markers and compares each route's set with the modules the map
measured as executed, naming every difference in both directions — a module
named but not executed on its route, a module executed but not named, a module
named but absent from the tree, and a Python file under `nova/` or `tests/`
changed since the map's recorded revision. The last of these is why the check
cannot rest on the lists alone: a change that makes a route execute another
module leaves the map's measured sets untouched, so a stale map would compare
clean while this section had drifted. When code has moved since the measurement
the remedy is a fresh `job.sh` run, not an edit here.

```
python scripts/prototypes/hot_path_map/section_route.py \
  --section nova/equilibrium/AGENTS.md \
  --map docs/evidence/fragments/code-reuse-and-hot-path/hot-path-written-down/route-modules.json \
  --root .
```

Its `--add BLOCK=MODULE` files one extra module into a block of a temporary copy
and shows the check reporting it, which is the control for a comparison that
would otherwise only ever pass. Re-measure with `job.sh` before changing a list,
and run the check in the same commit that changes one.

### Pending collapses from the duplicate register

`docs/evidence/fragments/code-reuse-and-hot-path/hot-path-duplicates.html`
proposes ten collapses on or beside this route. All ten are raised as reckon
decisions (`crew view decisions --project nova`) and every one is still
`ready`, so no collapse is accepted-and-pending: when an answer accepts one, the
module a route stops executing moves between the blocks below in the same commit
as the code change. The ten are: topology-read (`read_at_stationary_points`
removed in favour of the census), extreme-selection (`stencil_nulls`'s three
`_arg_extrem*` wrappers), storage-group-mode (`netCDF` and `ZarrStore` `subgroup`/`get_mode`), level-set-polish (`separatrix_clip`'s inline polish loop),
plot-columns (`PolyPlot.update_columns` against `VtkPlot.update_columns`),
route-resolution (which topology read each route executes), and the four
scan-flagged pairs `_tensor_bernstein`, `Aphi`, `_first_residual` and
`for_member`.

#### Modules both routes execute

<!-- hot-route:shared -->
**biot/** `nova/biot/__init__.py` `nova/biot/arc.py` `nova/biot/arcamplitude.py`
  `nova/biot/arcbandedcoupling.py` `nova/biot/array.py` `nova/biot/beam.py`
  `nova/biot/biotframe.py` `nova/biot/bow.py` `nova/biot/circle.py`
  `nova/biot/completeelliptic.py` `nova/biot/constants.py` `nova/biot/cylinder.py`
  `nova/biot/data.py` `nova/biot/elliptic.py` `nova/biot/error.py`
  `nova/biot/fieldnull.py` `nova/biot/gradedresidual.py` `nova/biot/greens.py`
  `nova/biot/grid.py` `nova/biot/groupset.py` `nova/biot/incompleteelliptic.py`
  `nova/biot/incompletemoments.py` `nova/biot/line.py` `nova/biot/matrix.py`
  `nova/biot/momentchannel.py` `nova/biot/null.py` `nova/biot/operate.py`
  `nova/biot/pairedfloat.py` `nova/biot/plasmagrid.py` `nova/biot/polybeam.py`
  `nova/biot/polybow.py` `nova/biot/polygon.py` `nova/biot/polygonanalytic.py`
  `nova/biot/polygonarc.py` `nova/biot/polysection.py` `nova/biot/rangefunction.py`
  `nova/biot/sectionaverage.py` `nova/biot/sheet.py` `nova/biot/solve.py`
  `nova/biot/symmetricelliptic.py` `nova/biot/target.py` `nova/biot/tiledassembly.py`
  `nova/biot/zeta.py`
**calibrate/** `nova/calibrate/__init__.py` `nova/calibrate/correction_model.py`
  `nova/calibrate/correction_set.py` `nova/calibrate/corrections.py`
  `nova/calibrate/scale_step.py`
**catalog/** `nova/catalog/__init__.py` `nova/catalog/mast_geometry.py`
**database/** `nova/database/content_store.py` `nova/database/datafile.py`
  `nova/database/filepath.py` `nova/database/zarrstore.py`
**equilibrium/** `nova/equilibrium/__init__.py`
  `nova/equilibrium/boundary_comparison.py` `nova/equilibrium/branch_selection.py`
  `nova/equilibrium/cell_partition.py` `nova/equilibrium/clip_quadrature.py`
  `nova/equilibrium/cold_seed.py` `nova/equilibrium/conductor_current.py`
  `nova/equilibrium/connectivity_boundary.py` `nova/equilibrium/conservation.py`
  `nova/equilibrium/constraint.py` `nova/equilibrium/convention.py`
  `nova/equilibrium/current.py` `nova/equilibrium/diagnostics.py`
  `nova/equilibrium/display_flux.py` `nova/equilibrium/domain.py`
  `nova/equilibrium/fixed_point.py` `nova/equilibrium/fixed_points.py`
  `nova/equilibrium/flux_surface_connectivity.py`
  `nova/equilibrium/flux_surface_extraction.py`
  `nova/equilibrium/flux_surface_geometry.py` `nova/equilibrium/forward.py`
  `nova/equilibrium/forward_operator.py` `nova/equilibrium/gs.py`
  `nova/equilibrium/labels.py` `nova/equilibrium/manifold_advance.py`
  `nova/equilibrium/map_extraction.py` `nova/equilibrium/measurement.py`
  `nova/equilibrium/moment.py` `nova/equilibrium/morphology.py`
  `nova/equilibrium/observation.py` `nova/equilibrium/observation_kernels.py`
  `nova/equilibrium/parallel_components.py` `nova/equilibrium/profile.py`
  `nova/equilibrium/reduced_newton.py` `nova/equilibrium/separatrix_branches.py`
  `nova/equilibrium/separatrix_clip.py` `nova/equilibrium/solve.py`
  `nova/equilibrium/solve_request.py` `nova/equilibrium/source.py`
  `nova/equilibrium/stencil_mesh.py` `nova/equilibrium/stencil_nulls.py`
  `nova/equilibrium/support_read.py` `nova/equilibrium/topology.py`
  `nova/equilibrium/wall_mask.py`
**frame/** `nova/frame/__init__.py` `nova/frame/arraylocindexer.py`
  `nova/frame/columnar.py` `nova/frame/dataarray.py` `nova/frame/dataframe.py`
  `nova/frame/error.py` `nova/frame/framedata.py` `nova/frame/framelink.py`
  `nova/frame/framesetloc.py` `nova/frame/framespace.py` `nova/frame/indexer.py`
  `nova/frame/metadata.py` `nova/frame/metaframe.py` `nova/frame/metamethod.py`
  `nova/frame/plasmaloc.py` `nova/frame/subspace.py`
**geometry/** `nova/geometry/frenet.py` `nova/geometry/geoframe.py`
  `nova/geometry/hexstencil.py` `nova/geometry/pointloop.py`
  `nova/geometry/polyframe.py` `nova/geometry/polygen.py` `nova/geometry/polygon.py`
  `nova/geometry/polyshape.py` `nova/geometry/rotate.py` `nova/geometry/section.py`
  `nova/geometry/select.py`
**graphics/** `nova/graphics/line.py` `nova/graphics/plot.py`
**imas/** `nova/imas/__init__.py` `nova/imas/database.py` `nova/imas/dataset.py`
  `nova/imas/diiid_current.py` `nova/imas/diiid_description.py`
  `nova/imas/edge_consistency.py` `nova/imas/ids_entry.py` `nova/imas/ids_index.py`
  `nova/imas/machine_artifact.py` `nova/imas/machine_drive.py`
  `nova/imas/machine_evidence.py` `nova/imas/mast_block_scale.py`
  `nova/imas/mast_chain_factory.py` `nova/imas/mast_corrections.py`
  `nova/imas/mast_parity_chain.py` `nova/imas/mast_passive_response.py`
  `nova/imas/mast_seed_parameters.py` `nova/imas/mast_solve_input_ids.py`
  `nova/imas/mast_solve_inputs.py` `nova/imas/mast_vacuum_cohort.py`
  `nova/imas/mast_vacuum_response.py` `nova/imas/parity_tolerances.py`
**io/** `nova/io/__init__.py` `nova/io/egress.py` `nova/io/ingest.py` `nova/io/json.py`
  `nova/io/signalstore.py` `nova/io/sourcemap.py` `nova/io/standardname.py`
  `nova/io/streaming.py`
**jax/** `nova/jax/config.py` `nova/jax/tree_util.py`
**linalg/** `nova/linalg/interpolant.py` `nova/linalg/split_spline.py`
  `nova/linalg/tensor_spline.py`
**media/** `nova/media/__init__.py` `nova/media/ink.py` `nova/media/poloidal.py`
  `nova/media/sources/__init__.py` `nova/media/sources/frame.py`
  `nova/media/sources/mast_thomson.py` `nova/media/sources/nova_labels.py`
  `nova/media/sources/plasma_mesh.py`
**nova/ (top level)** `nova/__init__.py` `nova/definitions.py`
**transport/** `nova/transport/__init__.py` `nova/transport/coupled_window.py`
  `nova/transport/current_diffusion.py` `nova/transport/ensemble.py`
  `nova/transport/evolved_state.py` `nova/transport/forward.py`
  `nova/transport/torax_geometry.py` `nova/transport/window_batch.py`
**utilities/** `nova/utilities/axes.py` `nova/utilities/importmanager.py`
<!-- /hot-route:shared -->

#### Modules only the forward route executes

None at this revision, and the check below enforces the block being empty: a
module moved here that the forward route does not execute is detected as another
route's module, and a module moved here that the inverse route also executes is
reported as executed but not named.

<!-- hot-route:forward -->
<!-- /hot-route:forward -->

#### Modules only the inverse route executes

<!-- hot-route:inverse -->
**biot/** `nova/biot/biot.py` `nova/biot/crosssection.py` `nova/biot/flux.py`
  `nova/biot/loop.py` `nova/biot/point.py` `nova/biot/reduce.py` `nova/biot/shape.py`
  `nova/biot/space.py`
**control/** `nova/control/__init__.py` `nova/control/control.py`
**database/** `nova/database/netcdf.py`
**equilibrium/** `nova/equilibrium/magnetics_fit.py`
**frame/** `nova/frame/coil.py` `nova/frame/coilset.py` `nova/frame/coilsetattrs.py`
  `nova/frame/energize.py` `nova/frame/frame.py` `nova/frame/frameset.py`
  `nova/frame/multipoint.py` `nova/frame/poloidalgrid.py` `nova/frame/polygeo.py`
  `nova/frame/polygrid.py` `nova/frame/polyplot.py` `nova/frame/select.py`
  `nova/frame/vtkgeo.py` `nova/frame/vtkplot.py`
**geometry/** `nova/geometry/polygeom.py`
**imas/** `nova/imas/test_utilities.py`
<!-- /hot-route:inverse -->

## Fit coefficients inside the iteration

`FluxIterationFit` in `flux_iteration.py` solves the least-squares fit on every
pass of the map iteration: each pass steps one coefficient at a time at the
held map, solves the whitened system, and applies one Grad-Shafranov
read-write cycle at the coefficients and fitted currents that solve just chose.
No forward solve is converged for a proposal the data did not choose, which is
the cheaper form beside the nested class. Three properties are load-bearing.

- **The map belongs to the coefficients that pass just fitted.** The operator
  applying the update is rebuilt from the corrected coefficient vector; reusing
  the loop head's operator lays down the pre-fit map, and the pass then reports
  an update of 2e-16 where the map sits 5e-3 of the fix's own map away.
  `tests/test_flux_iteration.py` builds the expected map independently and
  asserts the reviewer's probe, [1.0, -0.4] from [1.05, -0.42].
- **The fitted conductor currents reach the map as the operator's `current`.**
  The response columns use `conductor_response`; the update passes the solved
  fitted currents as `current`, so the rows and the laid-down map carry one
  conductor state. An operator left to its stored `current` measures one state
  and lays down another.
- **The pass is a fixed-point step, and its stability is a property of the
  state.** Measured on the three known states: MAST 22086 and ITER 135013 reach both
  known-answer bands in one pass (1.3e-15 and 1.5e-15 of the span); Solov'ev,
  started from the largest perturbation the recovery record uses, diverges at
  full step and is not recovered by damping at 0.5, 0.25 or 0.1: each arm
  improves to a residual near 0.14 and then leaves the basin, the half-step at
  pass 12, the quarter-step at pass 23, the tenth-step at pass 56, reading 0.98,
  1.01 and 0.41 at the ends of their budgets. Read `relaxation` before reading a
  failure as a fit defect.
- **A pass can be refused, and the receipt says so.** An update beyond
  `max_update` spans after the first pass keeps the previous map and records
  `rejected_pass` with its message. Read `flux_map_error_over_span` and
  `whitened_residual_sup` beside `converged`: the flag means the update reached
  `flux_tolerance` of the axis-to-boundary span, so a state inside a band
  while the flag reads false is a correct state whose final update was refused,
  not a recovered one. A case that misses its band is reported as failing.

## Keep the two reference lanes separate

### Closed-form analytic oracle

- The analytic lane is `scripts/analytic_oracle_fixtures/`. It selects the
  `moderate-rotation-conventional` member from
  `tests/rotating_equilibrium_references.py` and expresses its rotating source
  with `nova.equilibrium.rotation`.
- Its flux, profiles, current density, axis, separatrix, and exterior boundary
  supply are evaluated from that closed form. It has no stored map, IMAS read,
  or archive dependency.
- The exterior is the independently evaluated analytic total field minus the
  exact analytic-density plasma image on the same authored supports. This is
  what makes the closed-form state an accuracy oracle for the composed map.

### Stored DINA reproduction

- The stored-reference lane is
  `tests/test_equilibrium_forward_reference.py`. It reads DINA pulse/run
  `135011/7`, data-dictionary version `3.39.0`, time-slice index `353`, including
  the stored map, profiles, machine geometry, driven coils, and passive loops.
- This lane measures Nova's reproduction of that stored case. It is not an
  analytic oracle: the case's declared boundary constant and stored field are
  not assumed to be mutually exact.
- Never import an anchor, state, profile constant, cache identity, or acceptance
  floor from one lane into the other. Name every result and cache by its actual
  provenance.

## Build fixture machines through their semantic caches

- Standalone consumers request the lane's `cached_machine`; they do not call a
  direct machine builder. Only the cache-owned, advisory-locked miss path may
  construct geometry or evaluate a Green operator.
- The stored-reference identity includes its pulse/run/time locator, all
  reference scalars and array-content hashes, conductor geometry and currents,
  requested discretisation, precision, route policies, and cache schema. The
  analytic identity carries the closed-form constants, discretisation, wall
  sampling, precision, routes, and its distinct analytic schema. A changed
  semantic input must select a changed key.
- Store and restore native dtypes, shapes, and bytes. A warm hit is accepted
  only after descriptor, schema, payload inventory, digest, and semantic key
  validate. Publication and recovery of an interrupted group stay under the
  same lock.
- The measured stored-reference requests are expensive enough to make bypasses
  defects: coarse is `1,246.880145 s` cold and `0.265939 s` warm; fine is
  `3,335.325393 s` cold and `0.817662 s` warm. All `31` persisted arrays matched
  bitwise in both measurements. Expect warm loads for repeated work.
- A worktree consumes the shared root environment read-only:
  `UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync`.
  Never run `uv sync`, `uv venv`, or `pip` from a fixture worker. A missing
  capability is an environment-owner blocker, not permission to provision the
  shared environment.

## Preserve the serialized root banks

- Stored DINA reproduction roots are
  `scripts/root_gate_attribution/coarse-terminal-root.npz` and
  `scripts/root_gate_attribution/fine-terminal-root.npz`.
- Closed-form oracle roots are `scripts/oracle_rebaseline/root-coarse.npz` and
  `scripts/oracle_rebaseline/root-fine.npz`; their receipts retain the oracle,
  seed, terminal state, residual trajectory, locally normalized flux, topology,
  and current observations.
- Treat these files as banked evidence. Load and validate them before reuse; do
  not overwrite a bank while changing a fixture, solver budget, seed, or gauge.
  Bank a separately named measurement when the semantic experiment changes.

## Keep flux gauges and normalized anchors local

- The Biot-Savart composition sets Nova's poloidal-flux gauge. Do not re-zero a
  Nova state at a saddle or combine a Nova-gauge amplitude with a stored
  reference constant.
- `psi_norm` is gauge-safe only when its axis and boundary anchors are read from
  the same field being normalized. Normalize an oracle field with oracle-field
  anchors and a solved field with solved-field anchors.
- Compare raw flux amplitudes only when the states share a declared gauge, such
  as the same exact exterior supply. Record the gauge and anchor provenance in
  every receipt that reports a flux deviation.
- Pin `JAX_PLATFORMS=cpu` for every CPU-lane build, solve, test, or benchmark.
  Login and compute hosts may expose CUDA, and an implicit backend change is a
  different numerical measurement.

## Forward compilation cache root

`default_forward_compilation_cache_root` in
`nova/equilibrium/solve_request.py` selects where forward solves keep
compiled JAX programs.  Its default is
`~/.cache/nova/forward-compilation-cache/user-<uid>/host-<hostname>`
on the shared home filesystem, and a launch that must name its own location
sets `NOVA_FORWARD_COMPILATION_CACHE_ROOT` to that base directory.  The root
is deliberately **not** TMPDIR-derived: every SLURM step on this cluster
lands TMPDIR on the node-local filesystem, so a TMPDIR-derived root dies with
the allocation and no job banks a compile for its successor.  The base sits
under `~/.cache`, **not** under `~/.local/share/nova`: the per-user data tree
inherits a setgid group down the hierarchy whose name the cluster's name
service cannot resolve, and JAX's cache runtime calls `grp.getgrgid` on every
cached file while evicting, so a root inside that tree writes at most one
entry and then aborts every later write.  `~/.cache` is not setgid, so files
beneath it carry the user's own resolvable group.

Isolation and keying argument, kept here because it governs every consumer:

- **Host key** — `host-<hostname>` keeps each node's build inside its own
  subtree: a cache written by one node is never read by another, and
  allocations on different nodes never write the same file.  Reuse is
  therefore per node, which is how the reserved single-GPU lane pays: it
  always lands on the same node, so successive jobs hit the prior job's
  compiles.  A debug node that varies per allocation still cold-compiles,
  which is expected rather than a defect.
- **Concurrent writers, same node** — two allocations on one node share one
  subtree and are serialised by JAX's own cache entry lock (`.lockfile`).
  JAX writes an entry as a plain open+write under that lock, so neither the
  layout nor the writer depends on the atomic no-clobber create that GPFS
  does not provide.
- **Architecture key** — carried below the host key by the persistent cache
  configuration, which appends `nova/jax-compilation/runtime-<hash>` where
  the hash covers the jax/jaxlib versions, backend platform and version, the
  x64 flag, and the device topology.  A build for one runtime is therefore in
  a different directory than any other runtime and is never read by it.

Measurements of the root choice (node-local `/tmp` premise, cold-versus-warm
compile wall across two allocations on the same host) live under
`docs/figures/playable-forward-solve/compilation-cache/`.

## Keep compiled solve programs proportional to one map

Treat the optimized HLO and executable size as API constraints alongside the
temporary-memory analysis.  The measured 300- and 1000-cell whole solves held
654,832 and 662,331 optimized instructions against 9,908 and 10,024 for one
map application: 66.1 times the map at both cell counts.  Their source
sentinels found 120 traced current-moment paths and 8 topology reads in the
solve against one of each in the map.  Run
`benchmarks/solve_program_size_gate.py` beside the memory gate after changing a
compiled solve boundary; its map control must see the known-present sentinels
before an absent solve count can be believed.

- A trip, Newton budget, or other fixed-capacity iteration is a
  `jax.lax.scan`, `jax.lax.fori_loop`, or `jax.lax.while_loop` with fixed-shape
  telemetry slots.  Never express it as a Python loop inside a traced program.
  The measured production Newton and active-set budgets were already
  `fori_loop` bodies, so rewriting their host-only Python counterparts cannot
  reduce compiled size.  The rule is scan, never unroll; verify the optimized
  HLO rather than inferring the form from source syntax.
- A per-cell operation is vectorized over cells.  Do not trace a Python loop
  once per cell, candidate, trip, or budget.  Residual, merit, shadow, and
  Jacobian-vector-product paths share one traced operator body and carry their
  varying states through loop operands.
- Mesh-sized and interaction-matrix arrays are explicit program arguments,
  never closure constants.  Capturing the interaction blocks grew generated
  code from 0.44 GiB at 300 cells to 3.5 GiB at 1000 and 19 GiB at 2500 while
  the pre-optimization instruction count stayed constant.  A cache key may
  describe array shape and policy, but it must not close over the array values.
- **Every Newton update takes its state, operands and factor as runtime
  variables, never as compiled constants.** Use `nova.equilibrium.solve`:
  `factor(residual_fn, state, operands)` builds the dense Jacobian by a
  vmapped `jax.jvp` and LU-factors it, and `fixed_point(residual_fn, state,
  operands, factor=...)` runs cold or chord corrections. One jitted program
  then serves every correction and every state of a size, and it is compiled
  once.
  - Never wrap `jax.linearize`, or a closure over the current state, inside a
    per-correction `jit`. That bakes the linearisation point into the program,
    so every correction compiles a new executable.
  - Measured 2026-10-16 on the 2,098-cell diverted carrier: a 7.4 GB
    executable, about 60 s of compile and 60 to 95 s of apply per correction.
  - Programs above 2 GB also fail to serialise into the persistent compilation
    cache (`RESOURCE_EXHAUSTED` in the job log), so they recompile on every
    run.
  - The runtime-argument form of the same chord batch is 1.36 MB, compiles in
    about 33 s once, and costs 0.28 ms per state at width 128.
  - The guard is `tests/test_equilibrium_solve.py`, which lowers the program
    and asserts that no HLO constant scales with the state. Copy that check
    into any new solver program.
- Flux-function coefficients and normalization scalars enter as traced,
  fixed-shape arrays.  The profile evaluation rule may remain static; binding
  the profile object or its coefficient values into the callable makes a new
  program per slice.  Coil centroids may likewise be traced positions when the
  measured cost is negligible.  Do not differentiate through conductor or
  plasma-cell geometry.
  The analytic representation is `PolynomialFluxFunction`: pressure and
  diamagnetic functions each contribute one ascending-power coefficient vector
  and one scalar physical normalisation, four traced leaves in total.  Their
  vector lengths are static and equal to polynomial order plus one.  A request
  may replace their values on an existing `ForwardProfile` only when evaluator,
  shapes and dtypes match; an incompatible representation is refused before
  lowering.  `SampledFluxFunction` remains the fixed-node, piecewise-linear
  route and carries its coordinate and value tables as traced leaves.  Use a
  local higher-order spline only when its fixed knot and boundary policy are
  themselves part of the source declaration; do not infer one from samples at
  the solve boundary.
- One solve program contains one traced current-moment path and one traced
  topology-read path.  Repeated dynamic evaluations return to those shared
  loop bodies; a second source copy in optimized HLO is a failed size gate even
  when the terminal state and temporary-memory gate pass.

## Build the Biot-Savart operands once per carrier

The dense value and jet matrices that map cell moments to flux are the dominant
setup cost of a forward-solve measurement.
- At 2,098 cells they take 803 s and 286 s to assemble on the H200, and occupy
  1.9 GB and 2.2 GB.
- At 550 cells they take 146 s and 137 s.

Treat them as cached artefacts, not per-process work:

- **Load, do not rebuild.** The shared builders in
  `scripts/prototypes/lean_map` and `scripts/prototypes/read_route/jets.py`
  persist their matrices on GPFS under a content-addressed key: the sample
  nodes or Gauss points, the cell geometry, and the source of the coupling and
  polygon-kernel modules. A changed input or source is a new key, so a stale
  matrix is never served. A caller that builds jets directly passes
  `persist=True` to `build_jet_matrix`.
- **One process per carrier size.** Run every row that needs a size's
  operands, the map check, the cold solve, the warm and batch rows, in one
  process. Separate processes each reassemble or reload the matrices.
- **Never write a partial matrix where a reader can see it.** Build into a
  pending file and rename atomically, so concurrent jobs on one key cannot
  corrupt it.

The production forward route inherits the same rule. Its coupling operands
are explicit program arguments, under the closure-constant rule above, and
come from a persistent store keyed by carrier identity.

## Debug-lane handling for cache-sensitive reduced-state tests

Run a named `tests/test_reduced_newton.py` case in one `all_debug` allocation
when investigating a compilation-cache failure.  The allocation's shell
script exports `TMPDIR=/tmp`, `JAX_PLATFORMS=cpu`, and the worktree on
`PYTHONPATH`, then calls `/home/ITER/mcintos/Code/nova/.venv/bin/python -m
pytest` directly; it never invokes `uv` on the compute node.  Give the script
one test node ID and write its output to a file whose name includes
`$SLURM_JOB_ID`.

Normal measurements leave `NOVA_FORWARD_COMPILATION_CACHE_ROOT`,
`JAX_COMPILATION_CACHE_DIR`, and `JAX_ENABLE_COMPILATION_CACHE` unset.  That
selects the supported per-user, per-host cache under `~/.cache/nova`, where
JAX can resolve cache-file group metadata while it evicts entries.  Do not
point this lane at `/tmp` merely to make its cache private: that changes the
cache lifetime and does not diagnose a program failure.

For a cache-versus-program discriminator, additionally export
`JAX_ENABLE_COMPILATION_CACHE=0` in that one-test allocation.  This disables
the JAX compilation-cache read and write path; unsetting a cache-root override
alone does not disable it because Nova supplies its supported default.  A test
that passes in this mode but aborts with caching enabled is cache-path
evidence, while a failure in both modes belongs to the test or its compiled
program rather than the persistent cache.
