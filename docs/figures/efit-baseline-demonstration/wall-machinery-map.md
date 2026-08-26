# Wall sampling machinery map

## Quantitative verdict

The MAST reconstruction input supplies **37 limiter coordinate rows**, not 37
independent finite elements.  In the live shot store the first and last rows are
identical, so the polygon has **36 unique vertices and 36 authored straight
segments**.  Those 37 rows are passed unchanged to the current MAST parity
operator and remain fixed while its plasma-grid ladder changes.  The wall read
therefore has 37 target rows at every plasma resolution.

`g_wall` is a useful name for the wall part of the forward map, but it is not a
named array or function in the implementation.  It is assembled from the wall
`FluxTarget` blocks:

```text
authored limiter coordinates W (37 rows, closed duplicate)
              |                         |
              |                         +--> Null1D(W)
              |                              -> sampled wall extremum
              |                              -> 3-node quadratic flux fit
              |                              -> linearly interpolated contact position
              |
              +--> source-to-wall rows Gs(W)
              +--> plasma-to-wall rows Gp(W)
              +--> optional prescribed-circuit rows Gc(W)
                                      |
                                      v
g_wall(psi) = Gs(W) I_source + Gc(W) I_prescribed + Gp(W) J_plasma(psi)
```

For carriers with linear current moments, the last term additionally has the
existing radial- and vertical-moment blocks.  The MAST rectangular-lattice
parity builder sets `use_linear_moments=False`, so its plasma wall image is the
single matrix product `Gp(W) @ J_plasma(psi)`.

An **element-count sweep is feasible now without changing the plasma carrier**:
each rung derives a different fixed wall target cloud from the same authored
polygon and rebuilds only wall-target rows and the wall suffix of the seed.  An
**order sweep is feasible only after a wall-read policy is added**.  There is no
wall-element-order input today: geometry is piecewise linear and the flux
extremum fit is hard-coded quadratic.  This is therefore a **qualified YES**, not
a turnkey two-knob ladder.  Count is an existing construction knob; order needs
a wall-only selector extension, but neither requires changing plasma cells,
plasma-current moments, the source model, or the grid part of the seed.

## Where the MAST wall enters

1. `benchmarks/efit_forward_parity_slice.py:379-384` reads `efm/limiterr` and
   `efm/limiterz` and stacks them into `limiter`.  A direct read of
   `/work/projects/imas_gpu/mast/level1/shots/21978.zarr/efm` gives shape
   `(37,)` for both arrays, **37 coordinate rows, 36 unique rows, and exact
   first/last closure**.
2. `benchmarks/efit_forward_parity_slice.py:385` rasterizes that authored
   polygon once into the fixed plasma-grid `inside_material` mask.  Wall target
   refinement must not rebuild this mask if the goal is to isolate sampling
   resolution.
3. `benchmarks/efit_forward_parity_slice.py:403-408` evaluates both conductor
   and plasma Green responses at the 37 limiter coordinates.  The plasma-grid
   response is independently built at the same, unchanged plasma coordinates.
4. `benchmarks/efit_forward_parity_slice.py:425-449` installs those two wall
   matrices and `Null1D(limiter)` in the `DeclaredAnchorOperator`.
5. `benchmarks/efit_forward_parity_slice.py:455-456` appends wall flux sampled
   from the same frozen reference-map spline to the fixed grid seed.  A rung
   with a different wall count must resample this same spline; it cannot reuse
   the old wall suffix because the state length changes.
6. The passive-inclusive frozen arm stacks grid and wall targets at
   `benchmarks/efit_forward_parity_slice.py:2785-2804`, builds one prescribed
   response matrix over both, and installs it at lines 2818-2827 while zeroing
   the ordinary active-current path.  A wall rung must rebuild the wall rows of
   this matrix too while retaining the identical circuit sections and current
   vector.

The campaign label `lim37` and the stored `n_limiter == 37` describe stored
coordinate count.  Treating the closure row as a 37th physical segment would
be an off-by-one error.  It is also worth retaining an exact **37-row
reproduction control** before deduplicating, because the current operator
really does receive the repeated endpoint.

## How `g_wall` is assembled

| Component | Construction | Shape on a rung with `Nw` wall targets | What changes with wall count |
|---|---|---:|---|
| Active/source wall response | `_source_response()` at `benchmarks/efit_forward_parity_slice.py:300-304`, which calls `loop_response_matrix()`; the latter evaluates the polygon Green kernel directly at every target (`nova/imas/mast_vacuum_response.py:236-277`) | `Nw x Nsource` | Rows and target coordinates only |
| Plasma wall response | `_plasma_response()` at `benchmarks/efit_forward_parity_slice.py:280-288`, one `hybrid_greens()` rectangular-cell response column per unchanged plasma node | `Nw x Nplasma` | Rows and target coordinates only |
| Prescribed fitted-circuit response | `_stored_circuit_fields()` evaluates every stored section with `polygon_greens()` on the stacked grid+wall target cloud (`benchmarks/efit_forward_parity_slice.py:2657-2721`) | `(Ngrid + Nw) x Ncircuit` | Wall suffix only; grid prefix is invariant |
| Wall target | `FluxTarget(source_target, plasma_target, Null1D(W))` (`nova/biot/target.py:544-585`) | `Nw` physical rows | Coordinate array and response-row count |
| External wall image | `ForwardFluxOperator.external()` concatenates `grid.external(I)` and `wall.external(I)` and then adds the prescribed field (`nova/equilibrium/forward_operator.py:438-449`) | `Nw` wall values inside the full state | Recomputed from the rung's wall rows |
| Plasma wall image | `current_moment_image()` concatenates grid and wall internal images (`nova/equilibrium/forward_operator.py:847-852`) | `Nw` wall values inside the full state | Recomputed from the rung's wall rows |
| Total wall flux used by topology | `Topology.split_flux_map()` slices the wall suffix and `read_with_connectivity()` passes it to `Null1D` (`nova/equilibrium/topology.py:303-320,348-379`) | exactly `Nw` | State suffix and wall read change; grid flux does not |

The matrix multiplication itself is explicit in `FluxTarget.external()` and
`FluxTarget.internal()` at `nova/biot/target.py:570-585`.  Thus, for the
zeroth-moment MAST builder,

```text
psi_wall = source_to_wall @ source_current
         + prescribed_response_wall @ prescribed_current
         + plasma_to_wall @ cell_current(psi)
```

The Green kernels are exact point evaluations for their declared source
sections.  They do **not** have a wall quadrature order.  All wall approximation
order enters later through the target coordinates and the extremum selector.

## Current count and order semantics

### Count

There are two current count routes:

- The MAST parity/catalog route passes the 37 stored coordinates directly.  It
  has no `nwall` argument and does no resampling.
- The generic first-wall route uses `PlasmaWall.solve()` and `Sample`.
  `nwall` means **nodes per authored segment**, not a total node count
  (`nova/biot/plasmawall.py:65-76`; `nova/biot/field.py:65-102`).  The regression
  at `tests/test_plasmawall_first_wall.py:31-34` pins two nodes on each of four
  segments as eight wall targets.

For the MAST closed polygon, the generic endpoint-free sampler would realise
`36 * nwall` unique cyclic targets: 36, 72, 144, and 288 for per-segment counts
1, 2, 4, and 8.  A clean ladder can therefore carry:

| Rung | Wall targets | Purpose |
|---|---:|---|
| reproduction control | 37 including repeated closure | Prove the current operator is reproduced exactly |
| unique baseline | 36 | Remove only the duplicate closure while retaining every authored corner |
| count 2 | 72 | One midpoint per authored segment |
| count 4 | 144 | Quarter-segment spacing |
| count 8 | 288 | Eighth-segment spacing |

The count should be recorded as both total unique targets and samples per
authored segment.  A global count alone hides nonuniform physical spacing
because the 36 authored segments have unequal lengths.

### Order

The production read has two distinct approximation orders:

- **geometry order 1:** contact coordinates are interpolated along straight
  chords between sampled wall coordinates;
- **flux-fit order 2:** the maximum sampled wall flux selects a cyclic
  three-node cluster, a quadratic is fitted against cumulative chord length,
  and its stationary point supplies the flux and interpolated coordinate.

This is hard-coded in `nova/geometry/select.py:279-328` and reached through
`Null1D.__call__()` at `nova/biot/null.py:30-40`.  There is no policy object,
polynomial-order parameter, or alternative production selector.  Nor are the
wall points finite elements with shape functions.  Consequently, “order” must
be defined before implementation.  The isolated interpretation is
**wall-flux reconstruction order while geometry remains the same piecewise
linear authored polygon**:

- order 1: choose the best exact wall node, with no off-node extremum polish;
- order 2: retain the current cyclic three-node quadratic flux fit;
- any order above 2: add a fixed-size cyclic stencil and deterministic bounded
  stationary-root selection as a new wall-read policy.

Changing geometry to a spline at the same time would change the physical wall,
not only its sampling order, and would contaminate this ladder.  Geometry-order
experiments should therefore be a separate lane after a curve authority is
defined.

## Isolation contract for the ladder

### Vary per rung

Only these inputs may vary:

1. `wall_coordinate`: target coordinates sampled from the same 36-segment
   authored MAST polygon; retain a separate exact 37-row control.
2. `source_to_wall` and `plasma_to_wall`, plus
   `plasma_to_wall_r`/`plasma_to_wall_z` where the selected fixed carrier uses
   linear current moments, rebuilt at those coordinates.
3. The wall-row suffix of any `PrescribedCurrentField.response`, evaluated from
   the same circuit geometry and current order.
4. `Null1D(wall_coordinate)` and therefore `wall.node_number`.
5. The wall suffix of the initial flux vector, evaluated from the **same frozen
   seed field** at the rung coordinates.
6. The explicitly selected wall-read reconstruction order once that policy
   exists.

### Hold fixed bitwise or by physical identity

- plasma carrier: lattice radii/heights, plasma node coordinates, cell areas,
  cell polygons, stencils, moment geometry, direct support-sample coordinates,
  and every plasma-to-grid/source-to-grid/sample coupling block;
- material geometry: the `inside_material` mask rasterized once from the
  original authored polygon, including any material dilation policy;
- source treatment: `ForwardSource`, profile tables, boundary pressure and
  field-function values, declared/free-anchor mode, declared support and
  anchors, current-normalisation policy, requested topology class, mask-switch
  treatment, and solver route/budgets/tolerances;
- conductor state: machine-geometry registry selection, active/passive section
  geometry, source column order, all fitted currents, and whether ordinary or
  prescribed external response is authoritative;
- seed field: identical plasma-grid prefix and the same continuous seed-field
  generator for wall evaluation; no per-rung warm solve may become the next
  rung's seed;
- arithmetic and compilation policy: dtype, backend, device, and fixed trip
  counts.

Different `Nw` values necessarily produce different static array shapes and
therefore separate JAX compilations.  This does not violate the fixed-shape
requirement: each rung is a separately compiled fixed design.  Padding multiple
counts into one trace is unnecessary and would introduce a mask/selection
variable the isolated study does not need.

## Required per-rung receipts and verdict rule

For each `(wall target count, wall-read order)` pair, record at minimum:

- realised unique target count, per-segment sampling rule, minimum/maximum wall
  spacing, closure-duplicate status, and response-matrix shapes;
- limited-branch boundary flux and its change from the next-finer rung;
- wall-contact arc coordinate and `(R, Z)`, including change from the next-finer
  rung;
- achieved topology class and class margin from the achieved saddle-aware read;
- terminal relative residual and complete residual trajectory;
- fixed-carrier digests for the plasma-grid prefix, source policy, conductor
  currents, and seed-field identity.

The ladder is feasible when the count sweep rebuilds only the listed wall rows,
and the order sweep is implemented only inside `Null1D`'s wall selector.  A run
that changes `inside_material`, plasma cell count/order, plasma coupling blocks,
source normalization, solver treatment, or seed field is a whole-carrier
comparison and must not be credited as isolated wall-resolution evidence.

## Verification performed

- Direct store inspection: MAST shot 21978 `efm/limiterr` and `efm/limiterz`
  are each shape `(37,)`; first equals last exactly; 36 unique coordinate rows.
- Static source trace: every path and line anchor above was re-read in this
  worktree.
- Existing regressions establish the per-segment `nwall` semantics and exact
  wall-flux use: `tests/test_plasmawall_first_wall.py:31-34` and
  `tests/test_connectivity_boundary.py:580-599`.

