# Production forward-solve feature inventory

Scope: production reachability from `ForwardProfile.solve_branch` or
`ForwardProfile.solve_portfolio`, inspected at repository revision
`e34aa3618ebf69e4d6d01a1ab0fece15091879a9`. A definition or import is not
counted as production reachability.

## Quantitative verdict

| Capability | Verdict | Production fact |
|---|---|---|
| Quadratic current representation | **ABSENT** | The solve carries exactly three per-cell current arrays: current, radial first moment, and vertical first moment. Both the support integration and the coupling maps stop at first order; there are zero second-current-moment arrays and zero second-current-moment coupling blocks. |
| Curvature-aware clipping with sagitta correction | **ABSENT** | Separatrix crossings are linearly interpolated on straight atomic-cell edges. The only `sagitta` occurrence conditions a three-point circle fit for a three-dimensional Biot arc and is not called by clipping. |
| Limited-wall representation and refinement | **IN PRODUCTION, but not resolution-matched** | The limited branch reads a wall extremum from discrete wall samples: piecewise-linear wall geometry with a three-sample quadratic fit of flux versus chord length. The canonical global-default build has 121 wall samples against 2,210 plasma cells; the production DINA suite has 165 against 2,214; and the MAST parity/catalog carrier has 37 even at the 95 by 95, 9,025-node reference grid (2,161 stored-LCFS interior nodes). No isolated wall-resolution convergence study exists. |
| Neighbouring-slice warm start | **IN PRODUCTION** | The sequential equilibrium sweep cold-starts only its first portfolio. Every subsequent time sample passes two copies of the previously selected equilibrium flux to `solve_portfolio`; the seed is refreshed after every accepted selection. |

Thus two of four named capabilities are reached by production code, but one of
those two (the wall treatment) does **not** provide the resolution equivalence
being tested. The other two capabilities are absent rather than merely dormant.

## 1. Quadratic current representation — ABSENT

### Production call path

`ForwardProfile.solve_branch`
→ `_branch_receipt`
→ `_solve_accelerated`
→ `ForwardProfile.flux_map`
→ `ForwardFluxOperator.flux_map`
→ `ForwardFluxOperator.internal`
→ `ForwardFluxOperator.cell_current_moments`
→ `_support_partition`
→ `_partitioned_current_moments`
→ `ForwardSource.current_moments`
→ `ForwardFluxOperator.support_current_moments`
→ `InteriorCurrentMomentStencil.support_flux_moments`
→ `fixed_profile_current_moments`
→ `_direct_profile_current_moments`.

`solve_portfolio` reaches the same path by vmapping `_branch_receipt` over the
limited and diverted seeds.

The production type is decisive:

```python
class CellCurrentMoments(NamedTuple):
    cell_current: jax.Array
    radial_moment: jax.Array
    vertical_moment: jax.Array
```

The support rule integrates density and its two first moments and returns only
`current, first`. `support_flux_moments` then stacks exactly three rows. On the
map side, `FluxTarget.internal` contracts exactly the same three channels:
`plasma_target`, `plasma_target_r`, and `plasma_target_z`. The current global
carrier receipt likewise lists the coupling components as `Psi`, `PsiR`, and
`PsiZ` (and the analogous field blocks), with no quadratic current companions.

`MomentGeometry.second_moment` does not contradict this verdict. It is fixed
cell geometry used by `coupling_current_moments` to convert a physical *first*
current moment into coefficients of the three linear basis vectors. It is not a
second moment of the solved current density, is not carried in
`CellCurrentMoments`, and has no corresponding coupling channel.

The quadratic polynomial in `stencil_mesh.py` is also not a quadratic current
representation. It reconstructs the local **flux** sampled by
`profile.current_density`; after quadrature, the returned current information
still stops at zeroth and first moments.

Finally, `nova/biot/momentchannel.py` and
`nova/biot/gradedresidual.py` belong to analytic polygon/arc Biot reductions.
Neither is imported by `forward.py`, `forward_operator.py`, `source.py`,
`stencil_mesh.py`, or `separatrix_clip.py`, and neither adds a current-moment
field to the production solve. Their existence is therefore not reachability
evidence.

**Order comparison:** support = degree 1 in current moments; coupling maps =
degree 1 in current moments. The two sides are aligned, but both are one order
below the requested quadratic representation. Required quadratic channels:
at least the three independent second moments (`RR`, `RZ`, `ZZ`) in addition
to the existing zeroth and two first moments; present: **0 of 3** on each side.

## 2. Curvature-aware cell clipping — ABSENT

### Production call path

`ForwardProfile.solve_branch` / `solve_portfolio`
→ `_branch_receipt`
→ `_solve_accelerated`
→ `ForwardFluxOperator.flux_map`
→ `ForwardFluxOperator.internal`
→ `cell_current_moments`
→ `_support_partition`
→ `MomentGeometry.atomic_mesh.traced_clip`
→ `nova.equilibrium.separatrix_clip._traced_clip`.

The chord assumption enters at the crossing itself:

```python
fraction = jnp.where(crossing_edge, start_flux / denominator, 0.0)
crossing_point = start_point + fraction[..., None] * (end_point - start_point)
```

That is linear interpolation of flux along the straight segment joining the two
atomic nodes. The resulting clipped support is a polygon, and its area and
moments are polygon shoelace moments. No curvature, tangent, arc radius, or
sagitta reaches this path.

The sole sagitta calculation is
`nova.biot.arc.Arc._fit_leverage`: it computes
`2 * sin(sweep / 4) ** 2` to assess whether the three-point circumcircle fit of
a finite three-dimensional current arc is numerically resolvable. It governs
arc-source geometry validation for the Biot element; it neither corrects a
cell boundary nor calls any separatrix clipping function. This is unrelated
arc usage, not implemented-but-unreached cell clipping.

**Clipping order:** first-order straight chord in geometry and linear edge-flux
crossing. The code does compute exact moments of the polygon it constructed,
but exact integration of a chordal polygon does not restore the omitted curved
segment between the true level set and its chord.

## 3. Limited-wall order and refinement — IN PRODUCTION, not matched

### Production call path

`ForwardProfile.solve_branch` / `solve_portfolio`
→ `_branch_receipt`
→ `_solve_accelerated`
→ `ForwardFluxOperator.flux_map`
→ `ForwardFluxOperator.internal`
→ `ForwardFluxOperator.cell_current_moments`
→ `ForwardFluxOperator._support_partition`
→ `ForwardFluxOperator._fixed_design_topology.read_with_connectivity`
→ `Topology.read_with_connectivity`
→ `Topology.wall(...)`
→ `Null1D.__call__`
→ `nova.geometry.select.traced_wall_flux`.

The limited/diverted comparator consumes the wall extremum returned by this
path. `traced_wall_flux` selects the largest sampled wall flux, takes the three
adjacent samples, fits a quadratic in cumulative chord length, evaluates its
stationary point, and locates that point by `jnp.interp` on the sampled wall
coordinates.

There are consequently two different orders to report:

- wall geometry is piecewise linear between authored sample points;
- wall flux is locally quadratic in chord length over one three-point cluster.

This is a point-sampled wall read, not a high-order wall element discretization.
It is nevertheless in production because every limited branch uses its result
to set the binding surface.

### Resolution evidence

- Canonical global-default production build (`Frame.dplasma = -2100`): **2,210
  plasma cells and 121 wall samples**, a count ratio of **18.26 plasma cells per
  wall sample**. The carrier stores 5,427 direct support-sampling nodes
  separately; those are not wall refinement.
- Production DINA forward-reference suite: `WALL_NODES = 3` means three samples
  per authored first-wall segment, and the current 2,214-cell carrier realises
  **165 wall samples**, or **13.42 plasma cells per wall sample**. The parameter
  is per segment rather than a physical spacing, so the ratio changes if the
  authored wall polygon changes even when `WALL_NODES` does not.
- MAST reference-native parity/catalog carrier: **95 × 95 = 9,025 plasma-grid
  nodes**, **2,161 stored-LCFS interior nodes**, and **37 limiter samples**. Its
  33-, 65-, and 95-point plasma-grid ladder leaves the wall target count at 37,
  so the wall was not refined with the plasma grid. At the reference-native
  rung this is **58.41 interior nodes per wall sample** (or 243.92 total grid
  nodes per wall sample).
- The DIII-D explicit pseudo-wall fallback currently defaults to 33 points per
  side, hence **132 wall samples**. An evidence bundle proposes 65 per side,
  hence **260 samples** and 1.9697 times as many wall-target rows, but records
  the change as rerouted to its owner. The default in
  `benchmarks/diiid_forward_gs_match.py` remains 33, so 260 is a proposal, not
  a production result. Against its current 1,089-node solve grid, the fallback
  carries **8.25 grid nodes per wall sample**; the proposed 260 is explicitly
  meant to accompany restoration of the native 65 by 65 grid rather than being
  a measured accuracy optimum.

No code couples wall spacing or wall sample count to plasma pitch, cell count,
or `dplasma`. One Solov'ev coarse/fine comparison changed wall targets from 165
to 330 while also changing the plasma carrier from 566 to 1,069 nodes and the
source treatment; it is therefore a whole-carrier comparison, not an isolated
wall-refinement convergence study. No receipt varies wall resolution while
holding the plasma mesh and physical state fixed and reports convergence of
limited-surface position, wall flux, residual, or terminal topology. The MAST
plasma-mesh refinement receipts instead keep the wall fixed at 37; the DIII-D
bundle quotes only row cost for a proposed 132→260 change. Therefore **no
isolated wall convergence/refinement study exists**, and equivalence to
plasma-mesh resolution has never been demonstrated.

## 4. Neighbouring-slice warm start — IN PRODUCTION

### Production call path

`nova.transport.coupled_window.equilibrium_sweep`
→ construct `sampled_profile` for the current waveform sample
→ choose `portfolio_seed`
→ `sampled_profile.solve_portfolio(portfolio_seed, ...)`
→ `ForwardProfile.solve_portfolio`
→ vmap `_branch_receipt`
→ `_solve_accelerated`.

For an empty selection history, only the first time sample is cold-started:

```python
cold = sampled_profile.cold_seed_portfolio(...)
portfolio_seed = cold.branches.flux
```

After a branch has been selected, every later sample passes the neighbouring
accepted state to **both** branch solves:

```python
portfolio_seed = jnp.stack((seed, seed))
portfolio = sampled_profile.solve_portfolio(portfolio_seed, ...)
...
seed = equilibrium.flux
history = selection.next_history
```

The loop is ordered over the supplied increasing time array, so `seed` is the
immediately preceding accepted slice, not the original seed and not an
independent cold start. Quantitatively: for an `N`-slice successful sweep there
is **1 cold portfolio and N − 1 neighbouring-slice warm portfolios**. Applied
as one uninterrupted sweep to the catalog denominator of 1,341,435 slices,
that policy would mean 1 cold portfolio and 1,341,434 warm portfolios; campaign
or shard boundaries would each introduce another cold first slice.

This reachability result does not promote warm-start economy into a performance
pass. The live plan records that the forward map is strongly non-normal and
that neighbouring seeds improved three measured residuals and worsened two.
Production does warm-start, but proximity alone is not evidence that the seed
direction reduces iterations.

## Evidence files used

- `nova/equilibrium/forward.py`
- `nova/equilibrium/forward_operator.py`
- `nova/equilibrium/source.py`
- `nova/equilibrium/stencil_mesh.py`
- `nova/equilibrium/separatrix_clip.py`
- `nova/equilibrium/topology.py`
- `nova/biot/target.py`
- `nova/biot/null.py`
- `nova/geometry/select.py`
- `nova/biot/arc.py`
- `nova/biot/momentchannel.py`
- `nova/biot/gradedresidual.py`
- `nova/transport/coupled_window.py`
- `docs/figures/forward-operator-refinement/global-dplasma-gpu-rebuild.json`
- `docs/figures/forward-operator-refinement/reference-native-resolution-default.json`
- `docs/figures/forward-operator-refinement/alignment-bundle.json`
- `docs/figures/forward-operator-refinement/resolution-default-inventory.md`
- `docs/figures/plasma-edge-current-representation/solovev_contour_findings.json`
