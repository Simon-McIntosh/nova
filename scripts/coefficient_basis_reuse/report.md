# Coefficient-basis reuse map

## Scope and quantitative coverage

This is a solve-free implementation inventory prepared from Nova at
`de4b05373e810c317dbf9994c295b31c44ac60ed`, tracked Ambix prior art at
`c0c75b362dec63e8c25b15d5b072ab6a01978208`, and Reckon at
`bc015b56dcbc2c88b0439ba20bd9bc49e965d24e`. The search covered package code,
tests, benchmarks, measurement scripts, banked roots, and the four live
semantic machine-cache groups named below. It found **30 numbered reuse
candidates** and **10 explicitly absent capabilities**. No equilibrium solve,
Green-operator build, or cache mutation was run.

The search included:

- tensor-product splines, B-splines, Bernstein bases, polynomial bases,
  harmonics, SVD bases, and coarse spatial current bases;
- sampled-value-to-coefficient fits, least-squares and pseudoinverse
  projections, prolongation/restriction, and point evaluation;
- first, second, mixed, and elliptic derivative accessors;
- fixed-shape JAX evaluation, `jit`, `vmap`, `jax.linearize`, `jvp`, `jacfwd`,
  and matrix-free linear actions;
- structured-grid, unstructured-cell, local-stencil, and separatrix-aware
  representations;
- repeated knots, knot insertion/refinement, continuity reduction, and moving
  knot lines;
- stored-reference and closed-form machine carriers, response matrices,
  reference maps, and banked converged roots; and
- the coupled Ambix and Reckon repositories for compatible basis, projection,
  or traced-operator prior art.

A candidate below is a repository-owned reusable symbol, an existing
external-library route already used on equilibrium fields, a directly relevant
contract test, or an identity-pinned artifact the fit study must consume.
Unrelated one-dimensional time interpolation, quadrature rules, neural-network
projection heads, and geometry-only coordinate projections were searched but
excluded because they cannot contribute to a spatial flux coefficient state.

## Executive verdict

`TensorBSpline` **cannot carry a coarse-knot control-coefficient set as its
primary parameterisation as written and must be extended**. Its public
`coefficients` are already-expanded per-cell Bernstein blocks with shape
`(vertical_cells, radial_cells, 4, 4)`, while `fit_tensor_spline` accepts sampled
values at every supplied lattice node and derives those blocks with two traced
not-a-knot solves. Passing a coarse lattice to `fit_tensor_spline` is immediately
usable compression—a coarse **nodal-value** state—but it is not a literal
B-spline control-coefficient state.

The reuse map favours **re-parameterising the existing tensor-spline authority**
over authoring a second evaluator. Keep `TensorBSpline.evaluate`,
`TensorSplineEvaluation`, `_bernstein_matrix`, and `_tensor_bernstein`; add a
primary coarse lattice state and one fixed linear conversion to the cell
Bernstein blocks. Preserve `fit_tensor_spline` as the sampled-value constructor.
This keeps flux-surface extraction and coefficient-space work on one value and
derivative implementation while allowing the Newton state to be the coarse
unknown itself.

For the solve-free degree study, no product extension is required to form the
first design matrix. Because `fit_tensor_spline` is linear in its sampled-value
input, map the coarse identity basis through
`fit_tensor_spline(coarse_r, coarse_z, unit_state).evaluate(cell_r, cell_z)`
under `jax.jit(jax.vmap(...))`; transpose the results into the fixed
coefficient-to-cell matrix and project each banked cell field with a declared
least-squares solve. That proves or refutes the effective degree without
building a solver, while the product implementation can subsequently replace
coarse nodal values with literal control coefficients if that distinction is
worth retaining.

The derivative accessor is
`TensorBSpline.evaluate(...) -> TensorSplineEvaluation`. It supplies
`radial_derivative`, `radial_second_derivative`, `mixed_derivative`, and
`vertical_second_derivative` in addition to value and vertical first
derivative. The Grad--Shafranov contraction is
`radial_second_derivative - radial_derivative / R + vertical_second_derivative`;
the mixed term is available for the complete Hessian and derivative-consistency
checks but is not itself a term in this axisymmetric `Delta*` formula.

The existing fixed-shape route is
`tests/test_tensor_spline.py::test_jit_eager_and_vmap_keep_fixed_shapes`: its
`fit_and_evaluate` closure calls `fit_tensor_spline` and `spline.evaluate` at a
paired point array, once eagerly, once under `jax.jit`, and once under
`jax.jit(jax.vmap(...))` for two maps. The evaluator accepts arbitrary paired
coordinates, so supplying `machine.node[:, 0]` and `machine.node[:, 1]` is the
cell-point route; there is no separate cell-specific wrapper to invent.

## Numbered candidates

1. **Repo:** `nova`; **path:** `nova/linalg/tensor_spline.py`; **symbol:** `TensorBSpline`, `fit_tensor_spline`; **fitness:** **best implementation substrate, extension required for literal control coefficients** — the traced not-a-knot tensor spline is linear in sampled inputs and can already compress on a coarse sampled lattice, but its stored state is expanded cell Bernstein blocks rather than primary coarse controls.

2. **Repo:** `nova`; **path:** `nova/linalg/tensor_spline.py`; **symbol:** `TensorSplineEvaluation`, `TensorBSpline.evaluate`; **fitness:** **direct reuse as the sole value/derivative accessor** — it returns value, both first derivatives, both pure second derivatives, and the mixed derivative at arbitrary paired points with physical coordinate scaling.

3. **Repo:** `nova`; **path:** `nova/linalg/tensor_spline.py`; **symbol:** `_bernstein_matrix`, `_tensor_bernstein`, `_cubic_bernstein_control`; **fitness:** **direct internal reuse for the fixed coefficient operator** — these are the traced one-dimensional basis matrix, separable tensor contraction, and sampled-value-to-cell-block transform; expose or refactor them rather than duplicate their arithmetic.

4. **Repo:** `nova`; **path:** `nova/equilibrium/flux_surface_extraction.py`; **symbol:** `_tensor_bicubic_coefficients`, `_tensor_bernstein`, `_bicubic_derivatives`, `_bicubic_third_derivatives`; **fitness:** **reference and consolidation target, not a second authority** — this older C1 local bicubic path proves fixed-shape Bernstein derivative arithmetic through third order, but it is sampled-cell parameterised and duplicates the evaluator now owned by `TensorBSpline`.

5. **Repo:** `nova`; **path:** `nova/linalg/interpolant.py`; **symbol:** `Bernstein.coefficent_matrix`, `BSpline.inverse`, `Polynomial.forward`; **fitness:** **reuse the traced one-dimensional design pattern only** — it forms a JAX Bernstein matrix and solves coefficients with `jnp.linalg.lstsq`, but the class named `BSpline` is one global Bernstein polynomial, has no knot vector, and is not a two-dimensional spline.

6. **Repo:** `nova`; **path:** `nova/linalg/basis.py`; **symbol:** `Bernstein`, `Svd`; **fitness:** **host-study comparator only** — these NumPy/SciPy basis matrices can compare conditioning or empirical rank, but they are one-dimensional, mutable, and not traced production state.

7. **Repo:** `nova`; **path:** `nova/linalg/regression.py`, `nova/linalg/decompose.py`, `nova/linalg/lops.py`; **symbol:** `OdinaryLeastSquares`, `MoorePenrose`, `Decompose`, `Lops`; **fitness:** **generic host projection prior art, not the device route** — the modules provide dense least squares, SVD pseudoinverse, and a Pylops `LinearOperator`, but all are NumPy/host objects and none builds the required tensor collocation matrix.

8. **Repo:** `nova`; **path:** `nova/equilibrium/flux_surface_geometry.py`, `nova/biot/contour.py`, `nova/imas/profiles.py`; **symbol:** `RectBivariateSpline` call sites; **fitness:** **strong fp64 host referee, unsuitable as the Newton representation** — existing equilibrium code already evaluates smooth two-dimensional fields and derivatives through SciPy, but the objects are host-only, sampled-grid parameterised, and cannot enter `jit` or `vmap`.

9. **Repo:** `nova`; **path:** `benchmarks/diiid_corpus_conventions.py`, `benchmarks/diiid_vacuum_quiescent_gate.py`, `benchmarks/efit_flux_decomposition.py`; **symbol:** `RegularGridInterpolator` call sites; **fitness:** **linear-resampling comparator only** — these routes can check coordinate ordering and value interpolation on rectangular maps but supply neither a smooth coefficient basis nor second derivatives.

10. **Repo:** `nova`; **path:** `nova/equilibrium/harmonic.py`; **symbol:** `harmonic_columns`, `harmonic_flux_on_grid`, `ReconstructHarmonic.fit`; **fitness:** **valuable low-dimensional physics comparator, invalid as the full-domain state** — its fixed geometry matrix and coefficient fit represent source-free vacuum-annulus flux, but the basis is singular/invalid toward its focal ring, NumPy-only, and cannot represent sourced Grad--Shafranov curvature in the plasma.

11. **Repo:** `nova`; **path:** `nova/equilibrium/stencil_mesh.py`; **symbol:** `StencilMesh._fit_rings`, `StencilMesh.gradient`, `StencilMesh.delta_star`; **fitness:** **direct derivative/referee reuse, not a global coarse state** — precomputed pseudoinverse weights turn local ring values into traced gradients and `Delta*` on the production unstructured mesh, but every fit is local and its state remains one value per cell.

12. **Repo:** `nova`; **path:** `nova/equilibrium/stencil_mesh.py`; **symbol:** `SharedNodeFluxStencil`, `StencilMesh.shared_node_flux_stencil`; **fitness:** **direct fixed prolongation pattern** — one immutable gather-and-weight operator evaluates a cell field at arbitrary shared nodes under JAX, demonstrating exactly how geometry-owned fixed evaluation should be stored, though its source space is still the full cell vector.

13. **Repo:** `nova`; **path:** `nova/equilibrium/stencil_mesh.py`; **symbol:** `InteriorCurrentMomentStencil.sample_flux_field`, `_quadratic_flux_design`; **fitness:** **direct cell-point evaluation pattern, local quadratic only** — it contracts a precomputed local basis at fixed support points and returns value plus gradient without data-dependent shapes, but it has no global coefficient continuity or pure/mixed second-derivative payload.

14. **Repo:** `nova`; **path:** `nova/geometry/select.py`; **symbol:** `traced_quadratic_surface`, `traced_quadratic_wall`, `null_flux`; **fitness:** **traced local-fit reference only** — these `jnp.linalg.lstsq` quadratic fits show JAX-safe coefficient recovery and evaluation for small null/wall clusters, not a mesh-wide parameterisation.

15. **Repo:** `nova`; **path:** `nova/equilibrium/separatrix_clip.py`; **symbol:** `complete_polynomial_powers`, `padded_polynomial_current_moments`; **fitness:** **reuse for regional/current integration after evaluation, not for flux coefficients** — the complete two-dimensional monomial basis and fixed-capacity traced moments integrate polynomial current density on moving clipped supports but provide no smooth global flux basis.

16. **Repo:** `nova`; **path:** `nova/equilibrium/moment.py`; **symbol:** `build_moment_basis`; **fitness:** **low-dimensional current-image comparator only** — its masked zero-sum monomials are useful for asking whether current moments are low rank, but their coefficients parameterise toroidal current rather than poloidal flux and the implementation is NumPy.

17. **Repo:** `nova`; **path:** `nova/equilibrium/profile.py`; **symbol:** `ReconstructProfile._profile_basis`, `_least_squares_coefficients`, `_scaled_kkt`; **fitness:** **reuse the traced constrained-projection pattern, not the basis itself** — it builds flux-dependent current columns, forms their sensor response, and solves a scaled fixed-shape KKT system under JAX, but the unknowns are profile coefficients and the basis changes with topology.

18. **Repo:** `nova`; **path:** `nova/equilibrium/conservation.py`, `nova/equilibrium/map_extraction.py`; **symbol:** `FluxLattice.gradient`, `FluxLattice.delta_star`, `apply_delta_star`; **fitness:** **direct structured-grid derivative referee** — these fixed central-difference operators score a reconstructed field and derive current without a solve, but their state is the full raster and they expose no coefficient projection or mixed derivative.

19. **Repo:** `nova`; **path:** `nova/equilibrium/forward_operator.py`; **symbol:** `ForwardFluxOperator.flux_map`, `current_moment_image`; **fitness:** **direct production residual composition** — it captures the external field once and contracts current moments through cached Green response matrices under JAX; compose this map with coefficient-to-cell evaluation for a prototype rather than rebuild any physics response.

20. **Repo:** `nova`; **path:** `nova/equilibrium/fixed_point.py`; **symbol:** `newton_krylov`, `kink_aware_newton_krylov` (`jax.linearize` tangent closures); **fitness:** **direct traced-linear-operator pattern, not the coefficient solver** — exact map tangents already produce fixed-shape matrix-free `(I - J)` actions, and the same mechanism can differentiate a coefficient-composed map or form a small dense Jacobian with `jacfwd`.

21. **Repo:** `nova`; **path:** `tests/test_forward_operator_tangent.py`; **symbol:** `test_symmetric_null_fit_has_finite_exact_residual_action`; **fitness:** **direct tangent regression reuse** — it proves `jax.linearize` of the production forward map gives a finite residual action agreeing with central differences, so coefficient composition should extend this test rather than create a separate tangent oracle.

22. **Repo:** `nova`; **path:** `tests/test_tensor_spline.py`; **symbol:** `test_values_gradient_and_hessian_match_scipy`, `test_point_gradient_with_respect_to_map_matches_finite_difference`, `test_jit_eager_and_vmap_keep_fixed_shapes`; **fitness:** **direct spline contract reuse** — these tests pin fp64 values/Hessian entries, differentiability with respect to the complete sampled map, and fixed shapes under eager, `jit`, and `jit(vmap)` evaluation.

23. **Repo:** `nova`; **path:** `tests/test_equilibrium_forward_reference.py`; **symbol:** `machine_cache_identity`, `cached_machine`, `HexMachine`; **fitness:** **authoritative stored-reference machine and response carrier** — request the warm semantic groups `746fbe1553c4b242` (coarse: 566 cells, 2,396-state root) and `f0f96aa214aa9459` (fine: 1,069 cells, 4,216-state root), then consume `node`, source/plasma response blocks, stencil, samples, wall rows, and field blocks rather than calling `build_machine`.

24. **Repo:** `nova`; **path:** `scripts/root_gate_attribution/coarse-terminal-root.npz`, `scripts/root_gate_attribution/fine-terminal-root.npz`; **symbol:** `state` and regional/topology arrays; **fitness:** **authoritative stored-reference converged-root inputs** — use the grid prefix `state[:len(machine.node)]` with the matching cached carrier; the banks contain 21 arrays each and have SHA-256 `1b24fdc7c7f917b3a4a6a4a720541c3bb7ef69a98706f9d866dd6d77b1a804ad` and `a583cef5ec17c8dd8c3186cd5ff95ff4bca8a96128a9b08825fe68a637a9152a`.

25. **Repo:** `nova`; **path:** `scripts/analytic_oracle_fixtures/measure.py`; **symbol:** `cached_machine`, `OracleMachine`; **fitness:** **authoritative closed-form machine and response carrier, kept separate from the stored reference** — request semantic groups `030667d40f96d904` (coarse) and `7c94be9b777c55ba` (fine); never import their anchors, state, or acceptance floor into the stored-reference lane.

26. **Repo:** `nova`; **path:** `scripts/oracle_rebaseline/root-coarse.npz`, `scripts/oracle_rebaseline/root-fine.npz`; **symbol:** `root_state`, `oracle_state`, `seed_state`, `root_grid_psi_norm`, `oracle_grid_psi_norm`; **fitness:** **direct analytic fit controls** — each bank carries 10 arrays and has SHA-256 `79f4a3f507680b7531bf3a3dc6a1195acf321027f0c3f4d4ab103bc70e1a1e0c` and `512f8a4bce44b2701a76b587bf0740cd17c1b037839dbcfbde86863115b251de`, providing root-versus-closed-form representation error without a solve.

27. **Repo:** `imas-ambix`; **path:** `imas_ambix/latent/patch_basis.py`; **symbol:** `PatchBasis`; **fitness:** **conceptual fixed-operator reuse only** — it precomputes a geometry-keyed current-to-grid matrix and applies it batchwise on GPU, but it is Torch, its coefficients are per-cell currents, and its cache covers only `g_pg`, not a smooth flux basis.

28. **Repo:** `imas-ambix`; **path:** `imas_ambix/gs/operator.py`; **symbol:** `_default_plasma_basis`, `ForwardOperator`; **fitness:** **coarse spatial-current comparator only** — the limiter-masked 9-by-13 default proves a low-dimensional spatial basis can feed fixed Green columns, but it is a point-current/NumPy sensor operator rather than a flux spline or JAX map.

29. **Repo:** `imas-ambix`; **path:** `imas_ambix/statespace/discovery_sindy.py`; **symbol:** `ReducedBasis`, `build_reduced_basis`; **fitness:** **projection/lift design pattern only** — its orthonormal SVD project/lift pair is an exact reduced-subspace operator, but the learned coordinates are latent trajectories with no physical knot layout, derivatives, or Grad--Shafranov meaning.

30. **Repo:** `imas-ambix`; **path:** `imas_ambix/latent/current_diffusion.py`; **symbol:** `basis_projection_images`, `project_coefficients`; **fitness:** **host fit-study pattern only** — volume-weighted stacked least squares projects predicted one-dimensional current profiles onto declared coefficient images, but it is SciPy/NumPy, flux-surface rather than `(R,Z)` based, and does not evaluate a two-dimensional flux state.

## Cache and bank identity the degree study must consume

The live host contains all four semantic groups; existence was checked without
loading or mutating them. The stored carrier's 31 persisted arrays include 20
source/plasma coupling arrays plus mesh and field data, so a fit study that
rebuilds `CoilSet`, calls `plasmagrid.solve`, or evaluates a Green kernel is on
the wrong route.

| lane | request and semantic group | root/reference fields | required use |
| --- | --- | --- | --- |
| stored reference, coarse | `tests.test_equilibrium_forward_reference.cached_machine(case, -500, passive=True)`; `746fbe1553c4b242` | `case.grid_flux`; `root_gate_attribution/coarse-terminal-root.npz::state[:566]` | use `machine.node` as irregular cell points and the bank's same-carrier state; never rebuild responses |
| stored reference, fine | set the declared fine wall discretisation, then `cached_machine(case, -1000, passive=True)`; `f0f96aa214aa9459` | `case.grid_flux`; `root_gate_attribution/fine-terminal-root.npz::state[:1069]` | same, with the fine semantic identity and no cross-load from coarse |
| closed-form analytic, coarse | `scripts.analytic_oracle_fixtures.measure.cached_machine(case, -500, wall_nodes=3)`; `030667d40f96d904` | `oracle_rebaseline/root-coarse.npz::{root_state, oracle_state, root_grid_psi_norm, oracle_grid_psi_norm}` | use only analytic-lane anchors, gauges, and receipts |
| closed-form analytic, fine | `cached_machine(case, -1000, wall_nodes=6)`; `7c94be9b777c55ba` | `oracle_rebaseline/root-fine.npz` matching fields | use only analytic-lane anchors, gauges, and receipts |

For every ladder rung, the fit receipt should record the source bank path and
SHA-256, machine semantic key, realised cell count, knot axes, state convention
(raw total flux or lane-local normalized flux), coefficient count, projection
method, and regional masks. These fields make a warm response carrier and a
banked root one indivisible input identity without pretending the root itself
belongs in the machine-cache key.

## Explicit no-candidate results

- **No candidate — primary coarse control-coefficient tensor spline:** no Nova, Ambix, or Reckon symbol stores a two-dimensional knot vector plus primary B-spline control lattice and evaluates it as the physical state; `TensorBSpline` stores derived per-cell Bernstein blocks and `fit_tensor_spline` starts from nodal samples.
- **No candidate — irregular-cell coarse tensor projection API:** no symbol forms or solves the rectangular design that projects values at the production hex-cell coordinates onto a coarse tensor basis; the design can be assembled from the existing spline evaluator, but the reusable projector is absent.
- **No candidate — precomputed coefficient-to-derivative operator bundle:** no object owns fixed matrices for coefficient-to-value, `dR`, `dZ`, `dRR`, `dRZ`, and `dZZ` at all production cell points; the evaluator can generate every matrix, but there is no stored bundle or identity contract.
- **No candidate — knot insertion, repeated knots, or continuity control:** searches found no knot-multiplicity, knot-insertion/refinement, or C1-at-separatrix B-spline machinery in Nova or Ambix; all shipped tensor splines are globally not-a-knot C2 and the older bicubic path has fixed local continuity.
- **No candidate — moving or topology-conditioned knot lattice:** no fixed-shape JAX route moves a knot line with the separatrix or selects knot topology from an iterate; introducing one would make the basis state-dependent and threaten the shared-design `jit`/`vmap` contract.
- **No candidate — global spline directly on the unstructured hex mesh:** local quadratic ring fits and shared-node prolongation exist, but no globally continuous coarse basis spans the irregular cell centroids and clipped boundary.
- **No candidate — production cell-point tensor-spline wrapper:** arbitrary point evaluation and its `jit(vmap)` test exist, but no production object binds one coarse spline design to `ForwardFluxOperator.grid.coordinate`; the wrapper and its cache identity must be authored or kept study-local.
- **No candidate — coefficient-space fixed-point map and dense Jacobian receipt:** existing maps and exact tangents act on the full flux vector; no route projects the mapped cell field back to coarse coefficients or reports the explicitly formed dense coefficient Jacobian.
- **No candidate — fit-result cache keyed by bank plus knot ladder:** machine-response caches and immutable root banks exist, but no cache or receipt currently binds a projection result to both identities, its regional split, and its knot specification.
- **No candidate — coupled-repository JAX flux spline:** Ambix contains Torch/NumPy current bases and latent projections, while Reckon contains plan-state projections only; neither repository supplies a JAX spatial flux spline, coarse coefficient projector, or Grad--Shafranov derivative accessor.

## Recommended reuse boundary

Build the degree study around `TensorBSpline.evaluate` and the four immutable
carrier/root pairs above. Generate one fixed collocation matrix per carrier and
knot lattice by vmapping unit coarse states through the existing traced spline,
solve coefficients without running the forward map, and report the error in the
closed-flux region, stated separatrix band, and scrape-off layer separately.
Use the cached carrier's topology/geometry arrays to define those masks, and
retain raw-flux versus normalized-flux and gauge provenance in every row.

If the measured degree is usable, re-parameterise the existing tensor-spline
module so the coarse lattice is the primary state and the cell Bernstein blocks
are one derived fixed linear image. Compose that evaluator with
`ForwardFluxOperator.flux_map`; reuse `jax.linearize`/`jacfwd` to form the small
coefficient Jacobian. Do not fork the Bernstein derivative implementation, do
not rebuild a machine response, and do not treat a conditioning improvement as
evidence of a lower nonlinear terminal residual.

## Validation

- Inventory contract: **30** numbered candidates, each carrying repo, path,
  symbol, and one-line fitness verdict; **10** explicit no-candidate lines.
- Artifact identity: **4/4** root banks exist and expose **62** archived arrays
  in total; their SHA-256 values are recorded above.
- Live carrier identity: **4/4** semantic cache groups exist at their declared
  versioned user-data paths; no group was loaded, built, deleted, or written.
- Spline batching contract:
  `JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest tests/test_tensor_spline.py::test_jit_eager_and_vmap_keep_fixed_shapes`
  — **1 passed in 20.62 s** (one unrelated `jaxopt` deprecation warning).
- Document integrity:
  `git diff --cached --check -- scripts/coefficient_basis_reuse/report.md`
  — **pass**.
