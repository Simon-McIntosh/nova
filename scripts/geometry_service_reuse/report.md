# Geometry service reuse and compatibility map

This report is the dispatch authority for the traced geometry-service implementation nodes. It is based on the complete live `flux-function-forward-transport` plan at version 25, especially section 8, and on Nova commit `399094b6039beb346ea133cd7ddbecc371fd2fff`. It contains **seven numbered sections**. The service boundary is fixed: a structured lattice and its exactly evaluated total-poloidal-flux values enter `nova.equilibrium`; one traced flux-surface-averaged record leaves. A non-structured forward evaluates flux directly on that lattice. It must not interpolate mesh-node values at the seam.

## 1. Locked composition and ownership

The service is a composition of existing mechanisms, not a fresh contour implementation:

The inspected Nova source inventory is `nova/equilibrium/flux_surface_connectivity.py`, `nova/equilibrium/separatrix_clip.py`, `nova/linalg/interpolant.py`, `nova/biot/arcbandedcoupling.py`, `nova/biot/contour.py`, `nova/equilibrium/flux_surface_geometry.py`, and the transport-local salvage source `nova/transport/current_diffusion.py`.

| Concern | Existing authority to consume | What the service adds |
|---|---|---|
| Topology | `nova.equilibrium.flux_surface_connectivity.flood_fill_core` | One axis-connected core mask per level, applied to cell supports and reductions. |
| Representation | `nova.linalg.interpolant.Bernstein` and `Polynomial`; the host `FluxSurfaceGeometry.from_flux_map` spline as referee | A traced global tensor cubic spline fit and a static map from its control net to per-cell 4-by-4 Bernstein coefficients. |
| Moments | `AtomicCellMesh.traced_clip`, `TracedClippedSupports`, `padded_polynomial_current_moments`, `_arc_rule`, and the landed curved-clip Green corrections | Spline edge roots and interpolant-over-support contractions, qualified by the topology mask. |
| Shape | Landed constrained bicubic Newton machinery plus fixed-shape min/max reductions; `Contour` and host flux-surface geometry as referees | Topology-qualified masked reductions with coordinate payloads and an implicit derivative for each stationary point. |

There is no ordered contour in the device service. The moments are cell sums and the shape values are masked reductions. `nova.biot.contour` remains a host referee and the curve-producing interface for consumers that actually need ordered points.

The record name needs deliberate resolution before the extraction entry point is published. `nova.transport.current_diffusion.FluxSurfaceGeometry` is the current uniform-rho transport record (`rho_face`, `vpr_face`, `g2_face`, `g3_face`, `q_face`, signs and scalars). `nova.equilibrium.flux_surface_geometry.FluxSurfaceGeometry` is a different immutable host record on a toroidal-flux label. The new equilibrium service must return the transport-facing record or a clearly named equilibrium-owned replacement that `nova.transport` re-exports; it must not expose two incompatible classes under one unqualified name.

## 2. Topology: flood-fill mask contract

- **Consumed symbol:** `nova.equilibrium.flux_surface_connectivity.flood_fill_core(confined, seed, n_iter)`.
  - Contract: `confined` and `seed` are boolean arrays with one fixed two-dimensional lattice shape. `seed` is intersected with `confined`. `n_iter` is static and `nr + nz` is the documented sufficient cap. Associative row and column scans find the seed-connected component. The result is an exact zero-or-one `float32` mask, intentionally nondifferentiable.
  - **Interface mismatch:** the service works per cell and per spline level, while the current connectivity caller constructs a node mask. Define one fixed cell-mask convention and vmap it over levels; do not silently mix node and cell masks. Convert the returned zero-or-one weights to boolean before `TracedClippedSupports.qualify` or masked extrema so topology is never mistaken for a smooth weight.

- **Consumed symbol:** `nova.equilibrium.flux_surface_connectivity.flood_fill_core_with_steps(confined, seed, n_iter)`.
  - Contract: identical mask inputs, returning `(core, steps)`; the integer step count is an execution diagnostic from the early-stopping `lax.while_loop`.
  - **Interface mismatch:** shape and moment assembly need only the mask, but service receipts and benchmarks may want the diagnostic. Keep `steps` outside differentiable output fields and do not make convergence of the boolean fill a data-dependent output shape.

- **Do not consume as the new moment authority:** `traced_flux_surface_bins` combines the flood fill with the retired Gaussian-CDF coarea estimator. Its dictionary and regular-grid assumptions are useful compatibility evidence, but section 8 replaces its smoothed moment path with spline clips. Reusing the whole function would reintroduce the bandwidth and node-sampling mechanism that the accuracy work removed.

Seed and limiter rules are part of the public compatibility contract. The axis seed must be an explicit one-hot cell inside `confined`; the current helper inside `traced_flux_surface_bins` obtains it with a masked `argmin` of normalised flux. `inside_limiter` has the same fixed two-dimensional orientation as the flux map. Disconnected private pockets at comparable flux are rejected by connectivity, not by flux height, sign of vertical coordinate, or contour order.

The in-flight density work adds the missing composition point: `TracedClippedSupports.qualify(participation)`. Once that commit is integrated, the service should qualify each per-level support with the flood-fill result before moments or extrema are read. Until then, the implementation nodes must not invent a second mask application protocol.

## 3. Representation: global traced tensor spline

- **Consumed symbol:** `nova.linalg.interpolant.Bernstein(order=3).coefficent_matrix(coordinate)`.
  - Contract: a registered pytree with `order` as static auxiliary data; the method evaluates the complete one-dimensional Bernstein basis and returns the basis index in the last dimension. The misspelling `coefficent_matrix` is the actual callable name.
  - **Interface mismatch:** this is a one-dimensional basis on unit coordinates, not a knot spline, derivative bundle, tensor product, or per-cell coefficient extractor. The tensor-spline node should reuse its basis semantics while exposing a correctly named service-local matrix/evaluation API. Renaming the existing method is outside this node and would break current callers.

- **Consumed symbol:** `nova.linalg.interpolant.Polynomial`.
  - Contract: a registered pytree whose `model` coefficients are traced children and whose `order` is static; `forward` evaluates one Bernstein polynomial through a fixed `lax.scan`.
  - **Interface mismatch:** one `Polynomial` holds one one-dimensional polynomial and initializes its output with `zeros_like(coordinate)`. The service needs batched two-dimensional 4-by-4 cell coefficients, paired-coordinate values, gradients and Hessians. Reuse the pytree/static-metadata pattern, not a Python nest of `Polynomial` objects.

- **Compatibility warning:** `nova.linalg.interpolant.BSpline` is not the required global cubic B-spline. It performs a dense least-squares fit of one global Bernstein polynomial and has neither a knot vector nor banded interpolation solves. Its name must not be used as evidence that the section-8 representation already exists.

- **Consumed implementation seam:** `_tensor_bernstein` and `_bicubic_derivatives` in `nova.transport.current_diffusion` already evaluate per-cell 4-by-4 Bernstein coefficients and their first and second local derivatives.
  - **Interface mismatch:** these are private transport-local functions, and their coefficient producer `_tensor_bicubic_coefficients` is the rejected local centered-tangent Catmull-Rom/Hermite reconstruction. Move or re-express the evaluator in the equilibrium service, feed it coefficients extracted from the new global spline, and retire the local producer rather than carrying two surface authorities.

- **Consumed referee:** `nova.equilibrium.flux_surface_geometry.FluxSurfaceGeometry.from_flux_map` constructs `scipy.interpolate.RectBivariateSpline(kx=3, ky=3, s=0)` and then calls `_refine_axis` and `_trace_surfaces`.
  - **Interface mismatch:** it is host NumPy/SciPy code, expects a `FluxLattice`, flattens the state in the lattice's radius-by-height order, traces ordered rays, and returns the host toroidal-flux record. The device service convention used by `traced_flux_surface_geometry` is `psi2d[height, radius]`. Reference tests must transpose explicitly at this boundary and compare named physical fields, not pass one layout into the other by accident.

The tensor fit must therefore be new but narrow: two traced banded one-dimensional interpolation solves, static knots, control values differentiable with respect to the gridded flux, and a static linear extraction of every cell's 4-by-4 Bernstein block. Values, gradients and Hessians should share one evaluator so clipping and Newton shape calculations cannot disagree about the represented surface.

## 4. Moments: clipped supports, polynomial contractions and curved arcs

- **Consumed symbol:** `nova.equilibrium.separatrix_clip.AtomicCellMesh.traced_clip(signed_flux)`.
  - Contract: the host-created `AtomicCellMesh` owns fixed node, cell and support capacities; the traced method clips `signed_flux > 0` with shared atomic edges and returns a fixed-capacity `TracedClippedSupports`.
  - **Interface mismatch:** `_traced_clip` computes a linear root from the two edge endpoint values. That is sufficient for polygon topology but not for the global spline's curved level set. The service must either add an injectable spline-root/arc description to the shared clip API or treat the polygon support as the topological chord and compose the curved Green corrections beside it. It must not fork the vertex packing and padding logic into the service.

- **Consumed symbol:** `nova.equilibrium.separatrix_clip.TracedClippedSupports`.
  - Contract: fixed-shape JAX fields `support_vertices`, `vertex_count`, `centroids`, `included`, `boundary`, `area`, `full_area`, `first_area_moment`, `second_area_moment`, `contour_area`, and `patch_area_sum`. Padding is interpreted only through `vertex_count`. Areas and moments are about each fixed centroid.
  - **Interface mismatch:** the stored moments describe straight polygon chords. Spline-arc corrections must be carried separately or become explicit fields with the same fixed leading cell dimension. Callers must never infer validity by testing padded coordinates. After the in-flight `qualify` change, `vertex_count`, boolean flags and area moments are zeroed outside participation, but `support_vertices`, `full_area` and `contour_area` remain present; `vertex_count` is the authority.

- **Consumed symbol:** `nova.equilibrium.separatrix_clip.padded_polynomial_current_moments(support_vertices, vertex_count, centroids, coordinate_scale, coefficients, powers=None)`.
  - Contract: integrates a complete local monomial basis over every padded polygon using closed simplex moments and returns zeroth and first density moments. `coefficients` has shape `(cells, terms)`; `powers` is static and may be inferred from a complete-basis width. Coordinates are centered on `centroids` and divided by `coordinate_scale`.
  - **Interface mismatch:** the output and names are current-density-specific and integrate a polynomial over the straight support. Geometry integrands include inverse radius and spline-gradient functions and need curved-boundary contributions. Reuse its padding, local-coordinate and monomial-order conventions where an integrand is polynomial; provide a generic interpolant-over-support contraction for the geometry fields rather than disguising them as current.

- **Consumed symbol:** `nova.biot.arcbandedcoupling._arc_rule(nodes)`.
  - Contract: cached NumPy Gauss-Legendre nodes and weights mapped to the unit interval. The landed curved clips use `_arc_rule(12)` and convert those constants to the traced dtype.
  - **Interface mismatch:** this is a private symbol in a coupling module and returns host arrays. The geometry service should materialize it once as static constants, not call NumPy while tracing. A shared public quadrature location is desirable, but copying the numerical rule under a second authority is not.

- **Consumed symbol:** `nova.transport.current_diffusion._bicubic_arc_moment_correction`.
  - Contract: fixed edge-root bisection, branch-following Newton samples on the level set, and oriented Green-theorem chord-to-arc corrections for area, radial first moment, vertical first moment and the mixed moment; it also returns edge and arc samples plus a validity mask.
  - **Interface mismatch:** it is private to the transport module and currently receives coefficients from the rejected local reconstruction. Relocate the mechanism to equilibrium and feed the global spline's per-cell Bernstein block. Its validity mask must combine with the topology-qualified support mask, and an invalid curved correction must fail or produce a diagnostic rather than silently falling back to the chord.

- **Consumed symbol:** `nova.transport.current_diffusion._integrate_bilinear`.
  - Contract: contracts four corner values with polygon area/first/mixed moments plus the curved corrections.
  - **Interface mismatch:** only bilinear within-cell integrands fit this signature. The service needs a named policy per FSA field: exact polynomial contraction where possible, fixed arc quadrature for spline-derived or rational quantities, and volume-derivative identities where applicable. One generic silent bilinear approximation would recreate the representation error under a different name.

The history that led to this contract matters. Commit `3e46a605` introduced `padded_polynomial_current_moments`, integrating one interpolant over the actual support. Commit `5e838c92` generalized it from a fixed cubic to an inferred or explicit complete basis. Commit `9ef0e8dd` removed the separate full-cell/clipped evaluation weights and the transient smoothing fields from `TracedClippedSupports`. The service must target the post-removal API.

The ramped edge-crossing hand-off in commit `c6ffebe5` remains useful as a continuity pattern, not as a callable API. It formed `transition_weight = d^2(3-2d)` from the minimum normalized endpoint distance, blended surface and exact support moments, and was then retired when the one-interpolant-over-actual-support route made the two-path blend unnecessary. The current continuity pin is stronger: edge-vanishing polynomial moments and their composed fields cross a full-cell transition with matching first derivatives, with no smoothing option. Service tests should reproduce that invariant for spline moments rather than restore `transition_vertices`, `transition_vertex_count`, `transition_weight`, `evaluation_weights`, or `smoothing_width`.

## 5. Shape and permanent referees

- **Consumed symbol:** `nova.transport.current_diffusion._bicubic_stationary_point`.
  - Contract: ten fixed Newton iterations solve the level-set constraint together with the derivative condition for a radial or vertical extremum; steps and coordinates are clipped, and the result carries an explicit validity mask.
  - **Interface mismatch:** it currently relies on ordinary differentiation through the fixed iteration and local coefficients. Section 8 requires implicit differentiation of the converged constrained solve. Keep the residual, fixed shape and validity rules, but attach an implicit derivative with respect to the global spline coefficients. Invalid points must be masked before reduction.

- **Consumed reduction pattern:** the `extrema` closure inside `_clipped_surface_geometry` combines edge roots, arc samples and stationary points, then uses masked `min`, `max`, `argmin` and `argmax`; the vertical extrema carry radial-coordinate payloads for triangularity.
  - **Interface mismatch:** the implementation is nested in a private transport function and does not expose topology as a first-class argument. Extract a fixed-shape masked arg-extremum-with-payload primitive in equilibrium. Its mask is `support.included` after flood-fill qualification and curved-root validity. Ties use the selected JAX subgradient; tests must cover a non-tie shape gradient and a deterministic tie.

- **Consumed referee:** `nova.biot.contour.Contour`, particularly `levelset` and `closedlevelset`, with `Surface` carrying ordered points and its `closed` flag.
  - **Interface mismatch:** `contourpy`, NumPy arrays, mutable dataclasses and data-dependent polyline lengths are host-only. It is a comparison and visualization route, never an implementation dependency of the jitted service. `closedlevelset` returns the first closed contour found, so reference tests must still verify axis connectivity instead of assuming ordering selects the physical core.

- **Consumed referee:** `FluxSurfaceGeometry.from_flux_map`, `_refine_axis`, and `_trace_surfaces` in `nova.equilibrium.flux_surface_geometry`.
  - **Interface mismatch:** `_refine_axis` uses host `np.linalg.solve` on a SciPy spline Hessian; `_trace_surfaces` casts ordered rays and bisects the first outward crossing. They define reference behavior and output metrics, not reusable device functions. The service should compare axis, enclosed quantities and shape against this route while producing them from per-cell reductions.

The permanent three-route comparison is therefore: TORAX's independent reader; Nova's host global-spline contour/ray route; and the traced service. A two-against-one split remains diagnostic. No production method may call its referee internally, because that would turn the comparison into a self-comparison.

## 6. Committed in-flight histories and compatibility hazards

The histories were read through this worktree's Git object database; neither peer worktree was entered or modified.

### density-moment-projection

The detached density worktree head is `e1abc3061bccd3bb74e257d64cb8f39b4781941a`. Relative to this node's base, its substantive unique commits are:

- `c3ff6142`: replaces `stencil_mesh.fixed_profile_current_moments`' polynomial projection with a fixed degree-fifteen Duffy product rule that evaluates the actual density at quadrature points. It adds an optional topology `participation` mask. This is density-specific; the geometry service should not replace exact Green moments with that density quadrature, but it must tolerate the changed caller contract.
- `5d282612`: adds topology qualification across the moment route. Its reusable API contribution is `TracedClippedSupports.qualify(participation)`, which zeroes `vertex_count`, `included`, `boundary`, `area`, first and second area moments, and recomputes `patch_area_sum`. The same commit passes participation through the direct density integrator and topology/forward-operator path.

**Integration rule:** base the service's topology-to-support handoff on `qualify` after this head lands. Do not implement an overlapping qualifier in a new module. Because `qualify` deliberately leaves raw padded vertices, centroids, `full_area` and `contour_area` in the tuple, every downstream contraction must obey `vertex_count` and the qualified flags.

### Exact-kernel defaults

The exact-kernel-defaults history is already an ancestor of this node's base: merge `40d30fb6` and evidence commit `20fadf2c`. The merge changed the production section-coupling routes in `nova.biot.cylinder`, `nova.biot.polysection`, `nova.frame.coil` and their tests: exact polygon-section coupling became the default and the filament/standoff/banded truncation branches were removed. It did **not** modify `flux_surface_connectivity.py`, `separatrix_clip.py`, `interpolant.py`, `arcbandedcoupling.py`, `contour.py`, or `flux_surface_geometry.py`.

**Integration rule:** there is no content conflict with the service files, and `_arc_rule` remains byte-stable across that merge. There is, however, an ownership hazard: the helper is private inside a module whose banded production role has been reduced. Pin the unit-interval rule with a geometry-service test or move it once to a shared public quadrature utility; do not depend on the continued use of banded arc coupling as the reason the helper survives.

## 7. Coupled Ambix search and node handoff

The declared coupled repository `/home/ITER/mcintos/Code/imas-ambix` contains FSA implementations and consumers but no tensor-spline extraction to reuse.

- `imas_ambix.latent.flux_surface_connectivity.flux_surface_bins_jax` is an older contour-free FSA implementation with a fixed-iteration four-neighbour flood fill and the same smooth-CDF coarea bins. It is prior art and a legacy consumer path, not a second authority. Nova's `flood_fill_core` now uses associative scans and Nova's service is where deterministic extraction belongs.
- `imas_ambix.latent.current_diffusion.flux_surface_geometry` is a local host contour/coarea geometry implementation used by historical scripts and tests. It should remain a comparison or migration target; the service must not reproduce it inside Nova.
- `imas_ambix.physics.transport` already imports and re-exports Nova's `FluxSurfaceGeometry`, `flux_surface_geometry`, `traced_assemble_flux_surface_geometry`, and `traced_flux_surface_geometry`, and adds only mapping adapters. This is the consumer seam to preserve when the implementation moves from `nova.transport` to `nova.equilibrium`: keep the public Nova names stable through re-export or update this one adapter deliberately.
- Ambix's flux-evolution extraction script already imports `nova.equilibrium.flux_surface_connectivity.traced_flux_surface_bins`, demonstrating that cross-repository consumers can use a Nova equilibrium-owned geometry primitive.
- A search for `RectBivariateSpline`, `BSpline`, `make_interp_spline`, `splrep`, `splev`, `CubicSpline`, `Bernstein`, and `Polynomial` found no production spline-based surface extractor in Ambix. The only polynomial hit was unrelated reduced-order feature construction.

Implementation nodes should cite this report and obey the following handoff:

1. Fit one global tensor spline and expose per-cell Bernstein coefficients plus shared value/gradient/Hessian evaluation.
2. Build one flood-fill mask convention and vmap it over fixed levels.
3. Produce curved supports through the shared clip API, then call `qualify` before all moment and shape reads.
4. Reuse the Green correction and fixed Gauss rule, with an explicit integration policy per output field.
5. Reduce shape with topology-qualified masks and payloads; differentiate a stationary point implicitly.
6. Return the transport-facing FSA record from equilibrium and make transport and Ambix re-export that single authority.
7. Keep TORAX, `nova.biot.contour`, and host `FluxSurfaceGeometry.from_flux_map` independent and permanent as referees.

Done-when measure: **4 of 4 section-8 concerns mapped; 16 consumed symbols or callable patterns carry explicit interface-mismatch lines; 7 numbered report sections; 2 committed peer histories inspected without entering their trees; 1 coupled repository searched; 0 spline extraction implementations found there.**
