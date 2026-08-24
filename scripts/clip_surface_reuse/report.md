# Curved separatrix clip reuse map

## Outcome

The re-measurement does not need a new clip representation. The current production
extraction path already demonstrates the required composition: fit the complete flux
map with `fit_tensor_spline`, take each fixed `4 x 4` Bernstein cell block from
`TensorBSpline.cell_coefficients`, and pass those blocks through the landed bicubic
root, arc, Green-moment, and constrained-extremum machinery while leaving
`TracedClippedSupports` unchanged. The measurement work should reuse that composition
and the existing three reader factories in `tests/test_transport_geometry_reference.py`.

The one missing measurement convenience is a direct contour-to-TORAX comparator. It
can be composed in the measurement harness from `_contour_geometry`,
`_torax_geometry`, and `_relative_error`; it does not justify a production API.

## 1. Historical curved-clip primitives

Commit `47808904f51a3bf553112538030c47a1b80c531d` added the tensor-bicubic
clip implementation to `nova/transport/current_diffusion.py` and banked its real-EQDSK
measurements in `tests/test_transport_geometry_reference.py`. The implementation has
since been centralized in `nova/equilibrium/flux_surface_extraction.py`; the table maps
the historical primitives to their current homes.

| Capability | Candidate | Fitness verdict |
|---|---|---|
| Circular-arc sagitta leverage | `nova.biot.arc.Arc._fit_leverage`, specifically `sagitta_ratio = 2 sin(sweep / 4)^2` | **Diagnostic only.** It conditions a three-point circular-arc fit; it does not describe a tensor-spline level set and must not become the separatrix curve authority. |
| Cell polynomial evaluation | `_tensor_bernstein` and `_bicubic_derivatives` in `nova.equilibrium.flux_surface_extraction` | **Reuse.** They accept arbitrary `4 x 4` Bernstein blocks, so the evaluator is independent of whether coefficients came from the rejected local reconstruction or the global spline. |
| Edge roots | `_bicubic_edge_coordinates` and `_bicubic_edge_crossings` | **Reuse.** The current root finder is stronger than the original fixed endpoint-sign bisection: derivative-bounded intervals retain fixed capacities and detect same-sign root pairs. |
| Arc following | `_solve_bicubic_ordinate` | **Reuse.** Fixed-iteration, clipped Newton solves retain static shapes and follow the selected spline level set. |
| Curved area and moments | `_bicubic_arc_moment_correction`, `_integrate_bilinear`, and `_arc_rule(12)` from `nova.biot.arcbandedcoupling` | **Reuse.** The first function supplies oriented Green-theorem corrections and fixed quadrature samples; the second contracts them with the cellwise integrands. |
| Curved extrema | `_bicubic_stationary_point` plus `_masked_extremum` in the current extraction service | **Reuse.** Constrained fixed-iteration Newton and masked payload reductions preserve traced shapes for radial, vertical, and triangularity extrema. |
| End-to-end global curved clips | `_surface_clips` called by `extract_flux_surface_geometry` | **Reuse as the production seam.** It fits both normalized and physical maps globally, extracts cell blocks, qualifies topology, integrates spline arcs, and returns the fixed-shape transport record. |

The historical `_tensor_bicubic_coefficients` is deliberately excluded. It is the
centered-tangent local Catmull-Rom/Hermite producer whose surface authority the plan is
retesting. Its continued presence supports the isolated legacy
`_clipped_surface_geometry` route; it is not a candidate for the new measurement.

## 2. Global tensor-spline supply

`nova/linalg/tensor_spline.py` supplies all required surface information:

| Entry point | Supplied contract | Fitness verdict |
|---|---|---|
| `fit_tensor_spline(radial, vertical, values)` | Fits the full `values[vertical, radial]` map with global not-a-knot cubic solves in both axes. | **Use.** This is the weaker fitted-global-surface authority selected by the measurement goal. |
| `TensorBSpline.cell_coefficients` | Fixed `(vertical_cells, radial_cells, 4, 4)` Bernstein blocks, linear and differentiable with respect to the full sampled map. | **Use directly.** Reshape to `(cells, 4, 4)` as `_surface_clips` already does; no coefficient conversion is required. |
| `TensorBSpline.evaluate(radial, vertical)` | Returns `TensorSplineEvaluation(value, radial_derivative, vertical_derivative, radial_second_derivative, mixed_derivative, vertical_second_derivative)`. | **Use for pointwise checks and referees.** It supplies the value, both first derivatives, and the full symmetric Hessian requested by roots and constrained extrema. |
| `TensorBSpline.__call__` | Value-only paired-coordinate evaluation. | **Use only where derivatives are unnecessary.** The clip should otherwise share the coefficient evaluator so values and derivatives cannot diverge. |

`tests/test_tensor_spline.py` already verifies these entry points against SciPy on an
analytic map, ITER, and STEP; checks C2 continuity of the blocks; checks gradients with
respect to the source map; and checks eager/JIT/vmap fixed shapes. These are spline
fitness tests, not substitutes for the STEP clip gate.

## 3. Measurement harness and banked numbers

The exact bank source is the version of
`tests/test_transport_geometry_reference.py` at commit `47808904` and its parent. The
current file remains the correct harness but its characterization constants have moved
with later extraction-service changes, so the historical plan numbers must remain
quoted independently rather than being inferred from current constants.

| Harness symbol | Role | Fitness verdict |
|---|---|---|
| `_nova_input(filename, cocos)` | Reads and convention-normalizes ITER and STEP EQDSKs. | **Reuse.** It keeps all three routes on the same physical input. |
| `_nova_geometry(filename, cocos)` | Runs Nova's fixed-shape clipped extraction and adapts it to TORAX field names. | **Reuse.** On current main this reaches `extract_flux_surface_geometry`, whose curve comes from the global tensor spline. |
| `_contour_geometry(filename, cocos, rho_face)` | Builds Nova's independent host contour referee on the clipped route's face grid. | **Reuse.** It is the global `RectBivariateSpline` contour reader used to localize the clip discrepancy. |
| `_torax_geometry(filename, cocos)` | Builds TORAX's independent EQDSK geometry. | **Reuse.** It is the external second referee. |
| `_relative_error(actual, expected)` | Maximum absolute difference normalized by the maximum absolute referee magnitude, omitting the first two axis-dominated entries for arrays. | **Reuse unchanged.** This is the measure that generated the banked percentages. |
| `test_clipped_cells_match_independent_contour_geometry` | Measures clipped-to-contour fields for ITER and STEP. | **Reuse and report STEP explicitly.** It generated the decisive `vpr_face` and `g1_face` figures. |
| `test_nova_fsa_matches_torax_eqdsk_reader` | Measures clipped-to-TORAX fields for ITER and STEP and checks the four ITER coefficients. | **Reuse and retain all fields.** It also reports vertex capacity, which must remain `required <= capacity`. |

Quantitative bank to quote beside the new global-surface result:

| Comparison | Straight polygonal clip | Local bicubic clip at `47808904` | Gate or interpretation |
|---|---:|---:|---|
| STEP clipped-to-contour `vpr_face` | 16.46% | 18.14% | less than 3.92% |
| STEP clipped-to-contour `g1_face` | 9.78% | 22.31% | less than 6.20% |
| ITER clipped-to-TORAX `g2g3_over_rhon_face` | 2.63% | 2.96% | no regression in the bicubic characterization |
| ITER clipped-to-TORAX `g0_face` | 0.90% | 1.03% | no regression in the bicubic characterization |
| ITER clipped-to-TORAX `g1_face` | 2.08% | 2.36% | no regression in the bicubic characterization |
| ITER clipped-to-TORAX `g2_face` | 1.98% | 2.31% | no regression in the bicubic characterization |

The source constants underlying the rounded STEP values are
`_CONTOUR_CHARACTERIZATION_BASELINES["STEP_SPP_001_ECHD_ftop.eqdsk"]`: the
parent of `47808904` carries `vpr_face = 1.646e-1` and `g1_face = 9.776e-2`,
while `47808904` carries `1.8137e-1` and `2.2306e-1`. The parent ITER
`_CHARACTERIZATION_BASELINES` carries `2.628e-2`, `9.047e-3`, `2.082e-2`,
and `1.984e-2`; `47808904` carries `2.9602e-2`, `1.0296e-2`, `2.3632e-2`,
and `2.3136e-2`.

For the required three-way attribution, collect the same field arrays once from each
factory and report clipped-to-contour, clipped-to-TORAX, and contour-to-TORAX side by
side. Do not estimate the third side by subtracting percentage errors: the sup-norm
locations and referee scales can differ.

## 4. Fixed-shape contract consumed by transport

| Symbol or field | Contract | Fitness verdict |
|---|---|---|
| `AtomicCellMesh.traced_clip` / `_traced_clip` | Clips fixed node/cell arrays once per moving level and returns padded supports. | **Reuse without widening capacity.** The structured service uses `_SUPPORT_CAPACITY = 8`; the banked STEP requirement was 5 vertices, leaving 3 spare. |
| `TracedClippedSupports` | Stable named tuple carrying `support_vertices`, `vertex_count`, `centroids`, `included`, `boundary`, area/first/second moments, contour totals, two saddle branches, and saddle metadata. | **Read-only contract.** Keep field order and shapes stable; the curved geometry stays beside this polygonal topology record. |
| `TracedClippedSupports.qualify(participation)` | Zeros counts, masks, moments, and branch payloads outside the axis-connected topology and recomputes `patch_area_sum`. | **Reuse before any moment or extremum read.** It prevents disconnected lobes from masquerading as smooth weights. |
| `padded_polynomial_current_moments(...)` | Integrates one fixed-width complete polynomial over the padded support and returns zeroth and first current moments. | **Reuse unchanged for source transport.** The separatrix curve re-measurement must not alter this source-current contraction. |
| `ForwardSource.current_moments(..., support_moments, core_support, common_support)` | Consumes the clip and interpolant-over-support callback without owning clip construction. | **Read-only consumer.** Preserve the one-density-over-actual-support path and complementary domain selection. |
| `extract_flux_surface_geometry` result and `torax_geometry_from_fsa` | Fixed-shape dictionary adapted into transport geometry arrays. | **Read-only transport boundary.** The curve-source change belongs inside extraction, not in the transport adapter. |

The current global path composes curved Green corrections beside the polygonal support
rather than storing curved vertices in `TracedClippedSupports`. That separation is fit:
the polygonal support owns topology and padding, while spline roots, arc quadrature,
and constrained extrema own geometry. Repacking curved samples as support vertices
would unnecessarily change the consumer contract.

## 5. Explicit no-candidate lines

- **No candidate — separatrix sagitta curve generator.** `Arc._fit_leverage` is a circular-fit conditioning diagnostic, not a global tensor-spline level-set constructor; no such adapter is needed because the global spline supplies cell polynomials directly.
- **No candidate — acceptable local per-cell coefficient producer.** `_tensor_bicubic_coefficients` is precisely the rejected local surface authority. Use `fit_tensor_spline(...).cell_coefficients` instead.
- **No candidate — direct contour-to-TORAX comparison test.** The two factories and `_relative_error` exist, but no current test reports that pair directly; the measurement harness must compose it.
- **No candidate — curved-support storage type.** `TracedClippedSupports` intentionally stores fixed polygon topology and moments, not spline arc samples; keep curved corrections beside it.
- **No candidate — separate image-generation harness for the banked figures.** The cited STEP and ITER “figures” are scalar percentages printed and characterized by pytest, not plotted artifacts.

## 6. Recommended measurement composition

1. Use the current `_nova_input` once per machine and preserve the existing COCOS
   normalization.
2. Run `_nova_geometry` through `extract_flux_surface_geometry`; verify the route reaches
   `_surface_clips`, `fit_tensor_spline`, and `TensorBSpline.cell_coefficients`, not
   `_tensor_bicubic_coefficients` or `_clipped_surface_geometry`.
3. Build `_contour_geometry` on the identical `rho_face` and `_torax_geometry` at the
   existing 24-cell resolution.
4. Use `_relative_error` for all three pairings. Report at minimum STEP `vpr_face` and
   `g1_face`, the four ITER coefficient fields, and the clipped vertex
   `required / used / capacity` receipt.
5. Quote the straight and local-bicubic banks above beside the new result. The decisive
   verdict is whether the global curve closes the STEP gap; analytic spline tests remain
   primitive evidence only.

This composition reuses every available named capability, adds no production surface
authority, and leaves the transport-facing traced shapes unchanged.
