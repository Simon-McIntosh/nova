# Extraction seam map: spline geometry, chord clipping, and drawn contours

## Headline finding

There is no single Nova LCFS polyline shared by the solve, the production MAST
scorecard, transport geometry, and the corroboration figure. They currently use
four different geometric contracts:

| Consumer | Boundary representation it actually sees | Consequence |
| --- | --- | --- |
| Newton/fixed-point plasma-current map | A fixed atomic cell is clipped by `_traced_clip`; each edge crossing is a linear interpolation between shared-node flux samples, so the in-cell boundary is a straight chord. The samples themselves come from a quadratic reconstruction of the grid flux. See `nova/equilibrium/forward_operator.py:633-640`, `nova/equilibrium/separatrix_clip.py:612-621`, and `nova/equilibrium/stencil_mesh.py:645-670`. | Boundary-cell current changes with the chord-clipped support inside every residual evaluation. This path does **not** consume the global tensor spline. |
| Transport/flux-surface geometry extraction | `_surface_clips` fits the complete structured map with global not-a-knot tensor splines, takes their per-cell 4×4 Bernstein blocks, and corrects the provisional chord moments to the curved spline level set. See `nova/equilibrium/flux_surface_extraction.py:1947-1974`, `nova/equilibrium/flux_surface_extraction.py:2023-2062`, and `nova/linalg/tensor_spline.py:1-7`. | The consumer receives spline-corrected integrals, coarea samples, and extrema—not a reusable LCFS polyline. |
| Production MAST parity geometry (`TopologyLabels.lcfs_m`) | Eight fixed-angle radii found by a 512-sample ray march through a **bilinear** interpolation of the solve grid, assembled into an eight-vertex ring. See `nova/equilibrium/labels.py:22-42`, `nova/equilibrium/connectivity_boundary.py:231-288`, and `nova/imas/mast_chain_factory.py:68-95`, `nova/imas/mast_chain_factory.py:113-147`. | This is sub-grid in radius along each ray, but it is neither a marching-squares contour nor global-spline geometry. It is deliberately a fixed-shape geometry label. |
| EFIT corroboration figure and its boundary metrics | Matplotlib `contour` traces the selected binding level directly on the grid-valued `radius`, `height`, and `flux` arrays; one returned path is ranked and used as the Nova boundary. See `docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py:104-142` and `docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py:488-505`. | This is the grid-resolution polyline visible in the twelve panels and used by their sup/RMS metrics. It is not `TopologyLabels.lcfs_m` and is not `_surface_clips` output. |

The key scope result is therefore: **the coarse/wavy Nova curve in the
corroboration panels is a property of the figure-and-metric extraction path,
while the solve's boundary-cell current still has a separate straight-chord
geometric approximation.** The global spline already serves transport geometry
and wall-flux reads, but it does not currently supply either of those two
polylines.

## Mechanism and data-flow map

```text
trial total-flux vector ψ
  │
  ├─ topology read → boundary flux ψb + selected saddle / limiter
  │                    │
  │                    ├─ production MAST label
  │                    │    bilinear ray crossings → 8-point lcfs_m ring
  │                    │
  │                    └─ corroboration report
  │                         grid ψ + ψb → Matplotlib marching-squares path
  │                         → rank path near saddle/limiter → sup/RMS + figure
  │
  ├─ Newton map / residual
  │    quadratic shared-node flux → signed flux at atomic vertices
  │    → straight-edge _traced_clip → clipped support
  │    → quadratic in-cell flux → profile current density quadrature
  │    → zeroth + first current moments → Green response → g(ψ) → ψ - g(ψ)
  │
  └─ transport/FSA extraction (separate post-solve consumer)
       global C2 tensor spline → per-cell Bernstein blocks
       → provisional chord clip + bicubic arc correction
       → volume/coarea/shape columns (no contour polyline)
```

## Seam-by-seam anchors

### 1. Binding topology is selected before any display contour exists

For the corroboration bank, `_post_cutover_geometry` obtains the exact typed
saddle table and reachable-wall operand, determines the achieved class, and
chooses the binding flux from the selected saddle for a diverted result or from
the limiter for a limited result. It returns both the scalar binding flux and
the selected saddle coordinate at
`docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py:199-244`.
The solve-grid array later drawn is independently unpacked from the physical
state at
`docs/figures/primary-xpoint-evidence/real_equilibria_reachability.py:185-219`.

This separation matters: the topology read supplies **which flux level binds**
and **which point should anchor it**; it does not supply the vertices plotted by
the figure.

### 2. The corroboration figure manufactures its own grid-resolution polyline

`_binding_contour` calls `axis.contour(radius, height, flux,
levels=[boundary_flux])`, copies the returned path vertices, and ranks candidate
paths by axis containment, distance to the selected boundary point, and length
at `docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py:104-142`.
The main loop selects the saddle or limiter as the ranking point, calls that
helper, and immediately feeds its result to `_boundary_distances` at
`docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py:488-505`.
The same vertices are serialized as `nova_binding_contour_m` at
`docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py:537-558`
and drawn at
`docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py:426-454`.

Thus both the red line and the reported sup/RMS values see the same
marching-squares polyline. Resampling in `_boundary_distances` densifies its
segments for nearest-neighbour scoring but cannot recover geometry absent from
the original contour vertices
(`docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py:93-101`).

### 3. Production `lcfs_m` is a different, fixed-shape polyline

The connectivity read deliberately avoids general contour extraction. It
evaluates each ray sample with `_bilerp`, detects the first level crossing, and
linearly interpolates only along the ray at
`nova/equilibrium/connectivity_boundary.py:231-288`. The boundary read clamps
the emitted ring to at most 0.999 of the separatrix span so a ray cannot escape
down an open divertor leg
(`nova/equilibrium/connectivity_boundary.py:1131-1139`).

The MAST labeler sets 512 radial samples at
`nova/imas/mast_chain_factory.py:34-37`, invokes this read twice at the eight
fixed `LCFS_ANGLES`, then converts the eight radii into points about the refined
axis and publishes those points as `TopologyLabels.lcfs_m` at
`nova/imas/mast_chain_factory.py:68-95` and
`nova/imas/mast_chain_factory.py:113-147`. `run_parity_chain` carries that exact
array into the geometry scorecard at
`nova/imas/mast_parity_chain.py:626-658`.

The production ring is therefore sparse in poloidal angle (eight vertices) but
sub-grid along each ray. Calling the corroboration contour “the production
LCFS” is inaccurate: the corroboration path bypasses this labeler entirely.

### 4. `_traced_clip` is a straight-chord support clip

The clipper receives one signed-flux value per shared atomic node. For each
polygon edge it tests endpoint signs and places the crossing at

`start + start_flux / (start_flux - end_flux) * (end - start)`

at `nova/equilibrium/separatrix_clip.py:583-621`. The ensuing support polygon
and its area/first/second geometric moments are assembled from those straight
segments at `nova/equilibrium/separatrix_clip.py:656-745`. Nothing in this
function evaluates a tensor spline or bends a crossing-to-crossing segment.

There is already cell-local saddle handling: four edge crossings infer the
intersection of two chord lines (or accept a supplied `saddle_vertex`) at
`nova/equilibrium/separatrix_clip.py:633-654`, then the support is divided into
two fixed-capacity branch polygons at
`nova/equilibrium/separatrix_clip.py:697-740`. Those branches are geometric
support for one saddle-cut cell, not ordered global separatrix polylines.

### 5. Where the quadratic flux/current cell consumes the clipped boundary

The precise production semantics are slightly narrower than the shorthand
“quadratic-current cell”:

1. `MomentGeometry.from_cells` atomises the plasma polygons and builds a shared
   node-flux stencil at `nova/equilibrium/stencil_mesh.py:267-344`.
2. That stencil evaluates the same complete quadratic ring fit used by the mesh
   derivatives at arbitrary atomic nodes
   (`nova/equilibrium/stencil_mesh.py:645-670`).
3. Every map evaluation reads the topology, evaluates shared-node flux, forms
   `signed_flux = polarity * (shared_flux - boundary_flux)`, and calls
   `atomic_mesh.traced_clip` for complementary core/common supports at
   `nova/equilibrium/forward_operator.py:620-640`.
4. For each carried cell, `support_flux_moments` fits an own-node quadratic flux
   polynomial and passes the **clipped support vertices and vertex count** into
   the current integrator at `nova/equilibrium/stencil_mesh.py:142-177`.
5. The fixed quadrature evaluates that quadratic flux inside the clipped polygon,
   evaluates the profile current density there, and integrates current plus its
   first spatial moment at `nova/equilibrium/stencil_mesh.py:406-473`.
6. The source selects those support moments by topology domain at
   `nova/equilibrium/source.py:608-679`; the operator converts them into the
   plasma flux image and subtracts that image in the free-boundary residual at
   `nova/equilibrium/forward_operator.py:837-866`.
7. Both the host Newton-Krylov residual and the traced Newton-Krylov route invoke
   this map: `nova/equilibrium/forward.py:1082-1108` and
   `nova/equilibrium/forward.py:1117-1133`.

So the clipped boundary is consumed **inside the Newton residual**, before the
plasma Green response is formed. A boundary move changes the integration domain
and hence the current moments returned by `g(ψ)`.

Qualification: the quadratic object here is the reconstructed **flux field**
used to evaluate current density within a cell. The carried current response is
only zeroth plus radial/vertical first moments
(`nova/equilibrium/stencil_mesh.py:99-108`); it has no second-current-moment
channels. This distinction should remain explicit in later design text.

### 6. `_surface_clips` is the global-spline curved-geometry consumer

`_surface_clips` fits both normalized and physical flux using
`fit_tensor_spline` and extracts their global per-cell 4×4 Bernstein blocks at
`nova/equilibrium/flux_surface_extraction.py:1947-1974`. The spline itself is a
global not-a-knot cubic tensor fit whose cell blocks are derived from the whole
sampled map
(`nova/linalg/tensor_spline.py:121-146`,
`nova/linalg/tensor_spline.py:251-271`).

The route still uses `_traced_clip` as a provisional fixed-capacity polygon at
`nova/equilibrium/flux_surface_extraction.py:2023-2032`, but then replaces the
chord geometry in the relevant moments with `_bicubic_arc_moment_correction`
using the global spline coefficients at
`nova/equilibrium/flux_surface_extraction.py:2034-2089`. That correction finds
bicubic edge roots, follows the curved ordinate, and constructs arc-versus-chord
moment corrections at
`nova/equilibrium/flux_surface_extraction.py:554-690` and
`nova/equilibrium/flux_surface_extraction.py:800-829`.

The resulting surface samples feed coarea averages and geometric extrema
(`nova/equilibrium/flux_surface_extraction.py:2099-2159`,
`nova/equilibrium/flux_surface_extraction.py:2232-2295`) and are assembled as
transport/FSA columns. The public extractor enters `_surface_clips` at
`nova/equilibrium/flux_surface_extraction.py:2629-2642`. It does not return a
single ordered boundary vertex array, so the corroboration figure cannot reuse
it as-is.

One nearby spline use should not be confused with contour tracing: the
connectivity read evaluates a global tensor spline at wall points to refine the
wall binding flux at `nova/equilibrium/connectivity_boundary.py:857-881`. That
improves the scalar limiter operand; the published LCFS radii still come from
the bilinear `_ray_radii` path.

## X-point split attachment

The clean attachment is in the **extraction consumer**, after the topology read
has supplied `(binding_flux, selected_saddle)` and before any boundary metric or
plot consumes a polyline:

1. Keep `_post_cutover_geometry` as the authority for achieved class, binding
   flux, and selected saddle
   (`docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py:227-244`).
2. Replace the `_binding_contour` call at
   `docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py:497-505`
   with a production extraction API that traces the binding level on the global
   tensor spline at an independently chosen extraction resolution.
3. For diverted states, seed the split with the selected saddle, orient the
   incident branches, and return a typed fixed-shape result such as
   `closed_boundary`, `open_leg_a`, and `open_leg_b`. For limited states, return
   the single wall-binding closed boundary and empty leg branches.
4. Feed **only `closed_boundary`** to `_boundary_distances`; retain the two open
   legs for plotting and strike-point consumers. The current scoring call site
   that needs this substitution is
   `docs/figures/primary-xpoint-evidence/efit_topology_corroboration.py:501-505`.

Do not attach the global split inside `_ray_radii`: that API intentionally
returns one radius per fixed angle on a surface just inside the separatrix and
contains no branch connectivity. Do not treat `_traced_clip`'s existing
`branch_support_vertices` as the requested separatrix branches either: those
are per-cell support polygons for current integration
(`nova/equilibrium/separatrix_clip.py:708-740`). They are useful prior art for
fixed capacities and saddle insertion, but a global contour tracer still has to
order and connect cell arcs across the full surface.

For the Newton-current path, a later curvature change attaches separately at
`ForwardFluxOperator._support_partition`: replace or correct the straight-chord
support produced at `nova/equilibrium/forward_operator.py:633-640` with the same
global-surface boundary authority while preserving fixed capacities. That is a
residual-domain change, not a plotting change, and requires independent
conservation and convergence validation.

## Design consequences

- The corroboration figure can be made spline-precise without changing the
  Newton residual, but that alone does not improve solve physics; it only fixes
  geometric extraction and scoring.
- Curving the Newton support changes plasma current within every boundary cell
  and therefore changes the nonlinear map itself. It must be treated as a
  solver/current-representation change, even if it reuses the same spline
  contour authority.
- Production `TopologyLabels.lcfs_m` is an intentionally small fixed-shape
  label. A high-resolution closed boundary plus open legs should be a separate
  typed extraction product rather than silently changing the eight-radius
  scorecard contract.
- `_surface_clips` proves that global-spline Bernstein blocks, fixed boundary
  bands, and fixed-capacity arc samples already coexist under JAX. Its output
  shape is integral/diagnostic data, however; it is reusable machinery, not an
  existing separatrix-polyline API.
