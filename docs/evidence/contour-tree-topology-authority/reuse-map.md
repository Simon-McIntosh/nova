# Contour-tree sections 2–4 — prior-art reuse map

Node `ctta-prior-art-scout`, plan `nova:contour-tree-topology-authority` §2–§4.
Read-only survey: no source was edited, no GPU and no SLURM job was used.

One row per capability the plan's §2–§4 need. Each row names candidate
implementations found by bounded greps over named subtrees, with `file:line`,
what it does, whether it is traceable under `jit`, and a verdict: **reuse**,
**adapt**, or **fails** (with why).

## Revisions read

| Tree | Revision read | Note |
|---|---|---|
| nova (this worktree) | `b2cc96f4589839fc27de317fe982b6e1ed6b9956` | rows cite this tree |
| nova (main checkout) | `5fcfc53f5390` | ahead of this worktree; not read |
| imas-efit | `a1028cda5175` | |
| imas-ambix | `2cede036085c` | |
| imas-codex | `ddec238562ca` | |

## Searched subtrees (each named; no repository root was walked)

- **nova** (`b2cc96f4`): `nova/`, `tests/`, `benchmarks/`, `scripts/`, plus the
  whole-tree `grep` over those four roots for the tree-vocabulary patterns.
- **imas-efit** (`a1028cda`): `src/EFIT/`, `src/EFUND/`, `fit/`→`efit/` (python
  package), `tools/`, `output/m2-build/mergetree_oracle/`,
  `output/m2-build/mergetree_battery/`.
- **imas-ambix** (`2cede036`): `imas_ambix/`, `tests/`, `scripts/`.
- **imas-codex** (`ddec2385`): `imas_codex/`, `scripts/`, `benchmarks/`.

Positive control for every "absent" verdict below: the same patterns were run
over `imas-efit/src/EFIT/` in the same session and matched the 4131-line Fortran
`contour_tree.f90`, so a null result in a subtree is the pattern finding nothing
there, not the pattern failing to match.

## Capability rows

| # | Capability (§) | Best candidate | Verdict |
|---|---|---|---|
| 1 | Triangulate hex cell centres (three-direction mesh) and clip that triangulation to the multi-unit vessel polygon (§2) | `nova/nova/biot/plasmagrid.py:382` `PlasmaGrid.tessellate`; `nova/nova/biot/sectionaverage.py:120` `_covered_triangles`; `nova/nova/geometry/hexstencil.py:39` `hex_stencil` | **adapt** (Delaunay + ring packing is whole; the clip is not) |
| 2 | True polygon containment over the wall units (§2) | `nova/nova/equilibrium/connectivity_boundary.py:233` `_points_inside_wall_units` (jnp); `nova/nova/equilibrium/wall_mask.py:68` `inside_polygon`; `nova/nova/geometry/pointloop.py:15` `point_in_polygon` (numba) | **reuse** (jnp multi-unit form) / adapt host forms |
| 3 | Add wall vertices on the polyline, flux linearly interpolated on the containing triangle (§2) | `nova/nova/equilibrium/wall_mask.py:433` `densify_units`; `nova/nova/equilibrium/connectivity_boundary.py:981` `_sample_wall_polyline`, `:2644` `_densify_wall`; triangle-linear flux only via `matplotlib.tricontour` (`nova/nova/imas/equilibrium.py:688`) | **adapt** (resampling exists; the containing-triangle flux lookup does not) |
| 4 | Join tree, split tree, and their merge into a contour tree (§2) | `imas-efit/src/EFIT/contour_tree.f90:130` `build_contour_tree`, `:1132` `build_contour_tree_with_wall`, `:3389` `merge_tree_decide`; oracle `imas-efit/output/m2-build/mergetree_oracle/merge_tree.py` | **adapt** (port the algorithm; nothing in nova) |
| 5 | Simulation-of-simplicity tie breaking (§2) | none named as such; closest `imas-efit/output/m2-build/mergetree_oracle/merge_tree.py:189` born-earlier union tie-break and `nova/nova/equilibrium/connectivity_boundary.py:136` `_arg_extreme` first-index tie-break | **fails — absent**, must be built |
| 6 | Fixed-shape `jit`/`vmap` graph construction with stated capacities (§2) | `nova/nova/equilibrium/parallel_components.py:77` `label_parallel_graph_components_with_steps`; `nova/nova/equilibrium/flux_surface_connectivity.py:329` `hex_edge_admissibility`, `:473`/`:495` saddle-aware labelling; `nova/nova/equilibrium/stencil_nulls.py:1743` `critical_point_candidates_batch` | **reuse** the kernels and the fixed-slot/overflow receipt |
| 7 | Brute-force connectivity count of superlevel sets (§2's done-when) | no counter exists; primitives are `nova/nova/equilibrium/flux_surface_connectivity.py:435` `label_hex_connected_components_with_steps`, `nova/nova/equilibrium/connectivity_boundary.py:721` `_axis_component_before_level`; analytic-region test pattern `nova/tests/test_hex_flood_geometries.py:151` | **fails — absent**, build from the labelling kernel |
| 8 | Newton polish and Hessian classification of critical points (§3) | `nova/nova/equilibrium/stencil_nulls.py:1803` `_refine_selected_vertices`; `nova/nova/equilibrium/flux_surface_connectivity.py:703` `_polish_stationary_points_in_bounds`; `imas-efit/src/EFIT/null_detection.f90:36` `find_all_nulls` | **reuse** the two nova jax polishes; the Fortran is the accuracy reference |
| 9 | Private-region and wall-contact reads (§4) | `nova/nova/equilibrium/flux_surface_connectivity.py:509` `private_flux_mask`; `nova/nova/equilibrium/topology.py:69` `private_wall_node_read`, `:621` `_axis_connected_wall_candidates`; `imas-efit/src/EFIT/contour_tree.f90:1673` `mark_private_subtrees` | **adapt** (private by connectivity reuses whole; the raster/height-band wall reads are the ones §5 deletes) |

## Detail per row

### 1 — Triangulating hex centres and clipping to the vessel polygon

- `nova/nova/biot/plasmagrid.py:382` **`PlasmaGrid.tessellate(target, wall)`** —
  scipy `Delaunay` over the hex filament centroids (line 385), six-neighbour
  rings recovered from `tri.vertex_neighbor_vertices` plus
  `hex_ring_slots` (`:29`), and the triangulation *trimmed to the wall* by a
  **centroid test**: `PointLoop(centroids).update(np.array(wall.xy).T)`
  (line 397) then `triangles = tri.simplices[inside]` (line 398), stored as
  `data["triangles"]` (line 405) beside the packed `stencil` (line 406).
  *What it is*: the only existing three-direction mesh build in nova, and the
  only place a Delaunay result is clipped to a wall polyline.
  *jit*: no — host scipy at mesh-build time; the output is a static integer
  array, which is what §2 wants (the tree build is jitted, the mesh is not).
  *Gap*: the clip is a centroid test, not a true polygon clip — a triangle
  straddling the wall whose centroid lands inside is kept whole, and the wall
  takes one `LineString`, not the multi-unit wall. **Verdict: adapt** — reuse
  the triangulation and ring packing; replace the centroid test.
- `nova/nova/biot/sectionaverage.py:120` **`_covered_triangles(polygon)`** —
  the repo's *true* triangulation-to-polygon clip: Shapely
  `constrained_delaunay_triangles` where available (line 125), falling back to
  `triangulate` + per-triangle `intersection` + re-triangulation (lines 146–155),
  keeping only triangles `polygon.covers(...)` and asserting the areas sum to
  the polygon area (line 156). Handles holes (`_polygon_parts`).
  *What it is*: exactly the "clip the triangulation to the polygon" operation,
  already proven against polygonal material area in the Biot section solver.
  *jit*: no (Shapely, host). *Gap*: it clips a triangulation of *the polygon*,
  not of an exterior point set, and it is per-polygon — the vessel-unit union
  would be assembled from its parts. **Verdict: adapt** — this is the clipping
  idiom to port to the wall-unit union; reuse its area-sum assertion as the
  coverage receipt.
- `nova/nova/geometry/hexstencil.py:39` **`hex_stencil(shape)`** and
  `:36` `HEX_RING` — closed-form six-neighbour ring indices (centre-first,
  `(cells, 7)`) for a structured raster, angle-ordered offsets. *jit*: yes —
  pure index arithmetic, consumed inside `stencil_mesh._normalised_ring`
  and in the jitted `_raster_hex_partition_geometry`. **Verdict: reuse** for
  the raster-lattice carrier; it is the contract the Delaunay route also
  satisfies (`hexstencil.py:24–28` states the two routes are one contract).
- `nova/nova/equilibrium/cell_partition.py:123` **`cell_partition_geometry`** —
  the bridge: exact tensor-product lattices take the raster adapter
  (`connectivity_boundary._raster_hex_partition_geometry`), everything else
  takes authored rings plus **reciprocal physical shared edges** from
  `_shared_polygon_edge` (`:43`). *What it is*: the already-built adapter that
  lets a non-raster carrier present the same ring/edge contract. **Verdict:
  reuse** — §2's carrier should present rings through this function.
- `nova/nova/equilibrium/stencil_mesh.py:635` **`StencilMesh`** — DERIVATIVE
  rings (≥6 cells, centre-first) used for gradient/`delta_star`, not a
  triangulation; `:743` `shared_node_flux_stencil` fits fixed weights that
  reconstruct flux at arbitrary shared nodes from the nearest complete ring.
  *jit*: yes (`_scatter`/`_apply` are `jnp`). *Gap*: it is an interpolator over
  rings, not a triangle mesh, so it cannot answer "which triangle contains this
  wall vertex". **Verdict: reuse** as the flux-reconstruction machinery if the
  triangle route is not taken; not as the containing-triangle lookup.

### 2 — True polygon containment over the wall units

- `nova/nova/equilibrium/connectivity_boundary.py:169`
  **`_points_inside_polygon(point_r, point_z, polygon_r, polygon_z)`** — `jnp`
  ray-cast including an explicit `on_boundary` tolerance band
  (`tolerance = max(1e-12, 16·eps·scale)`). *jit*: yes — it is written in
  `jnp` and used inside the jitted boundary reads. **Verdict: reuse** — this is
  the "true polygon containment, never a bounding box" primitive in the
  traceable form. Single polygon only.
- `nova/nova/equilibrium/connectivity_boundary.py:233`
  **`_points_inside_wall_units`** — the **multi-unit** form: vessel interiors
  minus material units, over a concatenated unit table with
  `wall_unit_offsets`, `wall_unit_closed`, `wall_unit_vessel`, using
  `_wall_segment_geometry` (`:215`) so distinct units are never joined.
  *jit*: yes. **Verdict: reuse** — this is capability 2 as §1 defines the
  domain, already jittable.
- `nova/nova/equilibrium/wall_mask.py:68` **`inside_polygon`** — fully
  vectorised numpy ray-cast over points×edges, "no shapely dependency";
  documented as written for the boundary-contour push testing hundred-vertex
  rings tens of times per slice (sub-ms vs tens-of-ms). *jit*: no (numpy), but
  elementwise and trivially portable. **Verdict: reuse** for host-side mesh
  build; **adapt** if it must enter a jitted path.
- `nova/nova/geometry/pointloop.py:15` **`point_in_polygon`** (numba `@njit`,
  returns 2 for on-edge) and `:79` `points_in_polygon`, wrapped by `:87`
  `PointLoop`. *jit*: no — numba, and it takes a Python loop per point.
  **Verdict: fails for a jitted read** (numba ≠ JAX); **reuse** only where the
  caller is host code, as `plasmagrid.tessellate` does.
- `nova/nova/equilibrium/steering_frames.py:540` `_inside_occupiable_region`
  and `nova/nova/media/sources/frame.py:56` `inside_wall_units` — two further
  host-side spellings of vessel-minus-material (media's also treats open
  material units as zero-area lines). **Verdict: adapt** — same semantics,
  useful as cross-checks (`inside_wall_units` is the presentation-layer twin
  whose agreement with the solver is already asserted).
- `nova/nova/equilibrium/shape_inverse.py:432` `_point_in_polygon` — host
  point/one-polygon with on-edge test, used to keep a refined stationary point
  inside the vessel. **Verdict: reuse** as the §3 containment check idiom
  (bounds + containment), not as the domain test.

### 3 — Wall vertices on the polyline, flux interpolated on the containing triangle

- `nova/nova/equilibrium/wall_mask.py:433` **`densify_units`** — resamples
  every unit surface at ~`d/2` arc spacing (module docstring lines 29–31:
  "Wall nodes are every unit's surface resampled at ~d/2 arc spacing").
  *jit*: no (host mesh build). **Verdict: reuse** — the polyline-vertex
  generator for §2's wall vertices, multi-unit aware.
- `nova/nova/equilibrium/connectivity_boundary.py:981` `_sample_wall_polyline`
  and `:2644` `_densify_wall(grid, m=720)` — the raster read's own wall
  sampling; `:762` `_wall_nodes_touching_region` binds each wall node to a
  region by nearest in-material raster node. *jit*: yes. **Verdict: adapt** —
  the sampling is reusable, the nearest-raster-node binding is the raster
  coupling §2 replaces with a containing-triangle lookup.
- The **containing-triangle flux lookup itself does not exist in nova**. The
  only triangle-linear flux evaluation is matplotlib's `tricontour`
  (`nova/nova/imas/equilibrium.py:688` `plot_tri`, `nova/nova/biot/plasmagrid.py:477`),
  i.e. a plotter, and `sectionaverage`'s triangles carry material moments, not
  field values. **Verdict: fails — must be written**; the input it needs
  (cell-centre flux + the clipped triangulation) is what rows 1 supplies.
- Cross-repo: `imas-efit/src/EFIT/contour_tree.f90:693` `bilinear_interp`,
  `:2467` `quadratic_subvertex_1d`, `:919` `insert_wall_contact`, `:2752`
  `insert_limiter_node` — sub-cell flux at wall contacts, and the
  `merge_tree_decide` docstring (lines 3419–3440) describes **virtual
  wall-contact nodes** carrying "the sub-node tangency of the SAME interpolant
  the boundary contour is extracted on", which are "pure OBSERVERS: they never
  union". That is a *different but already-proven* answer to the same problem
  (§2 asks for one virtual outside node valued below every interior level).
  **Verdict: adapt/reference** — worth weighing against §2's single outside
  node before §2 is implemented.

### 4 — Join tree, split tree and their merge into a contour tree

- **Nothing in nova.** The pattern (contour tree, join tree, split tree, merge
  tree, Morse, persistence) matched only two non-implementations:
  `nova/nova/imas/contour_tree.py` — a module-level scratch script that opens
  pulse 135013 at import, builds a `pyvista` contour and calls
  `plotter.show()`; not a function, not importable, no tree. And
  `nova/nova/imas/contour_tree_layout.py` — a design sketch of the DD
  `contour_tree` structure (`node`, `edge`, `levelset`, ordering rules) with
  the sentence "Develop layout for contour_tree within equilibrium ids" and a
  trailing empty `class tree`. **Verdict for both: fails** — no reusable
  implementation; the layout file is nonetheless a record of the DD mapping
  §1 constrains.
- `imas-efit/src/EFIT/contour_tree.f90:130` **`build_contour_tree`** — the
  module header (lines 25–48) states the algorithm: detect critical points via
  `null_detection.find_all_nulls`, then "for each saddle, determine
  connectivity by gradient descent/ascent from its 4-connected grid
  neighbours", build a spanning tree with **Kruskal + union-find** to avoid
  cycles, connect isolated nodes. Helpers: `:474` `grid_descend`, `:516`
  `grid_ascend`, `:600` `uf_find`, `:614` `uf_union`. *jit*: not applicable
  (Fortran); it is a host algorithm on a `(nw, nh)` array + `inside` mask.
  **Verdict: adapt** — this is a working join/split/contour tree for exactly
  this field; the port target is §2, with the union-find sweep restructured to
  the fixed-shape labelling kernel of row 6.
- `imas-efit/src/EFIT/contour_tree.f90:1132` **`build_contour_tree_with_wall`**
  (+ `:742` `find_wall_contacts`, `:919` `insert_wall_contact`, `:3095`
  `merge_tree_inside_polygon`, `:3166` `merge_tree_inside_composite`) — the wall
  inside the tree. **Verdict: adapt** — §2's "connect every wall vertex to one
  virtual outside node" is this problem; the existing code masks cells inside a
  polygon rather than clipping a triangulation.
- `imas-efit/src/EFIT/contour_tree.f90:3389` **`merge_tree_decide`** — the
  module header at lines 58–62 says it is the "Geometry-free merge-tree
  (split-tree) limited-vs-diverted decision … Union-find sweep in outward-flux
  order; saddle = ≥2 distinct neighbour component roots (topological, no
  Hessian discriminant); the axis component's FIRST event (interior
  saddle-merge ⇒ DIVERTED, direct wall touch ⇒ LIMITED) is the boundary. This
  is the production reeb decision." Union-find carries `born` and `axis`
  (`:4105` `mt_uf_union`), with `mt_add_root` (`:4072`). **Verdict: adapt** —
  this is §4's "first event" rule, already implemented and in production use;
  the plan's §4 re-derives it, so the port should be checked against it rather
  than re-derived blind.
- `imas-efit/output/m2-build/mergetree_oracle/merge_tree.py` — the readable
  Python A/B oracle for the same decision (header lines 1–60 give the five
  steps; README shows 6.3e-6/5.3e-6 Wb agreement with the truth on TCV and
  HL-3). The README states it is a development/validation aid that "NEVER
  enters live `src/`". **Verdict: reuse as the §2/§4 reference implementation
  and test oracle** — it is the readiest executable statement of the merge-tree
  semantics, even though the plan needs a jitted nova-native build.
- `imas-ambix/imas_ambix/latent/topology.py:121` **`find_critical_points`**,
  `:638` `emergent_xpoints`, `:680` `classify_regions` (scipy `ndimage`
  connectivity for CORE/PUBLIC/PRIVATE), `:871` `_inside_polygon` — a numpy
  topology read whose module docstring says it was "steered by NOVA
  `biot/fieldnull.py` … and imas-efit `src/EFIT/contour_tree.f90`". No join
  tree. **Verdict: adapt** for the private/public region semantics and the
  `TopologyReadout` shape; it is not a tree.

### 5 — Simulation-of-simplicity tie breaking

- **No implementation anywhere searched**, in any of the four repositories:
  the pattern `simulation of simplicity` / `simplicity` returns nothing, and
  the ordering machinery that does exist is ad hoc:
  - `imas-efit/output/m2-build/mergetree_oracle/merge_tree.py:189` — union by
    rank with a documented "tie-break to the born-earlier root (older
    extremum)"; `mt_uf_union` carries `born` for the same purpose.
  - `imas-efit/output/m2-build/mergetree_battery/run_adversarial.py:499–572`
    case **A4** — the degenerate twin-column fixture: `A4sym` puts two
    axis-merge saddles at *identical* ψ on consecutive sweep steps and requires
    a deterministic (stable-sorted) first choice; `A4pert` deepens one well 1%
    to prove the tie-break is by genuine outward ψ. The case docstring is the
    clearest existing statement of the failure mode simulation of simplicity
    exists to remove.
  - `nova/nova/equilibrium/connectivity_boundary.py:136` `_arg_extreme` (used
    by `:159` `_argmax_exact` / `:164` `_argmin_exact`) — "the first maximum
    index without a default-dtype seed": deterministic, but first-index rather
    than value-perturbation.
  **Verdict: fails — absent; build it.** A `(σψ, index)` lexicographic key, or
  an explicit ε-perturbation, is new code, and the A4 fixture pair is a
  ready-made negative control (A4sym must not depend on column order; A4pert
  must select the deeper column).

### 6 — Fixed-shape `jit`/`vmap` graph construction with stated capacities

- `nova/nova/equilibrium/parallel_components.py:77`
  **`label_parallel_graph_components_with_steps`** — the hook-and-compress
  parallel component labelling, fixed iteration cap, returning
  `(labels, steps, settled)`. *jit*: yes. **Verdict: reuse** — the engine any
  level-set connectivity read is built on.
- `nova/nova/equilibrium/flux_surface_connectivity.py:329`
  **`hex_edge_admissibility`** — fixed-shape mask of hex links open at a level:
  evaluates each shared edge's own endpoints and a fixed-iteration stationary
  search (7 seeds × `stationary_steps`) and closes a link whose axis-side
  coverage is zero-measure. This is the sublevel-set link predicate a join-tree
  sweep needs, already fixed-shape. *jit*: yes (`static_argnums`).
  **Verdict: reuse.**
- `nova/nova/equilibrium/flux_surface_connectivity.py:473`/`:495`
  `label_saddle_aware_hex_connected_components[_with_steps]`, `:435`/`:464`
  `label_hex_connected_components[_with_steps]` — labelling on centre-first
  six-neighbour rings, with `n_iter` the caller's static cap and the documented
  bound `ceil(log2(cell_count)) + 2`. *jit*: yes. **Verdict: reuse** — this is
  the "fixed-shape graph construction" precedent, and its static-cap contract
  is the "stated capacity" pattern §2 asks for.
- `nova/nova/equilibrium/stencil_nulls.py:1743` **`critical_point_candidates_batch`**
  and `:1675` `_critical_point_candidates_batch` — fixed `k_slots` per state,
  returning `overflow`, `work_overflow`, `work_capacity` (lines 1659–1661)
  alongside `candidate_count` / `discarded_score_upper_bound` so a saturated
  census is *visible on the receipt* rather than silently truncated; the
  docstring at `:1756–1758` says exactly that. *jit*: yes, batched/vmapped.
  **Verdict: reuse as the capacity-and-refusal pattern** §2's "overflow refused
  visibly on the receipt" should copy field-for-field.
- `nova/nova/equilibrium/topology.py:621` `_axis_connected_wall_candidates`
  (docstring: "Preserving the fixed shape under `jit` and `vmap`") and `:1411`
  `read_batch` / `:1418` `update_batch` — the batched read surface.
  **Verdict: reuse** for the batching contract; `:158` `_carrier_polish_layout`
  is the same seam for carrier-aware polish.
- `nova/nova/equilibrium/connectivity_boundary.py:642` `_linear_flood_fill_core`
  (+ `flux_surface_connectivity.py:193/:231` `flood_fill_core[_with_steps]`) —
  fixed-iteration floods. **Verdict: reuse** as the reference/fallback fill.
- Fixed capacities stated as module constants: `nova/nova/equilibrium/flux_surface_connectivity.py:685`
  `EDGE_CROSSING_CAPACITY = 4`, `nova/nova/equilibrium/stencil_mesh.py:518` /
  `nova/nova/equilibrium/clip_quadrature.py:380 triangle slots. **Verdict: adapt** — the naming
  precedent for §2's stated node and edge capacities.

### 7 — Brute-force connectivity count of superlevel sets

- **No counter exists** in nova, imas-efit, imas-ambix or imas-codex: the
  closest machinery answers "is the axis component connected" rather than "how
  many components are there at this level".
  - `nova/nova/equilibrium/connectivity_boundary.py:721`
    `_axis_component_before_level` — confines at `u <= level - offset`, seeds
    at the nearest in-material cell to the axis, and returns a single component
    via `:707` `_saddle_aware_axis_component`. A count would be a one-line
    change over the same labels but does not exist. **Verdict: adapt.**
  - `nova/nova/equilibrium/flux_surface_connectivity.py:435`
    `label_hex_connected_components_with_steps` gives labels from which a count
    is `max(labels)`; no caller does that for a superlevel set.
    **Verdict: reuse the kernel, build the counter.**
  - `nova/tests/test_hex_flood_geometries.py:151`
    `test_saddle_aware_hex_flood_matches_analytic_geometry` with the
    `ManufacturedGeometry` battery (`:34` limited circle, `:50` lower single
    null, `:71` upper/lower double null, `:98` interior saddle with a wall
    notch) — the *existing pattern* for an independent region oracle:
    closed-form flux plus analytic core/private masks. **Verdict: reuse as the
    template** for §2's brute-force check on the Solov'ev fixtures.
  - `imas-efit/output/m2-build/mergetree_battery/battery.py:77`
    `critical_points_on_R_line(..., n=400001)` and `:125` `true_X_points` —
    a dense-sampling analytic truth for critical points;
    `.../mergetree_battery/truth_oracle.py` is the same idea generalised.
    **Verdict: adapt** as the analytic-comparison pattern.
  **Absence note:** the plan's done-when ("the tree matches an independent
  brute-force connectivity count of the superlevel sets at every critical
  level") therefore needs new code — a level sweep calling the existing
  labelling kernel. It is cheap; it simply is not there.

### 8 — Newton polish and Hessian classification of critical points

- `nova/nova/equilibrium/stencil_nulls.py:1803` **`_refine_selected_vertices`** —
  fits a 3×3 quadratic in a fixed 9-point cluster, solves for the stationary
  point from the fitted coefficients (`h00·h11 − e²` determinant guard,
  clamped to ±4 cells), evaluates the fitted flux there, and classifies by the
  Hessian: `ntype = -1` minimum, `+1` maximum, `0.0` for `determinant < 0`
  (saddle). Fully differentiable, fixed-capacity, batched. This is the
  "per-cell own-node quadratic" the plan's §3 open decision cites as measuring
  9/9 saddles and 15/15 axes. *jit*: yes. **Verdict: reuse** — one of the three
  named decision options, and the only one with a measured record on the hex
  carrier.
- `nova/nova/equilibrium/flux_surface_connectivity.py:703`
  **`_polish_stationary_points_in_bounds`** — fixed-slot Newton on ∇ψ = 0
  against a smooth spline, masked-lane convergence, caller-supplied coordinate
  bounds (so a polished node can be kept inside the vessel), differentiable via
  `custom_root`. *jit*: yes (`static_argnums`). **Verdict: reuse** — the §3
  polish-and-contain implementation, and the only existing one that polishes on
  a *smooth* representation with in-vessel bounds.
- Supporting census code in the same module: `:115` `ring_sign_changes`,
  `:175` `gradient_cell_degree`, `:1849` `xpoint_candidates`,
  `:1891` `magnetic_axis_subgrid` (polarity-aware extremum selection — the
  reversed-current case §4's last done-when needs). **Verdict: adapt** — these
  are the reads §3 replaces as the source of critical points, but they are the
  comparison arm and the source of seeds.
- `imas-efit/src/EFIT/null_detection.f90:36` **`find_all_nulls`** family — the
  8-neighbour cyclic sign-change census, quadratic patch via LAPACK `DGELS`,
  **Newton on ∇ψ = 0 against a bicubic `EZspline2`**, then a Hessian re-test at
  the refined position with disagreements discarded (module header lines
  1–36). Also `quadratic_subnull`, `select_primary_opoint`,
  `hessian_classify`, `newton_refine_null[_seeded]`, `find_min_gradient_seed`.
  **Verdict: reuse as the accuracy reference** — its two-stage (fit, then spline
  Newton) design is the strongest available evidence for what §3's polish
  should achieve, and it is the "placement accuracy < 1e-10 of grid spacing"
  claim the plan's 1e-9 Wb agreement target is comparable to.
- `imas-efit/efit/boundary_extraction.py:272` **`_scan_newton_saddles`**
  (python) — refined gradient-sign scan bracketing each quad-change, Newton via
  `scipy.optimize.root` on the analytic gradient, central-difference Hessian
  classification `det(H) < 0`; `:538` `find_saddles` exposes it per field
  (`BicubicField` at `:223` wraps `RectBivariateSpline` as "the Python proxy for
  the solver's EZspline psi_spl frame"). **Verdict: adapt** — the cleanest
  executable statement of scan→Newton→Hessian, in python, at grid resolution.
- `imas-ambix/imas_ambix/latent/topology.py:121` **`find_critical_points`** —
  grid bracket on the sign change of both partials, Newton on the local
  bilinear gradient/Hessian, classify by Hessian definiteness, `:222` `_dedup`.
  `imas-ambix/imas_ambix/latent/stencil_nulls.py:108` `subnull`, `:195`
  `magnetic_axis_subgrid`, `:238` `xpoint_candidates` are the same
  fixed-slot design as nova's `stencil_nulls` (ambix's copy/precursor).
  **Verdict: adapt/reference** — a second independent implementation to
  cross-check the census against.
- `nova/nova/biot/fieldnull.py:16` `DataNull` / `:161` `FieldNull`, `:207`
  `categorize_1d`, `:239` `categorize_2d`, `:99` `_subnull_2d` — the legacy
  host null finder both ambix's docstring and `nova/nova/geometry/hexstencil.py:11` cite.
  **Verdict: fails for the new read** (host, raster-indexed, `:79` `_unique`
  de-duplicates by rounding to 3 decimals) but it is the historical baseline
  the plan's defects were measured against.

### 9 — Private-region and wall-contact reads

- `nova/nova/equilibrium/flux_surface_connectivity.py:509`
  **`private_flux_mask(component_labels, axis_seed)`** — "return labelled
  confined cells disconnected from the magnetic axis": a confined cell is
  private iff its component label differs from the axis label. *jit*: yes.
  **Verdict: reuse** — this is exactly §4's private-region read, and §2's tree
  generalises it (a private region is a branch joining at the primary
  X-point).
- `nova/nova/equilibrium/topology.py:69` **`private_wall_node_read`** — reads
  private wall flux at *each* node independently of cell ownership, combining
  an admitted-saddle flux-side test with a height band from
  `x_point_height_limits` (`:52`). *jit*: yes. **Verdict: adapt** — the
  per-node independence is the good part; the height-band construction is the
  "shadowed" heuristic §4 replaces with "a wall contact inside a private region
  can never be a limiter".
- `nova/nova/equilibrium/connectivity_boundary.py:410`
  **`wall_height_shadow_mask`** — "the hysteretic private-wall exclusion from
  qualified saddles", plus `:373` `_print_wall_height_eligibility`.
  **Verdict: fails for the new authority** — this is the shadow machinery the
  plan's §5 retires (and §0 blames for "a shadowed contact sets the wall level
  to infinity").
- Wall-contact readers to be replaced: `:762` `_wall_nodes_touching_region`,
  `:781` `_wall_nodes_in_line_of_sight` (the geometric sightline heuristic §0
  names as defect 2), `:840` `_masked_wall_reachability`, `:1020`
  `_select_reachable_wall_limiter`. **Verdict: fails** — these are the reads
  §5 deletes; listed here so §4 does not accidentally reuse them.
- `nova/nova/equilibrium/topology.py:621` `_axis_connected_wall_candidates`,
  `:530` `_wall_anchor_candidates`, `:666` `wall_anchor_bracket`, `:670`
  `wall_anchor_data` — the current wall anchor selection with containment
  screening. **Verdict: adapt** — the anchor *brackets* and the containment
  screen are reusable; the selection rule is the thing §4 replaces.
- `imas-efit/src/EFIT/contour_tree.f90:1673` **`mark_private_subtrees`**
  ("Polarity-aware private-flux classifier"), `:1898`
  `qualify_separatrix_saddles`, `:1489` `saddle_is_on_main_plasma_lcfs`,
  `:1394` `path_avoids_saddles`, `:1593` `node_reachable`, `:2067`
  `saddle_gates_closed_inboard_surface`, `:2293` `classify_wall_basin`,
  `:2389` `classify_wall_polygon`, `:2192` `select_axis_node`. **Verdict:
  adapt** — a complete, production private/limiter classifier over a tree, and
  the closest thing to §4 that exists anywhere on this workstation.
- `imas-efit/output/m2-build/mergetree_oracle/merge_tree.py` step 5 — per
  wall-contact node `PUBLIC` iff it joins the axis component *before* its first
  saddle-merge, else `PRIVATE` (README §5, with the divertor private-flux case
  called out as "exactly where the old gradient-basin heuristic failed").
  **Verdict: reuse as the §4 reference semantics.**
- `imas-ambix/imas_ambix/latent/topology.py:680` **`classify_regions`** —
  CORE / PRIVATE / SOL by connectivity at the boundary flux, REGION codes at
  `:52–58`; `:280` `xpoint_set`, `:313` `boundary_flux`,
  `:338` `boundary_flux_robust`. **Verdict: adapt** — the connectivity-based
  private/public split, and `boundary_flux_robust` is a second opinion on the
  boundary level.
- `nova/nova/equilibrium/domain.py:194` **`classify_domains`**, `:172`
  `axis_connected_component`, `:222` `saddle_qualified_domains`, `:73`
  `PlasmaDomain` — the domain split the hex-flood battery already validates
  against analytic geometry. **Verdict: adapt** — the mask vocabulary §4's
  regions should keep speaking until §5 cuts over.

## What this map means for dispatch

- §2 can start today on three quarters of its work: the mesh and ring contract
  (rows 1, 6) and the connectivity kernels (rows 6, 7) are here and jittable.
  The two genuinely new pieces are the containing-triangle flux lookup (row 3)
  and simulation of simplicity (row 5).
- §4's definitions already exist as a working implementation in
  `imas-efit/src/EFIT/contour_tree.f90` plus its Python oracle. Porting the
  *semantics* is a reading task; re-deriving them is not required, and the
  oracle's two real-field self-tests (Δψ ≈ 5–6e-6 Wb) are a ready comparison.
- §3 has two viable in-repo jax polishes (row 8) and a strong accuracy
  reference (`null_detection.f90`), so the open decision
  `field-representation-on-hex-cells` can be settled against measurements
  rather than argument.
- Everything the plan's §5 deletes is listed in row 9 as **fails** so that
  §4's implementation does not inherit it by accident.