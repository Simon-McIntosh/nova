# Biot-Savart section-route guidance

This guidance applies to section-kernel selection, authored conductor geometry,
and FrameSpace access inside `nova/biot/`. Repository-wide development and git
rules remain in the root `AGENTS.md`.

## Preserve the exact production route

- `PolySectionPolicy()` defaults to the JAX closed-form ring lane. `Solve`
  selects `TiledPolySection`, and runtime JAX placement chooses CPU or GPU for
  the same fixed-shape graph. The scalar NumPy reduction remains the independent
  correctness reference, not a second production selector.
- `PolySection` integrates each source element's authored `poly`; full hexagons,
  clipped plasma cells, and non-rectangular conductors stay polygons. Do not
  replace authored material with a bounding rectangle or a point filament.
- Boundary quadrature remains an exact reference and compiled-device route.
  Approximate banded, standoff, and filament variants belong only in dedicated
  measurement modules and must not become production selectors.
- Route policy is immutable at machine construction and part of the machine and
  source-batch cache identity. A different kernel, backend, precision, device
  eligibility, or quadrature rule must produce a different semantic key.

## Enforce authored shape at construction

- `Coil._route_authored_sections` and
  `nova.frame.firstwall.PlasmaGrid._route_rectangular_cells` are build-time
  guards. They retain the rectangle shortcut only for complete, axis-aligned
  rectangular material and route every other authored polygon to
  `polysection`.
- `Cylinder` accepts only finite, positive dimensions whose authored polygon is
  an axis-aligned rectangle. Its constructor must raise for any other shape;
  never weaken the check to admit a bounding box.
- The rectangle shortcut is an exact shape-specific optimization. Build-time
  savings never authorize a shape substitution. Profile an exact route and
  improve caching or implementation when build cost matters.

## Read section authority from the owning FrameSpace

- For plasma-grid generator type and pre-clip sampling, the authority is the
  source assembly column `self.aloc["plasma", "section"]`. The `Target`
  section column carries schema defaults and is never the authored-section
  authority.
- Pass geometric targets from authoritative coordinates and polygons. Do not
  infer source material from target defaults or from a segment label after the
  build-time route guard has run.
- FrameSpace is columnar. Convert a column once and index it by integer position,
  for example `np.asarray(frame["poly"], dtype=object)[positions]`. Preserve row
  labels explicitly when rebuilding a column slice.
- Do not use `iterrows`. A bare `frame.iloc[position]` returns `RowView`, which
  is ordered and list-convertible but intentionally not subscriptable. Read
  named values from columns, and use tuple indexing only for the accessors that
  define it.
