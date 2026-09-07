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

## The device is already the default — do not build a route to reach it

Measured 2026-09-07 while sizing a wide-grid MAST response build. Read this
before writing any driver that assembles interactions, because the obvious
conclusion from reading `nova/biot/polygon.py` alone is wrong.

- **The production route runs on the device with no flag.**
  `PolySectionPolicy()` resolves `exact_kernel="closed_form"` to
  `backend="jax"` and `device_eligibility="axisymmetric_ring"`, `Solve` selects
  `TiledPolySection`, and both of its `tile_evaluator` calls in
  `polysection.py` already pass `batched=True`. `tile_evaluator` enables the
  persistent compilation cache itself through
  `configure_production_compilation_cache()`. So building a machine's
  interactions through the ordinary `CoilSet` / `Machine` path is a compiled,
  batched, cached device build already; JAX places it on a GPU whenever one is
  visible. Nothing needs assembling, and a bespoke driver only bypasses the
  route that is validated.
- **`nova.biot.polygon.polygon_greens` is the host reference, not the route.**
  It is pure numpy and always will be. Finding it host-only says nothing about
  device availability — the device path is `tiledassembly.tile_evaluator`
  beside it, whose fixed-shape padded trace exists precisely so the same code
  can be `vmap`-ed and sharded.
- **The low-level streaming helper is the one place with host defaults.**
  `tiledassembly.assemble` defaults to `backend="numpy"` with `workers`
  processes, and `tile_evaluator` defaults to `batched=False`. Those defaults
  belong to a diagnostic that streams tiles into a zarr store, and reaching the
  device there is the single flag `backend="jax"` (with `workers=1`, because the
  device is the parallelism). Prefer `batched=True` when you do: the module's
  own measurement is that the batched form is faster on both CPU and GPU,
  because `scan` denies the compiler the parallelism it would find across
  blocks. A batched tile does not respect `TilePlan.peak_bytes`, so size it
  from a measurement rather than from the byte budget.
- **A P100 is a usable float64 device.** Measured on the `titan` partition:
  jax 0.11.0 drives compute capability 6.0, `default_backend()` is `gpu`, and a
  float64 matmul sustains 3.5 TFLOP/s. When the `betelgeuse` reservation cores
  are fully allocated, titan is the faster lane despite the older card. Treat a
  cross-backend result as a different numerical measurement from an H200
  receipt rather than as the same number.

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
