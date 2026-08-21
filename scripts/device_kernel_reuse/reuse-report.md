# Device-kernel reuse map

## Executive verdict

The single-source port can reuse the accepted Part V mathematics, most of its
array-namespace plumbing, and the existing fixed-shape JAX launch machinery.
The production section API exposes nine logical exact blocks—ψ, B_R, and B_Z,
each at uniform, radial-moment, and vertical-moment order—and all nine already
contract through the same ninth-order harmonic, complete-elliptic/pole, and two
graded `arsinh` residual channels. The closed-form *uniform* triple also already
has a packed `xp`-threaded JAX path. What does not yet exist is the packed
single-source form of the six moment companions, a production dispatcher that
uses that form on both CPU and GPU, or a compute-provenance descriptor in the
semantic carrier-cache identity.

The Solovev profile makes that seam worth targeting. Exact polygon-moment
evaluation plus fixed moment geometry accounts for 286.788/294.217 s (97.475%)
on the 551-cell coarse fixture and 769.949/778.044 s (98.960%) on the 1,076-cell
fine fixture. The report's H200 contraction estimate of 0.124–0.154 ms per dense
moment block is only a feasibility estimate—no device build kernel was measured—
but it confirms that the authored-shape and carrier residue (7.429 s coarse,
8.096 s fine) is not the port target.

## What “nine blocks” means

There are two related counts in the evidence and they must not be conflated.

1. The reusable section-kernel API has nine logical outputs per target family:
   `(Psi, PsiR, PsiZ)`, `(Br, BrR, BrZ)`, and `(Bz, BzR, BzZ)`. The first suffix
   is the source's uniform-current block; `R` and `Z` are first-moment companions
   about the section expansion point. [`PolySection`](../../nova/biot/polysection.py#L109)
   exposes exactly these properties.
2. The profiled Solovev carrier builds only the three **flux** moment orders, for
   each of three target families (`grid`, `wall`, `sample`): 3 × 3 = nine dense
   matrices. [`_flux_blocks`](../analytic_oracle_fixtures/measure.py#L366) returns
   `(G0, GR, GZ)` and the carrier stores them as `plasma_to_{grid,wall,sample}`
   plus `_r` and `_z` companions.

The logical section blocks and their current entry points are:

| logical block | production entry | shared exact reduction | fitness verdict |
|---|---|---|---|
| `Psi` / ψ·G0 | `PolySection._coupling[0]` → `polygon_analytic_greens` | `_Vertex` + `_Edge.terms` | **reuse with a thin dispatch change** — the packed `xp` form already returns this row. |
| `PsiR` / ψ·GR | `PolySection._moment_coupling[0]` → `polygon_analytic_flux_moments()[1]` | `_Edge.flux_and_moment_terms` | **port wrapper, reuse mathematics** — accepted direct antiderivative exists, but its public driver is NumPy/mutation shaped. |
| `PsiZ` / ψ·GZ | `PolySection._moment_coupling[1]` → `polygon_analytic_flux_moments()[2]` | same, plus reflection-conditioned contour sum | **port wrapper, reuse mathematics** — preserve algebraic translation and reflection pairing exactly. |
| `Br` / B_R·G0 | `PolySection._coupling[1]` → `polygon_analytic_greens` | `_Vertex` + `_Edge.terms` | **reuse with a thin dispatch change** — already emitted by `packed_analytic_greens`. |
| `BrR` / B_R·GR | `PolySection._moment_coupling[2]` → `polygon_analytic_field_moments()[0][1]` | `_central_field_moments_direct` | **port wrapper, reuse direct field formula** — do not resurrect AD-through-flux. |
| `BrZ` / B_R·GZ | `PolySection._moment_coupling[3]` → `polygon_analytic_field_moments()[0][2]` | `_central_field_moments_direct` | **port wrapper, reuse direct field formula** — same shared moment channel; packed form is absent. |
| `Bz` / B_Z·G0 | `PolySection._coupling[2]` → `polygon_analytic_greens` | `_Vertex` + `_Edge.terms` | **reuse with a thin dispatch change** — already emitted by `packed_analytic_greens`, including the axis limit. |
| `BzR` / B_Z·GR | `PolySection._moment_coupling[4]` → `polygon_analytic_field_moments()[1][1]` | `_central_field_moments_direct` | **port wrapper, reuse direct field formula** — maintain the direct contour identity and axis handling. |
| `BzZ` / B_Z·GZ | `PolySection._moment_coupling[5]` → `polygon_analytic_field_moments()[1][2]` | `_central_field_moments_direct` | **port wrapper, reuse direct field formula** — packed form is the final companion gap. |

## Exact evaluation call tree

The current host carrier calls the flux-only branch directly; the general
production element reaches the same primitives through `PolySection`:

```text
Solovev carrier build
└── _flux_blocks(targets, polygons, expansion centres)       [3 target families]
    └── for each source polygon
        └── polygon_analytic_flux_moments(...)               [G0, GR, GZ]
            ├── pack_section(vertices)
            ├── _Vertex(...) once per live corner
            │   ├── elliptic.harmonic_moments(..., count = 9 + headroom + 2)
            │   │   ├── elliptic._complete_kind(complement, xp)
            │   │   │   └── completeelliptic.complete_kind
            │   │   │       ├── _descent(..., trips = fixed TRIPS)
            │   │   │       └── _accumulate(...)             [Bulirsch cel arrangement]
            │   │   └── ninth-order harmonic recurrence
            │   ├── cn_pole_moment / sn_pole_moment
            │   │   └── completeelliptic.complete_pole
            │   │       ├── same fixed-trip _descent
            │   │       └── _accumulate(...)                  [third-kind cel]
            │   ├── momentchannel.Channel
            │   │   ├── harmonic_root_moments
            │   │   ├── harmonic_pole_moments
            │   │   └── rangefunction contractions
            │   └── _first_residual
            │       └── gradedresidual.graded_residual        [arsinh β1, 128 nodes]
            ├── _Edge(...) once per live edge
            │   ├── vertex.split(plane denominator)
            │   ├── _second_residual
            │   │   └── gradedresidual.graded_residual        [arsinh β2, 128 nodes]
            │   └── flux_and_moment_terms(lower/upper)
            ├── signed, ordered contour accumulation
            ├── reflection-conditioned vertical moment
            └── translate central moments to requested expansion point
```

For the complete nine-output section API, `polygon_analytic_greens` supplies
the three uniform rows, `polygon_analytic_flux_moments` supplies the two flux
companions, and `polygon_analytic_field_moments` supplies the four field
companions. The field companions are direct antiderivatives, not derivatives of
the flux evaluator. Their slanted-edge `J0/J1/J2` and horizontal-edge
`Q0/QR/QZ` primitives still use the same `_Vertex` ninth-order harmonic, pole,
and graded-residual channel.

Important structural facts for the port:

- [`completeelliptic.py`](../../nova/biot/completeelliptic.py) already expresses
  complete first, second, and third kinds in the fixed-trip Bulirsch `cel`
  arrangement and threads `xp`; this is the trace-stable special-function core.
- [`elliptic.py`](../../nova/biot/elliptic.py) already threads `xp` through the
  harmonic family and pole seeds; the harmonic ceiling is `_HARMONICS = 9` in
  [`polygonanalytic.py`](../../nova/biot/polygonanalytic.py#L173).
- [`gradedresidual.py`](../../nova/biot/gradedresidual.py) already threads `xp`
  and uses fixed node counts. Both residuals are evaluation quadratures after
  logarithmic endpoint layers have been removed analytically.
- [`_Vertex`](../../nova/biot/polygonanalytic.py#L217) and
  [`_Edge`](../../nova/biot/polygonanalytic.py#L508) already accept `xp`; their
  mathematics is not the missing substrate.
- [`packed_analytic_greens`](../../nova/biot/polygonanalytic.py#L1794) already
  turns dead edges, wrap topology, corner residual ownership, and axis handling
  into arithmetic suitable for a static JAX trace. It returns `(ψ, B_R, B_Z)`
  for a whole pair tile, but not the six companions.

## Existing JAX substrate

| candidate | reusable capability | limitation at this seam | fitness verdict |
|---|---|---|---|
| `nova.jax.config.configure_dtypes` + `resolve_precision` | enables x64 once, resolves precision before arrays/tracing, and permits distinct compiled dtype variants | does not describe backend/compiler provenance | **fit as-is for fp64 policy** — call before construction and keep dtype explicit. |
| `TilePlan`, `plan_tiles`, `_device_blocks` | fixed target/source/block shapes, padding, and deterministic pair indexing | its memory model covers one block, not a mapped tile's true HBM peak | **fit with measured tile sizing** — reuse layout, do not infer HBM from `peak_bytes`. |
| `tile_evaluator` / `_warm_evaluator` | `jit`, `vmap` or `lax.scan`, optional `pmap`, static edge-count identity, memoized executable | product accelerator currently authorises quadrature; closed form is diagnostic and uniform-only | **fit as launcher substrate** — point it at the single-source nine-row evaluator after companion packing. |
| `packed_analytic_greens` | existing JAX/NumPy dual-namespace closed-form Part V uniform evaluator | eagerly forms some dead-lane work and lacks moment companions | **best kernel seed** — extend/replace the host wrappers around this shared arithmetic rather than create a new device module. |
| `compilation_cache` | bounded persistent XLA executable cache whose key already depends on graph/JAX/XLA/device | executable reuse is not matrix-artifact identity | **fit as-is, but orthogonal** — retain for compile economy; never treat a hit as Green-matrix provenance. |
| `bound_compilation_retention` | prevents a long-lived process retaining unbounded executables | no matrix/cache semantics | **fit as-is for service hygiene** — useful but not part of numerical acceptance. |
| JAX regression corpus in `tests/test_biotcompleteelliptic.py` and `tests/test_biottiledbackend.py` | fixed-trip `cel` value/gradient checks and packed NumPy/JAX parity | current assertions are tolerance-oriented, not the required full ULP distribution and byte-identical fraction | **fit as a base, insufficient as the go gate** — add block-level per-device receipts. |

The existing closed-form tile route is especially valuable because its geometry
is passed as runtime data, edge count is part of executable identity, and pair
indices are static. Moving a section therefore does not retrace. The port should
preserve that property while making the exact closed-form moment evaluator the
one production implementation for both CPU and GPU.

## Profiled build decomposition and artifacts

The additive measurement is implemented in
[`scripts/accuracy_cost_ladder/measure.py`](../accuracy_cost_ladder/measure.py#L71).
It bypasses the semantic cache, reconstructs the carrier, and then requires all
18 persisted production arrays to match the warm cached carrier bitwise.

| fixture | realised cells | section shape | fixed moment geometry | exact kernel families | assembly | total | exact-kernel share | device-eligible share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| coarse | 551 | 7.402 s | 0.765 s | 286.023 s | 0.027 s | 294.217 s | 97.216% | 97.475% |
| fine | 1,076 | 8.045 s | 1.841 s | 768.108 s | 0.051 s | 778.044 s | 98.723% | 98.960% |

Within the kernel stage, coarse target-family costs are grid 76.677 s, wall
46.014 s, and sample 163.332 s; fine costs are grid 229.930 s, wall 79.439 s,
and sample 458.738 s. These are family totals for all three flux moment orders,
not independently timed per-order kernels.

Artifacts and their fitness:

| artifact | evidence retained | fitness verdict |
|---|---|---|
| [`build-coarse.json`](../accuracy_cost_ladder/build-coarse.json) | additive stages, three target-family timings, 551 cells, 18-array bitwise identity | **authoritative port ranking input** — use as the host baseline. |
| [`build-fine.json`](../accuracy_cost_ladder/build-fine.json) | same decomposition at 1,076 cells | **authoritative projection input** — its 98.960% eligible share bounds host residue. |
| [`results.json`](../accuracy_cost_ladder/results.json) | merged build/GPU receipts and explicit `estimate_only_no_port_attempted` status | **fitness evidence, not a speedup claim** — preserves the 0.124–0.154 ms contraction estimate with its caveat. |
| [`gpu-coarse.json`](../accuracy_cost_ladder/gpu-coarse.json) and [`gpu-fine.json`](../accuracy_cost_ladder/gpu-fine.json) | H200 kind/backend/JAX version, solve timings, phase probes, StableHLO and trace references | **compute-receipt seed** — reuse its device record shape, but add compiler/runtime fields required for cache identity. |
| [`README.md`](../accuracy_cost_ladder/README.md) | method and environment contract (`JAX_PLATFORMS=cpu` for CPU measurements) | **fit as measurement protocol** — keep backend selection explicit. |

The merged feasibility receipt defines one dense block as one target family ×
one moment order and estimates 8,036.8 blocks/s coarse (0.1244 ms/block) and
6,508.4 blocks/s fine (0.1536 ms/block) from the inclusive H200 map. Because the
map also pays topology, clipping, and density work, this is deliberately a
conservative contraction-side estimate and explicitly says that a device-native
exact polygon build kernel is still required.

## Cache identity and compute provenance

The semantic cache machinery is reusable, but the compute field itself is not
implemented yet.

| candidate | in-reach machinery | fitness verdict |
|---|---|---|
| `PolySectionPolicy.key` | canonical JSON already includes exact kernel, backend, precision, device eligibility, and quadrature | **fit as route identity** — it separates current NumPy/closed and JAX/quadrature routes, but cannot identify CPU vs GPU code generation within a single-source JAX route. |
| `CoilSet.route_attrs` → fixture `machine_cache_identity(...)["routes"]` | route policy, discretisation, precision, source/reference content, and schema already feed the semantic descriptor | **fit to extend** — add compute provenance beside semantic route; do not overload geometry or policy fields. |
| `FilePath.canonical_key` + `hash_attrs` | deterministic type-tagged recursive serialization and xxh64 keying | **fit as-is** — a nested compute descriptor automatically changes the group key. |
| fixture `_machine_dataset` / `_machine_from_dataset` | stores schema, key, canonical semantic descriptor, payload inventory/digest; rejects drift before warm reuse | **fit as validation pattern** — require the compute descriptor to round-trip and reject a mismatched producer. |
| `_machine_cache_lock` + `cached_machine` | serializes miss/build/store/validation; warm artifacts are bitwise checked | **fit as publication mechanism** — retain separate CPU/GPU siblings under distinct keys. |
| `measurement_stamp` | refuses a source stamp for a dirty or non-git checkout | **fit for source provenance** — include the clean commit in measurement receipts, but it is not compute provenance. |
| accuracy-cost GPU `device` receipt | currently records `platform`, `device_kind`, `jax_backend`, and `jax_version` | **partial fit** — use as the minimum seed; it does not yet fingerprint jaxlib/XLA/compiler flags or precision mode. |

The current fixture keys (`746fbe1553c4b242` coarse and `f0f96aa214aa9459`
fine in the older stored-reference cache audit) prove the semantic-key and
bitwise-load path, not eligibility for cross-device reuse. The analytic profile
also proves its direct rebuild matches 18 cached arrays bitwise. Neither cache
identity presently carries a compute producer, so publishing GPU matrices under
an unchanged descriptor would permit the contamination the plan forbids.

The compute descriptor to author should be a stable data object, captured before
the build and included verbatim in both the semantic key and persisted attrs. At
minimum it needs backend/platform (`cpu`/`gpu`), device kind, resolved dtype,
JAX and jaxlib versions, and the XLA/compiler configuration that can change fp64
code generation. Hostname, process id, reservation, and wall-clock time belong
in the receipt but **not** in cache identity because they do not define numerical
semantics. Verification pins and ULP histograms must be indexed by the same
descriptor.

## Banked differentiation conviction

The 12.2 GiB result applies to a different composition and must remain visible
without being generalized into a ban on plain kernel evaluation.

- Forward-mode AD of the accepted flux antiderivative composition for one target
  died independently at R = 2.6, 3.5, and 5.0 m on the login node, excluding a
  single geometric regime and target accumulation as explanations.
- The matching parity job on a 32 GiB `all_debug` node reached 20m23s and
  MaxRSS 12,235,772 KiB (about 12.24 GiB) for one target before termination
  (SLURM 1249084).
- Production therefore derives B_R/B_Z moment companions directly through the
  shared ninth-order harmonic/pole/graded-residual channel. That direct route
  costs 0.119 s in the banked near/far check and leaves no JAX derivative reachable
  from the current field-moment API.

**Fitness verdict:** the conviction rules out restoring “differentiate the flux
antiderivative to obtain field companions.” It does **not** rule out JIT-compiling
plain evaluation, nor does it answer geometry-gradient feasibility after the
packed port. The correct follow-on remains a fresh, batched-target `jacfwd`
measurement of one ported block with peak memory and wall time per target, with
the 12.24 GiB result reported beside it rather than silently replaced.

## What must be authored new

1. A packed, `xp`-threaded moment evaluator that emits all nine logical rows from
   the accepted `_Vertex`/`_Edge` reductions. It must preserve ordered contour
   arithmetic, corner residual ownership, reflection conditioning, translation
   algebra, direct field companions, and axis limits.
2. One production dispatch path that calls that evaluator under NumPy or JAX by
   runtime placement. The existing NumPy implementation remains only as the
   prototype reference until parity passes; no permanent host/device module pair.
3. Fixed-shape carrier assembly for all three Solovev target families and all
   three flux moment orders, using the existing `TilePlan`/`vmap` launcher while
   measuring HBM to choose the mapped tile size.
4. A compute-provenance data contract and collector, included in semantic cache
   identity and persisted attrs, with producer/consumer mismatch tests. The
   descriptor must distinguish CPU and H200 artifacts even when source geometry,
   route policy, and precision match.
5. Device parity receipts for every built matrix: full element-wise ULP
   distribution, byte-identical fraction, maxima and locations, arithmetic-only
   byte-identity classification, and Richardson pin results per provenance.
6. CPU-lane performance measurement for the single-source evaluator using the
   same staged snapshot protocol. Existing JAX-vs-NumPy unit parity is not a CPU
   build-regression result.
7. Cold H200 build receipts for both fixtures, including compile time, evaluation
   time, host assembly, peak HBM, artifact key, compute descriptor, source stamp,
   and proof that CPU and GPU warm requests cannot cross-load.
8. The measure-only geometry `jacfwd` probe after the evaluation port: one block,
   batched targets, peak memory and wall time per target, compared explicitly with
   12,235,772 KiB / 20m23s. Adoption remains outside this build node.

## Bottom line

The mathematical port surface is narrow: reuse `completeelliptic`, `elliptic`,
`momentchannel`, `gradedresidual`, `_Vertex`, `_Edge`, and the packed uniform
closed-form evaluator; reuse JAX precision, tile, compilation-cache, and launch
conventions; reuse semantic hashing, locked publication, descriptor validation,
and measurement source stamps. Author the packed six-companion extension, the
single production dispatcher, the compute descriptor, and the measurement gates.
No evidence supports a new special-function implementation, a CuPy/CUDA fork,
or AD-derived field companions.
