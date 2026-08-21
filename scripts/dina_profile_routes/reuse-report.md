# DINA dual-route flux-function reuse map

## Outcome

The dual-route receipt does not need a new flux-function extractor, COCOS algebra,
stored-reference reader, forward solver, or source primitive integrator. It needs one
thin receipt adapter that keeps two coordinates explicit: the map-shell coordinate
used by affine extraction and the prescribed-reference coordinate used to compare and
force the reproduction. Route disagreement must remain an output, never a correction.

The live native-DD read of ITER 135011/7, time-slice index 353, found 50 finite samples
for each of `psi`, `dpressure_dpsi`, `f_df_dpsi`, `pressure`, and `f`, plus a finite
65 by 129 `profiles_2d.psi` map. The declared COCOS-11 anchors are
`psi_axis = 86.0164541041778 Wb` and `psi_boundary = 5.12276312270973 Wb`;
the existing mapping produces Nova total-flux anchors -86.0164541041778 Wb and
-5.12276312270973 Wb, with an 80.8936909814681 Wb outward span. This check used
`imas.DBEntry(..., dd_version="3.39.0")` through imas-python only.

## Affine map extraction

`extract_flux_functions` requires strictly increasing, uniformly spaced one-dimensional
radius and height axes; a flux map and `psi_norm` map of shape `(radius.size,
height.size)` (or C-order flattened equivalents); and optionally an equally shaped
plasma mask, explicit increasing shell centres in `[0, 1]`, and qualification limits
([`nova/equilibrium/map_extraction.py:107`](../../nova/equilibrium/map_extraction.py#L107),
[`nova/equilibrium/map_extraction.py:181`](../../nova/equilibrium/map_extraction.py#L181)).
It returns `p_prime`, `ff_prime`, per-profile uncertainties, projection RMS, scaled
design condition, gradient/axis inflation, sample count, reliability, and the underlying
Delta-star/current receipt ([`nova/equilibrium/map_extraction.py:64`](../../nova/equilibrium/map_extraction.py#L64),
[`nova/equilibrium/map_extraction.py:326`](../../nova/equilibrium/map_extraction.py#L326)).

**Fitness verdict — fit with adapter:** the extractor and its receipt are fit as-is once
the DINA adapter supplies mapped Nova total flux in `(R, Z)` order, an explicit
map-derived shell coordinate, and the declared-coordinate mapping retained beside it;
the extractor must not be taught IDS access or reference-anchor policy.

There are six benchmark call sites and three direct test calls. They establish the
following reusable input patterns:

| Call site | Inputs already demonstrated | Fitness verdict |
|---|---|---|
| [`benchmarks/diiid_plasma_subtraction_gate.py:167`](../../benchmarks/diiid_plasma_subtraction_gate.py#L167) | Convention-mapped total-flux label, normalized label, polygon core mask, `min_samples=6` | **fit as-is** — closest complete pattern for extracting, masking by `current.valid`, interpolating reliable profiles, and reconstructing current. |
| [`benchmarks/diiid_forward_gs_match.py:451`](../../benchmarks/diiid_forward_gs_match.py#L451) | `(R,Z)` label, interpolated axis value, contour-median boundary value, normalized map, plasma mask, `min_samples=6` | **fit with adapter** — reuse the extraction/reliability path, but retain DINA declared anchors beside the map-derived axis/boundary instead of replacing them. |
| [`benchmarks/diiid_label_resolve_gate.py:427`](../../benchmarks/diiid_label_resolve_gate.py#L427) | Operator axes, label, axis/label-span normalization, explicit 5–95% surfaces, mask, `min_samples=6` | **fit as-is** — explicit surface selection is useful for a fixed comparison base. |
| [`benchmarks/diiid_label_resolve_gate.py:577`](../../benchmarks/diiid_label_resolve_gate.py#L577) | Operator axes, label, interpolated axis, contour-median boundary, default surfaces, mask, `min_samples=6` | **fit with adapter** — useful map-anchor pattern, but its scalar return path discards uncertainty and reliability metadata that the DINA receipt must bank. |
| [`benchmarks/diiid_forward_gs_illustration.py:142`](../../benchmarks/diiid_forward_gs_illustration.py#L142) | Operator axes, label, interpolated axis, known span, explicit 5–95% surfaces, mask, `min_samples=6` | **fit as-is** — suitable figure-data preparation after route provenance is attached. |
| [`benchmarks/diiid_root_existence.py:268`](../../benchmarks/diiid_root_existence.py#L268) | Convention-mapped total-flux label, normalized label, core mask, default surfaces, `min_samples=6` | **fit as-is** — demonstrates evaluation of extracted sources back on the same label without fitting. |
| [`tests/test_map_extraction.py:111`](../../tests/test_map_extraction.py#L111) | Prescribed synthetic total-flux map, known `psi_norm`, mask, explicit surfaces | **fit as-is** — regression precedent for profile round-trip and vacuum qualification. |
| [`tests/test_map_extraction.py:139`](../../tests/test_map_extraction.py#L139) | Axis-inclusive surfaces and a deliberately stationary label region | **fit as-is** — directly verifies uncertainty inflation and `reliable=False` near the axis and vanishing gradient. |
| [`tests/test_map_extraction.py:155`](../../tests/test_map_extraction.py#L155) | Same synthetic map with locally stationary normalized labels | **fit as-is** — preserves the required low-gradient failure mode in the banked receipt. |

The prose occurrence at `benchmarks/diiid_forward_gs_match.py:191` names the extractor
in receipt metadata but is not a call. No additional invocation exists outside these
six benchmark calls and three tests.

## Stored-reference reader and solve lane

The stored-reference lane pins pulse/run, DD version, and slice at
[`tests/test_equilibrium_forward_reference.py:237`](../../tests/test_equilibrium_forward_reference.py#L237).
Its reader opens the URI with `imas.DBEntry(..., dd_version="3.39.0")`, reads
`equilibrium`, `pf_active`, `pf_passive`, and `wall`, and closes the entry
([`tests/test_equilibrium_forward_reference.py:714`](../../tests/test_equilibrium_forward_reference.py#L714)).
It exposes declared `psi_axis`/`psi_boundary`, normalizes the 1-D profile grid from the
stored `profiles_1d.psi` endpoints, and retains all four requested source arrays plus
the two-dimensional map ([`tests/test_equilibrium_forward_reference.py:725`](../../tests/test_equilibrium_forward_reference.py#L725)).
The lane then constructs traceable absolute source functions and boundary primitives
([`tests/test_equilibrium_forward_reference.py:1362`](../../tests/test_equilibrium_forward_reference.py#L1362))
and drives the reference-seeded Newton-Krylov solve without measurements
([`tests/test_equilibrium_forward_reference.py:1619`](../../tests/test_equilibrium_forward_reference.py#L1619)).

**Fitness verdict — fit with adapter:** reuse the native-DD reader, machine/cache
construction, source construction, and solve unchanged; add a route-selected
`ReferenceCase` view so declared and extracted gradients can drive identical cached
geometry while the prescribed declared anchors and boundary primitives remain explicit.
Do not change `ReferenceCase.psi_norm` silently, because it currently means endpoint-
normalized declared `profiles_1d.psi`, not a map-saddle coordinate.

## Existing DINA extractions

`benchmarks/dina_reference_mapping.py` already performs the strongest independent IDS
read. It reads the five `profiles_1d` quantities, declared anchors, grid axes, and map
directly through imas-python ([`benchmarks/dina_reference_mapping.py:248`](../../benchmarks/dina_reference_mapping.py#L248));
then composes documented COCOS 11 to COCOS 17 factors, converts flux, gradients, field
function, and the map, and forms declared-anchor `psi_norm`
([`benchmarks/dina_reference_mapping.py:296`](../../benchmarks/dina_reference_mapping.py#L296)).
Its comparison against the stored-reference reader is explicitly an audit performed
after the independent derivation ([`benchmarks/dina_reference_mapping.py:349`](../../benchmarks/dina_reference_mapping.py#L349)).

**Fitness verdict — fit with adapter:** reuse `StoredProfiles`, `derive_mapping`, and
the post-derivation equality audit for the declared route; wrap their values in a
route receipt instead of importing benchmark CLI output as runtime authority.

`benchmarks/dina_pack_current_distribution.py` reuses `reference.reference_case()`,
builds one cached machine, and solves two current-distribution variants with identical
stored absolute `p_prime` and `ff_prime`
([`benchmarks/dina_pack_current_distribution.py:569`](../../benchmarks/dina_pack_current_distribution.py#L569),
[`benchmarks/dina_pack_current_distribution.py:612`](../../benchmarks/dina_pack_current_distribution.py#L612)).

**Fitness verdict — fit with adapter:** its fixed-machine/two-arm orchestration is the
right precedent for declared-versus-extracted forcing, but the present code varies
conductor distribution rather than source profiles and records no extraction receipt.

## Generic IMAS exposure

`nova.imas.equilibrium.Profile1D` currently exposes only `dpressure_dpsi` and
`f_df_dpsi`, creates a uniform `psi_norm`, optionally retains raw `profiles_1d.psi` as
`psi1d`, and interpolates the two gradients onto the uniform base
([`nova/imas/equilibrium.py:405`](../../nova/imas/equilibrium.py#L405)). It does not
expose `profiles_1d.pressure` or `profiles_1d.f`, does not carry declared axis/boundary
anchors into the profile object, and performs no explicit COCOS conversion. The module's
global-quantity reader does expose `psi_axis` and `psi_boundary` separately
([`nova/imas/equilibrium.py:89`](../../nova/imas/equilibrium.py#L89)).

**Fitness verdict — unfit because the generic `Profile1D` path drops both primitives
and convention/anchor provenance:** use the direct native-DD DINA reader for this
receipt; widening the generic xarray layer is a separate API change, not required here.

## Flux convention and primitive checks

Nova's convention module pins total poloidal flux, the sign of `p_prime` and
`ff_prime`, current-density reconstruction, and Delta-star closure
([`nova/equilibrium/convention.py:1`](../../nova/equilibrium/convention.py#L1)).
`toroidal_current_density` and `grad_shafranov_source` provide pointwise consistency
checks, while `flux_function_pressure` and `flux_function_toroidal_field` reconstruct
the pressure and squared-field primitives from boundary values, flux span, and inward
gradient tails ([`nova/equilibrium/convention.py:68`](../../nova/equilibrium/convention.py#L68),
[`nova/equilibrium/convention.py:85`](../../nova/equilibrium/convention.py#L85)).

**Fitness verdict — fit as-is:** use these helpers for integral cross-checks after both
routes are expressed in Nova's negated-total-flux convention; they deliberately do not
replace the external COCOS mapping or choose anchors.

## Declared-anchor versus map-saddle precedent

`scripts/normalization_discriminator/measure.py` already loads this DINA reference,
separates absolute in-cell flux values from affine normalization constants, and compares
the production topology's map-derived axis/boundary against the stored case's declared
anchors ([`scripts/normalization_discriminator/measure.py:127`](../normalization_discriminator/measure.py#L127),
[`scripts/normalization_discriminator/measure.py:190`](../normalization_discriminator/measure.py#L190)).
It banks the production-minus-declared offsets and the clip-to-Newton-saddle offset
([`scripts/normalization_discriminator/measure.py:418`](../normalization_discriminator/measure.py#L418)).
The existing result records a 0.4109918832381485 Wb boundary offset, 0.005080641991379802
of the declared span, and a map saddle at `psi_norm = 0.9999446994667243`
([`scripts/normalization_discriminator/results.json:191`](../normalization_discriminator/results.json#L191)).

**Fitness verdict — fit with adapter:** reuse its explicit affine-coordinate comparison
and receipt vocabulary, but not its heavy tangent-response experiment; the profile
receipt only needs the declared anchors, independently read map axis/saddle, absolute-
flux coordinate transform, and the three banked offsets.

## Ambix ownership boundary

`~/Code/imas-ambix` does not contain Nova's affine map extraction, a `profiles_1d`
native-DD reader, or a DINA declared-anchor-versus-map-saddle receipt. It does own the
downstream typed source contract: `RadialCoordinate` carries normalized flux, absolute
axis/separatrix total flux, and the exact physical Jacobian
([`imas_ambix/fluxstate/contract.py:79`](../../../../../../imas-ambix/imas_ambix/fluxstate/contract.py#L79));
`FluxFunctionState` carries pressure, both derivatives, `f`, units, COCOS provenance,
derivative validation, and absolute-source policy
([`imas_ambix/fluxstate/contract.py:658`](../../../../../../imas-ambix/imas_ambix/fluxstate/contract.py#L658));
and `to_nova_forward_payload` freezes those arrays without renormalizing them
([`imas_ambix/fluxstate/consumer_contract.py:126`](../../../../../../imas-ambix/imas_ambix/fluxstate/consumer_contract.py#L126)).
Therefore Ambix owns a suitable eventual handoff schema, not the DINA extraction or
reference-qualification machinery. **Fitness verdict — fit with adapter:** emit enough
provenance for a later `FluxFunctionState` construction, but keep the banked dual-route
receipt and deterministic forward forcing in Nova.

## Minimal composition for the receipt

1. Read the stored slice with the existing native-DD imas-python reader and independently
   derive the COCOS-17 mapped declared profiles and map.
2. Preserve the declared coordinate
   `psi_N_declared = (Phi - Phi_axis_declared) / (Phi_boundary_declared - Phi_axis_declared)`
   for reproduction support and forcing.
3. Independently read the map axis and saddle, form `psi_N_map`, and call
   `extract_flux_functions(radius, height, mapped_grid_flux, psi_N_map,
   plasma_mask=declared_support)`. Bank the complete `SurfaceExtractionReceipt`.
4. Convert every extracted shell centre back to absolute total flux through the map
   anchors, then into `psi_N_declared`. Interpolate both routes onto one explicitly
   declared comparison base while retaining the original coordinates and reliability
   mask. This exposes the anchor disagreement instead of erasing it.
5. Reconstruct pressure and `F^2` from both gradient routes using the stored boundary
   primitives and convention helpers; report deviation profiles and primitive-integral
   closure. Unreliable extracted shells remain masked, not filled as evidence.
6. Construct two route-selected source views over the same cached reference machine and
   run the existing reference-seeded solve under each. Bank forcing, solver, gauge,
   declared-anchor, map-anchor, and route provenance together.

Quantitatively, this reuses 9 existing extractor invocations, one native-DD DINA reader,
two DINA benchmark orchestration patterns, four convention helpers, and one banked
anchor/saddle comparison. The only missing implementation is the thin dual-route adapter
and its serialized receipt/figures; no existing component is authoritative for silently
reconciling the two routes.
