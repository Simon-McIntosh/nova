# Criteria derivation machinery reuse map

Scope: candidate machinery for deriving an independent fixed-point criterion, retaining the frozen-six re-score, and replacing write-dependent artifact pins with semantic identity. Paths are relative to the Nova checkout unless an absolute coupled-checkout path is shown.

Scan provenance: Nova `e2c20ed8eb30ed11f412c95bc866110aa4e2a11e`; coupled imas-ambix `2f3dcd0c87be908f3f7912c50168c6fc0dae2b79`.

The governing constraint is stronger than merely having three mesh points: the residual being judged must not be an input to its own threshold. The two observed-order strata must therefore be fitted separately, and a target row must be held out from the order/coefficient ladder used to construct its bound.

## Candidate bank

| Area | Path | Symbol | Fitness verdict |
|---|---|---|---|
| Order measurement | `benchmarks/efit_parity_tared_external_field.py` | `_classify_mesh_floor` | Reuse its spacing validation and row schema, but not its two-residual `observed_mesh_order` as independent evidence because the gated fine residual participates directly. |
| Order measurement | `benchmarks/efit_operator_consistency_order.py` | `_fit_power_order` | Best existing fit core: it estimates `coefficient * h**order` over four resolutions and banks standard error, a 95% order interval, residual diagnostics, and R²; invoke once per stratum with the target row held out. |
| Order measurement | `benchmarks/efit_interior_stencil_quadrature.py` | `_fit_order` | Reuse for stratum-specific ladders where cancellation is possible because it rejects fewer than two positive errors and labels non-improving series as a numerical floor, though it lacks the confidence interval of `_fit_power_order`. |
| Discretisation bound | `benchmarks/efit_analytic_roundtrip_floor.py` | `_fit_order`, `measure_analytic_floor` | Strong pattern for turning an independently measured `coefficient` and `observed_order` into a mesh-scale bound or required resolution; substitute fixed-point residual ladders per stratum and retain its floor qualification. |
| Discretisation bound | `benchmarks/efit_parity_criterion_provenance.py` | `richardson_fine_error` | Reuse only as arithmetic over an independently banked order and a pair that excludes the gated target residual; the current same-pair order makes `E_f` collapse to the tested fine residual and is inadmissible. |
| Frozen-cohort re-score | `benchmarks/efit_parity_criterion_provenance.py` | `_mesh_rows` | Reuse the frozen-six join, cohort guard, round-off slack, dual registered/derived verdict columns, and low-order qualification, replacing only the circular criterion source. |
| Receipt assembly | `benchmarks/efit_parity_criterion_provenance.py` | `build_receipt`, `_provenance_table`, `_units_audit` | Reuse the banked-input-only receipt, five-bound provenance table, protected-artifact checks, and explicit refusal of unsupported cross-scale conversion; extend rather than fork it. |
| Mesh error envelope | `nova/equilibrium/observation_kernels.py` | `_map_interpolation_error` | Reuse as an example of a bound computed from resolved-grid curvature and flux span rather than a solver residual, but do not equate this interpolation envelope with the fixed-point defect. |
| Mesh error envelope | `nova/equilibrium/observation_kernels.py` | `_profile_interpolation_error`, `_profile_signal_error` | Reuse the curvature-times-spacing-squared construction and propagated slope term as the local discretisation-bound pattern; applicability is limited to interpolation observables, not whole-map convergence. |
| Residual domain | `nova/equilibrium/conservation.py` | `FluxMesh`, `conservation_ledger` | Reuse its mesh-validity masks and truncation-floor interpretation to state each stratum's validity domain, but it provides qualification machinery rather than a fixed-point threshold. |
| Semantic identity key | `nova/database/filepath.py` | `canonical_key` | Reuse its deterministic, type-tagged, insertion-order-independent encoding for a criterion artifact's semantic descriptor so identity changes when inputs or discretisation parameters change. |
| Semantic identity digest | `nova/database/filepath.py` | `FilePath.hash_attrs` | Suitable for local cache grouping over the semantic descriptor, but its 64-bit xxHash is not by itself a publication-grade pin or a verifier of artifact contents. |
| Semantic identity constructor | `nova/imas/machine_artifact.py` | `MachineArtifactManifest.semantic_identity` | Directly reusable design: hash canonical semantic fields while excluding materialisation-specific file and OCI tables, producing a re-authoring-stable SHA-256 identity. |
| Pytest collection gate | `conftest.py` | `_SLOW_FILES`, `_SLOW_NODEIDS`, `pytest_collection_modifyitems` | Add any multi-resolution solve tests to the curated slow set so default collection stays bounded while the explicit full lane still collects and runs them; never disable the tests. |
| Coupled semantic pin | `/home/ITER/mcintos/Code/imas-ambix/imas_ambix/gs/artifact_resolution.py` | `PINNED_SEMANTIC_IDENTITY`, `find_revision`, `_resolve_stored`, `resolve_machine_description` | Best end-to-end pinning exemplar: discover by semantic identity, verify physical and registry identity plus manifest contents, fail closed on mismatch, and record the materialisation digest only as provenance. |
| Coupled receipt provenance | `/home/ITER/mcintos/Code/imas-ambix/imas_ambix/spine_bench/machine_artifact_arm.py` | `MachineArtifactGeometrySource.provenance`, `MachineArtifactGeometrySource.revision` | Reuse the consumer pattern that publishes both semantic identity and the resolved materialisation digest while treating the semantic identity as the revision visible to downstream evidence. |

## Recommended composition

1. Build one three-or-more-resolution ladder per observed-order stratum and fit it with `_fit_power_order`; hold each frozen target residual out of the ladder that supplies its `coefficient` and order interval.
2. Form the target-mesh criterion from the independently fitted power law (`tau(h_target) = C * h_target**p`), carrying a conservative choice from the banked uncertainty interval and the stratum validity range; use `richardson_fine_error` only if its inputs remain independent of the target residual.
3. Feed those six per-reference criteria into `_mesh_rows`/`build_receipt` so the registered `1e-8` count, derived count, circular-collapse algebra, qualifications, source identities, and protected-artifact audit remain together.
4. Describe the receipt's semantic inputs with `canonical_key`, publish a SHA-256 semantic identity following `MachineArtifactManifest.semantic_identity`, and resolve/verify that identity following Ambix's fail-closed resolver rather than pinning a write-dependent artifact digest.
5. Put any solve-bearing validation under the existing `slow` collection policy; keep banked-receipt unit tests in the default lane where they do not launch equilibrium solves.

## Coverage count

The bank contains **16 candidates across 10 Nova source files plus 2 files in the coupled imas-ambix checkout**. It explicitly covers **3 order-fit/classification candidates, 7 discretisation/re-score/receipt candidates, 3 semantic-identity primitives, 1 pytest collection gate, and 2 coupled semantic-pin/provenance candidates**; the receipt-assembly row spans both provenance and units machinery and is counted once.
