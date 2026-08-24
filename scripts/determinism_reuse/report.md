# Same-device determinism reuse map

## Outcome

Nova already has almost all of the measurement spine needed to decide whether
the registered-bound route reproduces its own terminal state:

- the exact six held-out case builder, solve options, result-tree names and
  69-bound registration;
- direct access to each solve's complete terminal flux array;
- an acceptance driver that retains source-identical repetition summaries and
  compares every case, batch member and registered observable;
- in-process repetition loops on the production `ForwardProfile` route; and
- one explicit fixed reduction plus persistent/in-process compilation reuse
  controls.

The missing capability is narrow but decisive: **no existing driver, receipt or
test retains the raw terminal flux arrays from multiple repetitions**. The two
committed acceptance receipts retain two repetitions of verdicts and maximum
differences, but `_repetition_snapshot` deliberately discards both terminal
flux and raw observable values. The state verdict therefore cannot be recovered
from banked evidence.

There is also **no existing acceptance-path setting that requests deterministic
GPU kernels, pins an autotuned algorithm, or fixes all solver reduction
associations**. `layout_invariant_sum` fixes the association of the published
moment reductions only. The compilation cache reuses an executable but is
documented not to change its answer. These are useful controls or control arms,
not evidence that the complete solve is deterministic.

The shortest implementation is to extend the existing
`benchmarks/observable_batch_acceptance.py` measurement shape, not create a
second acceptance route: build each frozen case once, invoke the same selected
production route three times inside one process, block each result, retain a
copy of every `ForwardEquilibrium.flux`, flatten every full receipt through the
existing `_named_tree`/`_leaves` helpers, and compare every later repetition to
the first with the existing exact/absolute/relative difference machinery.

## Quantitative inventory

- Scope searched: the production `ForwardProfile` solve/receipt path, its
  registered-bound drivers and receipts, forward-throughput repetition
  harnesses, JAX runtime configuration, relevant equilibrium reductions and
  scatters, and their focused tests. Unrelated timing loops that never execute
  `ForwardProfile` or the registered acceptance were excluded.
- Direct terminal-state access or near-persistence sites: **7**.
- Same-process or same-allocation repetition sites/receipts: **8**.
- Reduction, scatter, precision, executable-reuse or kernel-control sites:
  **11** (including limited or negative-fit controls).
- Committed repeated acceptance measurements: **2 receipts**, each with
  **2 repetitions × 6 cases × 2 batch widths × 69 registered bounds**.
- Raw terminal arrays retained across repetitions: **0**.
- Explicit deterministic GPU-kernel or autotune controls on the acceptance
  route: **0**.
- Existing bitwise same-route state regression: **1 test**, covering two
  synthetic in-process solves rather than the six H200 cases.

## Direct terminal-state access and persistence map

| Module, symbol, driver, receipt or test | What is already retained | Fitness verdict |
|---|---|---|
| `nova.equilibrium.forward.ForwardEquilibrium.flux`, `nova/equilibrium/forward.py:208-223` | The complete terminal flux array is a first-class field of every published solve result. | **BEST RAW-STATE SOURCE.** Copy this field after `jax.block_until_ready`; no solver instrumentation is needed. It is in memory only unless the caller persists it. |
| `nova.equilibrium.fixed_point.FixedPointResult.state`, `nova/equilibrium/fixed_point.py:90-125` | The fixed-point result carries the terminal solver state and trace; `ForwardProfile._receipt` publishes the same terminal state as `ForwardEquilibrium.flux`. | **REUSE AS AN IDENTITY CHECK, NOT A SECOND STATE.** Assert `equilibrium.fixed_point.state` equals `equilibrium.flux`; persist one canonical copy rather than doubling the artifact. |
| `ForwardProfile.solve`, `ForwardProfile._solve_accelerated` and `ForwardProfile._receipt`, `nova/equilibrium/forward.py:970-1010,1117-1208` | The production route returns full flux, fixed-point trace, labels and qualification from one terminal state. | **REUSE AS THE MEASURED ROUTE.** It is the scalar reference route already named by the acceptance receipt. Do not substitute `observe`, which would measure labels without testing solve-state reproducibility. |
| `ForwardProfile.solve_batch`, `nova/equilibrium/forward.py:1418-1449` | A leading-axis `vmap` returns one full terminal flux array and receipt per batch member. | **REUSE FOR EACH REGISTERED WIDTH.** It measures the production batch route for one static profile. It does not by itself repeat the same invocation or persist member fluxes. |
| `benchmarks.observable_batch_acceptance._case_measurement`, `benchmarks/observable_batch_acceptance.py:106-167` | Holds `scalar.flux` and `transformed.flux` transiently before flattening their label trees. Its returned case record keeps only registered observable arrays. | **BEST INSERTION SEAM.** The exact raw arrays already exist at the right case/route boundary. Retain them per repetition before returning; current code drops them. |
| `benchmarks.observable_route_discriminator._case_measurement`, `benchmarks/observable_route_discriminator.py:372-437` | Holds eager flux, compiled flux and `shared_flux`; emits only `terminal_state_difference` and label/localisation summaries. | **ADAPT DIFFERENCE LOGIC ONLY.** `_difference` already reports exact equality and max absolute/relative differences, but this is eager-versus-JIT, not same-route reruns, and its receipt retains no raw flux arrays. |
| `benchmarks.forward_solve_throughput.SolveBundle.reference_flux` and `measure_latency`, `benchmarks/forward_solve_throughput.py:117-217,449-503` | The bundle persists one reference flux and the measurement materialises one compiled result state for parity. | **REFERENCE PRECEDENT, NOT CROSS-REPETITION STORAGE.** It proves raw arrays can live in a fixture bundle, but the repeated timing calls discard each individual result and retain only one final deviation. |

**NO CANDIDATE — raw terminal flux retained across repetitions.** A scan of the
registered acceptance, discriminator, conditioned-convergence and parity JSON
receipts found zero scalar paths containing a raw flux value, and no JSON with
both a repetition collection and a flux array. `integrated-acceptance.json` and
`batch-acceptance.json` each retain two repetition summaries, but neither
contains terminal flux. The new measurement must run and persist the arrays; it
cannot be reconstructed.

## Same-process and same-allocation repetition map

| Module, symbol, driver, receipt or test | Repetition contract | Fitness verdict |
|---|---|---|
| `benchmarks.observable_batch_acceptance.measure`, `benchmarks/observable_batch_acceptance.py:515-693` | One invocation measures all six cases at widths 1 and 4. When the output exists, it appends source-identical `_repetition_snapshot` rows and derives `_repetition_stability`. | **PRIMARY HARNESS, SMALL ADAPTATION.** Add an internal repetition count of at least three so one process and allocation are guaranteed by construction. The current append-on-invocation scheme cannot itself prove that separate invocations shared a process or allocation. |
| `_repetition_snapshot`, `_repetition_stability` and `_repeated_remaining_failures`, `benchmarks/observable_batch_acceptance.py:245-380,464-512` | Retains source identity, backend, per-width counts, all 69 per-observable statuses, per-case statuses and maximum differences for every prior run. | **REUSE METADATA AND VERDICT HISTORY.** Preserve this structure, but add raw-state identity/difference fields and raw observable differences. It currently discards the data needed for the state verdict. |
| `docs/figures/derived-observable-parity/integrated-acceptance.json` | Two source-identical H200 repetitions, 6 cases, widths 1 and 4, 69 bounds; counts moved 67→68 at both widths and case passes moved 410→412 and 408→410. | **AUTHORITATIVE FAILURE RECEIPT, NOT STATE EVIDENCE.** It proves the label route varied in one allocation according to the landed record, but the JSON has no allocation/job identity and no raw terminal state. Use its cohort, source identity and canary, not its data as a state verdict. |
| `docs/figures/derived-observable-parity/batch-acceptance.json` | Two earlier measurement repetitions with the same 6-case/2-width/69-bound shape. | **SECOND REPEATED-LABEL PRECEDENT.** Useful for schema compatibility and regression tests; still no terminal flux or raw label values. |
| `benchmarks.forward_solve_throughput.time_call` and `measure_published_route`, `benchmarks/forward_solve_throughput.py:388-410,737-792` | Repeats synchronised calls inside one interpreter; the published route is timed three eager and three compiled times on one profile and seed. | **REUSE THE BLOCKED IN-PROCESS LOOP.** It already avoids asynchronous timing mistakes and demonstrates same-process production-route repetition. It discards outputs, so add result collection rather than reusing its timing summary. |
| `benchmarks.diiid_batched_throughput.measure`, `benchmarks/diiid_batched_throughput.py:256-371` | Compiles one `ForwardProfile.solve_batch`, resets to one initial state for every repeat, and retains every repeat/frame/member solve qualification; CLI default is two repeats. | **STRONG LOOP AND RECEIPT PRECEDENT.** It is the closest existing in-process batched production loop and retains repeat indices. Its synthetic bootstrapped workload, warm-start frames and branch-only receipts are not the frozen six-case registered-bound measurement, and it discards flux. |
| `tests/test_equilibrium_forward_solve.py::test_the_solve_refuses_to_enforce_a_moment`, lines 903-911 | Runs the same synthetic `ForwardProfile.solve` again in one pytest process and asserts bitwise equality of terminal flux with the fixture solve. | **DIRECT BUT NARROW REGRESSION.** Reuse the `np.testing.assert_array_equal` standard; it covers two synthetic solves on the test backend, not three repetitions of six H200 cases. |
| `tests/test_observable_acceptance.py::test_committed_receipt_covers_two_real_batch_sizes_and_all_bounds`, lines 239-266 | Requires the committed acceptance receipt to retain at least two repetition snapshots and a matching repetition count. | **EXTEND AFTER THE HARNESS LANDS.** It validates receipt presence, not same-process/allocation provenance or raw-state coverage. Add assertions for three repetitions, six state rows, raw array identities and all 69 observable difference rows. |

**NO CANDIDATE — committed one-command three-repetition acceptance driver.**
There is no checked-in SLURM script or Python entry that runs the exact
registered-bound measurement three times inside one process or proves the
allocation identity in its receipt. The existing driver is safe to call
repeatedly, and prior orchestration did so twice in one allocation, but the
guarantee currently lives outside the driver.

## Reduction, scatter and kernel-selection control map

| Module, symbol, driver, receipt or test | Control surface | Fitness verdict |
|---|---|---|
| `layout_invariant_sum`, `nova/equilibrium/observation.py:116-145` | Expresses a fixed binary tree of scalar additions, preserving association when a leading batch axis is introduced. | **ACTIVE, NARROW CONTROL.** Reuse unchanged as the deterministic moment-reduction arm. It controls association, not GPU kernel selection, solver matrix products or conservation calculations. |
| `observe_moments`, `nova/equilibrium/observation.py:565-620` | Uses `layout_invariant_sum` for volume, plasma current, major radius, pressure and field integrals. | **ACTIVE IN THE ACCEPTANCE ROUTE.** It already made `moments.volume` and `moments.major_radius` bitwise scalar/vmap equal on all six H200 cases. It did not stop the run-to-run movement and does not govern `conservation.divergence_b`. |
| `tests/test_observable_reduction_parity.py::test_repaired_moment_reductions_are_invariant_under_a_leading_batch`, lines 50-112 | Compares scalar observation with `jax.jit(jax.vmap(observe))` and requires exact equality for volume and major radius. | **REUSE AS CONTROL REGRESSION.** It proves transformation-layout parity for the fixed reductions, not repeatability of the nonlinear solve. |
| `StencilMesh._scatter` plus the unique-centre invariant, `nova/equilibrium/stencil_mesh.py:538-589,621-644` | Places one derivative per unique centre with `.at[centre].set`, while construction rejects repeated centre indices. | **ATOMICS-SUM HYPOTHESIS NOT SUPPORTED ON THIS OBSERVATION SEAM.** Unique indexed sets cannot race by accumulation order; this is not a segment/scatter-add reduction. Upstream solver arithmetic can still vary. |
| `tests/test_equilibrium_stencil_mesh.py::test_a_cell_may_centre_at_most_one_ring`, lines 629-635 | Rejects a repeated scatter centre explicitly. | **REUSE AS THE UNIQUE-INDEX PROOF.** It guards the property that makes `_scatter` unlike an atomics-based accumulating scatter. |
| `conservation_ledger`, `_sup`, and `conservation.divergence_b`, `nova/equilibrium/conservation.py:315-399` | The canary derives poloidal field and its divergence, then applies a masked `jnp.max`; it does not use the fixed summation helper. | **CANARY AND LOCALISATION SEAM, NOT A CONTROL.** Because the terminal operation is a maximum and the scatter indices are unique, a flip should first be correlated with raw state and per-cell divergence arrays rather than blamed on the repaired moment sum. |
| `nova.biot.tiledassembly.compilation_cache`, `nova/biot/tiledassembly.py:554-603` | Selects a bounded JAX persistent compilation cache through `NOVA_COMPILATION_CACHE`/`JAX_COMPILATION_CACHE_DIR`; it can also be disabled. | **USEFUL COMPILE-REUSE ARM, NOT A DETERMINISM GUARANTEE.** The symbol's contract says cached executables cannot change answers. It is not called by the acceptance driver, so a study must opt in explicitly and record the directory/configuration. |
| `nova.biot.tiledassembly.tile_evaluator`, `nova/biot/tiledassembly.py:606-709` | Memoises an evaluator by plan, mapping, kernel, geometry, device count, precision and shape, reusing one executable inside a process. | **LIMITED REUSE.** It controls the Biot tile evaluator's executable identity, not the full `ForwardProfile` solve/receipt graph used by the acceptance driver. |
| `benchmarks.forward_solve_throughput.configure_compilation_cache`, `benchmarks/forward_solve_throughput.py:1003-1018` | Exposes an explicit cache/off CLI choice for a forward-solve benchmark. | **BEST CONFIGURATION PRECEDENT.** Mirror its explicit, receipt-visible choice if cache reuse is tested; do not silently inherit a user cache and call that a deterministic control. |
| `tests/test_biottiledbackend.py`, lines 379-391 and 501-570 | Verifies in-process evaluator identity, cache hits, explicit directory precedence and cache disablement. | **REUSE FOR CACHE PLUMBING ONLY.** These tests establish configuration behaviour, not numerical reproducibility. |
| `nova.jax.config.configure_dtypes`, `nova/jax/config.py:98-111` | Enables x64 once per process and forbids dtype toggling. | **REQUIRED BASELINE, NOT A DETERMINISM CONTROL.** It holds float64 policy fixed, as the banked H200 route requires, but does not pin reduction association, library algorithms or GPU scheduling. |

**NO CANDIDATE — deterministic GPU kernel/autotune setting on the acceptance
route.** Searches found no acceptance-path `XLA_FLAGS` deterministic-kernel
option, cuBLAS workspace determinism setting, algorithm id, autotune level,
deterministic-ops switch or equivalent. A few unrelated diagnostic benchmarks
disable XLA GPU command buffers; the acceptance, discriminator and conditioned
drivers do not, and command-buffer disablement is not documented here as a
deterministic-kernel control.

**NO CANDIDATE — fixed compilation cache on the acceptance route.** Persistent
cache machinery exists, but `observable_batch_acceptance.py`,
`observable_route_discriminator.py` and `conditioned_convergence_observables.py`
never call it. The current two repetitions may therefore have used separately
compiled executables even inside one allocation.

**NO CANDIDATE — fixed association for the complete nonlinear solve.** The
fixed-point ladder and flux operator still contain backend reductions, dense
matrix products, Gram solves and norms (`nova/equilibrium/fixed_point.py` and
`nova/biot/target.py:568-585`). Only the selected moment reductions have an
explicit association. Do not describe the complete route as reduction-fixed.

**NO CANDIDATE — atomics-based accumulating scatter in the registered
observation seam.** The relevant stencil scatter is unique-index `.set`, and
searches of the equilibrium path found no segment sum or scatter-add. This does
not exclude nondeterministic library kernels elsewhere; it makes an
atomics-accumulation explanation for `divergence_b` less fit than state or
upstream arithmetic variation.

## Acceptance and comparison machinery to reuse

| Capability | Existing site | Fitness verdict |
|---|---|---|
| Frozen six-case identity and profile reconstruction | `_case_rows`, `build_profile`, `_with_moment_geometry`, `_stored_lcfs` from `benchmarks/jitted_eager_parity_gate.py:266-302`, already consumed by `_case_measurement` | **REUSE WITHOUT RESELECTION.** It is the accepted cohort and avoids turning determinism measurement into a data-selection change. |
| Result-tree naming | `_named_tree` and `_leaves`, `benchmarks/jitted_eager_parity_gate.py:130-157` | **REUSE.** They name all full-receipt leaves stably, including `flux`, every registered observable and the `conservation.divergence_b` canary. |
| Exact state/leaf difference | `_difference`, `benchmarks/observable_route_discriminator.py:156-175` | **REUSE.** It reports `exactly_equal`, maximum absolute difference and reference-scaled maximum relative difference for finite equal-shaped arrays. Extend it with unequal-element count for the terminal-state requirement. |
| Every-registered-observable scoring | `evaluate_observable_bound_acceptance`, `nova/equilibrium/observable_acceptance.py:168-230` | **REUSE FOR LABEL VERDICTS.** It validates shapes/dtypes and scores all cases and batch members. It currently returns only differences and verdicts, so raw values must be retained separately when required for diagnosis. |
| Canary computation | `conservation_ledger` → `ConservationLedger.divergence_b`, `nova/equilibrium/conservation.py:280-399` | **REUSE AND TRACE FIRST.** This is the coarsest known flip: failed both widths in repetition 1 and passed both in repetition 2. If raw state repeats but this leaf moves, retain radial/vertical field, divergence field, checked mask and sup input. |
| Source identity | `_git`, `_sha256` and `source_identity` in `benchmarks/observable_batch_acceptance.py:63-90,587-599` | **REUSE AND STRENGTHEN.** Current filtering requires identical commit, tree, driver and acceptance hashes. Add runtime/JAX/XLA environment and one allocation identity to distinguish source sameness from executable/runtime sameness. |

## Recommended consumption boundary

Keep the measurement in benchmark/evidence code and consume the existing
production interfaces. The minimum reliable shape is:

1. call `configure_dtypes()` before tracing and record JAX/JAXLIB, device,
   environment flags, cache setting, source/tree/driver digests and allocation;
2. build one frozen profile and seed for each of the six accepted cases;
3. create the selected solve callable once per case/width, compile once where
   the production route compiles, then run it **three times in one process** on
   byte-identical copied inputs;
4. block each result and retain every raw `ForwardEquilibrium.flux` array,
   preferably in a lossless binary companion artifact with repetition and case
   axes; never retain only the latest state;
5. compare repetitions 2 and 3 to repetition 1 with bitwise unequal-element
   count, maximum absolute difference and maximum relative difference;
6. flatten every complete receipt and compute the same three figures for all
   **69 registered observables**, naming the first differing leaf;
7. emit `STATE_REPRODUCIBLE` only when all three flux arrays are bitwise equal,
   otherwise `STATE_VARIES`; report label movement independently in either
   case; and
8. run cache-off and fixed-cache arms only as named controls. Do not call cache
   reuse or fixed moment association a deterministic production route unless
   the three-repetition state and all-label evidence demonstrates it.

This preserves the exact production solve, cohort, acceptance definitions and
canary while adding only the missing repetition axis and lossless state
persistence.
