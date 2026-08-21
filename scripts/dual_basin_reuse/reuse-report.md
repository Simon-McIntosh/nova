# Dual-basin reuse map

## Bottom line

The repository already supplies the shared fixed-point engines, fixed-shape batch route,
topology/domain read, branch-aware open-domain source continuation, moving-separatrix clip,
and rich equilibrium receipts. It does **not** yet supply a topology pin, a two-branch result
type, per-branch convergence qualification, or a history selector. The closest complete design
precedent is imas-efit's independent forced-limited/forced-diverted re-solves followed by a pure
receipt-and-history selector. Ambix already freezes the upstream source payload and records
Nova integral results, but its seam has no branch portfolio or selected-branch provenance.

Quantitative inventory: all **6 required Nova modules** are covered below, with **24 reusable
symbols/mechanisms**, **7 focused Nova test/fixture groups**, **4 banked oracle artifacts**, **3
imas-efit precedent surfaces**, and **3 Ambix seam surfaces**. The author-new list contains **9
gaps**. Fitness is `reuse`, `adapt`, or `reference-only`; none is presented as already satisfying
the complete two-branch done-when.

## Nova production candidates

| Module | Candidate symbols/mechanisms | Fitness verdict |
|---|---|---|
| `nova/equilibrium/fixed_point.py` | `FixedPointResult`; `picard`; `anderson`; `newton_krylov`; `kink_aware_newton_krylov` | **Reuse.** All routes have one fixed-shape `(state, residual, trace)` contract and are jit/vmap-safe; wrap the same engine twice under branch-specific maps, but add explicit success/iteration fields at the branch receipt layer rather than changing these generic solvers. |
| `nova/equilibrium/continuation.py` | `OUTWARD_SENSE`; `SeparatrixContinuation`; `ContinuedDomainProfile`; `continuation_record` | **Reuse.** It already distinguishes common-SOL and private-flux branches, validates separatrix anchors, and emits a fixed-shape policy record; it is source-domain continuation, not a diverted/limited solve controller. |
| `nova/equilibrium/topology.py` | `TopologyState`; `BoundaryMode`; `boundary_mode`; `TopologySolveReceipt`; `topology_solve_receipt`; `Topology.read` / `read_with_connectivity` | **Adapt.** It names limited/diverted outcomes, records topology history and failed traversal honestly, and exposes the wall or X-point boundary read; it currently observes emergent topology and cannot require a declared class or return a device-side pin-consistency flag. |
| `nova/equilibrium/domain.py` | `PlasmaDomain`; `DomainMasks`; `classify_domains` | **Reuse.** The integer, fixed-shape partition is the correct shared substrate for both branch maps and keeps core/common-SOL/private-flux/material exclusion on one code path. |
| `nova/equilibrium/forward.py` | `ForwardEquilibrium`; `ForwardProfile.flux_map`; `observe`; `_receipt`; `solve`; `solve_batch` | **Adapt.** This is the one production solve/receipt path and already batches fixed-shape routes, but it accepts one seed and returns one emergent branch; introduce a portfolio wrapper and branch pin without duplicating `_receipt` or the physics map. |
| `nova/equilibrium/forward_operator.py` | `ForwardFluxOperator.read`; `_support_partition`; `current_moments_and_observation`; `internal`; `flux_map` | **Adapt.** This is the narrow topology-to-source seam: the pin should select the boundary anchor/mask here while retaining exactly one current and coupling path. Today `read()` always delegates to the emergent `Topology.read`. |
| `nova/equilibrium/separatrix_clip.py` | `TracedClippedSupports`; `AtomicCellMesh.traced_clip`; `ClippedSupports`; `padded_linear_current_moments` | **Reuse.** Fixed-capacity traced supports already jit/vmap over moving separatrices and preserve continuous clipped moments; feed each branch's pinned boundary flux (and diverted saddle vertex when needed) to the same clip implementation. |

### Receipt machinery already in reach

| Candidate | What it banks | Fitness verdict |
|---|---|---|
| `fixed_point.FixedPointResult` | terminal state, relative residual, fixed-length residual trace | **Reuse inside each branch receipt.** Iteration count/success must be derived or stored explicitly by the portfolio layer. |
| `forward.ForwardEquilibrium` | flux, current, domains, topology, fixed-point trace, moments, current/conservation ledgers, normalisation, rotation, continuation and finite checks | **Reuse as each branch payload.** Do not invent a second equilibrium result family; compose two of these with pin-consistency and selection receipts. |
| `topology.TopologySolveReceipt` | named final class, boundary/wall point, class history, transition count and solver-success qualification | **Adapt for the pin.** The semantics are nearly exact, but the solve path does not currently collect every device iterate and the receipt lacks `requested_class` / `class_consistent`. |
| `source.ContinuationRecord` via `ContinuedDomainProfile.continuation_record` | domain, continuation form/order/support/width, edge anchors and truncated fraction | **Reuse unchanged.** It is already strict provenance for the branch-owned source continuation. |
| `scripts/oracle_rebaseline/measure.py` `_history_receipt`, `_measure_root`, `_array_identity`, `_json_write` | criterion, evaluation budget, finite trace indices, topology, moments, conservation, local-gauge identity and strict JSON/array digests | **Lift the schema ideas, not the script API.** These fields are the strongest existing model for honest per-branch convergence and bank integrity. |

## Tests and banked oracle substrate

| Path / test | Existing evidence | Fitness verdict |
|---|---|---|
| `tests/test_fixed_point.py::{test_vmap_batch_matches_per_slice_solves,test_trace_layout_shares_one_evaluation_axis}` | Generic schemes batch identically and retain fixed trace layout. | **Reuse as engine coverage; extend with a leading branch axis and two branch maps.** |
| `tests/test_equilibrium_forward_solve.py::{test_the_accelerated_solve_reaches_its_fixed_point,test_the_batched_ensemble_solve_matches_the_per_slice_solve,test_an_externally_supplied_flux_map_is_qualified_by_the_same_receipt}` | Production convergence, batch parity and common receipt qualification. | **Reuse fixtures and assertions; add a portfolio cold-seed case without weakening the current single-solve tests.** |
| `tests/test_topology_boundary.py` | Wall-binding and X-point-binding truth cases, polarity, private-flux shadow and limited Gaussian normalization. | **Reuse as the pin's low-level truth table.** It has both anchor mechanisms but no requested-class API. |
| `tests/test_limiter_topology_receipts.py` | Every solve publishes a class; failed solves do not claim successfully traversed topology changes; limited contact is banked independently of EFIT inputs. | **Reuse receipt invariants; add contradiction-to-pin and explicit non-convergence cases.** |
| `tests/test_domain_participation_continuity.py` and `tests/test_equilibrium_separatrix_clip.py` | Continuous participation and current moments through label/LCFS motion; traced clip jits once and vmaps. | **Reuse as the no-parallel-physics and fixed-shape rail for both branches.** |
| `tests/test_solovev_recovery_gates.py` | Independent seeds, local gauge discipline, strict recovery registry, serialized root digests and warm cache sentries. | **Reuse as the dual-root bank and recovery gate harness; add branch-class qualification rather than interpreting residual alone.** |
| `tests/test_equilibrium_analytic_oracle.py` plus `scripts/analytic_oracle_fixtures/` | Closed-form production boundary read and dependency-free analytic carrier. | **Reuse as the analytic branch oracle.** It does not presently prove an opposite topology class. |

### Banked dual-root facts and qualification

The reusable bank is:

- `scripts/oracle_rebaseline/root-coarse.npz` and `root-fine.npz`: each contains
  `oracle_state`, `root_state`, `seed_state`, residual trace, locally anchored normalized flux,
  axes and cell currents, all accompanied by SHA-256 identities in the JSON receipts.
- `scripts/oracle_rebaseline/receipt-coarse.json` and `receipt-fine.json`: the production
  current-centroid seed reaches a criterion-qualified alternate root at residuals
  `2.494170604898324e-15` and `2.172012894849535e-15`.
- `scripts/oracle_rebaseline/results.json`: merged gates, artifacts and the explicit
  `alternate-root-hold` verdict; the alternate root differs from the analytic root by
  `0.53331` and `0.53384` of analytic flux span.
- `scripts/oracle_rebaseline/gates.py`: `validate_registry`, `validate_gauge_discipline` and
  `validate_artifacts` are ready-made integrity and no-false-pass checks.

Important qualification: the current bank proves **two genuine fixed points**, separated in
axis radius by about **74.84 mm coarse** and **74.19 mm fine**, but both the `oracle_topology`
and `root_topology` fields currently say `class: limited` with no X-point. It therefore does
not yet, by itself, prove one diverted and one limited branch. The two-branch test must either
add a genuinely diverted analytic fixture/root or explain and correct this bank/plan mismatch;
changing the topology label in the receipt is not evidence.

## imas-efit precedent

| Candidate | Mechanism | Fitness verdict |
|---|---|---|
| `src/EFIT/fit.f90::fit_dual_stream` | Snapshots the complete slice seed, runs independent forced-L and forced-D solves, qualifies convergence/realization/plausibility, then commits a clean winning re-solve. | **Reference architecture.** Port the separation of branch solve from selection and the independent-state invariant; do not port Fortran global state, marginal-only arming, or measurement-chi-squared policy into Nova. |
| `src/EFIT/contour_tree.f90::select_dual_stream_branch` and `basin_fork_should_arm` | Pure hierarchical selection: sole valid stream; separated fit score; tied streams use prior pulse anchor; explicit neither case and sub-rule receipt. | **Reference and translate.** The temporal-anchor/tie pattern is the strongest hysteresis precedent, but Nova needs its own declared admissibility and cold-start criterion over forward receipts. |
| `src/EFIT/bound.f90` forced-branch controller plus `get_dual_stream_anchor` / `update_dual_stream_anchor` | Pins the solve's boundary mode and stores a pulse-relative unambiguous branch anchor. | **Reference semantics only.** Re-express as immutable JAX inputs/outputs; module `SAVE` state and in-loop topology latches would break Nova's jit/vmap contract. |
| `tests/unit/fortran/test_contour_tree.f90` selector matrix | Exercises sole convergence, neither convergence, separated scores, tied scores, prior limited/diverted anchors, validity/plausibility disqualification and deterministic repeats. | **Port the case matrix.** It is a direct template for a pure Python/JAX selector test, with Nova-specific receipt fields and no EFIT labels in Nova symbols. |

The older `tests/unit/test_lcfs_hysteresis.cpp` and `lcfsHysteresis` control are **reference-only**:
they suppress in-iteration class toggles and mirror the gate outside the Fortran binding. The
locked Nova strategy is post-convergence branch selection, so this mechanism is useful as a
warning against chattering but is not the implementation to copy.

## imas-ambix seam

| Candidate | Existing contract | Fitness verdict |
|---|---|---|
| `imas_ambix/fluxstate/contract.py::FluxFunctionState` | Immutable force-balance state with explicit COCOS/sign provenance, domain-qualified profiles, derivatives, ensemble/member identity, temporal provenance and integral policy. | **Reuse unchanged upstream.** Branch choice belongs to the deterministic solve/result side, not the source-state contract. |
| `imas_ambix/fluxstate/consumer_contract.py::to_nova_forward_payload` / `NovaForwardPayload` | Freezes arrays and source identity, preserves absolute versus upstream-conditioned policy, requested moments and ensemble/member ids; explicitly leaves algorithm selection to the consumer. | **Reuse as the input seam.** A single payload can feed both branch maps; no duplicate or renormalized source state is needed. |
| `imas_ambix/fluxstate/consumer_contract.py::record_nova_integrals` / `NovaForwardReceipt` | Records returned integral residuals against the exact source digest and pinned Nova revision without modifying profiles. | **Adapt after Nova lands.** Add a separate selected-branch/portfolio result record (or extend through a versioned contract change); the current receipt cannot identify both candidates, selection reason, history anchor or switch. |
| `tests/fluxstate/test_consumer_contract.py` | Pins immutability, exact source handoff and before/after source identity around returned integrals. | **Reuse and extend.** Require dual evaluation to leave one input digest unchanged and bank candidate plus selected-branch provenance. |

## What must be authored new

1. A device-compatible topology-pin value (`limited` / `diverted`) passed into the one
   `ForwardFluxOperator` read/partition path, selecting wall-contact versus saddle boundary
   normalization without cloning current, Green-matrix, source or receipt physics.
2. A pin-consistency result computed on-device at each branch's terminal read, plus explicit
   non-convergence when the requested topology is unavailable; no emergent class switch may
   silently become success.
3. A branch receipt containing requested class, achieved class, converged flag, residual,
   iterations/evaluations, topology consistency, and the existing `ForwardEquilibrium` payload.
4. A fixed-shape two-branch portfolio result and solve entry point that batches limited and
   diverted states under jit/vmap while preserving the existing single-solve API.
5. Cold production seed construction for each class. Existing current-centroid seeds and banked
   roots are useful inputs, but no production helper currently constructs a saddle-anchored
   diverted cold seed separately from a wall-anchored limited seed.
6. A pure history selector over branch receipts with an explicit cold-start rule, persistence
   threshold, disappearance/admissibility transition rule, no-chatter behavior and a selection
   receipt naming why a switch occurred.
7. A synthetic time sequence and selector test matrix, including one-valid, both-valid,
   neither-valid, tied/admissibility cases and a limited-to-diverted transition without chatter.
8. A genuinely opposite-topology oracle pair (or a corrected, evidenced reclassification of the
   existing analytic bank). The current root arrays are numerically excellent but both receipts
   are limited, so they cannot close the topology-pinned two-class acceptance test.
9. A versioned Ambix consumer-result addition for both candidate identities, selected branch,
   history input, selection criterion/switch reason and unchanged source digest; the upstream
   `FluxFunctionState` and payload need no branch duplication.

## Recommended assembly order

Build the pin at `ForwardFluxOperator.read`/partition and prove wall/X anchor truth first; wrap
the existing `ForwardProfile` solver twice into a fixed-shape portfolio; compose existing
equilibrium and topology receipts into branch receipts; then port the imas-efit pure selector
case matrix with Nova-specific admissibility and history. Close on the banked root-integrity
tests plus a newly evidenced opposite-topology fixture. This order keeps one code path per
physical quantity and makes selection depend only on completed, honest receipts.
