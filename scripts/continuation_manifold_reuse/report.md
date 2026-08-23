# Continuation-manifold and seed-direction reuse map

## Scope and search basis

This is an implementation reuse map, not a claim that a topology-manifold
predictor-corrector already exists. It was prepared from Nova at `bacbbd81` and
from tracked Ambix prior art at `c0c75b3`. The search covered production modules,
benchmarks, tests, banked receipts, and available datasets in both repositories.
The five-frame DIII-D store and the MAST shot store are present at their declared
paths; no numerical solve was rerun for this inventory.

Searched capabilities:

- topology classification and the existing admissibility predicate;
- fixed-shape nonlinear stepping under `jit` and `vmap`;
- solve-state predictor-corrector, pseudo-arclength advance, manifold tangent,
  and projection/correction;
- Krylov action qualification and local projected-operator information;
- topology-pinned terminal branch receipts;
- seed construction, controlled seed-direction perturbation, and candidate
  enumeration;
- seed-alignment observables computable from a seed and operator without a
  converged answer;
- fixed-shape candidate ranking and rank-correlation machinery;
- contract tests for admission, batching, branch receipts, and seed-direction
  perturbations;
- banked receipts for the five score-blind frames, transient growth, warm-seed
  outcomes, and branch recovery;
- DIII-D, MAST, and synthetic root/state datasets suitable for remeasurement;
- coupled-repository topology, continuation, tangency, and ranking prior art.

## Numbered candidates

1. **Repo:** `nova`; **path:** `nova/equilibrium/topology.py`; **symbol:** `TopologyClass`, `TopologyState`; **fitness:** **direct reuse** — these are the production limited/diverted identity and fixed-shape emergent topology payload that the advance route must preserve.

2. **Repo:** `nova`; **path:** `nova/equilibrium/forward_operator.py`; **symbol:** `ForwardFluxOperator.read`; **fitness:** **direct reuse** — this is the existing class predicate over a candidate state, already shared by the production map and receipts, so a second topology test would be duplication.

3. **Repo:** `nova`; **path:** `nova/equilibrium/fixed_point.py`; **symbol:** `kink_aware_newton_krylov(..., admissibility_fn=...)`; **fitness:** **strong scaffold, replace its direction policy** — it supplies fixed-count JAX control flow, exact tangents, action qualification, and caller-owned admission, but its `nonmonotone` arm only scales the departing Newton direction and is therefore the measured stall mechanism rather than manifold advance.

4. **Repo:** `nova`; **path:** `nova/equilibrium/fixed_point.py`; **symbol:** `KinkAwareResult.candidate_admissibility`, `KinkAwareResult.accepted_factors`; **fitness:** **extend rather than fork** — the fixed-shape per-update receipt already records admission and promoted fraction, but it needs predictor length, corrected length, and projection outcome fields to report curvature-limited advance.

5. **Repo:** `nova`; **path:** `nova/equilibrium/fixed_point.py`; **symbol:** `_qualified_krylov_step`, `KrylovActionQualification`; **fitness:** **direct reuse** — every predictor/corrector linear action should pass through the landed fail-closed checks for nonfinite action, solver status, achieved residual, and zero material step.

6. **Repo:** `nova`; **path:** `nova/equilibrium/fixed_point.py`; **symbol:** `_projected_krylov_condition`; **fitness:** **reuse as local subspace evidence, not as the manifold tangent** — its fixed-shape Arnoldi basis and projected singular values can support conditioning or an alignment observable, but they describe `(I-J)` and do not themselves differentiate the topology boundary.

7. **Repo:** `nova`; **path:** `nova/equilibrium/forward.py`; **symbol:** `ForwardProfile.flux_map`; **fitness:** **direct reuse** — this exposes the constrained, requested-class production operator needed by both predictor residuals and corrector residuals without introducing a second physics path.

8. **Repo:** `nova`; **path:** `nova/equilibrium/forward.py`; **symbol:** `ForwardBranchReceipt`, `ForwardProfile._branch_receipt`, `ForwardProfile.solve_branch`; **fitness:** **direct reuse and extend only if required** — requested/achieved class, convergence, residual, iteration count, topology consistency, finite state, and terminal topology already share one typed receipt.

9. **Repo:** `nova`; **path:** `nova/equilibrium/forward.py`; **symbol:** `ForwardProfile.solve_diverted_perturbations`, `ForwardPerturbedSeedReceipt`; **fitness:** **strong seed-study reuse** — it builds fixed-shape direction-normalized seed ladders against the production topology-pinned solve, although it measures one supplied direction rather than ranking several directions.

10. **Repo:** `nova`; **path:** `nova/equilibrium/stencil_nulls.py`; **symbol:** `_native_candidate_stage` (`jax.lax.top_k` candidate compaction); **fitness:** **implementation-pattern reuse only** — it demonstrates deterministic fixed-shape masked ranking with overflow evidence under JAX, but its score ranks critical-point cells, not solve seeds.

11. **Repo:** `nova`; **path:** `benchmarks/diiid_repaired_solve_remeasure.py`; **symbol:** `COHORT`, `_prepare_frame`, `_solve_frame`; **fitness:** **direct remeasurement harness** — it reconstructs the exact five score-blind 24-current constrained diverted inputs and already reports residual, admissions, terminal class, X-point finiteness, and action qualification.

12. **Repo:** `nova`; **path:** `benchmarks/event_resolved_amplification.py`; **symbol:** `_arnoldi_event`; **fitness:** **best prototype for the seed-alignment observable** — it lifts the smallest-singular projected `(I-J)` direction into state space and computes seed/incoming-direction overlap from the seed and local operator, but it is NumPy/host benchmark code and must be recast for production JAX use.

13. **Repo:** `nova`; **path:** `benchmarks/event_resolved_amplification.py`; **symbol:** `_event_discrimination`; **fitness:** **direct study reuse** — it already computes Spearman correlations against growth and distinguishes Krylov condition from active-mode overlap, exactly the analysis shape needed to adjudicate a proposed ranking observable.

14. **Repo:** `nova`; **path:** `benchmarks/event_resolved_amplification.py`; **symbol:** `_alternate_seed`, `_seed_prediction`; **fitness:** **direct paired-direction study reuse** — it constructs a real same-shot alternate moment seed and banked a direction change from three bursts to none and cumulative growth from `5.379792627e9` to `0.06186094`; it supplies evidence, not a general candidate enumerator.

15. **Repo:** `nova`; **path:** `benchmarks/efit_parity_warm_neighbour.py`; **symbol:** `_candidate_rows`, `_find_mast_newton_warm_source`, `measure_newton_reference`; **fitness:** **reuse the corpus and outcome qualification, reject the ordering rule** — it enumerates bounded same-shot candidates and records their own solve outcomes without label-based selection, but temporal adjacency is precisely the variable the new ranking must replace.

16. **Repo:** `nova`; **path:** `benchmarks/diiid_constrained_cold_start.py`; **symbol:** `NEIGHBOUR_FRAME_OFFSETS`, `_neighbour_candidates`, `_solve_public_seam`; **fitness:** **reuse as a shared enumeration/solve seam** — the declared ladder and constrained terminal checks are already imported unchanged by MAST, while any alignment score should reorder or filter the returned candidates without copying these helpers.

17. **Repo:** `imas-ambix`; **path:** `imas_ambix/latent/topology_objectives.py`; **symbol:** `SliceAnchor`, `build_slice_anchor`, `terminator_penalty`; **fitness:** **conceptual projector pattern only** — it builds fixed candidate slots and X-point/limiter tangential projectors, but it is a Torch training penalty in physical gradient space, not a JAX corrector onto Nova's solve-state topology manifold.

18. **Repo:** `imas-ambix`; **path:** `imas_ambix/fluxstate/contract.py`; **symbol:** `DomainProfilePolicy`; **fitness:** **semantic reuse only** — it is the typed contract for physical source continuation on a topology-qualified domain and must not be confused with nonlinear solve-state continuation.

19. **Repo:** `nova`; **path:** `tests/test_topology_qualified_admission.py`; **symbol:** `test_an_offered_limited_candidate_is_refused`, `test_an_offered_nonfinite_candidate_is_refused`, `test_an_admissible_diverted_candidate_is_accepted`, `test_an_unqualified_krylov_action_selects_no_backtracking_trial`; **fitness:** **direct regression reuse** — these pin fail-closed candidate admission and state non-promotion and should remain green for every new step strategy.

20. **Repo:** `nova`; **path:** `tests/test_factor_ladder_extension.py`; **symbol:** `test_fixed_ladder_reaches_measured_fraction_under_jit_and_vmap`; **fitness:** **direct shape regression, negative algorithm comparator** — it proves the smallest safe fraction is reachable under `jit`/`vmap`, while the associated measurement demonstrates that merely extending backtracking does not improve convergence.

21. **Repo:** `nova`; **path:** `tests/test_equilibrium_forward_solve.py`; **symbol:** `test_the_topology_portfolio_matches_per_branch_solves_under_jit_and_vmap`, `test_a_terminal_class_contradiction_stays_in_its_branch_receipt`; **fitness:** **direct integration reuse** — these pin batched branch identity and honest terminal contradictions on the production `ForwardProfile` seam.

22. **Repo:** `nova`; **path:** `tests/test_two_class_recovery.py`; **symbol:** `test_diverted_near_basin_perturbation_ladder_recovers_banked_root`, `test_banked_diverted_state_is_a_machine_precision_pinned_root`; **fitness:** **direct seed-direction fixture reuse** — these provide a digest-pinned diverted root, normalized perturbation ladder, `jit`/`vmap` contract, and topology/root-parity qualifications.

23. **Repo:** `nova`; **path:** `tests/test_krylov_route_conditioning.py`; **symbol:** `test_conditioning_receipt_is_fixed_shape_under_jit_and_vmap`, `test_kink_aware_route_carries_the_same_conditioning_receipt`; **fitness:** **direct solver-shell regression** — these ensure a new advance arm cannot silently drop the shared local conditioning and fixed-shape receipts.

24. **Repo:** `nova`; **path:** `tests/test_amplification_observation.py`; **symbol:** `test_sustained_growth_is_reported_without_blocking_terminal_promotion`, `test_qualified_contracting_trajectory_has_its_own_observation`; **fitness:** **reuse as a coarse outcome label only** — the advisory enum is useful receipt context, but binary sustained-growth classification is too weak to rank candidate seed directions.

25. **Repo:** `imas-ambix`; **path:** `tests/latent/test_topology_objectives.py`; **symbol:** `test_tangency_projector_ignores_normal_component`, `test_terminator_penalty_invalid_candidates_are_inert`; **fitness:** **port the invariants, not the implementation** — tangent-only response and inert padded candidates are useful corrector tests, but the test substrate and state space differ from Nova.

26. **Repo:** `nova`; **path:** `docs/figures/diiid-forward-onboarding/repaired-solve-five-frame-remeasure.json`; **symbol:** `frame_records[]`; **fitness:** **authoritative pre-extension comparator** — all five terminals stay diverted with finite X points, while only `4-9` of `89` updates promote and residuals remain `0.0854-0.1644`.

27. **Repo:** `nova`; **path:** `docs/figures/diiid-forward-onboarding/factor-ladder-extension.json`; **symbol:** `solver_contract`, `frame_records[]`; **fitness:** **authoritative banked baseline and negative** — it fixes the six-factor JAX-safe route and the required admitted counts `7, 7, 10, 5, 9`, but confirms the `0.03125` rung does not close convergence.

28. **Repo:** `nova`; **path:** `docs/figures/diiid-forward-onboarding/topology-qualified-admission.json`; **symbol:** `admission_contract`, `verdict`; **fitness:** **direct topology receipt reuse** — it proves a finite diverted X point across the qualified run while explicitly retaining `2.925438173e-4` as a nonconverged result rather than wording topology preservation into a solve pass.

29. **Repo:** `nova`; **path:** `docs/figures/forward-operator-refinement/event-resolved-amplification.json`; **symbol:** `event_discrimination`, `predictions.different_seed_direction`, `predictions.warm_ladder_directionality`; **fitness:** **authoritative seed-direction evidence input** — condition/growth Spearman is `0.6636`, active-mode-overlap/growth is only `0.1273` on the single event path, and the five-frame warm split is qualified by the `21985/51` outlier rather than treated as a universal law.

30. **Repo:** `nova`; **path:** `docs/figures/efit-forward-parity/warm-neighbour-stall-lift.json`; **symbol:** `references[].cold_control`, `references[].warm_solve`, `references[].warm_neighbour_search`; **fitness:** **direct ranking-study outcome corpus** — it contains cold/warm terminal residuals and candidate provenance for the frozen MAST stalls, but its selected source was adjacency-ordered and must not be accepted as an alignment label without rescoring every candidate.

31. **Repo:** `nova`; **path:** `docs/figures/dual-basin-solve/portfolio-warm-start-receipt.json`; **symbol:** `seed_policy`, `matrix_rows`, `catalog_projection`; **fitness:** **direct branch-safe seed-distance comparator** — it qualifies both branches at relative seed distance `1e-3` and two Newton steps, but direction is fixed by cold-seed-minus-root and the `19.07 ms` projected portfolio result is a performance receipt, not an alignment score.

32. **Repo:** `nova`; **path:** `scripts/dual_basin_fixtures/diverted-state.npz`; **symbol:** `state`; **fitness:** **direct synthetic diverted root fixture** — its digest-pinned machine-precision root is suitable for tangent/corrector unit and perturbation tests, but it does not represent the five real score-blind stalls.

33. **Repo:** `nova`; **path:** `scripts/oracle_rebaseline/root-coarse.npz`, `scripts/oracle_rebaseline/root-fine.npz`; **symbol:** `root_state`, `seed_state`; **fitness:** **direct limited-root controls** — these provide independent root/seed pairs for checking that a generic continuation component is not diverted-only.

34. **Repo:** `nova`; **path:** `docs/figures/diiid-forward-onboarding/diverted-root/host_large_budget_terminal_state.npz`; **symbol:** `state`, `current`, `seed`; **fitness:** **direct real-carrier state bank** — it avoids rebuilding the known favourable DIII-D carrier and can exercise advance diagnostics, but it is one frame rather than the fixed five-frame done-when cohort.

35. **Repo:** `nova`; **path:** `/work/projects/imas_gpu/sophelio/raw/data/diii_d_train`; **symbol:** `benchmarks.diiid_repaired_solve_remeasure.COHORT`; **fitness:** **authoritative real five-frame dataset** — the five declared Parquet files and frame indices are present and already rebuild the constrained production operator without consulting an achieved score.

36. **Repo:** `nova`; **path:** `/work/projects/imas_gpu/mast/level1/shots`; **symbol:** `benchmarks.parity_divergence_attribution.TRAJECTORY_CASE`, `benchmarks.event_resolved_amplification.ALTERNATE_SEED_ROW`; **fitness:** **authoritative seed-direction dataset** — shot/slice `21985/51` plus alternate row `49` supplies the extreme paired-direction case, and the frozen warm-neighbour references add four more candidate-outcome cases.

## Explicit no-candidate results

- **No candidate — production topology-manifold predictor-corrector:** no Nova or Ambix production/test symbol implements predictor-corrector or pseudo-arclength solve-state stepping; the only production Nova globalization scales a Newton direction.
- **No candidate — solve-state arclength parameter:** searches found physical contour arclength and source/profile continuation only, with no state-space arclength coordinate, bordered continuation system, or tangent normalization.
- **No candidate — topology-manifold tangent basis:** no symbol differentiates the admissibility/class boundary into a solve-state tangent space; `_projected_krylov_condition` and Ambix's limiter tangency projector act in different spaces.
- **No candidate — manifold corrector/projector:** no fixed-shape JAX routine projects a predicted state back to the requested topology manifold while retaining material advance.
- **No candidate — production seed-alignment ranking API:** the event-resolved benchmark computes useful local directions and correlations, and the null finder demonstrates fixed-shape `top_k`, but there is no reusable production API that scores and ranks equilibrium seed candidates from seed-plus-operator inputs.
- **No candidate — predictor-corrector contract test:** no test covers tangent prediction, correction convergence, curvature-limited step size, or `jit`/`vmap` parity for such a method because the method is absent.
- **No candidate — complete advance receipt:** existing results expose `accepted_factors` and admission counts, but no receipt directly records mean advance fraction, predictor/corrector lengths, or achieved Newton-step-equivalents; those fields must be added or derived explicitly for the five-frame measure.

## Recommended reuse boundary

Implement the new strategy inside the existing fixed-shape solver shell, call
the existing `ForwardFluxOperator.read` predicate, route every linear action
through `_qualified_krylov_step`, and return through the existing branch receipt.
Prototype seed alignment from the event-resolved Arnoldi active direction and
reuse its Spearman study shape, but preregister the observable before scoring all
available candidates. Keep source-domain continuation, adjacency order, and the
Ambix physical-space tangency penalty as explicit non-substitutes.
