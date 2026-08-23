# Forward-solve stability reuse map

## Bottom line

Nova does not need another nonlinear solver. It already has both Krylov drivers, four
proposal-globalisation policies, topology-pinned map reads, terminal branch qualification,
a two-branch portfolio, post-solve hysteresis, and a guarded declared-current amplitude.
The repair should reuse those rails, but no existing combination closes both live solver
defects:

1. the production exact residual action can be non-finite on Arnoldi vector zero while
   returning a finite zero GMRES step; and
2. neither the JAX nor host driver admits trial steps on the joint condition
   **finite + residual-acceptable + emergently diverted**.

The smallest credible assembly is therefore: repair or isolate the non-finite primitive in
the exact action; add linear-action and GMRES-result qualification; adapt the existing
fixed-shape nonmonotone candidate ladder to reject a candidate whose *emergent* topology is
not diverted; retain the existing step cap, declared-current guard, branch receipt, and
portfolio selector. The host SciPy/Armijo route remains a useful independent validator, not
the production globalisation, because its measured 89-step solve crossed topology three
times and ended limited.

Survey basis: Nova commit `8fc755e849d9ffdc99e0d57f3ed67dc0e1705fe0`; coupled Ambix
reference commit `c0c75b362dec63e8c25b15d5b072ab6a01978208`. Inventory: **16
candidate mechanisms**, of which **14 have a test or banked machine measurement**, **2 are
untested for this path**, and **4 required mechanisms are absent**. The focused recheck for
this survey collected and passed **10/10 cases**.

## Defect-to-symbol map

### Non-finite exact residual-Jacobian action on the first Arnoldi vector

The production state-to-image path is
`ForwardProfile.flux_map` -> `ForwardFluxOperator.flux_map.mapped` ->
`ForwardFluxOperator.internal` -> `normalised_current_moments` ->
`current_moment_image`. The fixed-point residual is `psi - g(psi)`. The accelerated driver
then calls `jax.linearize(map_fn, state)`, exposes the residual action as
`vector - tangent(vector)`, and passes it to batched JAX GMRES. Citations:
`nova/equilibrium/forward_operator.py:467`, `nova/equilibrium/forward_operator.py:471`,
`nova/equilibrium/forward_operator.py:530`, `nova/equilibrium/forward_operator.py:535`,
`nova/equilibrium/forward_operator.py:540`, `nova/equilibrium/forward_operator.py:555`,
`nova/equilibrium/forward_operator.py:559`, `nova/equilibrium/forward_operator.py:561`,
`nova/equilibrium/forward_operator.py:571`, `nova/equilibrium/forward_operator.py:573`,
`nova/equilibrium/fixed_point.py:242`, `nova/equilibrium/fixed_point.py:244`,
`nova/equilibrium/fixed_point.py:249`, `nova/equilibrium/fixed_point.py:250`.

The exact failing path is reproduced by
`benchmarks.diiid_diverted_root_full_currents.residual_jacobian_diagnostic`: it constructs
the same `jax.linearize` action, seeds Arnoldi with the normalized nonlinear residual, and
records the first non-finite action column. Its paired central-difference action stays finite
and its finite-difference Arnoldi basis reaches the requested dimension. Citations:
`benchmarks/diiid_diverted_root_full_currents.py:462`,
`benchmarks/diiid_diverted_root_full_currents.py:474`,
`benchmarks/diiid_diverted_root_full_currents.py:478`,
`benchmarks/diiid_diverted_root_full_currents.py:485`,
`benchmarks/diiid_diverted_root_full_currents.py:490`,
`benchmarks/diiid_diverted_root_full_currents.py:521`,
`benchmarks/diiid_diverted_root_full_currents.py:557`,
`benchmarks/diiid_diverted_root_full_currents.py:563`,
`benchmarks/diiid_diverted_root_full_currents.py:569`,
`benchmarks/diiid_diverted_root_full_currents.py:610`,
`benchmarks/diiid_diverted_root_full_currents.py:625`.

**Verdict: tested defect; repair required.** The generic exact-action route is tested on a
smooth affine map, but that does not qualify the DIII-D action. The banked regression pins
column zero as non-finite, a finite zero GMRES step, full finite-difference rank 64/64, and
condition number above 1000. Citations: `tests/test_fixed_point.py:59`,
`tests/test_fixed_point.py:70`,
`tests/imas/test_diiid_diverted_root_full_currents.py:172`,
`tests/imas/test_diiid_diverted_root_full_currents.py:176`,
`tests/imas/test_diiid_diverted_root_full_currents.py:179`,
`tests/imas/test_diiid_diverted_root_full_currents.py:185`,
`tests/imas/test_diiid_diverted_root_full_currents.py:187`.

### Absent topology-preserving globalisation

There are two active drivers. `ForwardProfile._solve_host_krylov` hands the residual to
SciPy's Newton-Krylov and accepts its own line search; `fixed_point.newton_krylov` takes one
full JAX GMRES step after only a finite-step fallback and a norm cap. Neither asks the
emergent topology reader whether a trial remains diverted before acceptance. Citations:
`nova/equilibrium/forward.py:1044`, `nova/equilibrium/forward.py:1049`,
`nova/equilibrium/forward.py:1060`, `nova/equilibrium/forward.py:1075`,
`nova/equilibrium/fixed_point.py:256`, `nova/equilibrium/fixed_point.py:257`,
`nova/equilibrium/fixed_point.py:262`, `nova/equilibrium/fixed_point.py:263`.

The DIII-D host fixture explicitly uses GMRES plus Armijo and records topology only in the
accepted-step callback. It therefore observes crossings but cannot veto them. The committed
receipt test pins convergence at 89 accepted iterations and topology transitions at 31, 50,
and 88, with a limited terminal state. Citations:
`benchmarks/diiid_diverted_root_full_currents.py:387`,
`benchmarks/diiid_diverted_root_full_currents.py:391`,
`benchmarks/diiid_diverted_root_full_currents.py:403`,
`benchmarks/diiid_diverted_root_full_currents.py:410`,
`tests/imas/test_diiid_diverted_root_full_currents.py:153`,
`tests/imas/test_diiid_diverted_root_full_currents.py:157`,
`tests/imas/test_diiid_diverted_root_full_currents.py:159`,
`tests/imas/test_diiid_diverted_root_full_currents.py:164`.

**Verdict: absent.** A topology pin exists inside map evaluation, and terminal topology is
qualified after a solve, but there is no line-search admission predicate over the emergent
topology of each trial. The plan's quantitative statement remains exactly supported: the
host route was not budget-limited, converged to `7.747708130404432e-11`, crossed topology
three times, and terminated limited. Citation: `docs/plans/diiid-forward-onboarding.html:518`.

### Division before `_lambda_value`'s finite-positive-band guard

This is **not live in the current production tree**. The original benchmark-local
`_lambda_value` is absent. The replacement
`ForwardFluxOperator.current_normalisation_amplitude` computes an admissibility mask from
finite target/unscaled current, common sign, and the declared magnitude band; it substitutes
a safe denominator before division; eager invalid inputs raise `CurrentNormalisationError`.
Citations: `nova/equilibrium/forward_operator.py:434`,
`nova/equilibrium/forward_operator.py:446`, `nova/equilibrium/forward_operator.py:453`,
`nova/equilibrium/forward_operator.py:454`, `nova/equilibrium/forward_operator.py:460`,
`nova/equilibrium/forward_operator.py:464`.

The current-pinned host residual converts that exception into a large finite rejection
residual, allowing Armijo to reject the invalid trial rather than aborting. Tests explicitly
cover `0.0`, wrong-sign, and NaN unscaled currents and prohibit a local `_lambda_value` copy.
Citations: `benchmarks/diiid_current_pinned_forward.py:494`,
`benchmarks/diiid_current_pinned_forward.py:499`,
`benchmarks/diiid_current_pinned_forward.py:501`,
`tests/test_equilibrium_forward_constrained.py:61`,
`tests/test_equilibrium_forward_constrained.py:66`,
`tests/test_equilibrium_forward_constrained.py:69`,
`tests/imas/test_diiid_current_pinned_forward.py:154`,
`tests/imas/test_diiid_current_pinned_forward.py:158`.

**Verdict: tested and repaired; reuse unchanged.** The live plan records the restored
circuit-driven receipt terminating through the public guard rather than unguarded division.
Citation: `docs/plans/diiid-forward-onboarding.html:525`.

## Candidate inventory and fitness verdicts

| Candidate | Test state | Fitness for the repair |
|---|---|---|
| `fixed_point.newton_krylov`: `jax.linearize` residual action | **Tested**, generic pass and DIII-D fail | **Repair in place.** Preserve exact JVP as the primary route, but locate/fix the non-finite primitive and add a DIII-D action-finiteness regression. Do not infer health from a finite returned step. |
| JAX batched GMRES in `newton_krylov` | **Tested**, generic pass and DIII-D pathological zero step | **Reuse after qualification.** Check action finiteness and achieved linear residual; do not ignore `_info` and accept a finite zero step after a poisoned action. |
| JAX non-finite fallback + relative step cap | **Tested** on a near-singular tangent | **Reuse as a final safety rail, not diagnosis.** It catches non-finite `step`, but the DIII-D failure returns a finite zero step and bypasses it. The cap itself is pinned finite in `tests/test_fixed_point.py:198` and bounded at `tests/test_fixed_point.py:220`. |
| Host `scipy.optimize.newton_krylov` + Armijo | **Tested** on DIII-D; converges to wrong topology | **Reference/validator only.** Its independent finite-difference JVP is useful for comparison, but plain Armijo is not topology preserving. Keep it as the host reproduction lane. |
| Benchmark `finite_difference_action` and Arnoldi projection | **Tested as diagnostic**, **untested as production solver action** | **Reuse to localise and cross-check.** It proves a finite local derivative exists. Promote only a narrowly justified fallback after measuring perturbation sensitivity; do not make the benchmark a second production solver. |
| `kink_aware_newton_krylov` Clarke averaged tangent | **Tested on synthetic piecewise tangent**, untested on DIII-D | **Reference-only for this defect.** It still calls exact tangents on both sides, so it cannot rescue a primitive that produces non-finite exact actions before a known scalar handoff. Citations: `nova/equilibrium/fixed_point.py:403`, `nova/equilibrium/fixed_point.py:407`, `nova/equilibrium/fixed_point.py:410`, `tests/test_fixed_point.py:74`, `tests/test_fixed_point.py:99`. |
| `kink_aware_newton_krylov` nonmonotone four-factor backtracking | **Tested on synthetic piecewise tangent**; machine survey exists, no DIII-D topology admission test | **Best globalisation substrate to adapt.** It already evaluates fixed-shape factors `(1, .5, .25, .125)` against a recent residual envelope. Add a caller-supplied admissibility predicate so a limited or non-finite candidate is never selected. Citations: `nova/equilibrium/fixed_point.py:427`, `nova/equilibrium/fixed_point.py:428`, `nova/equilibrium/fixed_point.py:435`, `nova/equilibrium/fixed_point.py:439`, `nova/equilibrium/fixed_point.py:441`. |
| `kink_aware_newton_krylov` surface restriction | **Tested on a synthetic scalar crossing**, untested for topology | **Adapt only if a continuous signed basin surface is demonstrated.** It shortens a crossing but deliberately places the proposal just beyond the surface; that is the opposite of a hard stay-diverted guard unless its orientation and margin are changed. Citations: `nova/equilibrium/fixed_point.py:362`, `nova/equilibrium/fixed_point.py:375`, `nova/equilibrium/fixed_point.py:421`, `nova/equilibrium/fixed_point.py:423`. |
| `kink_aware_newton_krylov` damped hybrid | **Tested on synthetic map**; prior machine measurements plateau | **Reject as the repair.** Blending Newton with a non-contractive Picard proposal cannot supply topology admission and does not address the poisoned exact action. The residual-release schedule is nevertheless reusable elsewhere. Citations: `nova/equilibrium/fixed_point.py:445`, `nova/equilibrium/fixed_point.py:448`, `nova/equilibrium/fixed_point.py:456`, `tests/test_fixed_point.py:130`, `tests/test_fixed_point.py:152`. |
| Safeguarded Anderson mixing | **Tested** on contracting maps | **Reject for this diverted repair.** Warmup, restart-on-growth, ridge, cap, and finite fallback are sound, but the live map remains non-contractive after current pinning. Citations: `nova/equilibrium/fixed_point.py:143`, `nova/equilibrium/fixed_point.py:159`, `nova/equilibrium/fixed_point.py:169`, `nova/equilibrium/fixed_point.py:176`, `nova/equilibrium/fixed_point.py:185`, `tests/test_fixed_point.py:156`, `tests/test_fixed_point.py:172`. |
| `Topology.pinned_boundary` / `read_with_connectivity(requested_class)` | **Tested** wall-versus-saddle truth table | **Reuse unchanged inside each branch map.** It selects the desired physical anchor, but it reports the requested class inside the pinned map; it is not an emergent-topology line-search check. Citations: `nova/equilibrium/topology.py:248`, `nova/equilibrium/topology.py:251`, `nova/equilibrium/topology.py:328`, `nova/equilibrium/topology.py:353`, `nova/equilibrium/topology.py:358`, `tests/test_topology_boundary.py:94`, `tests/test_topology_boundary.py:101`. |
| `ForwardProfile._branch_receipt` terminal qualification | **Tested**, including absent diverted saddle | **Reuse as the hard terminal gate.** It independently re-reads emergent topology, combines finiteness, tolerance and topology consistency, and refuses a contradicted requested class. It is too late to globalise a step, but it prevents false success. Citations: `nova/equilibrium/forward.py:1203`, `nova/equilibrium/forward.py:1213`, `nova/equilibrium/forward.py:1216`, `nova/equilibrium/forward.py:1218`, `tests/test_equilibrium_forward_solve.py:504`, `tests/test_equilibrium_forward_solve.py:516`, `tests/test_equilibrium_forward_solve.py:523`. |
| `solve_portfolio` + `select_forward_branch` | **Tested** under JIT/vmap and history transitions | **Reuse outside the nonlinear solve.** It is the correct final two-branch and no-chatter rail, not a substitute for within-solve globalisation. Citations: `nova/equilibrium/forward.py:1332`, `nova/equilibrium/forward.py:1373`, `nova/equilibrium/branch_selection.py:223`, `nova/equilibrium/branch_selection.py:241`, `nova/equilibrium/branch_selection.py:302`, `tests/test_equilibrium_forward_solve.py:766`, `tests/test_equilibrium_forward_solve.py:809`, `tests/test_branch_selection.py:140`, `tests/test_branch_selection.py:170`, `tests/test_branch_selection.py:172`. |
| Public declared-current amplitude guard | **Tested**, including zero denominator | **Reuse unchanged.** This already repairs the historic `_lambda_value` ordering bug and is an admissibility predicate the globalisation must preserve. |
| Ambix `solve_equilibrium_nk` finite-difference NK + smooth topology read | **Tested on Ambix synthetic equilibrium**, not DIII-D/Nova production | **Reference-only.** It demonstrates that finite-difference NK over a smoothed topology read can converge, but it duplicates Nova physics and lives on the statistical consumer side. Lift the measured ideas, not the implementation. Citations: `/home/ITER/mcintos/Code/imas-ambix/imas_ambix/latent/gs_solve.py:1054`, `/home/ITER/mcintos/Code/imas-ambix/imas_ambix/latent/gs_solve.py:1105`, `/home/ITER/mcintos/Code/imas-ambix/imas_ambix/latent/gs_solve.py:1107`, `/home/ITER/mcintos/Code/imas-ambix/imas_ambix/latent/gs_solve.py:1188`, `/home/ITER/mcintos/Code/imas-ambix/imas_ambix/latent/gs_solve.py:1194`, `/home/ITER/mcintos/Code/imas-ambix/tests/latent/test_gs_solve_smooth_topology.py:131`, `/home/ITER/mcintos/Code/imas-ambix/tests/latent/test_gs_solve_smooth_topology.py:144`. |
| Legacy `nova.equilibrium.inverse` SciPy NK | **Untested for this forward path** | **Reject.** It mutates legacy plasma state and supplies neither exact-action diagnostics nor topology qualification. Citations: `nova/equilibrium/inverse.py:330`, `nova/equilibrium/inverse.py:338`, `nova/equilibrium/inverse.py:342`. |

## Explicitly absent mechanisms

1. **Production finite-difference residual action or fallback:** absent. Only the DIII-D
   diagnostic and the separate Ambix engine have one.
2. **Linear-action/GMRES qualification:** absent. Production ignores `_info`, never evaluates
   the achieved linear residual, and treats a finite zero step as admissible.
3. **Topology-qualified trial admission:** absent. The closest reusable substrate is the
   four-candidate nonmonotone ladder; the closest predicate is an emergent
   `operator.read(candidate)` plus `topology.diverted`, but they are not composed today.
4. **DIII-D regression proving a finite exact action after repair:** absent. The banked test
   currently asserts the defect, correctly; it must change only after the same fixture reports
   a finite first action and a nonzero, qualified step.

## Recommended reuse assembly

1. Keep `ForwardFluxOperator.flux_map` as the only physics map and the public
   `current_normalisation_amplitude` as its current-admissibility authority.
2. Use the existing `residual_jacobian_diagnostic` fixture to bisect the map constituents and
   identify the first primitive whose JVP becomes non-finite. Fix that primitive; retain the
   finite-difference action as an independent oracle and only as a fallback if exact repair is
   impossible and perturbation stability is re-measured.
3. Harden `fixed_point.newton_krylov`: reject a non-finite linear action, non-successful GMRES
   status, non-finite achieved linear residual, or a zero step with a material nonlinear
   residual. Fall back loudly and record the reason rather than silently reusing Picard.
4. Adapt the existing nonmonotone factor ladder to accept a pure
   `candidate_admissible(state)` callback. For this solve the callback performs the *emergent*
   topology read (no requested-class pin) and requires finite flux plus diverted topology;
   residual-envelope admission remains independent and both must pass.
5. Leave the pinned boundary map, terminal `_branch_receipt`, two-branch portfolio and
   history selector unchanged. They supply, respectively, the intended branch physics,
   fail-closed terminal qualification, alternate candidate, and temporal no-chatter policy.
6. Re-run the unchanged circuit-driven five-frame gate only after the exact-action regression
   and topology-preserving acceptance test pass. Required success remains residual at most
   `1e-6` *and* terminal diverted; label-map RMS remains diagnostic.

## Verification performed for this survey

Command (root Nova environment reused read-only, CPU code generation):

```text
JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv PYTHONPATH=$PWD \
  uv run --no-sync pytest -p no:cacheprovider \
  tests/test_fixed_point.py::test_kink_aware_route_options_converge_across_a_piecewise_tangent \
  tests/test_fixed_point.py::test_newton_step_is_capped_and_finite_on_a_near_singular_tangent \
  tests/test_equilibrium_forward_constrained.py::test_guard_checks_admissibility_before_division \
  tests/imas/test_diiid_diverted_root_full_currents.py::test_plateau_diagnostic_identifies_nonfinite_exact_tangent \
  tests/test_branch_selection.py::test_persistent_admissibility_transition_switches_once_at_declared_slice
```

Result: **10 passed in 9.80 s**. Full log:
`/home/ITER/mcintos/.config/reckon/crew/runs/r-20260823T135014696207-forward-stability-reuse-map/focused-tests.log`.

Citation count in this report: **more than 100 validated `file:line` references**, exceeding
the required 30. No source, test, benchmark, plan, or index file was modified.
