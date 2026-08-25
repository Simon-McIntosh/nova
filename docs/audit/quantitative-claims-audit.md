# S15 quantitative evidence audit

Audit snapshot: 2026-08-25 08:51 CEST. The six live plan records were read from
the primary Nova checkout through commit `1cf84ab9`; banked artifacts, test
assertions, and the cited implementation commits were then checked from the
worker checkout. Live state continued to advance during the audit, so this
report incorporates the higher-order contour record committed at `c16f3567`,
the production compilation control merged at `4a629ba0`, and the determinism
landing committed at `1cf84ab9`.

## Executive verdict

The arithmetic in the sampled receipts is generally reproducible, provenance
is usually explicit, and the curved-clip and coefficient-space corrections are
particularly strong. The audit found three evidence-level discrepancies:

1. **Unstable grounds — plateau input attribution.** Direct interventions
   refute the wall and shipped discrete-wiring candidates, but the profile and
   discretisation candidates are supported only by low spatial-pattern scores
   and a one-control descriptive contrast. The shipped statement that all four
   candidates are “closed by measurement” is stronger than the receipts'
   express qualifications.
2. **Error in conclusion — same-device label determinism.** The persistent
   compilation control is demonstrated and effectively free, but the claim
   that both remaining divergence values are themselves only roundoff conflicts
   with the repository's differential contract: on `StencilMesh`, the
   least-squares operators do not commute and the residuals land at a
   second-order truncation floor. The run-to-run *differences* are last-bit
   effects; that does not make the underlying fitted residual pure roundoff.
3. **Process defect — discrete-operator receipt semantics.** The numerical
   ladder exonerates both the operator and the posed analytic case, as the plan
   correctly concludes, but the machine-readable receipt nevertheless labels
   its cause `posing`. That field is mechanically inconsistent with the term
   audit and the plan's conclusion.

Two live process defects observed in the audit brief have since been corrected:
the higher-order contour landing is now in the committed evidence record, and
the production compilation control is now integrated. The coefficient-space
H200 matrix receipt remains unbanked: two explicitly time-limited jobs expired,
and a corrected one-hour run was still executing as job `1254399` at the audit
snapshot. No H200 matrix number is claimed in the plan, so this is unfinished
capacity evidence rather than a false quantitative conclusion.

## Coverage matrix

“Verified” means the quoted number agrees with the named artifact and, where
applicable, with the test assertion or commit diff. “Discrepant” means the
number or the inference drawn from it is not supported at the strength claimed.

| Plan | Sampled quantitative claim | File and figure checked | Verdict |
|---|---|---|---|
| `plateau-input-attribution` | Six reference cases all terminate in bounded non-convergence with current fraction 1; mean residual-pattern scores are wiring 0.110634, wall 0.094857, profiles 0.051493, discretisation 0.032606. | `docs/figures/plateau-input-attribution/label-seed-residual-field.json`, residual-field regional-score table | **Verified, spatial only.** The receipt explicitly denies that a spatial match proves an input change reaches a root. |
| `plateau-input-attribution` | The real-wall substitution is worse: physical-ring axis error 342.437 mm versus 164.595 mm for the pseudo-wall, ratio 2.08048. | `docs/figures/plateau-input-attribution/wall-topology-surface.json`, pseudo-wall versus physical-ring comparison | **Verified.** This directly refutes the real-wall substitution used in the arm. |
| `plateau-input-attribution` | The shipped discrete-wiring family has no viable member: best 83/84 circuits; implied current corrections range from about -8.05 to -53.995 of the programmed current. | `docs/figures/plateau-input-attribution/circuit-projection-arm.json`, circuit projection arm | **Verified.** All shipped definitions are one-section, one-turn, positive-gain objects; the required corrections are incompatible with one fixed wiring scale. |
| `plateau-input-attribution` | One converged control frame separates from the five plateau frames on none of four candidate scores. | `docs/figures/plateau-input-attribution/converged-frame-contrast.json`, control-versus-plateau range table | **Verified as descriptive non-separation, not causal closure.** The control lies inside all four five-frame ranges, but the receipt explicitly states that one control plus five plateau frames has no population or causal power. |
| `plateau-input-attribution` | Ten margin remeasurements are all limiter-class; 7/10 wrong-class proposals reduce the residual, with a physical-ring minimum 2.409e-16. | `docs/figures/plateau-input-attribution/margin-frame-remeasure.json`, pseudo-wall/physical-ring terminal table | **Verified and correctly qualified.** The plan does not promote the residual minimum into convergence evidence. |
| `plateau-input-attribution` | “All four documented candidates are now closed by measurement” and the plateau “belongs to none of them.” | The four figures above plus `docs/plans/plateau-input-attribution.html`, §7 landing | **Discrepant — unstable grounds.** Wall and shipped wiring have direct negative arms. Profiles and discretisation have only weak pattern ranking plus descriptive non-separation. Supported wording is “not implicated by this screen,” not “causally closed.” |
| `curved-clip-global-surface` | Production-route STEP disagreement is 2.34937% for VPR and 3.88481% for G1, below independent gates 3.92% and 6.20%. | `docs/figures/curved-clip-global-surface/global-surface-clip-rescore.json`, STEP rescore table | **Verified.** The gate values correspond to the separately banked contour-to-TORAX discrepancies, 3.93215% and 6.22067%, rounded to 3.92% and 6.20%; they were not derived from the achieved production values. |
| `curved-clip-global-surface` | The comparison uses the production `extract_flux_surface_geometry` route and six tests pass in 833.2 s. | `tests/test_transport_geometry_reference.py`, production-route assertions; receipt route-audit fields; commit `9317ef50` | **Verified.** The test imports and calls the production extractor, asserts `error < limit`, and the cited commit changes only the reference test. |
| `coefficient-space-newton` | The original support-order ranking gives arm A errors of roughly 1.22e-15 to 2.49e-15 against the banked converged root. | `docs/figures/coefficient-space-newton/support-order-arms.json`, carrier-region table | **Discrepant as an arm ranking; correctly retracted.** The reference root is arm A's own fixed point, so those values are self-comparison error and cannot rank A versus B. |
| `coefficient-space-newton` | Against independent analytic truth, arm B wins relative-sup error in all six carrier-region comparisons by margins 6.07e-6 to 3.17e-4, and wins RMS in five of six. | `docs/figures/coefficient-space-newton/analytic-truth-rescore.json`, independent-truth rescore | **Verified arithmetic, limited inference.** The receipt has independent analytic authority, common normalization and partitions, frozen hashes, and a fresh reproduction; it is a two-resolution result, not systematic order evidence. |
| `coefficient-space-newton` | A seven-rung iso-accuracy ladder resolves no accuracy order or cell-count advantage; second order wins only 3/7 analytic and 1/7 low-aspect rungs, with 1.0x exclusive-node memory. | `docs/figures/coefficient-space-newton/support-order-iso-accuracy.json`, seven-rung comparison | **Verified.** This correctly supersedes the two-resolution rescore for the systematic support-order decision. Commit `b774c28b` adds only the benchmark and receipt. |
| `coefficient-space-newton` | Earlier analytic-state errors of 26–45% become 19–38% after common-gauge alignment, so gauge is not the explanation. | `docs/figures/coefficient-space-newton/analytic-truth-rescore.json`, gauge-aligned global errors | **Verified.** The correction reduces but does not remove the discrepancy; the plan's conclusion is appropriately negative. |
| `coefficient-space-newton` | Higher-order extraction measures order 3.74479 +/- 0.139 versus 1.86927 +/- 0.071, finest RMS 1.190e-9 versus 8.377e-5, and zero tangent jumps over 276 checks versus 0.03506. | `docs/figures/coefficient-space-newton/higher-order-contour.json`, convergence and tangent-continuity panels; commit `479667ed` | **Verified.** Worker tests reported 7 focused plus 3 reference tests passing. The evidence record is now durably committed by `c16f3567`. |
| `coefficient-space-newton` | H200 matrix throughput/memory evidence exists for the dense versus iterative crossover. | Expected figure `docs/figures/coefficient-space-newton/h200-matrix-benchmark.json`; driver commit `2c8e02a2`; jobs `1254380`, `1254382`, `1254399` | **Not claimed and not yet verifiable.** The first two jobs reached their 35- and 45-minute limits without a receipt; the corrected one-hour job was still running. This is a capacity/process defect, not a banked result. |
| `same-device-label-determinism` | Cache-off fresh processes produce counts 410/409/411 and 412/409/410 of 414, with 11 changing case verdicts; persistent cache reuse gives 69/69 aggregate and 828/828 per-case verdicts identical. | `docs/figures/same-device-label-determinism/executable-boundary-arms.json`, paired fresh-process arms | **Verified.** The receipt retains three processes per arm and records cache hits on the warm repetitions. |
| `same-device-label-determinism` | Cold-driver control cost is +0.113009 s/candidate slice and +2.19713%, explicitly an upper bound over 30 candidate slices. | `docs/figures/same-device-label-determinism/executable-boundary-arms.json`, timing decomposition | **Verified and correctly qualified.** Startup, scalar references, and compilation are included, so the plan does not present this as steady-state store cost. |
| `same-device-label-determinism` | Production control costs 0.08703% steady-state: 0.0406050 s/slice enabled versus 0.0405380 disabled, paired median delta 3.52794e-5 s/slice; cache hits are 0/1/1 and all 69 observables are bitwise identical. | `docs/figures/same-device-label-determinism/production-compilation-control.json`, production-route timing and reproducibility panels; commits `e4093c4e`, `4a629ba0` | **Verified.** The production code and six cited tests are included in the merge; the live plan now ships determinism only and preserves the 67/69 acceptance failure. |
| `same-device-label-determinism` | Both remaining divergence residuals are identically zero in exact arithmetic, therefore their computed values are roundoff and relative bounds are passable only by last-bit luck. | `docs/figures/same-device-label-determinism/executable-boundary-arms.json`, remaining-failure rows; `tests/test_equilibrium_stencil_mesh.py:28`; `benchmarks/observable_batch_acceptance.py:553` | **Discrepant — error in conclusion.** The physical continuum identities are zero, but the shipped `StencilMesh` least-squares derivatives do not commute; the repository contract says these residuals sit at a second-order truncation floor. The discriminator also classifies `divergence_j` as a fitted-gradient computation difference and `divergence_b` as unadjudicated. Only the cross-process changes are demonstrated to be last-bit scale. |
| `discrete-operator-analytic-error` | One analytic map application stays at 2.69e-15 to 3.06e-15 relative-sup error over 342, 551, 815 and 1,076 carrier cells, below the 9.095e-13 roundoff gate; fitted order is -0.083 +/- 0.129 with 95% interval [-0.637, 0.471]. | `docs/figures/discrete-operator-analytic-error/operator-refinement-ladder.json`, refinement ladder | **Verified.** Boundary, external field, gradients, and current-density term audits are also at roundoff. The plan correctly concludes that the discrete map admits the analytic fixed point. |
| `discrete-operator-analytic-error` | The receipt's machine-readable `cause: posing` identifies the remaining source of the 26–45% solved-state discrepancy. | Same receipt, aggregate verdict and term-audit sections; benchmark `_aggregate` logic in `benchmarks/operator_refinement_ladder.py` | **Discrepant — process defect.** The term audit and plan both exonerate the posed analytic case. The field is assigned mechanically when the map error is at roundoff and contradicts the actual conclusion: solver/root selection remains open. |
| `discrete-operator-analytic-error` | No sampled downstream conclusion silently requires the banked discrete solve to equal analytic truth. | Curved-clip route comparison; same-device self-comparison; plateau margin qualification; MAST scope and acceptance holds | **Verified by dependency audit.** The downstream claims use route parity, self-consistency, or one-map evidence and retain separate convergence/acceptance holds. |
| `mast-catalog-gpu-solve` | The FAIR-MAST level-2 scope has 11,573 catalog shots, 11,378 equilibrium-bearing stores, 195 without an equilibrium group, and 1,341,435 equilibrium slices; one hour on eight devices requires 372.621 slices/s aggregate or 46.5776 per device. | `docs/figures/mast-catalog-gpu-solve/slice-census.json`, coverage and throughput-requirement tables | **Verified.** The receipt records 11,573/11,573 catalog-to-mirror reachability, hashes the catalog and count vector, reads 11,378 metadata files, downloads no bulk arrays, and runs zero solves. |
| `mast-catalog-gpu-solve` | Existing evidence has fired the sharding, output-store, or convergence-policy decisions. | `docs/mast-catalog-gpu-solve.html`, decision cards; `slice-census.json`; coefficient H200 worker receipt state | **Verified negative: no measured trigger has fired.** Sharding explicitly waits for catalog production-path evidence. The convergence-policy branch waits for a catalog iteration distribution showing a heavy tail. Neither exists. Output-store policy is open but is not stated as quantitatively triggered by the coefficient matrix benchmark. |
| `mast-catalog-gpu-solve` | Missing coefficient-space H200 matrix evidence blocks a MAST policy decision. | `docs/plans/coefficient-space-newton.html`, §12; `docs/mast-catalog-gpu-solve.html`, decisions and blockers | **Discrepant if asserted; the plans do not assert it.** The coefficient benchmark measures a matrix-arm crossover, whereas MAST sharding waits on production catalog throughput. The MAST hold remains topology/branch-root existence plus scientific acceptance, not this missing matrix receipt. |

## Detailed findings

### 1. Plateau attribution: the negative is useful, but two candidates are not causally closed

The evidence cleanly removes two concrete hypotheses. The physical-wall arm is
an intervention and worsens the axis error by a factor of 2.08048. The discrete
wiring search evaluates 84 shipped circuits and shows that the corrections
needed by the best projections are large, negative, and inconsistent across
references. Commits `8aaa505f`, `c44ebc4f`/the wall landing sequence, and
`fdffd8e9` have artifact-and-benchmark scopes consistent with the plan's cited
work; no production implementation is smuggled into these negative studies.

The remaining evidence is a screen. `label-seed-residual-field.json` explicitly
says the regional spatial attribution does not establish that changing an input
will produce a root. `converged-frame-contrast.json` likewise labels its one
control plus five plateau cases descriptive and non-causal. The control value is
inside every plateau range:

| Candidate | Control | Five-frame range |
|---|---:|---:|
| wall | 0.095069 | 0.086403–0.108929 |
| wiring | 0.144053 | 0.059829–0.198177 |
| profiles | 0.049404 | 0.027849–0.079092 |
| discretisation | 0.032246 | 0.026265–0.037396 |

That absence of separation is worth recording, but it does not refute profiles
or discretisation. The shipped conclusion should preserve the asymmetry:
physical wall and the enumerated wiring family are refuted; profiles and
discretisation were not implicated by the available spatial/descriptive screen.
The plan does correctly weaken the solver pillar: the historical machine-
precision result is not treated as current evidence because 32 production
Python paths differ and its source revision is unavailable.

### 2. Curved clip: clean independent gates and a production-route assertion

This is the strongest sampled landing. The achieved production-route errors are
not used to set their own gates. The 3.92% and 6.20% limits are the rounded
reader discrepancy measured independently between the contour bank and TORAX,
while the production clipped-to-contour route achieves 2.34937% and 3.88481%.
`tests/test_transport_geometry_reference.py` imports
`extract_flux_surface_geometry`, calls it, and asserts each error is below the
corresponding limit. Commit `9317ef50` is test-only and replaces the prior
helper-path comparison with the production route; the cited six-test result and
receipt route audit agree.

### 3. Coefficient-space Newton: correction history is honest; H200 evidence is still open

The original banked-root accuracy ranking was circular because the banked root
is the zeroth-order arm's own fixed point. The plan now says so plainly. The
analytic-truth rescore fixes reference independence and its arithmetic checks,
but the later seven-rung ladder is the appropriate authority for a systematic
order claim. It finds neither an order nor cell-count advantage and correctly
retires the support-order lever. Retaining the earlier two-resolution result in
the cumulative record is not misleading because the later landing explicitly
marks it superseded.

The higher-order contour result is fully banked and its test/commit scope is
consistent with the claim. The earlier audit-brief defect—implementation merged
but no durable landing record—was repaired at `c16f3567` before this report was
frozen.

The H200 matrix lane is not complete. Commit `2c8e02a2` contains the driver but
there is no `h200-matrix-benchmark.json`. Jobs `1254380` and `1254382` stopped at
their explicit 35- and 45-minute limits without traceback or artifact. A
corrected one-hour, five-H200 job `1254399` was running at 11:26 elapsed when
checked. The important scientific safeguard is intact: the plan claims no
throughput or memory number from those attempts.

### 4. Determinism: the control is proved; the acceptance diagnosis is not

The fresh-process and production-control receipts support the causal mechanism:
independent compilation changes last-bit results and persistent executable reuse
eliminates those changes. The production implementation also overrides
conflicting inherited cache paths and marks the resolved configuration explicit.
Three processes return the same terminal-flux digest, zero unequal elements,
and identical values for all 69 observables. Its 0.08703% steady-state delta is
far below the deliberately conservative 2.19713% cold-driver upper bound.

The separate acceptance explanation overreaches. Lines 28–35 of
`tests/test_equilibrium_stencil_mesh.py` are an explicit contract: continuum
divergence identities cancel under commuting central differences, while the
ring-mesh least-squares fits do not commute and therefore land at a
second-order truncation floor. In the acceptance driver,
`_failure_verdict()` calls `divergence_j` a fitted field-function gradient
computation difference and `divergence_b` unadjudicated. Receipt values reinforce
the distinction: process-to-process changes are around 1e-16, while the
`divergence_j` observable itself is around 1e-10. Thus determinism is shipped on
sound evidence, but the new `roundoff-scale-acceptance-bounds` plan must first
classify the discretisation floor; it cannot begin from the conclusion that the
entire computed residual is rounding error.

### 5. Discrete operator: numerical conclusion clean, receipt label contradictory

The refinement ladder is decisive for its narrow question. One application of
the production map preserves the analytic fixed point at approximately 3e-15
relative-sup error across all four meshes, and the fitted convergence interval
includes a constant roundoff floor. The term-by-term audit shows no boundary,
external-field, gradient, or current-density discrepancy large enough to explain
the solved-state error. The plan therefore correctly relocates the open question
to which fixed point the nonlinear solve reaches.

The receipt's aggregate `cause: posing` must not be consumed as an attribution.
It contradicts the receipt's own term audit and is set by a mechanical branch in
the benchmark. A future schema/consumer should distinguish “map admits analytic
fixed point; solver cause unresolved” from an actual case-posing defect.

### 6. MAST catalog: the census is banked and decision gates remain genuinely open

The catalog count and one-hour rate arithmetic reproduce exactly:
`1,341,435 / 3,600 = 372.620833` slices/s aggregate and division by eight gives
`46.577604` slices/s/device. Coverage is explicit: all 11,573 catalog shots are
reachable, 11,378 contain equilibrium slices, and 195 lack the group rather than
silently disappearing.

None of the three open owner decisions has a hidden quantitative trigger already
satisfied. Sharding awaits production-path catalog throughput; convergence
policy awaits an observed iteration distribution showing whether the tail is
heavy. The open output-store choice is a policy decision, but the live plan does
not define the coefficient matrix crossover as its trigger. The MAST dependency
should therefore remain held on topology/branch-root existence and acceptance,
while reproducibility alone is now released by the production compilation
control.

## Commit-diff and assertion audit

- `9317ef50` changes only `tests/test_transport_geometry_reference.py` and
  routes the assertions through the production extractor.
- `231659ea` and `b774c28b` add the coefficient comparison drivers and receipts;
  the latter is the systematic seven-rung authority.
- `ae723e28` adds the analytic one-map benchmark and receipt. Its values support
  the plan, but its `cause` field does not.
- `8aaa505f`, the wall/contrast landing sequence, and `fdffd8e9` bank negative
  plateau experiments without altering the production solver.
- `91ac686c` banks the fresh-process determinism comparison. `e4093c4e`, merged
  by `4a629ba0`, adds the production cache configuration, six focused tests, and
  the production receipt.
- `479667ed` contains the higher-order contour implementation, benchmark, and
  tests; `c16f3567` supplies the durable cumulative landing record that was
  missing when the audit brief was written.

## Required corrective actions

1. Amend the plateau landing so profiles and discretisation are recorded as
   *not implicated by the sampled screen*, not closed/refuted by causal
   measurement. Preserve the direct negative verdicts for wall and the shipped
   wiring family.
2. In `roundoff-scale-acceptance-bounds`, separate the exact continuum identity,
   the non-commuting `StencilMesh` truncation floor, and the last-bit
   process-to-process variation before defining any absolute floor. Do not treat
   the `divergence_j` value itself as pure roundoff.
3. Correct the operator-ladder receipt schema or regeneration logic so its
   aggregate cause is unresolved solver/root selection rather than `posing`.
   Until then, consumers should ignore that field and use the term audit plus
   the plan conclusion.
4. Let the running H200 matrix job either bank a complete receipt or remain an
   explicit capacity result. Do not infer a crossover, memory envelope, or MAST
   sharding policy from the two timed-out attempts.

Overall classification: **three unresolved discrepancies across 24 sampled
claims**—one unstable causal conclusion, one incorrect numerical
interpretation, and one machine-readable receipt defect. One additional sampled
historical ranking was circular but is explicitly retracted and superseded in
the live plan. The remaining 20 samples are verified, explicitly unclaimed, or
negative dependency checks whose asserted implication is absent from the live
plans.
