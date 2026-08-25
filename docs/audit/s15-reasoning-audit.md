# S15 reasoning audit — do the landed conclusions follow from the recorded evidence?

Audited 2026-08-25 against worktree base `5e3d56ea`. Scope: the reasoning chain
of the six S15 plans — whether each landed conclusion is entailed by the
evidence actually banked for it. Evidence *existence and reproducibility* is a
sibling node's subject; this report grades *inference*.

Grades: **stable** (the conclusion follows from the cited evidence),
**unstable** (the conclusion may be right but the cited evidence does not
establish it), **in error** (the evidence contradicts the conclusion).

| Plan | Grounds |
|---|---|
| discrete-operator-analytic-error | **stable** |
| curved-clip-global-surface | **stable** (one stale-pin caveat) |
| coefficient-space-newton | **stable** (one superseded figure still quoted) |
| plateau-input-attribution | **stable on direction, unstable on 2 of 4 closures** |
| same-device-label-determinism | **unstable** on the cost premise the lock was taken on; stable elsewhere |
| mast-catalog-gpu-solve | **stable** — correctly blocked, no missed trigger |

---

## 1. discrete-operator-analytic-error — stable

Admitted to S15 at `67cb20d5`. §2 landed at `ae723e28`.

The inference that matters here is a negative one, and it is the right shape.
§2 poses two branches (operator, posing) and closes both by measurement rather
than by argument:

- Operator admits the analytic fixed point: one application of the exact
  production forward map to an independently evaluated closed-form field
  returns relative-sup residuals `2.6922745652091497e-15`,
  `3.054874184360744e-15`, `2.83332350384108e-15`, `2.893824705250372e-15`
  across realised cells 342 → 1076, every rung under a **preregistered**
  4096-binary64-epsilon tolerance.
- The fitted order `-0.0831452960470665 ± 0.1287518999975828`, 95 % interval
  `[-0.637120010032096, 0.4708294179379629]`, is used correctly: it is read as
  *including a constant and excluding the 0.5 converging floor*, i.e. as
  evidence that nothing is left to converge — not as a measured convergence
  order.
- Posing is exonerated **term by term**, not in aggregate: external completion
  `4.440892098500626e-16` Wb, boundary `-8.673617379884035e-19` Wb against
  closed-form zero at every rung, pressure gradient `-72400.04058623493`
  Pa/Wb and FF gradient `-0.06997315389737607` T m²/Wb both at zero quoted
  delta, current density `3.8707986461934123e-16`. This is the difference
  between a finding and an absence of evidence, and the plan says so.

§4's surviving claim — the residue is in *which fixed point the solver
reaches* — is entailed rather than asserted: map admits the root (2.7e-15),
inputs match the closed form (term by term), solve does not arrive (stalls
unqualified at `1.2549724178436488` and `1.2998050585857959` against 1e-10;
converges 19–38 % away under a common gauge). Three mechanisms are named with
a discriminating measurement each, none preferred without one.

**The one thing worth noting is already self-caught.** The plan's own followup
states "THE VERDICT THIS PLAN INHERITS IS NARROWER THAN ITS NAME": the
inherited receipt verdict `genuine_discrete_operator_error` separated the
operator from the *comparison*, never from the *posing*. The plan refused the
stronger reading and made the narrower one its first node. That is the correct
handling of an over-named upstream verdict, and it is why §2 came out
"neither".

Directional check on refinement, which carries the posing argument: coarse and
fine discrete anchors agree to 3–4 digits (`1.0241986753014287` /
`1.0227451468118647` on axis; `-0.24590656947069325` /
`-0.24697114012695603` at the boundary). Refinement does not move it — so a
discretisation account is excluded by the data, not by preference.

---

## 2. curved-clip-global-surface — stable, with a stale-pin caveat

Shipped `8e661dfc`. The brief's question is whether the gates were *measured*
or merely *asserted*, and whether the regression lock asserts a measured value
(which would be circular) or an independent bound.

**Verified: the gates are independent bounds, and they are real assertions.**
`9317ef50` introduced, in `tests/test_transport_geometry_reference.py:127-129`:

```
_STEP_CLIPPED_TO_CONTOUR_LIMITS = {"vpr_face": 3.92e-2, "g1_face": 6.20e-2}
```

enforced at line 555 as `assert error < limit`. `git show 9317ef50 --
tests/test_transport_geometry_reference.py` confirms all four lines are
additions — the gate assertion did not exist before that commit.

**Provenance of the bound values, traced.** `3.92 %` and `6.20 %` are not
achieved values of the quantity they score. They are the *TORAX ↔ contour*
inter-reference disagreements from the "two against one" table in
`docs/evidence/archive/flux-function-forward-transport-landed.html` (vpr
`3.92 %`, g1 `6.20 %` in the third column). The criterion is therefore "the
clipped route must agree with the contour reader at least as well as the two
independent reference readers agree with each other" — a bound on a *different
pair* than the one scored. That satisfies the plan's own standing rule and is
the strongest form available on this lane.

The measured values sit separately: `vpr_face 2.349368719e-2` and
`g1_face 3.884813932e-2` are banked as characterization baselines at
`_CHARACTERIZATION_FRACTION = 0.10`, distinct from the hard limits. Measured
and asserted are kept apart. Headroom 1.571 and 2.315 pp.

**§6's finding is the real one and it is correctly framed.** The pre-lock suite
was green at 6 passed / 560.13 s while `_nova_geometry` called
`traced_flux_surface_geometry` rather than the shipped
`extract_flux_surface_geometry`, so STEP was characterised at
`18.137000000 %` and `22.306000000 %` — values *above* the quoted gates. This
is direct evidence that for the whole prior programme the "gate" was a
receipt-level criterion and not a test; the plan's section title says exactly
that. All 72 constants were rebanked with a retained
previous/measured/signed-difference receipt (STEP vpr −15.787631281 pp, g1
−18.421186068 pp).

**Caveat — the gate is pinned to a stale snapshot of a drifting quantity.**
§4's fresh three-way pairing measures contour↔TORAX at `3.932154 %` (vpr) and
`6.220674 %` (g1), i.e. the reference-reference spread the gate is *meant* to
encode has since moved to `+0.012` and `+0.021` pp above the pinned `3.92` /
`6.20`. The gate's numeric value and its stated semantics have separated. No
consequence for the verdict — the measured 2.349/3.885 pass either way — but a
future tightening should re-derive the bound rather than inherit it.
Severity: process-defect, cosmetic.

**One claim is understated rather than overstated**, which is the safe
direction. §4 claims the two-against-one reading inverts "on STEP g1"
(3.885 % clipped↔contour vs 6.221 % contour↔TORAX). It also inverts on STEP vpr
(2.349 vs 3.932) and the plan does not claim it. On ITER the clipped route
remains the outlier (g2g3 3.110 vs 0.779), and the plan correctly scopes the
inversion to STEP.

---

## 3. coefficient-space-newton — stable; the circular-scoring incident was
handled correctly and its correction was itself correctly superseded

This is the deepest chain in the sprint and it is the one place the audit
expected to find a conclusion resting on refuted scoring. It does not.

**The chain, as recorded.**

1. `support-order-arms` (`d6be878b`) scored arms A and B on accuracy against
   the banked converged root — which is arm A's own fixed point. The archive
   states the defect in its own words: "It scored both arms against the banked
   converged root, which is baseline A's own fixed point — so A reproduces it
   at 1.9e-15 to 3.1e-15 by construction while B reads 7.9e-4 for having a
   different discrete fixed point. That measures disagreement with A, not
   accuracy." Ownership is taken ("that dispatch defect was mine"), not
   diffused.
2. `analytic-truth-rescore` (`231659ea`) reversed the verdict: arm B closer to
   analytic truth in all six carrier-region sup comparisons by margins
   `6.066e-6` to `3.169e-4`. **The rescore's own construction is sound on the
   axes that matter** — identical plasma-cell supports, identical ψ_N region
   masks, the same full analytic grid-field span denominator, so the two
   reference columns compare directly; and the circularity is *demonstrated*
   rather than asserted by publishing A's banked-root self-error
   (1.221e-15–2.494e-15) beside its analytic error (0.258–0.451).
3. The rescore correctly flagged its own scope limit: measured at source
   revision `162f208b` where only the exact-value arms were integrated, so
   arms C and D still owed rescoring.
4. **The reversal was then correctly retired.** `support-order-iso-accuracy`
   (`b774c28b`) states: "its margins of 6.066e-6 to 3.169e-4 were two-point
   comparisons against errors of 0.26 to 0.45, and a seven-rung fit with
   uncertainty now shows no systematic improvement." Verdict
   `no_resolved_order_or_constant_improvement`: analytic
   `2.03495 ± 0.06477` (first) vs `2.03493 ± 0.06797` (second), difference
   `-0.00003 ± 0.09389`; low-aspect difference `0.00228 ± 0.12958`; second
   order better on only 3 of 7 and 1 of 7 rungs.

**Verdict on the incident: no conclusion banked before the reversal still
stands on the refuted scoring, and no conclusion stands on the reversal
either.** The margins were ~0.1 % of the distance both arms sit from the
reference — comparing two points against a reference neither is near. The
seven-rung fit is the right instrument and it replaced the two-point one. The
accuracy axis of the matrix now carries no arm preference, which is the honest
state.

**Arm B's memory figure is superseded but still quoted in one place.** §11's
execution-evidence entry records "40 to 51 percent more peak memory"; §14
corrects it to "peak process memory is identical at 778.97 MiB, ratio 1.0×,
not the 40–51 % previously reported" — the earlier figure was a shared-CPU
allocation artefact, remeasured on an exclusive node. The correction is
present and explicit, and §11's *evergreen* body carries no memory figure, so
the live plan does not assert the stale number. The stale figure survives only
in the append-only execution log and in the frozen archive, which is what
those surfaces are for. Severity: none.

**Convergence and conditioning are kept separate, as §6 demands.** §18/§19:
dense 36×36 coefficient Jacobian conditions at `119.02` against the banked
projected Krylov `4087.40844825275` — 34.3× better, solved to relative linear
residual `3.16e-16`; and convergence is lost, one advance, terminating limited
at `3.708e-2` against arm A's `2.7110550053242652e-15`. The verdict string
itself carries both halves. §19's attribution is the strongest single piece of
reasoning in the sprint because it is *symmetric*: the square system matched
finite differences of the projected residual to `1.0181914475518321e-07` (a
correct Jacobian of the wrong function) while the same columns against
exact-residual finite differences gave `0.06899077124977254`; the corrected
2253×36 object matched every directional derivative to `5.350605353071424e-08`
at fitted scale `0.9999999797120991`. Both directions are shown, so the
attribution is not an inference from one residual.

**§12 is the one open hole and the plan names it.** The matrix's speed axis is
declared "indicative only" until measured on the H200, and no arm preference at
campaign scale is claimed. See §6 below for why it has not landed.

---

## 4. plateau-input-attribution — stable on its shipped direction, unstable on
two of the four candidate closures

Shipped `f243a3e3` on a named negative. Two questions from the brief: is the
negative stable enough to ship on, and is the "solver pillar weakens"
admission reflected in the shipped conclusion?

**The solver-pillar admission is reflected, in both places it needs to be.**
§1 carries it in the evergreen prose: "From the solver — and this pillar has
since weakened, 2026-08-25 … The existence argument therefore rests on the
physics alone, not on two independent legs, and the solver leg should not be
cited as current evidence until something reproduces it." §7 does not re-cite
it. The `28b98147` node's own record states the amendment was owed and made.
Attribution for the loss of reproduction is **declined** rather than invented:
the machine-precision artifact at `a0aee18c` did not embed its execution source
revision, and 32 production Python paths differ from the measured tree
`f8f1f310` — more than one mechanism in the diff, so the method requirement
forbids attributing it. Correct call; the alternative would have been a named
regression the evidence cannot support.

**Power is declared, not oversold.** `converged-frame-contrast` reports
"a descriptive null at n=6 rather than a population effect", with 0 of 4
substitution-pattern properties separating frame 22086/43 under a declared
inclusive-range rule (wall `0.09506945818225918` inside
`[0.08640290979900724, 0.10892907063904789]`; conductor wiring `0.14405290`
inside `[0.05982879, 0.19817707]`; profile `0.04940377` inside
`[0.02784862, 0.07909236]`; discretisation `0.03224643` inside
`[0.02626486, 0.03739640]`). Naming a null as descriptive at its own n is the
correct handling; shipping on it is defensible because the plan's *closure*
does not rest on the null — it rests on the four candidate refutations.

**Two of those four refutations are strong and power-independent.**

- *Conductor wiring* is refuted structurally, not statistically: the implied
  equivalent-current correction spans `-8.259178669094224` to
  `-53.99500490603694` of drive across six frames of **one** machine sharing
  **one** circuit-definition digest `c26682ad` — a span of `45.7358`. A wiring
  error is a fixed factor; a sevenfold-varying multiplier is not one. And there
  is no discrete choice to be wrong: every best circuit has one section,
  `turns=[1.0]`, `gains=[1.0]`, positive polarity, no series-parallel branch.
  This argument does not need n.
- *Wall* is refuted by direct substitution across all five score-blind frames:
  the governed physical ring puts the axis at mean `342.437` mm against the
  banked pseudo-wall `164.595` mm — worse by `177.842` mm, factor `2.08048` —
  with two frames going actively pathological at terminal residuals `0.9659205`
  and `3.1727126` against an `0.087`–`0.155` plateau, and a four-expansion
  rectangle sweep worse at every rung. Retained without tuning.

**The finding: the other two candidates were never substituted.** Profiles and
discretisation are closed on *spatial pattern scores* of `0.0514925944` and
`0.0326058989` from the label-seed residual field — a proxy — and that
receipt's own qualification says "this is spatial attribution only, not proof
that changing either input yields a converged parity root". §7's enumeration
is accurate about each ("profiles and discretisation score 0.051 and 0.033 and
are **not implicated**"), but the sentence that governs it — "All four
documented candidates are now closed by measurement rather than by argument" —
and the shipped followup's "NONE of the four documented substitutions moved the
plateau — each closed by measurement" both flatten the distinction between
*refuted by substitution* (2 candidates) and *not implicated by a proxy the
receipt disclaims* (2 candidates). No profile substitution and no
discretisation substitution appears in any S15 receipt.

Severity: **unstable-grounds, narrow.** It does not move the plan's direction
— that is carried independently by the operator ladder (map admits the root)
and by §6's margin remeasure (solver reaches `2.408801293e-16` the moment the
class refusal is relaxed, and lands in the wrong class 10 of 10 times). But the
shipped summary claims a stronger closure for 2 of 4 candidates than the
evidence gives, and a reader planning a fifth arm would reasonably conclude
profiles and discretisation had been tested when they were only scored.

**The margin-grading result is graded correctly and this is worth recording as
a positive.** `fdffd8e9` had every incentive to read as a win — admission rises
up to twelvefold, traversal up to thirtyfold, residual falls to
`1.747136782e-16` / `2.408801293e-16`. It is reported as closing nothing,
because all ten terminals are limited where diverted was requested (0 of 5
diverted in both arms, seven wrong-class lower residuals explicitly named). The
exact Boolean terminal gate `_branch_receipt` was not edited. A zero-residual
wrong-class state is rejected — which is precisely what makes the negative
trustworthy rather than a near-miss. The continuous merit function carries
**zero tuned thresholds** (unit penalty weight), so it cannot have been shaped
to the result.

---

## 5. same-device-label-determinism — unstable on the premise the
production-wide lock was taken on

The control-scope decision was locked production-wide at `c861e70d`, after the
cost was measured and not before, which is the ordering §3 insisted on. The
process discipline is right. The **number** the decision was taken on does not
support the framing put on it.

### 5a. The 67-of-69 claim is CONFIRMED

The brief asks whether the two remaining failing bounds are on divergence
residuals identically zero in exact arithmetic. Read directly from
`docs/figures/same-device-label-determinism/executable-boundary-arms.json`,
`arms.*.invocations[*].batch_results[*].per_observable`, the failing set is
identical in every invocation of both arms at both widths:

```
FAIL: ['conservation.divergence_b', 'conservation.divergence_j']   (obs_pass 67 of 69)
```

Both are divergence residuals whose exact value is zero, so their computed
value *is* roundoff. The mechanism is visible in a single retained record
(`pass_status_changes[0]`, case `21978/35`, width 1): absolute values by
invocation `2.220446049250313e-16`, `6.106226635438361e-16`,
`5.551115123125783e-16` — all three **below** the `absolute_bound`
`9.83631031098047e-16` — yet `passes_by_invocation = [true, false, false]`,
because the criterion is a `banked_dual_envelope` carrying a *relative* bound
of `0.8425616333270692`. A last-bit move on a roundoff-scale quantity blows a
relative envelope. §1's "a bound on them is a bound on roundoff" is therefore
demonstrated by the receipt, not merely argued. **Stable.**

### 5b. But "the two bounds whose verdicts moved" is imprecise

The locked decision text and §1 both say the two bounds whose verdicts moved
are `divergence_b` and `divergence_j`. The receipt says something different:
`pass_status_changing_observables` is `[]` in **all four** arm×width summaries,
and `observable_pass_count_by_invocation` is `[67, 67, 67]` in every one. No
aggregate bound verdict moved anywhere. What moved is 11 **per-case** verdicts
(`conservation.divergence_b` ×3, `conservation.divergence_j` ×8) in the
cache-off arm, and zero in the fixed-cache arm. The two failing bounds and the
two observables carrying the moving cases happen to be the same pair, so the
sentence lands on true facts by coincidence of identity rather than by
derivation. Severity: process-defect. The conflation of "aggregate bound" with
"per-case verdict" is exactly the granularity confusion §4's method
requirement ("report a pass count only with its repetition count beside it")
exists to prevent.

### 5c. The cost figure is a mismatched-estimator difference

`throughput_cost` reports `control_cost_seconds_per_candidate_slice =
0.11300887924929537`, `control_cost_percent = 2.1971256764727887`, from
`unconstrained_median = 5.143487259714553` and
`controlled_median = 5.256496138963849`. The per-invocation arrays are:

```
cache_off        : [5.429310244973749, 5.143487259714553, 5.125127385867139]
persistent_cache : [5.427942570795616, 5.275857901386917, 5.237134376540780]
```

The "unconstrained median" is the median of all three cache-off invocations.
The "controlled median" is **not** a median of its arm — `median([5.4279,
5.2759, 5.2371]) = 5.2759` — it is the mean of invocations 2 and 3, as
`controlled_route = 'persistent_cache_warm_invocations_2_and_3'` discloses:
`(5.275857901386917 + 5.237134376540780)/2 = 5.256496138963849`. So the
headline difference subtracts a 2-sample warm mean from a 3-sample median, and
the two arms' cold first invocations — `5.4293` and `5.4279`, within 0.03 % of
each other — enter the two sides asymmetrically: discarded by the median on one
side, excluded by construction on the other.

Recomputed like-for-like:

| estimator | cache off | fixed cache | delta | % |
|---|---|---|---|---|
| cold invocation 1 only | 5.42931 | 5.42794 | −0.00137 | −0.03 % (cache faster) |
| warm invocations 2–3, mean | 5.13431 | 5.25650 | +0.12219 | +2.38 % |
| all three, mean | 5.23264 | 5.31364 | +0.08100 | +1.55 % |
| **as published** | 5.14349 (med of 3) | 5.25650 (mean of 2) | +0.11301 | **+2.20 %** |

The published figure is bracketed by the consistent estimators (1.55 %–2.38 %),
so the mismatch does not distort the magnitude and the *sign* is robust
warm-to-warm (the warm-only effect, 2.38 %, exceeds the within-arm warm spread
of ≤0.74 %). The number is usable. Calling it a "median" is not accurate.
Severity: process-defect.

### 5d. "UPPER BOUND" is a framing claim the receipt does not establish

The decision rests on the cost being an upper bound: "the per-slice figure is
dominated by terms that amortise across a batched catalog run instead of
recurring on all 1,341,435 slices." The receipt's own qualification is weaker
and does not use the phrase: `"negative cost denotes measured time saved; this
cold-driver figure is not steady-state solve-only throughput"`, with
`denominator = "six cases times the sum of batch widths, 30 candidate slices;
process startup, scalar references and compilation are included"`.

The amortisation argument requires the cost to be a **per-process** term. But
the measured sign is the wrong way round for that reading: reusing a
compilation cache should *reduce* per-process compile time, and the fixed-cache
arm is **slower** than cache-off at the same warm position. Two accounts fit,
and they imply opposite amortisation behaviour:

- *(a) cache read/deserialise overhead* — a per-process term, which does
  amortise, and the upper-bound framing holds.
- *(b) the pinned executable is itself slower* — a per-solve term, which does
  **not** amortise, and the framing inverts: 2.2 % would then be a floor on
  1,341,435 slices, not a ceiling.

The receipt records only whole-invocation `elapsed_seconds` over 30 slices per
process; it does not separate compile time from execute time within an
invocation, so it cannot discriminate (a) from (b). The upper-bound claim is
therefore **not verified** — not refuted either, and I make no claim about
which account is right.

**Mitigating, and this is the reason the grade is not worse:** the plan has the
correct guard already installed. §5's done-when requires "the per-slice cost is
re-quoted from a steady-state measurement that separates one-time compilation
from per-slice execute, against the 0.113 s/slice and 2.197 % cold-driver upper
bound this decision was taken on", and states that "a steady-state cost
materially above the cold-driver upper bound would be a surprising result and
reopens the scope decision rather than being absorbed quietly." The reopen
condition is pre-registered and it is exactly the right one. This is a decision
knowingly taken on a provisional cost with an explicit unwind, not a decision
resting on an unexamined premise.

Grade: **unstable** on the cost premise; **stable** on the mechanism
(`independent process compilation selecting a different acceptance executable`,
established by a paired arm — 11 changing case verdicts cache-off vs 0
fixed-cache, with the within-executable control retained at 12 of 12
`STATE_REPRODUCIBLE`, 0 unequal flux elements, 0 label movement — rather than
by elimination alone), and **stable** on what the lock does not buy (acceptance
still fails 67 of 69 once deterministic, stated in the decision itself).

---

## 6. mast-catalog-gpu-solve — stable, correctly blocked, and the blocking
evidence is a self-inflicted walltime

No missed trigger. All three open decisions say "measure L1 first" and none of
their triggers is satisfied by anything banked in S15:

- **Sharding** (`single-process jax.sharding` vs `per-GPU process array`,
  "decide at L2 from L1 evidence"). Its evidence source is precisely
  coefficient-space-newton §12, whose done-when requires "a card ladder from
  one card to the largest the reservation grants, reporting throughput and
  scaling efficiency at each rung **together with the sharding arrangement
  used**". §12 has not landed. The decision is correctly open and correctly
  waiting; it is not being pre-empted.
- **Output store** (zarr vs IDS vs both). Nothing in S15 measures it.
- **Convergence policy** (one global budget vs two-pass, "likely winner if the
  iteration-count distribution from L1 is heavy-tailed"). S15 banks advance
  counts on DIII-D frames (arm A 10 advances; margin remeasure 7–12 and 16–89
  of 89), not a MAST catalog L1 iteration distribution. Trigger unmet.

**The §12 stall is a walltime defect, and it is verifiable.** `sacct` on the two
cancelled jobs, beside two jobs on the same partition from the same sprint:

```
   1254380   betelgeuse    TIMEOUT     00:35:18   Timelimit 00:35:00
   1254382   betelgeuse    TIMEOUT     00:45:19   Timelimit 00:45:00
   1254203   betelgeuse     FAILED     01:05:56   Timelimit 02:00:00
   1254245   betelgeuse  COMPLETED     00:08:49   Timelimit 02:00:00
```

Both §12 jobs died at **their own requested limit**, on a partition that
accepted and ran a 2-hour request twice in this same sprint. The requested
times, 35 and 45 minutes, track the reckon dispatch time budget rather than the
worker's own 32–35 minute GPU estimate — 1254382 overran by 19 seconds, i.e.
it was submitted with essentially zero margin. Nothing about the partition
capped it.

The repository already holds the banked demonstration, landed in this sprint on
the plateau plan: `mast-response-carrier-warm` (`29e24759`) records "the same
build that a one-hour debug cap truncated twice, at 48:07 and 40:15, finishes
in eight and a half minutes where there is no cap", on betelgeuse under
`--reservation=gpu_0003_grpA`. So the trigger the brief asks about is not
merely available — it is *already written into an S15 receipt* and was not
applied one section over. Severity: **process-defect**, and the cheapest fix in
the sprint: a single generous `--time` on betelgeuse ends it.

Downstream consequence, stated precisely: the sharding decision waits on §12's
card ladder, so this walltime is on the critical path of a blocked plan's
decision — not merely of a benchmark number.

---

## 7. Cross-cutting: does any landed verdict depend on the discrete fixed point
being close to analytic truth?

Checked, because a 26–45 % analytic sup error would invalidate anything scored
against a reference. **Answer: no landed S15 verdict depends on it, and the
blast radius is scoped correctly.**

discrete-operator-analytic-error §3 states the boundary in the right place:
"Every parity, accuracy and reproduction claim that scores Nova's forward map
against an analytic or reference solution inherits this residue … Claims about
self-consistency are untouched — a converged fixed point is still converged, a
batched route still matches a scalar one, and a deterministic route is still
deterministic."

Applying that test to the S15 landings:

| landed verdict | scored against | inherits the residue? |
|---|---|---|
| topology margin seam `9cc2d0f3` | agreement between two of Nova's own classifiers (53 of 53) | no — self-consistency |
| margin frame remeasure `fdffd8e9` | Nova's own relative residual + class margin | no — self-consistency |
| plateau four-candidate closure | Nova's own label-seeded residual field | no — self-consistency |
| clip gate lock `9317ef50` | TORAX and Nova's contour reader (both external/independent readers, not a closed form) | no |
| matrix convergence axis (arms A–D) | each arm's own terminal residual | no |
| **matrix accuracy axis (`analytic-truth-rescore`)** | **closed-form truth** | **yes — and it was superseded by `b774c28b`** |

The single verdict that did depend on analytic closeness is the one the
campaign already retired, for exactly this reason. That is the correct outcome
and it was reached by measurement (a seven-rung fit) rather than by the
argument being noticed.

---

## 8. Process observations that are not grounds defects

- **Unpromoted completed run.** `higher-order-contour-extraction` (`479667ed`)
  reported complete at 06:27:33Z and was still a pointer at 06:40Z, carried as
  a `completed_unpromoted` waiver in `docs/state/nova/crew.json` with
  `next_action` naming the exact `reckon crew complete` invocation. Its content
  has since been written into §15 of the live plan and merged at `95344e75`, so
  the plan-write half of the landing beat is done. Landing-beat discipline says
  promote and plan-write in the same beat; here they separated by one beat.
  Severity: process-defect, resolved.
- **A mis-routed dispatch is recorded rather than hidden.** The execution log
  carries "stopped 1 min after launch: dispatch routed to the codex default
  backend where the coordinator required a claude opus worker; redispatched
  with default_backend forced", and `5e3d56ea` reconciles a mis-routed audit
  dispatch. Both are recorded as attempts rather than dropped, which is what
  makes the four-attempt history of `support-order-iso-accuracy` (two target
  construction defects, one login-lane CUDA capacity failure with unrelated
  processes holding 21.7 of 23.0 GiB, one numerically complete run whose
  reporting logic mislabelled mixed rung errors as a systematic improvement,
  "caught by validation rather than published") readable at all.

---

## Summary of findings, most severe first

1. **unstable-grounds** — same-device-label-determinism §3/§5: the
   production-wide lock's cost premise. The published `0.113 s / 2.197 %` is a
   2-sample warm mean minus a 3-sample median (consistent estimators give
   1.55–2.38 %, so the magnitude survives), and the "UPPER BOUND" framing is
   **not verified**: the receipt records only whole-invocation elapsed time over
   30 slices and cannot distinguish a per-process cache overhead (amortises,
   framing holds) from a slower pinned executable (does not amortise, framing
   inverts). Guarded by §5's pre-registered reopen condition.
2. **unstable-grounds** — plateau-input-attribution §7 and its shipped
   followup: "all four documented candidates closed by measurement" /
   "NONE of the four documented substitutions moved the plateau". Two of four
   (profiles `0.0514925944`, discretisation `0.0326058989`) were never
   substituted; they are closed by non-implication on a spatial pattern score
   whose own receipt says "this is spatial attribution only, not proof that
   changing either input yields a converged parity root". Direction unaffected.
3. **process-defect** — coefficient-space-newton §12: SLURM 1254380/1254382
   TIMEOUT at their own 00:35:00 and 00:45:00 requests on betelgeuse, a
   partition this same sprint ran a 2-hour request on twice (1254203, 1254245).
   Blocks mast-catalog-gpu-solve's sharding decision, not just a number.
4. **process-defect** — same-device-label-determinism: "the two bounds whose
   verdicts moved" conflates aggregate bounds with per-case verdicts.
   `pass_status_changing_observables` is `[]` and
   `observable_pass_count_by_invocation` is `[67,67,67]` in all four
   arm×width summaries; what moved is 11 per-case verdicts.
5. **process-defect (cosmetic)** — curved-clip-global-surface: gates `3.92e-2`
   / `6.20e-2` are pinned from a superseded snapshot of the reference-reference
   spread, which §4 re-measures at `3.932154` / `6.220674`.
6. **process-defect (resolved)** — `higher-order-contour-extraction` promoted
   one beat after its plan-write.
7. **clean** — the circular-scoring incident. The defect was named and owned,
   the reversal was sound on its own construction, and the reversal was then
   itself correctly superseded by a seven-rung fit rather than left standing on
   two-point margins. No conclusion banked on either side of it still stands.
8. **clean** — the analytic-truth blast radius. No landed verdict outside the
   superseded accuracy axis depends on the discrete fixed point being close to
   the closed form; the self-consistency / agrees-with-physics distinction is
   drawn correctly and in the right document.
9. **clean** — curved-clip gates are measured and asserted against independent
   bounds installed at `9317ef50` (`assert error < limit`), not against chosen
   values; and §6 found and fixed a green suite measuring the wrong route.
10. **clean** — mast-catalog-gpu-solve: three decisions correctly open, no
    trigger satisfied, none pre-empted.
