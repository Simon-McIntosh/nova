# Analytic-start production-iteration probe

This probe separates the reference's one-application allocation floor, the
Jacobian used by the inner Newton iteration, and Newton-Krylov globalisation.
It changes no production code.

## Map floor and decomposition

The fixture exterior is the analytic total flux minus the frozen-block image of
the analytic density integrated on the fixture's whole-cell partition.  On
every finite row, the decomposition closes to binary64 zero: the fixture
external array minus the exterior implied by the production booking is the same
field as the production internal image minus the fixture's analytic-moment
image, and on residual carriers both equal the measured map residual.  The
external and internal columns below are therefore two views of one booking
incompatibility, not additive errors.

| Row | Mode | Floor rms / span | Affine rms | Remainder rms | External rms | Internal rms | Booked / analytic current |
|---|---|---:|---:|---:|---:|---:|---:|
| weak 110 | whole-cell | 0.201692 | 0.176044 | 0.098427 | 0.201692 | 0.201692 | 0.91621203 |
| weak 110 | exact | 0.284933 | 0.226770 | 0.172517 | 0.284933 | 0.284933 | 0.98819755 |
| weak 300 | whole-cell | 0.126709 | 0.118591 | 0.044625 | 0.126709 | 0.126709 | 1.04847671 |
| weak 300 | exact | 0.078681 | 0.069046 | 0.037727 | 0.078681 | 0.078681 | 0.99998729 |
| moderate 110 | whole-cell | 0.047462 | 0.042524 | 0.021079 | 0.047462 | 0.047462 | 0.98017771 |
| moderate 110 | exact | 0.163751 | 0.128100 | 0.102004 | 0.163751 | 0.163751 | 0.99998693 |
| moderate 300 | whole-cell | 0.131455 | 0.123055 | 0.046237 | 0.131455 | 0.131455 | 1.05202230 |
| moderate 300 | exact | 0.076056 | 0.068646 | 0.032743 | 0.076056 | 0.076056 | 0.99998712 |
| single-null 300 | whole-cell | refused | n/a | n/a | n/a | n/a | refused |
| single-null 300 | exact | refused | n/a | n/a | n/a | n/a | 0 before refusal |
| single-null 500 | whole-cell | 4.87e-15 | 3.67e-15 | 3.20e-15 | 4.77e-15 | 4.75e-15 | 1.00000000 |
| single-null 500 | exact | 2.063231 | 1.107142 | 1.741023 | 2.063231 | 2.063231 | 0.60878945 |
| weak 1000 | whole-cell | 0.233437 | 0.225314 | 0.061045 | 0.233437 | 0.233437 | 1.08573216 |
| weak 1000 | exact | 0.136088 | 0.133436 | 0.026732 | 0.136088 | 0.136088 | 0.99998738 |
| moderate 1000 | whole-cell | 0.225459 | 0.217562 | 0.059148 | 0.225459 | 0.225459 | 1.08590932 |
| moderate 1000 | exact | 0.127486 | 0.124527 | 0.027307 | 0.127486 | 0.127486 | 0.99998563 |
| single-null 1000 | whole-cell | 4.98e-15 | 3.64e-15 | 3.40e-15 | 4.82e-15 | 4.79e-15 | 1.00000000 |
| single-null 1000 | exact | 2.077132 | 1.144776 | 1.733195 | 2.077132 | 2.077132 | 0.60888480 |

The weak-1000 exact per-cell census locates the difference: inactive cells
carry 50.36% of the absolute booked-minus-analytic current, cut cells carry
49.04%, and whole cells carry only 0.602%.  Across all 823 whole cells with
nonzero analytic current, the unnormalised production cell current divided by
the analytic integral over the identical polygon is one to within 6.66e-16,
with no trend in R.  Production and analytic density therefore agree on whole
cells to about 7e-16.  Participation—the cells made inactive and the retained
region in cells classified as cut—is the only material difference.  Fixture
construction is at `scripts/analytic_oracle_fixtures/measure.py:904-908`;
production density evaluation reaches `profile.current_density` through
`ForwardFluxOperator._partitioned_current_moments`,
`ForwardSource.current_moments`, and
`ForwardFluxOperator.support_current_moments`.

## X-point candidates

| Requested cells | Production class at analytic flux | x_candidate_count |
|---:|---|---:|
| 300 | limited | 0 |
| 500 | diverted | 1 |
| 1000 | diverted | 1 |

The single-null 300 refusal is absence of a retained saddle candidate, not
rejection of a candidate already present.

## Jacobian-vector products

The residual is the state minus the certificate's shadow-frozen,
target-normalised production map.  Each value is the maximum relative
discrepancy across four fixed-seed smooth directions between `jax.linearize`
and central finite differences at relative steps 1e-5 and 1e-7 of the analytic
flux span.

| Row | Mode | Discrepancy at 1e-5 | Discrepancy at 1e-7 |
|---|---|---:|---:|
| weak 110 | whole-cell | 5.2422e-12 | 5.5889e-10 |
| weak 110 | exact | 1.6618013e-2 | 1.6618585e-2 |
| weak 300 | whole-cell | 5.6286e-12 | 5.9751e-10 |
| weak 300 | exact | 1.5105548e-2 | 1.5106323e-2 |
| moderate 110 | whole-cell | 5.4149e-12 | 5.1202e-10 |
| moderate 110 | exact | 1.5952467e-2 | 1.5953780e-2 |
| moderate 300 | whole-cell | 5.8748e-12 | 6.0220e-10 |
| moderate 300 | exact | 1.5249046e-2 | 1.5250551e-2 |
| single-null 300 | whole-cell | unavailable | unavailable |
| single-null 300 | exact | unavailable | unavailable |
| single-null 500 | whole-cell | 7.5290e-12 | 7.7919e-10 |
| single-null 500 | exact | 2.0116965e-1 | 2.0116973e-1 |

The whole-cell JVP agrees with finite differences to below 8e-10 at the finer
step.  Exact mode has a step-independent discrepancy of 1.5–1.7% on limited
rows and 20.1% on single-null 500, so its differentiated residual is not
consistent with the finite-difference residual at the analytic field.

## Positive-control contraction

Single-null 500 with the whole-cell fixture allocation has a map floor of
1.18e-15 rms and 2.43e-15 sup of span.  Every perturbed start had a nonzero
production-map residual, made one attempted and accepted Newton promotion, and
returned to the same analytic discrete flux.

| Perturbation / span | Start residual | Terminal residual | Distance / span | Trips | Accepted / attempted | Returned |
|---:|---:|---:|---:|---:|---:|---|
| 1e-4 | 6.7880253e-5 | 6.2029009e-15 | 1.4282481e-14 | 1 | 1 / 1 | yes |
| 1e-3 | 6.7880253e-4 | 6.2029009e-15 | 1.4282481e-14 | 1 | 1 / 1 | yes |
| 1e-2 | 6.7880253e-3 | 6.2029009e-15 | 1.4282481e-14 | 1 | 1 / 1 | yes |
| 1e-1 | 6.7880253e-2 | 6.2029009e-15 | 1.4282481e-14 | 1 | 1 / 1 | yes |

The terminal axis and X-point match the production read of the analytic flux
to 7.35e-15 m and 4.97e-16 m.  The iteration verdict is: **where the fixed
point is the reference, production Newton-Krylov contracts back to it from
every tested start, including one tenth of the flux span away.**

![Single-null 500 whole-cell terminal iterate against analytic contours](/nova/figures/cut-cell-current-attribution/oracle-start/diverted-single-null-cells-500-fixture_exterior_control.png)

## Exact-clip solve-memory block

| Requested cells | Row and variant | Free card memory at launch | Refused boolean allocation | Completed perturbation arms |
|---:|---|---:|---:|---:|
| 300 | weak, re-posed exact | 143.771 GiB | 97.31 GiB | 0 |
| 500 | weak, re-posed exact | 143.771 GiB | 131.47 GiB | 0 |
| 1000 | weak, re-posed exact | sufficient at launch | 278–279 GiB | 0 |

The weak-300 re-posed map itself closes to 1.21e-16 rms and 2.84e-16 sup of
span, so the reference was successfully made a fixed point before the solve
entered the failing path.  Nevertheless, `jit_bitwise_and` exhausted the H200
at every tested cell count.  Variant (ii) therefore has no contraction result:
all exact-clip production solves are blocked until the boolean temporary is
removed.  The 97, 131 and 279 GiB measurements are handed to the live plan's
exact-clip solve-memory-scaling follow-up; the separate HLO analysis owns
naming the array and its producer.  No further job is submitted by this node.

## Final attribution

- Limited fixture rows: the map floor is the production participation/booking
  against a whole-cell-posed exterior; whole-cell density itself is exact.
- Exact rows: the fixture allocation floor is joined by a 1.5–20% Jacobian
  discrepancy, and re-posed globalisation cannot be measured because of the
  solve-memory defect.
- Single-null 300: the topology read retains no X-point candidate and refuses
  current normalisation.
- Single-null 500 whole-cell: allocation and JVP are correct to roundoff and
  Newton globalisation contracts to the reference from 0.1 span away.

