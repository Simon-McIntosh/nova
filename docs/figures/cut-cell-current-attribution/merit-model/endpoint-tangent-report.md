# Endpoint-resolved tangent audit at the 342-cell stalled terminal

**Verdict: the tangent is not wrong — the map is discontinuous at the stalled
state.** The central difference at the 342-cell terminal measures a jump, not a
derivative. Three cells — **128, 156 and 255** — change their confined-or-open
classification between the two endpoints of the difference, and the pinned map's
current moments change discretely with them. The finite difference divides that
fixed jump by the increment, so `‖fd‖ ∝ 1/h` and the discrepancy-to-fd ratio
sits at one for every increment that straddles the flip. Along the analytic
direction the flip leaves the increment at `h = 1e-8` and the tangent then agrees
to `2.5e-7`; along the Newton and map-defect directions the crossing is at or
within `1e-8` of the state, so the ratio stays at one all the way down.

## What was run

One `all_debug` allocation, job **1275518**, `COMPLETED`, scheduler exit `0:0`,
elapsed `00:01:06`, 8 CPUs, 96 GiB, peak RSS within the request; one process,
`JAX_PLATFORMS=cpu`, x64 asserted, `TMPDIR=/tmp` in submit and payload, the root
interpreter run directly with no `uv` on the node. Driver
`endpoint_tangent.py` at revision `b64a3db64`, which imports the same
instrument (`measure.py`) that produced the recorded verdict, builds the same
342-cell machine (requested 300, realised 342), and starts from the same state:
the asserted `state_digest` equals the compact receipt's terminal digest and the
analytic state matches elementwise. `support_clip_mode()` reads **chord** in this
run, and the receipt's class counts confirm its consequence: all 342 cells are
included with `area == full_area` (0.0796455 m² each) at every measured state, so
**no cell is cut and no cell is outside** — the discontinuity this audit finds is
a classification flip, not a clip-area transition. The base merit from this
instrument is `0.0546254885059`, which is the value the recorded ladder's
smallest steps converge to.

Both endpoint states of every central difference are retained in the raw receipt
(`/home/ITER/mcintos/.config/reckon/crew/reports/nova/s19-codex-20260916/cca-why-the-342-cell-terminal-fails-the-tangent-check/endpoint-tangent-cells-342.json`,
1 298 829 bytes, SHA-256 recorded in the compact receipt
[endpoint-tangent-cells-342.json](endpoint-tangent-cells-342.json)), together
with each endpoint's confined, open and label masks, its cell support areas and
full areas, its raw and scaled current moments, and its state digest. The
process log is [endpoint-tangent-300.log](endpoint-tangent-300.log) and the
launcher is [endpoint-tangent-allocation.sh](endpoint-tangent-allocation.sh).
Attempt 1 (job 1275517) failed in 51 s on an argument-count defect in the driver
itself and measured nothing; its log is preserved beside the live one in the
reports directory.

## Reproduction of the recorded verdict

| direction | measured `‖fd−jvp‖/‖fd‖` at 1e-6 | recorded | absolute delta |
|---|---|---|---|
| analytic | 0.9995651827060339 | 0.999565182706 | 3.4e-14 |
| newton | 0.995715816448768 | 0.995715816449 | 2.3e-13 |
| map_defect | 1.0000923807574222 | 1.00009238076 | 2.6e-12 |

## The increment ladder

`h` is the fraction of the direction vector; `‖fd‖` and `‖jvp‖` are state-vector
2-norms; `2h‖fd‖` is the jump the difference implies; flips are cells whose
confined flag differs between the two endpoints (the open and label masks flip
on exactly the same three cells, and no cell's shadow flag flips).

| direction | h | ‖fd‖ | ‖jvp‖ | rel over fd | ‖fd‖/‖jvp‖ | 2h‖fd‖ | flips | support change | max cell gap (A) |
|---|---|---|---|---|---|---|---|---|---|
| analytic | 1e-4 | 8.50151e3 | 373.702 | 0.958221 | 22.75 | 1.7003 | 3 | 0 | 2.93068e6 |
| analytic | 1e-6 | 8.14992e5 | 373.702 | 0.999565 | 2180.9 | 1.62998 | 3 | 0 | 2.93067e8 |
| analytic | 1e-7 | 8.14673e6 | 373.702 | 0.999957 | 21800.1 | 1.62935 | 3 | 0 | 2.93067e9 |
| analytic | 1e-8 | 373.702 | 373.702 | 2.50314e-7 | 1.00000 | 7.47e-6 | 0 | 0 | 2.55091e-3 |
| newton | 1e-6 | 8.18138e5 | 3850.3 | 0.995716 | 212.49 | 1.63628 | 3 | 0 | 2.93066e8 |
| newton | 1e-7 | 8.14987e6 | 3850.3 | 0.99957 | 2116.7 | 1.62997 | 3 | 0 | 2.93067e9 |
| newton | 1e-8 | 8.14672e7 | 3850.3 | 0.999957 | 21158.7 | 1.62934 | 3 | 0 | 2.93067e10 |
| map_defect | 1e-6 | 8.14563e5 | 102.053 | 1.000092 | 7981.8 | 1.62913 | 3 | 0 | 2.93068e8 |
| map_defect | 1e-7 | 8.14630e6 | 102.053 | 1.00001 | 79824 | 1.62926 | 3 | 0 | 2.93067e9 |
| map_defect | 1e-8 | 8.14637e7 | 102.053 | 1.00000 | 798247 | 1.62927 | 3 | 0 | 2.93067e10 |

Flip ids at every straddling row, all three directions: `[128, 156, 255]`.
Nonzero-current support (177 cells at the base state): `+only` 0, `−only` 0, and
no cell differing from the base support, at every row — the flip changes current
values, not the support.

## Which of the two roots it is

**The analytic direction is a classification flip inside the increment: it
vanishes at a smaller increment.** At `h = 1e-8` the flip count is zero, the
implied jump collapses from 1.63 to `7.5e-6`, `‖fd‖/‖jvp‖` is 1.00000, and the
residual discrepancy `2.5e-7` is the ordinary truncation-order term. The flip
surface lies between `1e-8` and `1e-7` of the analytic step from the state.

**The Newton and map-defect directions put the state on the flip surface: the
discrepancy persists and grows as `1/h`.** Their flips stay at three cells and
their rel-over-fd stays at 0.99996 / 1.0000009 down to `1e-8`.

**A wrong local-coefficient tangent is excluded, by two independent facts.**
First, a coefficient error predicts `‖fd‖/‖jvp‖` tending to a *finite* value as
`h → 0`; the measured ratios instead grow by exactly one decade per decade of
`h` — 2180 → 21800 → 1 (analytic, which then leaves the flip entirely), 212 →
2117 → 21159 (newton), 7982 → 79824 → 798247 (map defect) — with no limit.
Second, the implied jump `2h‖fd‖` is the *same number* across four decades of
increment and across all three directions: 1.62913–1.63628 for the six
straddling fine rows and 1.7003 at the `1e-4` control, against `7.5e-6` at the
one clean row. A direction-dependent coefficient error cannot produce a
direction-independent jump of constant size while the three JVP norms differ by
a factor of 38 (102.053 to 3850.3). The jump is a property of the field, not of
the linearisation.

## Which cells carry the discrepancy

Every named cell is **wholly inside** at the base state, at the plus endpoint and
at the minus endpoint: `class_counts` is `{outside: 0, cut: 0, inside: 342}` at
all three, and each named cell's area equals its full area (0.0796455 m²) at all
three. The chord clip mode admits no cut cell, so no named cell is cut and none
is outside.

At `h = 1e-6`, analytic direction, the six cells with the largest gap between the
finite-difference current and the JVP prediction:

| cell | class | base (A) | plus (A) | minus (A) | fd (A) | jvp (A) | gap (A) |
|---|---|---|---|---|---|---|---|
| 255 | inside/inside/inside | 29902.1 | 29902.3 | 29315.9 | 2.93221e8 | 1.53596e5 | 2.93067e8 |
| 51 | inside/inside/inside | 44211.3 | 44211.5 | 43660.9 | 2.75298e8 | 1.40421e5 | 2.75158e8 |
| 43 | inside/inside/inside | 26723.8 | 26724.0 | 26181.4 | 2.71299e8 | 1.33169e5 | 2.71166e8 |
| 256 | inside/inside/inside | 20045.6 | 20045.8 | 19510.1 | 2.67826e8 | 1.37542e5 | 2.67688e8 |
| 284 | inside/inside/inside | 29791.6 | 29791.8 | 29276.4 | 2.57695e8 | 1.20378e5 | 2.57575e8 |
| 44 | inside/inside/inside | 33039.3 | 33039.4 | 32531.7 | 2.53862e8 | 1.15652e5 | 2.53746e8 |

Cell 255 is one of the three flip cells; the others are not. The pattern is the
same in every named row and in the Newton and map-defect directions: the plus
endpoint's current is within a few tenths of the base value, the minus endpoint's
is lower, and the difference is the jump. The moment vector moves as a whole —
the raw current total goes from 9.20315e6 A (plus) to 9.18517e6 A (minus), a
0.195 % shift, while individual cells move by 1.2–2.0 % — so the flip
redistributes the moment values without adding or removing a current-carrying
cell. In the flux map the jump is largest at grid nodes 1065, 181, 1064 and 1060,
where the apparent discrepancy is 6.45e4 Wb at `h = 1e-6`, falling as `1/h` with
the same constant: the node-level jump is 0.129 Wb.

## Positive controls

- **Reproduction** of all three recorded values to 3.4e-14, 2.3e-13 and 2.6e-12.
- **Classification detector**: a seeded single-bit flip in the confined mask is
  reported as exactly 1 changed cell — the detector is live.
- **Support detector**: 177 nonzero-current cells reported at the base state —
  the support comparison is not trivially empty.
- **Coarser control increment** `1e-4` on the analytic direction reports the same
  three flips and the same jump magnitude (1.7003), so the straddling set is
  stable three decades further out; the effect is not a fine-increment artifact.
- **Clean rung**: at analytic `h = 1e-8` the flip and the jump are both absent and
  the per-cell current gap falls to 0.0026 A against a 6.26e4 A difference — the
  same instrument reports the tangent as correct when no flip occurs.

## What this means for the merit model

The recorded model-trust ladder and this audit agree on the mechanism. In the
recorded table the analytic direction at `α = 0.01` predicts the actual merit to
1.00023 with zero flips, and at `α = 1` it is off by 3851× with 129 flips: the
linear model is accurate exactly while its step does not cross a classification
boundary, and wrong by orders of magnitude when it does. This audit locates the
first crossing on the analytic ray between `1e-8` and `1e-7` of that step, and
shows that at the stalled terminal the Newton and map-defect rays begin *on* the
surface. So the refusal of these directions is not a coefficient defect to
repair in the tangent; it is the map's own branch structure. The shapes the
evidence supports are a step that stays inside a branch — a branch-frozen
linearisation with re-classification only at acceptance, or a continuation in the
clip geometry that moves the support before the flux. There is no measured
support here for fixing a local coefficient.

The two flip counts are associated rather than cross-checked: the recorded
table's flip column and this audit's flip count are computed on different state
pairs, so the co-movement above is the safe reading and not a quantitative
agreement.

## Limits

One fixture, one stalled terminal state, one machine, CPU JAX, chord clip mode.
Because chord mode admits no cut cells, nothing here resolves any cut-cell
quadrature question; the discontinuity is the confined/open partition. The flip
surface is located only within a factor of ten along the analytic direction and
only bounded above along the Newton and map-defect directions; both endpoint
states of each difference are retained with their digests, so a bisection could
tighten that distance without re-measuring the base state. The 135-cell terminal
passes the same instrument at 1.7e-9, which this audit does not revisit and does
not claim to explain.

No figure accompanies this record: the result is tabular, and the node's lane is
text only.