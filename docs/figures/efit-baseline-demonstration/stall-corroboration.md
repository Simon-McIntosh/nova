# Independent stall corroboration

## Result

All four named EFIT-disagreeing MAST arms are **stalled limited terminals, not converged topology misclassifications**. The current replay leaves every terminal between `2.763141e-3` and `7.425989e-3`, far above the fixed-point criterion of `1e-8`. Reading achieved class through the post-cutover saddle-aware partition gives **limited on 4/4**, while the independent EFIT label is diverted on 4/4. This independently agrees with the banked stalled-versus-misclassified account on all four arms.

The late confined-mask evidence is not uniform. The two 21985/51 arms retain an empty confined mask through all four late states, so their zero-change result is stable but vacuous. The 21986/46 pure arm changes 3 cells over the final three transitions, and the 21989/55 mixed arm changes 29. Late mask switching is therefore directly present on 2/4 arms, but it is not a universal explanation of all four terminal residual floors.

## Per-arm measurements

The residual trajectories list the twelve promoted-state residuals in order. Mask counts are the confined-cell population at promotions 9, 10, 11, and 12; `Δ cells` gives cell-by-cell symmetric differences for 9→10, 10→11, and 11→12. Achieved class is determined from `traced_margin_candidate_diagnostics.class_margin` after the saddle-aware partition; `ForwardTopologyState.diverted` is never used.

| MAST arm | Twelve-promotion residual trajectory | Terminal residual / criterion | Saddle-aware achieved class | Late confined cells | Δ cells | Stalled versus misclassified verdict |
|---|---|---:|---|---|---|---|
| 21985/51 pure | `6.220624e-3 → 6.105431e-3 → 5.994427e-3 → 5.896188e-3 → 5.805185e-3 → 5.712869e-3 → 5.621505e-3 → 5.528925e-3 → 5.437807e-3 → 5.346619e-3 → 5.256728e-3 → 5.167458e-3` | `5.167458e-3 / 1e-8` | limited, margin `−0.2024463`; EFIT diverted | `0, 0, 0, 0` | `0, 0, 0` | **Stalled, not a converged misclassification; agrees with banked account.** Stable mask evidence is vacuous because the confined mask is empty. |
| 21985/51 mixed | `3.404299e-3 → 2.297791e-3 → 2.200630e-3 → 2.730232e-3 → 2.104020e-3 → 2.545402e-3 → 2.705266e-3 → 2.942901e-3 → 2.521126e-3 → 2.742040e-3 → 3.005782e-3 → 3.179711e-3` | `3.179711e-3 / 1e-8` | limited, margin `−0.2069079`; EFIT diverted | `0, 0, 0, 0` | `0, 0, 0` | **Stalled, not a converged misclassification; agrees with banked account.** Stable mask evidence is vacuous because the confined mask is empty. |
| 21986/46 pure | `9.499851e-3 → 9.267295e-3 → 9.038080e-3 → 8.814090e-3 → 8.608701e-3 → 8.407961e-3 → 8.213984e-3 → 8.040867e-3 → 7.870796e-3 → 7.704196e-3 → 7.549070e-3 → 7.425989e-3` | `7.425989e-3 / 1e-8` | limited, margin `−0.1482236`; EFIT diverted | `261, 261, 263, 264` | `0, 2, 1` | **Stalled, not a converged misclassification; agrees with banked account.** Three late cell switches support a state-dependent-domain contribution. |
| 21989/55 mixed | `4.774947e-3 → 4.576544e-3 → 4.482717e-3 → 2.341446e-3 → 1.758303e-3 → 2.070433e-3 → 2.082711e-3 → 2.442292e-3 → 2.518198e-3 → 2.684594e-3 → 2.743984e-3 → 2.763141e-3` | `2.763141e-3 / 1e-8` | limited, margin `−0.1614419`; EFIT diverted | `271, 264, 272, 264` | `9, 10, 10` | **Stalled, not a converged misclassification; agrees with banked account.** Twenty-nine late cell switches coincide with the residual's post-minimum rise. |

The saddle-aware class remains limited at every one of the four late promotions on every arm, including the two arms whose confined masks change. The topology verdict is therefore stable even when the discrete residual domain is not.

## Measurement and independence

The study freshly replays the selected banked references and persisted 101-circuit response carrier at the current checkout. Pure arms use the established retained-state equivalent of production `solve_portfolio`: twelve sequential one-promotion `newton_krylov` calls. Mixed arms use the established retained-state residual-plus-margin ladder. For each arm the final four promoted states are passed through the same topology snapshot machinery used by the discrete mask-switch diagnosis, and achieved class is then recomputed through the post-cutover saddle-aware class reader.

This is an independent adjudication rather than a transcription of the older summary:

- Mixed-arm residual trajectories reproduce the pinned receipt within `5e-12` on 2/2 arms.
- Current pure-arm replay does not reproduce the older pinned receipt byte-for-byte: maximum trajectory drift is `8.290e-7` on 21985/51 and `1.338e-6` on 21986/46. Terminal shifts are only `−2.539e-7` and `+6.528e-7`, respectively, so both remain more than five orders of magnitude above the convergence criterion and the stall verdict is unchanged.
- Independently recomputed saddle-aware class agrees with the current class bank on 4/4 arms.
- No production source, test, bank, or response carrier was modified.

Exact arrays, input digests, class margins, late mask counts, and replay deltas are in [`stall-corroboration.json`](stall-corroboration.json).

## Design evidence and qualification

The evidence supports first-class per-outer-iteration mask instrumentation: it would expose real late domain motion on 21986/46 pure and 21989/55 mixed instead of inferring it from residual oscillation. It does **not** justify claiming one common mask-switch mechanism for all four arms. On 21985/51 the late confined mask is empty and unchanged, so those arms need either an earlier-window mask census or a separate continuous-residual attribution before a treatment is credited with their stall.

The window is deliberately narrow: the last four promoted states and three adjacent transitions. Stability there does not exclude an earlier mask crossing, and a changing confined mask does not by itself prove causality. It establishes that the nonlinear residual domain is still moving at the tail on two arms while the achieved saddle-aware class stays limited on all four.
