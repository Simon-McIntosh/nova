# Map-fidelity re-measure at the analytic Solov'ev state

Revision `fa8af260e8adb154a7df4191d0c5308e6b675911` in worktree `/home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/s21-nova/cca-map-fidelity-h200-remeasure`; one H200 allocation, job 1279817 (betelgeuse, reservation gpu_0003_grpA, 1x H200). Every certificate rung of `diverted-single-null` and `weak-rotation-reactor-static` is evaluated in both `exact` and `chord` clip modes. Pass bound: sup relative and RMS relative both below 1e-2. Receipts and per-row state archives are beside this file.

## Rows

| case | clip | requested | realised | sup_relative | pass | rms_relative | dR [mm] | dZ [mm] |
|---|---|---|---|---|---|---|---|---|
| diverted-single-null | exact | -110 | 132 | 0.180568 | fail | 0.163203 | -2.0939 | -11.0855 |
| diverted-single-null | chord | -110 | 132 | 0.261242 | fail | 0.223655 | -5.1773 | 4.34173 |
| weak-rotation-reactor-static | exact | -110 | 135 | 0.000165923 | pass | 0.000183025 | -0.0555209 | -2.71174e-11 |
| weak-rotation-reactor-static | chord | -110 | 135 | 0.0186063 | fail | 0.019408 | -0.64553 | 7.67148e-07 |
| diverted-single-null | exact | -300 | 340 | 0.154141 | fail | 0.130062 | -1.51143 | -8.41419 |
| diverted-single-null | chord | -300 | 340 | 0.215919 | fail | 0.20754 | 0.984305 | 12.5668 |
| weak-rotation-reactor-static | exact | -300 | 342 | 0.00143219 | pass | 0.0011433 | -0.0754499 | 0.566991 |
| weak-rotation-reactor-static | chord | -300 | 342 | 0.0245458 | fail | 0.0166505 | -6.63793 | 2.23951 |
| diverted-single-null | exact | -500 | 550 | 0.138457 | fail | 0.110956 | -1.27397 | -7.35171 |
| diverted-single-null | chord | -500 | 550 | 0.113035 | fail | 0.114456 | -2.13561 | 1.07217 |
| weak-rotation-reactor-static | exact | -500 | 555 | 5.93661e-05 | pass | 8.56986e-05 | -0.00399313 | 1.47497e-12 |
| weak-rotation-reactor-static | chord | -500 | 555 | 0.0258684 | fail | 0.0287746 | -12.825 | -1.60613e-10 |
| diverted-single-null | exact | -1000 | 1074 | 0.0879875 | fail | 0.0755735 | -0.929041 | -4.78468 |
| diverted-single-null | chord | -1000 | 1074 | 0.108867 | fail | 0.06489 | 1.92442 | -0.195678 |
| weak-rotation-reactor-static | exact | -1000 | 1072 | 0.000177483 | pass | 0.000216791 | -0.00685704 | -0.0274404 |
| weak-rotation-reactor-static | chord | -1000 | 1072 | 0.0167256 | fail | 0.00794152 | -2.91924 | 3.37026 |
| diverted-single-null | exact | -2500 | 2616 | 0.0658035 | fail | 0.0540696 | -0.628764 | -3.21176 |
| diverted-single-null | chord | -2500 | 2616 | 0.0597244 | fail | 0.0347732 | -0.417378 | 0.730409 |
| weak-rotation-reactor-static | exact | -2500 | 2608 | 6.16073e-05 | pass | 8.40992e-05 | -0.000361769 | 5.52273e-11 |
| weak-rotation-reactor-static | chord | -2500 | 2608 | 0.00560465 | pass | 0.00285007 | 0.950804 | 2.6772e-12 |

## Comparison with the committed receipt

Committed receipt revision `de29a1a303098542006dfe1b96575fce86ba535a` (predates the chord-membership merge 92613fd3d of 2026-10-02 00:56).

| case | clip | requested | committed sup | this sup | ratio |
|---|---|---|---|---|---|
| diverted-single-null | exact | -110 | 0.447979 | 0.180568 | 0.4031x |
| diverted-single-null | chord | -110 | 0.0950848 | 0.261242 | 2.747x |
| weak-rotation-reactor-static | exact | -110 | 0.0930861 | 0.000165923 | 0.001782x |
| weak-rotation-reactor-static | chord | -110 | 9.54029e-05 | 0.0186063 | 195x |
| diverted-single-null | exact | -300 | 0.299535 | 0.154141 | 0.5146x |
| diverted-single-null | chord | -300 | 0.02291 | 0.215919 | 9.425x |
| weak-rotation-reactor-static | exact | -300 | 0.0248543 | 0.00143219 | 0.05762x |
| weak-rotation-reactor-static | chord | -300 | 9.59681e-05 | 0.0245458 | 255.8x |
| diverted-single-null | exact | -500 | 0.120879 | 0.138457 | 1.145x |
| diverted-single-null | chord | -500 | 0.0338673 | 0.113035 | 3.338x |
| weak-rotation-reactor-static | exact | -500 | 0.0209503 | 5.93661e-05 | 0.002834x |
| weak-rotation-reactor-static | chord | -500 | 4.84589e-05 | 0.0258684 | 533.8x |
| diverted-single-null | exact | -1000 | 0.149589 | 0.0879875 | 0.5882x |
| diverted-single-null | chord | -1000 | 0.012676 | 0.108867 | 8.588x |
| weak-rotation-reactor-static | exact | -1000 | 0.025369 | 0.000177483 | 0.006996x |
| weak-rotation-reactor-static | chord | -1000 | 4.67932e-05 | 0.0167256 | 357.4x |
| diverted-single-null | exact | -2500 | 0.120138 | 0.0658035 | 0.5477x |
| diverted-single-null | chord | -2500 | 0.00409184 | 0.0597244 | 14.6x |
| weak-rotation-reactor-static | exact | -2500 | 0.0163335 | 6.16073e-05 | 0.003772x |
| weak-rotation-reactor-static | chord | -2500 | 4.82138e-05 | 0.00560465 | 116.2x |

## Verdicts

- **Diverted chord below 1e-2 at every rung: FAIL.** Every diverted-single-null chord rung is outside the bound (sup 0.261 at 110, 0.216 at 300, 0.113 at 500, 0.109 at 1000, 0.0597 at 2500). Against the committed receipt the diverted chord map is worse at every rung (2.7x to 14.6x).
- **Weak chord no worse than its committed receipt: FAIL.** Every weak-rotation chord rung is worse, by 116x (2500) to 534x (500). The weakest rung, -2500, is the only weak chord row inside 1e-2; all others fail.
- **Exact rows (decide gate `g-cca-exact-default-waits-on-map-fidelity`): the gate is not satisfiable.** Weak exact passes at every rung (1.66e-4, 1.43e-3, 5.94e-5, 1.77e-4, 6.16e-5), a large improvement over the committed receipt (0.093 to 0.016, now 2e-4 to 1e-3) consistent with the moment-reduction repair. Diverted exact fails at every rung (0.181, 0.154, 0.138, 0.088, 0.066). The gate additionally requires the chord rows to be bit-identical, which they are not: chord changed materially at every rung.

## Controls

- Instrument controls pass (identity accepted, a two-percent perturbation refused, a non-finite value refused, the non-finite counter sees its injected NaN).
- The declared negative control (the analytic state shifted vertically by one pitch) raises both mismatch norms on every row (`negative_control_all_rows_detected` true).
- Per-row `support_current_centroid_offset_mm` (booked minus analytic, radial and vertical, mm) is in the row table and in each row receipt.
