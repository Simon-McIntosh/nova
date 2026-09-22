# Measured candidate acceptance under a pessimistic model

The production selector admits a finite candidate whose local model predicts no decrease only when the candidate independently passes the unchanged incumbent-relative sufficient-decrease threshold and a strict sup-residual decrease on the incumbent's own mask. A decrease outside that mask alone is insufficient. Such an admission marks the local model unreliable and carries an explicit rebuild requirement into the next Newton trip, where the existing rebuilt-model promotion path is used. Predictions of positive decrease retain the existing realized-versus-predicted trust test and do not request a rebuild.

The recorded-state replay calls `nova.equilibrium.fixed_point._select_backtracking_candidate`, the selector kernel used by the production backtracking path. It does not reproduce the rule separately.

| Witness | Actual merit | Predicted merit | Actual residual | Production-selector verdict |
|---|---:|---:|---:|---|
| Historical full analytic | 9.62674026395e-5 | 0.348071530520 | 9.53950481603e-5 | accepted at index 0; model unreliable |
| Authoritative 135-cell full analytic | 9.62674026395e-5 | 0.271049525734 | 9.53950481603e-5 | accepted at index 0; model unreliable |
| Authoritative 342-cell full analytic | 7.58066918431e-5 | 0.291954563011 | 9.59626538977e-5 | accepted at index 0; model unreliable |
| Historical half defect | 0.0621299454943 | 0.0624428163974 | 0.0535951962998 | accepted at index 1 by the existing trusted-model rule |

All seven measured Newton candidates on each authoritative row worsen actual merit and are refused by that same production selector: 14 of 14. The committed synthetic cases additionally pin the authority boundary in eager and JIT execution: the otherwise identical pessimistic candidate is refused when its apparent residual decrease exists only outside the incumbent's own mask. A promotion-sequence case counts exactly one rebuilt-model invocation after one pessimistic admission and none after an ordinary trusted admission.

The final weak-rotation 300-requested-cell production row realizes 342 cells and exhausts sixteen active-set trips at residual 0.05529921020198606, converged false, termination reason `10`, and digest `bfdc797c2b9cc85a36ed04124019b8d4baf7ad0af145d861e88b6808fcd5a5b7`. It supersedes the selector-only intermediate digest `69f8805e…`, which converged in two trips before the unreliable-model flag was made causal. The movement from that intermediate terminal, as well as from the historical pointwise-quadrature digest `d730c294…` and continuous-support terminal `3e9fa335…`, is explained by forcing the existing rebuilt-model path on the trip after pessimistic admission.

This single-row outcome is not the locked cross-row convergence claim. Production still proposes only Newton directions; this change decides an already evaluated candidate and, when appropriate, rebuilds the model used for the next Newton proposal. The second authoritative row, analytic-axis/null bounds, joint qualification, and the locked cross-row convergence gate remain outside this node.

One all_debug allocation, job 1275537, ran the root interpreter directly with `JAX_PLATFORMS=cpu` and `TMPDIR=/tmp` in the submit environment and payload. Five fresh pytest processes passed 164 tests: 10 measured-candidate cases, 41 fixed-point cases, 28 route-wiring cases, 69 default-wiring cases, and 16 scan-shape cases. The base revision passed the corresponding pre-existing 154 tests. The selector replay and terminal measurement returned zero. The declared scratch mutation, `Require model trust even when a finite candidate passes strict measured decrease.`, restored the veto and failed both eager and JIT pessimistic-admission cases, exit 1. No budget, current pin, sufficient-decrease slope, residual guard, backtracking factor, Newton-step limit, Krylov limit, or active-set trip limit changed.

Receipts: [selector outcomes](/nova/figures/cut-cell-current-attribution/measured-candidate-acceptance/selector-receipt.json) and [terminal identity](/nova/figures/cut-cell-current-attribution/measured-candidate-acceptance/terminal-receipt.json). The numerical relationship is most directly represented by the table; no additional figure would clarify it.
