# Measured candidate acceptance under a pessimistic model

The production selector now admits a finite candidate whose local model predicts no decrease only when the candidate independently passes the unchanged incumbent-relative sufficient-decrease threshold and a strict own-mask sup-residual decrease. Such an admission marks the local model unreliable; the next Newton iteration rebuilds the linearization at the promoted state. Predictions of positive decrease retain the existing realized-versus-predicted trust test.

| Witness | Actual merit | Predicted merit | Actual residual | Verdict |
|---|---:|---:|---:|---|
| Historical full analytic | 9.62674026395e-5 | 0.348071530520 | 9.53950481603e-5 | accepted; model unreliable |
| Authoritative 135-cell full analytic | 9.62674026395e-5 | 0.271049525734 | 9.53950481603e-5 | accepted; model unreliable |
| Authoritative 342-cell full analytic | 7.58066918431e-5 | 0.291954563011 | 9.59626538977e-5 | accepted; model unreliable |
| Historical half defect | 0.0621299454943 | 0.0624428163974 | 0.0535951962998 | accepted by the existing trusted-model rule |

All seven measured Newton candidates on each authoritative row worsen actual merit and remain refused: 14 of 14. The committed synthetic cases pin pessimistic acceptance, preservation of the trusted half-step, and refusal of worsening candidates in eager and JIT execution.

The weak-rotation 300-requested-cell production row now terminates after two active-set trips with residual 2.70492925964e-15, converged true, and digest `69f8805e240e6472209b0a27ce73132bf35c5ab0e239e68c7558472242adcdff`. This differs from the historical pointwise-quadrature digest `d730c294…` and the continuous-support terminal `3e9fa335…`. The selector is the only production change in this node, so the new terminal is the measured effect of admitting an actually decreasing candidate despite its pessimistic local prediction.

This single-row outcome is not the locked cross-row convergence claim. Production still proposes Newton directions; this change only decides an already evaluated candidate. The second authoritative row, analytic-axis/null bounds, joint qualification, and the locked cross-row convergence gate remain outside this node.

One all_debug allocation, job 1275519, used the root interpreter directly with `JAX_PLATFORMS=cpu` and `TMPDIR=/tmp` in submit and payload. The focused suite passed 160 tests in 122.11 seconds, against 154 passed at base revision 192fa3b15ccafc80f3874c69bcdd7ca9734c0271. The declared scratch mutation, `Require model trust even when a finite candidate passes strict measured decrease.`, restored the veto and failed both eager and JIT acceptance cases. No budget, current pin, sufficient-decrease slope, residual guard, backtracking factor, Newton-step limit, Krylov limit, or active-set trip limit changed.

Receipts: [selector outcomes](/nova/figures/cut-cell-current-attribution/measured-candidate-acceptance/selector-receipt.json) and [terminal identity](/nova/figures/cut-cell-current-attribution/measured-candidate-acceptance/terminal-receipt.json). The numerical relationship is most directly represented by the table; no additional figure would clarify it.
