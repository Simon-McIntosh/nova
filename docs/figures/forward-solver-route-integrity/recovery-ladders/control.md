# Control: does the instrumentation perturb the solve?

The census wraps two solver rungs so a runtime callback fires on each
execution.  A callback is a side effect inside the compiled program, so the
counts are only meaningful if the wrapped solve terminates at the terminal
residual it would have reached unwrapped.  The pairing below establishes that.

| run | terminal fixed-point residual |
| --- | --- |
| census, instrumented | 0.27948523220464483 |
| `tests/test_reduced_newton.py::test_diverted_normalised_certificate_rung_retains_finite_fallback`, uninstrumented | 0.27948523220464483 |

Both runs are `diverted-single-null` at requested cells -300 on `all_debug` with
`JAX_PLATFORMS=cpu`, with the certificate part roots redirected so the row is
recomputed instead of read from a persisted part.  The two agree bit for bit,
so the callback does not perturb the solve.

## A stale expectation in the test, independent of this work

The test asserts `terminal_fixed_point_residual == 2.248351582217136`.  At the
base revision of this worktree the uninstrumented run returns
`0.27948523220464483` and the assertion fails, so the expected constant is stale
at this revision rather than an artifact of the census.  The solve terminates
with `active_set_settled`, `converged` false, and
`qualification_reason = fixed_point_residual_outside_qualification_bound`, over
6 trips and 10 Newton steps.  Repairing the expectation is outside this node's
write scope and expects an adjudication of which value the row should carry.

## The test itself is not the row counter

The census count is taken from the instrumented solve; the control is a way to
show the instrumented and uninstrumented solves agree, not a second count.
