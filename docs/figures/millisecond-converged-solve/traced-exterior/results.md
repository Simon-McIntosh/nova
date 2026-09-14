# Traced exterior compilation result

The exterior field is now an explicit traced input to the forward solve programs. One retained program serves changing exterior values on a fixed mesh. The certificate and compiled-slice lowering identities pass, and the H200 same-mesh timing records no tracing, backend compilation, or persistent-cache activity for the second exterior.

## Cause and repair

The exterior array was not present among the differing StableHLO constants after the traced-argument change. The remaining divergence came from constructing a fresh solve on every request: fresh `frozen_map` and `acceptance_map` callables acquired different Python identities, and JAX treated that identity as a varying static input. The result was a duplicated closed-call graph with different structure even when the mesh and policy were unchanged.

Commit `ae7a4fdf` retains one jitted accelerated-history callable per `ForwardProfile` and static solve configuration. The initial flux and exterior field remain explicit array arguments to that callable.

## Structural identity

CPU lowering job 1270462 completed successfully in 1 minute 6 seconds in production-default whole-cell `chord` mode.

| Route | First input | Second input | First digest | Second digest | Identical |
|---|---|---|---|---|---:|
| Certificate solve | Fixture exterior | Fixture exterior scaled by 0.9 | `4af3be210b2f3c1bf5fa579520d886f727bddeae247d853f22601069adc78ff3` | `4af3be210b2f3c1bf5fa579520d886f727bddeae247d853f22601069adc78ff3` | Yes |
| Compiled slice | First prescribed current | Second prescribed current | `194c2fa7a8526721892b0f781fc63f19d0890dd7c6ed70eebd4647685beb8341` | `194c2fa7a8526721892b0f781fc63f19d0890dd7c6ed70eebd4647685beb8341` | Yes |

The corrected certificate constant census contains zero differing constants.

## Terminal-state comparison

H200 job 1270467 ran the base revision and candidate weak 300-cell whole-cell solve and persisted both terminal arrays. The original comparison process exited 1 because it divided the residual difference by the small residual and then applied the relative flux tolerance to that ratio. Re-evaluating the persisted arrays applies the stated contract: relative tolerance `1e-14` to terminal flux and absolute tolerance `1e-14` to the scalar residual.

| Measure | Base | Candidate | Difference | Verdict |
|---|---:|---:|---:|---|
| Terminal residual | `0.005558930033419142` | `0.0055589300334190185` | `1.231653667943533e-16` absolute | Within `1e-14` absolute tolerance |
| Terminal flux | — | — | `3.552713678800501e-14` maximum absolute; `7.052709766269945e-16` maximum relative | Within `1e-14` relative tolerance |

The stored arrays contain no NaNs and both use `float64`. A raw `uint64` comparison is not bit-identical: 1,118 of 1,529 terminal-flux elements differ in bit pattern, and the residual scalar differs in one bit-pattern position. `np.array_equal(..., equal_nan=True)` also returns false. Thus the result is numerical equivalence under the accepted tolerance, not exact bit identity; rounding both residuals to twelve decimal places produces the same displayed `0.005558930033` but does not make their underlying bits equal.

Both base and candidate solves are unconverged at a terminal residual of approximately `0.0056`. This is the known state of the committed whole-cell row and is independent of the compile-identity change.

## Same-mesh H200 timing

The timed pair used one `ForwardProfile`, one weak 300-cell mesh, and exterior fields differing only by the 0.9 scale.

| Exterior | Wall time | Backend compile | Persistent-cache hits | Persistent-cache misses |
|---|---:|---:|---:|---:|
| Fixture | `118.6078579230234 s` | `98.20729207992554 s` | 3 | 0 |
| Scaled by 0.9 | `6.8922181241214275 s` | `0.0 s` | 0 | 0 |

The second solve therefore reused the already-loaded executable: it recorded zero backend compilation, zero persistent-cache retrievals, and zero persistent-cache misses.

The first timed solve spent `84.49194737244397 s` retrieving three persistent-cache hits. The principal cached solve executable is `jit_solve_program-f49f7fe4deda697d5b7bb8cc851df86f7ae5deb16671df598b9e59ca77ed37c0-cache`, stored under runtime directory `runtime-0745ecebea4cb264e054`. It occupies 419,095,023 logical bytes (`399.680159569 MiB`; approximately `400M` reported by `du`). That retrieval latency and executable size are material costs for a millisecond-scale serving target even though changing the exterior no longer causes another retrieval or compile in the same process.

## Test accounting

- Job 1270462: certificate same-mesh lowering identity passed; compiled-slice prescribed-current identity passed; zero differing certificate constants.
- Job 1270467: H200 allocation and both whole-cell solves completed; the same-mesh timing gate passed. The job-level exit was 1 solely from the original residual-relative comparison predicate, and the persisted-array reevaluation passes the stated numerical tolerance without executing another solve.
- The focused candidate suite recorded 15 passes and one failure in `test_solve_and_solve_branch_trace_prescribed_current_replacements`. The failing `if equilibrium.constraints` access and the test fixture that omits that attribute are unchanged at the base revision, so it is not attributable to these commits. The base suite was interrupted before reaching that test; it is absent execution evidence rather than a green baseline.

The cache-reuse conclusion does not depend on the numerical predicate: the second same-mesh solve independently recorded exactly `0.0 s` of backend compilation and no cache activity.
