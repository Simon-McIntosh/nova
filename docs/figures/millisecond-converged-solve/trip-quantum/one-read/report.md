# H200 active-set trip quantum

Job `1262715` completed every device-synchronised timer for 22086/43 pure. Its production median was **40.803044 s**, or **5.829006 s per active-set trip**.
The solve executes **7 active-set trips**, **588 residual evaluations** (84.000/trip: 84 primal plus 504 line-search grades), and **1008 Jacobian-vector products** (144.000/trip). The source receipt's count of 24 is the attempted-Newton-promotion count, not the active-set trip count.
Independent job `1262501` spans 41.353373 to 41.537134 s over three warm solves, or 5.919389 s/trip at its median.

## Additive timer attribution

| rank | component | wall / trip [s] | share | direct wall / evaluation [ms] |
|---:|---|---:|---:|---:|
| 1 | host dispatch or device sync | 5.632057 | 96.62% | 5632.057402 |
| 2 | jacobian vector product | 0.078317 | 1.34% | 0.543868 |
| 3 | forward evaluation | 0.049651 | 0.85% | 0.591083 |
| 4 | topology read | 0.038774 | 0.67% | 38.773551 |
| 5 | gmres orthogonalisation | 0.018288 | 0.31% | 1.524041 |
| 6 | line search | 0.011919 | 0.20% | 0.993240 |

## Topology attribution

| rank | sub-part | direct probe [ms] | attributed / trip [s] | share |
|---:|---|---:|---:|---:|
| 1 | wall reachability | 25.611235 | 0.021462 | 55.35% |
| 2 | flood fills | 13.823818 | 0.011584 | 29.88% |
| 3 | spline fits | 3.364272 | 0.002819 | 7.27% |
| 4 | separatrix | 3.110746 | 0.002607 | 6.72% |
| 5 | candidate census | 0.359015 | 0.000301 | 0.78% |

the directly timed candidate census, spline fit, flood fill, wall reachability, and separatrix-exclusive medians are normalized to partition the separately timed topology-read wall; these overlapping isolated probes are relative attribution weights, not additive wall.

## Launch and scan floor

GPU kernel-launch and transfer counts are **not available**: both CUPTI replays failed before a complete interval was written. The partial trace is unpromoted. The branch-minimum static census has **11958.1 fixed scan iterations/trip**; at 7.6 µs each, their arithmetic floor is 0.090882 s and measured trip wall is **64.14×** that floor. Dynamic while-loop iterations are not guessed into it.
One forward evaluation contains at least **14 fixed scan iterations** (14 when all compiled branches are summed). The frozen-partition census assigns 1 topology read/trip and 63 split-spline fits/read.

| evaluation | GPU launches | transfers | fixed scan lower bound | compiled-branch scan sum |
|---|---:|---:|---:|---:|
| forward evaluation | not captured | not captured | 14 | 14 |
| jacobian vector product | not captured | not captured | 14 | 14 |
| gmres orthogonalisation | not captured | not captured | 0 | 0 |
| line search | not captured | not captured | 124 | 214 |
| topology read | not captured | not captured | 11690 | 11690 |
| candidate census | not captured | not captured | 0 | 0 |
| spline fits | not captured | not captured | 843 | 843 |
| flood fills | not captured | not captured | 2499 | 2499 |
| wall reachability | not captured | not captured | 3110 | 3110 |
| separatrix | not captured | not captured | 4031 | 4031 |

## Jacobi SVD replay finding

The `cusolverDnDgesvdj` failure exposes production work, not profiler setup alone. The dominant authored call site is `nova.linalg.split_spline._conditioned_fit`: `jnp.linalg.cond(normal)` runs once for each of the level and field normal equations. At 63 fits/topology read and one read/trip, that is **126 Jacobi SVDs/active-set trip**. `nova.equilibrium.fixed_point._projected_krylov_condition` owns one additional singular-value-only SVD per Newton residual linearisation.
The timer-only route did not request CUPTI replay, so kernel-launch counts remain unavailable by construction.

## Ranked bottlenecks and implied repairs

1. **host, synchronization, or unmodelled fused-device remainder** — 5.632057 s/trip. Repair: move reconciliation under one compiled boundary and add a profiler route that isolates solver-library replay from component timing. Owner: `millisecond-converged-solve`.
2. **Jacobian-vector products** — 0.078317 s/trip. Repair: reuse or compress the linearized operator and reduce the Krylov action budget before changing nonlinear acceptance. Owner: `solver-convergence-regression`.
3. **topology read: wall reachability** — 0.038774 s/trip. Repair: land the shared TensorBSpline authority so each topology read fits once rather than 63 times, then reuse one topology partition across residual grades. Owner: `null-identification-authority`.

## Host callbacks and synchronization

The closed production jaxpr contains **0 host callback primitives**. Every timed invocation ends in one `jax.block_until_ready` device synchronization. Without a complete device trace, the timer remainder is 5.632057 s/trip (96.62%). It is an upper bound on host plus synchronization, not a pure-host measurement, because it retains fused device work absent from the isolated model.

## Measurement boundaries

The additive table scales direct synchronized component medians by exact evaluation counts. Isolated probes do not prove that the fused program adds identically. The receipt preserves both raw timer distributions, terminal-log hashes, the unpromoted trace identity, and the static control-flow census. No `nova/` source was changed.
