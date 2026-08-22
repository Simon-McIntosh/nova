NEEDS-HELP: the H200 gentle window exhausts the declared iteration cap at residual 0.0277048 instead of converging at 0.005

tried: Ran the same landed gentle configuration in exactly two H200 jobs. Job `1253074` correctly refused the original CUDA-only launch at the host grid-validation callback. Under the amended `JAX_PLATFORMS=cuda,cpu` contract, job `1253082` cleared that seam, executed all ten declared exchanges with the equilibrium solve leaves and returned TORAX state channels guarded as CUDA-only, then returned `WindowConvergenceError`: residual `0.0277048` after cap `10`. The landed CPU run converged at maximum residual `0.0049860186`. No tolerance, damping, source, window, or iteration knob was changed.

options: (1) add a product diagnostic that banks the non-converged GPU residual trace, branch receipts, per-field exchanged values, callback timings, and CPU/GPU precision provenance before raising, then compare it with the landed CPU trajectory; (2) isolate the first CPU/GPU exchange whose residual or branch selection diverges in a focused equivalence test; (3) integrate the post-cut boundary-band implementation and rerun both hardware lanes as a separately versioned comparison rather than mixing it with this pre-band baseline.

leaning: Option 1 followed by option 2, preserving cap `10`, tolerance `0.005`, and damping `0.5`. The observed residual is 5.54 times the convergence limit, so weakening the gate would turn a hardware-dependent trajectory change into a false pass.

cost-if-wrong: If the discrepancy is caused only by benchmark synchronization or unbanked precision configuration, the product diagnostic can be removed after attribution and this exact job rerun; if the equilibrium branch or TORAX state genuinely differs on CUDA, the affected solver seam needs repair and both timing and receipt evidence must be regenerated.

# Gentle coupled-window H200 outcome

The amended execution contract successfully passed the host callback boundary, but the physics convergence contract did not pass. Job `1253082` ran for 249 s allocation wall time on one H200 and exhausted all ten exchanges at residual `0.0277048`; it did not return a `WindowReceipt`. The result is therefore a measured non-convergence, not an end-to-end timing or receipt success.

## Allocation and declared knobs

| item | value |
|---|---:|
| amended SLURM job | `1253082` |
| node | `98dci4-gpu-0003` |
| partition | `betelgeuse` |
| reservation | `gpu_0003_grpA` |
| requested accelerator | `gres/gpu:h200:1` |
| CPUs / memory | `7` / `64 GiB` |
| TMPDIR | `/tmp` |
| JAX platforms | `cuda,cpu` |
| job state / exit | `FAILED` / `1:0` |
| allocation wall time | `249 s` |
| window length | `0.0025 s` |
| auxiliary source multiplier | `0.5` |
| iteration cap | `10` |
| convergence tolerance | `0.005` |
| damping | `0.5` |

The first job, `1253074`, ran for 97 s and established the callback refusal that motivated the amended platform contract. It remains provenance, not a timing sample.

## Device and solver assertions

The amended driver requires JAX backend `gpu` and a separately registered CPU callback device. During each exchange it walks every JAX leaf in the returned `ForwardEquilibrium` and each of the five returned TORAX state channels—rho, psi, ion temperature, electron temperature, and electron density—and raises immediately if any device platform is not GPU. Job `1253082` reached the convergence gate only after ten transport and ten equilibrium updates without a device-placement exception: 20 coarse-sample equilibrium receipts and 10 returned TORAX states traversed those CUDA assertions.

The source-level owner path is also intact:

```text
nova.transport.coupled_window.equilibrium_sweep
  -> sampled_profile.cold_seed_portfolio(
       observed.moments.plasma_current, ...)
  -> sampled_profile.solve_portfolio(...)
  -> nova.equilibrium.forward.ForwardProfile.solve_portfolio
```

The relevant definitions and calls are at `nova/transport/coupled_window.py:948`, `nova/transport/coupled_window.py:1015`, `nova/transport/coupled_window.py:1039`, and `nova/equilibrium/forward.py:978`.

## Guard callback measurement

The driver synchronously wraps `_validated_grid_callback`, blocks its returned array, and records dispatch, CUDA-to-host grid transfer, CPU validation, and returned-array handoff once per adapter construction. Those ten in-memory measurements were not serialized because the driver correctly stopped before its success-only artifact writer after receiving `WindowConvergenceError`. Their numerical cost is therefore **unavailable**, not reconstructed from the 249 s allocation time. A diagnostic rerun would be required to bank them, but the worker contract requires stopping after the same batch command has failed twice following different fixes.

This unavailable number means no evidence-based recommendation can yet be made about a provenance-gated guard bypass. The guard stays as-is until its measured cost is durable.

## Pre-band CPU baseline and post-cut sparsification

| measurement | landed CPU, pre-band | H200 amended job | finding |
|---|---:|---:|---|
| complete window wall time | `423.032716 s` | unavailable | no converged `WindowReceipt` |
| equilibrium plus FSA | `422.454568 s` | unavailable | per-iteration values were not serialized on error |
| TORAX | `0.578148 s` | unavailable | per-iteration values were not serialized on error |
| iterations used | `10` | `10` | both reached the declared cap |
| maximum exit residual | `0.0049860186` | `0.0277048` | H200 is `0.0227188` higher and 5.54 times the tolerance |
| measured contraction | `0.5371039633` | unavailable | non-converged receipt was not serialized |
| flux-consumption ledger closure | `0` | unavailable | terminal ledger was not serialized |
| plasma-current ledger closure | `0` | unavailable | terminal ledger was not serialized |
| cold-compile cost | not separately banked | unavailable | first-versus-warm timing rows were not serialized |

The `423.032716 s` figure is the landed pre-band CPU window and remains the only directly comparable CPU measurement. Boundary-band sparsification landed on main after this worktree was cut in commit `32942ac3`; warm ITER CPU assembly is now `24.6 s`. The complete CPU window has not been remeasured after that change, so this report does not infer a post-band CPU window time or speedup. The H200 job also executed the pre-band worktree, keeping the failed receipt comparison on the same code lineage.

## Quantitative verdict

The declared success threshold was `0.005`; the observed H200 exit residual was `0.0277048`, an excess of `0.0227048` over the limit and a ratio of `5.54096`. The H200 result therefore fails convergence. End-to-end wall time, per-iteration sweep times, guard callback cost, cold-compile amortisation, contraction, exchanged-field residuals, and the two ledger closures remain unavailable because the driver never promoted the non-converged trajectory to a success artifact.
