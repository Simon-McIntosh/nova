NEEDS-HELP: the TORAX uniform-grid guard requires a CPU pure_callback inside the CUDA-only H200 window

tried: Submitted exactly one SLURM job with the landed gentle configuration to `betelgeuse` under reservation `gpu_0003_grpA`, requesting one H200 and exporting `TMPDIR=/tmp` and `JAX_PLATFORMS=cuda`. Job `1253074` reached the TORAX geometry adapter, then failed after 97 s because `_require_uniform_normalized_grid` could not place its `jax.pure_callback` on a local CPU device. I did not add the CPU platform, retry, or report the aborted attempt as a completed benchmark.

options: (1) make the uniform-normalised-grid guard trace-safe and device-native, or validate this static mesh invariant before the CUDA-traced adapter call; (2) deliberately permit a CPU callback and reclassify the measurement as hybrid GPU/host execution; (3) give the geometry adapter a validated mesh type whose construction proves the invariant once, outside the transport solve.

leaning: Option 1, with a static preflight where the grid is concrete and a device-native traced assertion where it is not. Adding `cpu` to `JAX_PLATFORMS` would conceal the exact host dependency this benchmark is required to expose.

cost-if-wrong: If the callback is semantically required during every traced adapter call, the product needs an explicit host-orchestration contract and the end-to-end GPU claim and timing decomposition must be rewritten; after any product repair, this single job must be rerun to obtain all unavailable H200 timings and receipts.

# Gentle coupled-window H200 attempt

The requested end-to-end measurement is blocked. The only H200 attempt stopped before the first TORAX interval completed, so no converged window receipt, per-iteration decomposition, cold-compile amortisation, or device placement for the equilibrium exchange solves exists to report. The 97 s allocation duration is a measured **time to refusal**, not a window wall time.

## Allocation and declared knobs

| item | value |
|---|---:|
| SLURM job | `1253074` |
| node | `98dci4-gpu-0003` |
| partition | `betelgeuse` |
| reservation | `gpu_0003_grpA` |
| requested accelerator | `gres/gpu:h200:1` |
| CPUs / memory | `7` / `64 GiB` |
| TMPDIR | `/tmp` |
| JAX platform selection | `cuda` only |
| job state / exit | `FAILED` / `1:0` |
| elapsed to refusal | `97 s` |
| window length | `0.0025 s` |
| auxiliary source multiplier | `0.5` |
| iteration cap | `10` |
| convergence tolerance | `0.005` |
| damping | `0.5` |

The SLURM allocation itself proves that one H200 was assigned. It does not prove that the equilibrium exchange solve arrays stayed on that H200: the driver was instrumented to record their JAX devices, but the initial transport leg refused before the first equilibrium update returned. That required evidence remains explicitly blocked rather than inferred from the allocation.

## Refusing operation

The observed call path was:

```text
solve_window
  -> transport_sweep
  -> ForwardTransport.solve
  -> _solve_torax
  -> _prepare_torax_config
  -> torax_geometry_from_fsa
  -> _require_uniform_normalized_grid
  -> jax.pure_callback
```

JAX reported:

```text
RuntimeError: jax.pure_callback failed to find a local CPU device to place the inputs on. Make sure "cpu" is listed in --jax_platforms or the JAX_PLATFORMS environment variable.
```

The failing product operation is the uniform normalised radial-grid validation in `nova.transport.torax_geometry._require_uniform_normalized_grid`. It forces CPU placement from inside the CUDA-only adapter path. The product source is outside this node's write fence, and enabling the CPU backend would be the prohibited host-round-trip workaround.

## Solver identity audit

The source-level owner identity is intact:

```text
nova.transport.coupled_window.equilibrium_sweep
  -> sampled_profile.cold_seed_portfolio(
       observed.moments.plasma_current, ...)
  -> sampled_profile.solve_portfolio(...)
  -> nova.equilibrium.forward.ForwardProfile.solve_portfolio
```

The relevant definitions and calls are at `nova/transport/coupled_window.py:948`, `nova/transport/coupled_window.py:1015`, `nova/transport/coupled_window.py:1039`, and `nova/equilibrium/forward.py:978`. This discharges the architectural solver-identity assertion: the window equilibrium leg owns a plasma-current-bearing cold portfolio and routes through `ForwardProfile.solve_portfolio`. It does not substitute for the blocked runtime device assertion.

## CPU baseline and unavailable H200 comparison

| measurement | landed CPU | H200 job | finding |
|---|---:|---:|---|
| end-to-end window wall time | `423.032716 s` | unavailable | job refused before one window completed |
| equilibrium plus FSA wall time | `422.454568 s` | unavailable | no completed iteration |
| TORAX wall time | `0.578148 s` | unavailable | first interval did not complete |
| measured contraction | `0.5371039633` | unavailable | no convergence receipt |
| maximum exit residual | `0.0049860186` | unavailable | no convergence receipt |
| flux-consumption ledger closure | `0` | unavailable | no terminal transport receipt |
| plasma-current ledger closure | `0` | unavailable | no terminal transport receipt |
| cold-compile cost | not separately recorded | unavailable | compilation cannot be separated from a failed trace |
| ten-iteration compile amortisation | not applicable | unavailable | zero completed iterations |

No H200 deviation from the landed CPU receipt can be calculated, and none is claimed. The full traceback is retained in the named SLURM log; the companion TSV records the same blocked cells as explicit `unavailable` values.
