# Which input to the current-normalisation guard is nan

## Result

The guard's second input is the nan one.

```
"unscaled_current": {"shape": [], "finite": false, "is_nan": true, "is_inf": false, "value": NaN},
"target_current":   {"shape": [], "finite": true,  "is_nan": false, "is_inf": false,
                     "value": 16314773.311828371}
```

`ForwardFluxOperator.current_normalisation_amplitude` divides the declared current
by `jnp.sum(moments.cell_current)`. The declared current is finite, the recorded
row-300 target being 1.6314773311828371e7 A. The summed cell current is nan, so
the quotient is nan and the guard raises against its declared band.

Guard location: `nova/equilibrium/forward_operator.py:3590`.

## The first nan array, and what produced it

The first non-finite array the pipeline carried is the whole cell-moment
structure returned by

```
clipped_support_current_moments   nova/equilibrium/clip_quadrature.py:713
```

All of it is non-finite: shape `[3, 342]`, size 1026, nan 1026, inf 0. The `342`
is the realised cell count of the row-300 state, so the nan covers every realised
cell and every moment component — not one bad cell among good ones.

The producer is not performing a division that went wrong. Its tail emits nan
deliberately:

```python
overflow = cut_count > capacity
return ClippedCurrentMoments(
    *(jnp.where(overflow, jnp.nan, value) for value in (...))
)
```

So the nan is a **capacity-overflow sentinel**: `cut_count` exceeded
`cut_cell_capacity` for this state and the kernel returned nan for the entire
structure rather than a partial measure. The two values to inspect next are that
pair.

This also explains why the guard, not the kernel, is where the symptom surfaces:
an overflowing measure is silently converted to nan at the kernel boundary and
travels up as a plausible-looking float until the quotient it feeds is compared
against the band.

## The state it appears in

| field | value |
|---|---|
| row | 300, `weak-rotation-reactor-static`, `TopologyClass.LIMITED` |
| arm | B, production iteration from current-aligned cold seed |
| realised cells (`grid_count`) | 342 |
| reference trip count | 2, converged |
| reference part | `docs/figures/cut-cell-current-attribution/exact-clip-seed/parts/weak-rotation-reactor-static-production-route-cells-300.json`, sha256 `c1a0c713a8b8` |
| terminal flux | shape (1529,), all finite |

The reference part is itself finite: the nan is produced by the pipeline run on
that state, not inherited from it.

## Reproduction

Previously recorded defect, reproduced at HEAD `6fc9ff5b1`:

```
LOW_STATE_AMPLITUDE_PROBE_RAISED
"error_message": "source amplitude nan is outside (1e-06, 1000000.0)"
"outcome": "raised"
PROBE_EXIT=0
```

That message is the one the predecessor run records for this row, so the
reproduction is the recorded defect and not a neighbouring failure.

Run as one H200 job (98dci4-gpu-0003) in the lane shape the predecessor manifest
records: `betelgeuse`, reservation `gpu_0003_grpA`, account `grpa`, one card,
8 cores, 64G, prewarm cache root
`/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm`, interpreter
`/home/ITER/mcintos/Code/nova/.venv/bin/python` with `PYTHONPATH` pointing at
this worktree and no `uv` on the compute node.

## How the instrument works, and why it is outside every module it observes

No file under `nova/` changes. The driver patches three names at run time and
then calls the benchmark's own context build and reference-seed arm:

- `ForwardFluxOperator.current_normalisation_amplitude`, so both guard inputs are
  recorded before the raise;
- `nova.equilibrium.forward_operator.flux_field_polynomial`;
- `nova.equilibrium.forward_operator.clipped_support_current_moments` (the import site, line 47).

The producer's reported location is the module that defines it, not the
`jax/_src/source_info_util.py:712` path the first instrument returned: `inspect`
resolves a jax-wrapped callable to jax internals, so a location must be read from
the import site or the defining module instead.

## Reading the record

`nan-inputs.json` carries the structured result: the guard calls with both
inputs, the environment, the driver and reference-part digests, and the state
block. `probe-gpu.log` is the job's own log.

## Outstanding

- The negative control has not been run: it is the same driver with the guard
  left unpatched, and it belongs beside this positive result.
- The evidence fence names an `all_debug` CPU job. That lane cannot run this
  benchmark: `benchmarks/exact_clip_low_state_discriminator.py:138-150` accepts
  no lane but an H200 on `betelgeuse` under `gpu_0003_grpA`. Job 1276152 died
  before building a single array with
  `the discriminator requires one H200, got cpu:0`. The reproduction therefore
  ran on the lane the fence's own lane-shape clause names.