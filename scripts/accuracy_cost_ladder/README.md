# Accuracy and cost measurement

`measure.py` profiles the closed-form oracle fixtures without changing the
forward model or an accuracy bound. It has three entry points:

- `profile-build` directly reconstructs one carrier and times authored section
  geometry, fixed moment geometry, exact polygon-moment kernels, and residual
  carrier assembly. The timed path bypasses the semantic cache, then requires
  bitwise identity with the cached production carrier.
- `profile-gpu` constructs the independent production moment seed and runs the
  fixed-shape ten-Newton, thirty-GMRES solver on a CUDA device. It records warm
  batch timings, independent inclusive phase probes, a device trace, and the
  compiled StableHLO from which loop trip counts are extracted.
- `finalize` merges the two CPU and two GPU receipts into `results.json` and
  renders `accuracy-cost-ladder.png`.

The repository environment is externally managed. Every invocation uses
`uv run --no-sync`; CPU measurements additionally set `JAX_PLATFORMS=cpu` so a
login or compute node with CUDA cannot silently change the measured backend.

The accuracy table retains the proposed recovery bounds as read-only inputs.
It reports signed margin and pass/fail against those proposals, but neither
locks nor modifies them. The independently seeded alternate root therefore
remains visible as a basin-selection failure even when its terminal fixed-point
residual is at machine precision.

## Banked measurement

The cold CPU carrier builds took 294.217 seconds for 551 cells and 778.044
seconds for 1,076 cells. Exact polygon-moment kernels carried 97.22% and 98.72%
of those totals. Both direct builds matched all 18 persisted production arrays
bitwise.

On the reserved NVIDIA H200 NVL, the genuine coarse ten-by-thirty solve took
198.545 milliseconds at batch one, 56.311 milliseconds per state at batch four,
and 30.244 milliseconds per state at batch sixteen. The fine batch-one solve
took 257.393 milliseconds. The one-millisecond target is therefore missed by
198.5 times at the genuine coarse batch-one operating point; batching narrows
but does not close the gap. Elided StableHLO receipts independently expose loop
bounds 10 and 30, while compact Perfetto traces retain the map, tangent, and
Krylov-matvec regions without duplicating executable constants.

The cheaper zeroth-moment arm is not an operating point: although its batch-one
solve is 12.206 milliseconds, it reaches a non-finite terminal residual and a
3.361-span flux error. The production first-moment arms retain round-off
exact-state forcing, but their independently seeded roots reproduce the known
limited alternate basin and remain on scientific hold. `results.json` therefore
makes a measured recommendation for owner review; it does not lock a bound or
an operating point.
