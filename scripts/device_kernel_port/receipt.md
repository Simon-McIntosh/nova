# Packed flux-kernel device receipt

## Outcome

The fp64 packed closed-form G0 evaluator processed 303,601 coarse-fixture pairs on NVIDIA H200 NVL at 1,139,817.252 pairs/s (0.266359 s kernel wall). The measured rate projects the complete nine-block fine fixture to 11.475397 s (0.191257 min).

The performance projection passes the two-minute input, but the numerical parity input is a **HOLD**. This receipt does not authorise the full port.

## Numerical receipt

Against the same single-source JAX graph on CPU, 4,353/303,601 elements were byte-identical (1.433789744%). Absolute ULP percentiles were p50=2550, p90=13602, p99=622673, p99.9=1875420127380, max=36374763089312. The complete element-wise ULP histogram is stored in `receipt.json`.

Against the production NumPy-built G0 reference, the device block's maximum absolute difference was 6.8215253832279113e-10 and its maximum relative difference was 0.00090009000900095372.

## CPU lane

The same compiled fp64 graph took 24.329035 s on CPU, while the independently timed production NumPy uniform-G0 path took 58.701973 s (ratio 0.414450x). The banked current NumPy grid-family stage for all three flux orders is 76.676843 s.

## Method and boundary

Every launch used a fixed 32×32 pair tile, fp64, 128 residual nodes, and 14 fixed Bulirsch `cel` descent trips. The packed arithmetic is the existing `xp`-threaded production candidate; it evaluates the uniform ψ, B_R, and B_Z triple, of which only the flux G0 row was retained in this receipt. CPU and H200 ran in separate processes so each backend compiled the same source graph independently. The production NumPy path was not modified.
