# Packed flux-kernel device receipt

## Outcome

The fp64 packed closed-form G0 evaluator processed 303,601 coarse-fixture pairs on NVIDIA H200 NVL at 1,051,041.422 pairs/s (0.288857 s kernel wall). The measured rate projects the complete nine-block fine fixture to 12.444663 s (0.207411 min).

The performance projection passes the two-minute input, but the numerical parity input is a **HOLD**. This receipt does not authorise the full port.

## Numerical receipt

Against the same single-source JAX graph on CPU, 4,706/303,601 elements were byte-identical (1.550060771%). Absolute ULP percentiles were p50=2536, p90=13410, p99=455335, p99.9=6467816, max=81308413723. The complete element-wise ULP histogram is stored in `receipt.json`. The banked baseline was 1.434% byte-identical with p99.9=1,875,000,000,000 ULP.

Against the production NumPy-built G0 reference, the device block's maximum absolute difference was 1.7825410786849265e-10 and its maximum relative difference was 9.9369994237391388e-06.

For source column 184, the conditioned production reference differs from the 1024-rung arbitrary-precision oracle by at most 8.0380528684983649e-05 relative overall and 1.3237819231401374e-10 away from the retained finite-section self evaluation. The prior production reference differed by 0.0046287193027980409 relative.

## CPU lane

The same compiled fp64 graph took 24.873608 s on CPU, while the independently timed production NumPy uniform-G0 path took 55.430824 s (ratio 0.448732x). The banked current NumPy grid-family stage for all three flux orders is 76.676843 s.

## Method and boundary

Every launch used a fixed 32×32 pair tile, fp64, 128 residual nodes, and 14 fixed Bulirsch `cel` descent trips. The packed arithmetic and the production NumPy reference apply the same cancellation-conditioned uniform-section rule; both evaluate the ψ, B_R, and B_Z triple, of which only the flux G0 row was retained in this receipt. CPU and H200 ran in separate processes so each backend compiled the same source graph independently.
