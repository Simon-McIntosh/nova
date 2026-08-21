NEEDS-HELP: the one authorized sized attempt remained in XLA compilation on both CPU and H200 when the worker's 60-minute wall required cancellation, so no cross-device matrices exist to merge.

tried: The unchanged `measure.py` pipeline prepared the 551 by 551 reference in job 1252995 (15:51.97 wall, 539,384 KiB process peak RSS, digest `64b0067358bcc1362f948b0a5ffb2d5185d792e150b670173f677f66da8109f1`). The NumPy lane completed in job 1252998 at 895.378692 s for 303,601 pairs (339.075525 pairs/s, 385,336 KiB process peak RSS). The CPU lane used 16 CPUs, 180 GiB and the full 60-minute allocation; job 1252996 remained in XLA compilation through 38:55 and reached 149,811,136 KiB scheduler RSS before worker-fence cancellation. The H200 lane used eight CPUs, one H200, 200 GiB, the `gpu_0003_grpA` reservation and the full 60-minute allocation; job 1252997 remained in XLA compilation through 38:55 and had reached an observed 58,935,472 KiB resident lower bound before the same cancellation. Neither lane was OOM-killed or scheduler-timed-out. This was the single authorized sized attempt; no retry was made.

options: (1) Treat the sized receipt as the definitive compile-cost wall and decline the exact-paired device path in its current graph form. (2) Extend scope to `nova/biot` and reduce the exact-paired graph's compiler size before any further device measurement. (3) Authorize a separately budgeted measurement whose worker wall can cover preparation plus the full one-hour compile allocation, while keeping the same 180/200 GiB requests.

leaning: Option 2. The 180 GiB CPU request removed the 96 GiB OOM but compiler RSS grew to 142.87 GiB and neither backend produced an executable after 38:55. The limiting mechanism is now graph compilation cost rather than kernel execution or a marginal memory request.

cost-if-wrong: Declining without graph reduction forfeits an unknown chance that compilation would finish between 39 and 60 minutes. Refactoring the graph changes production code, requires a scope extension, and reopens the exact-section oracle and kernel-suite validation. A longer measurement-only run costs another 15-16 minute preparation plus up to one hour on each concurrent compile lane and can still return no matrices.

## Definitive sized compile-cost wall receipt

- Original cross-device baseline: 1.434% byte-identical; p99.9 1.875e12 ULP.
- Filament-rule baseline retained for comparison only: 1.550% byte-identical; p99.9 6.468e6 ULP.
- Exact-paired sliver context: source-self relative deviation 2.19e-5; far relative deviation 1.976e-3; maximum absolute deviation 1.49e-9.
- Preparation: job 1252995, 16 CPUs, 64 GiB, 60-minute allocation; completed 15:53 scheduler wall / 15:51.97 process wall; 568,848 KiB scheduler peak RSS.
- NumPy reference: job 1252998, 16 CPUs, 64 GiB, 60-minute allocation; completed 15:04 scheduler wall; measured stage 895.378692 s, 339.075525 pairs/s; 416,064 KiB scheduler peak RSS.
- CPU compile: job 1252996, 16 CPUs, 180 GiB, 60-minute allocation; canceled at the worker time fence after 38:55, still compiling; 149,811,136 KiB scheduler peak RSS (142.87 GiB). This exceeded the banked 32:25 / 96 GiB OOM floor without memory failure.
- H200 compile: job 1252997, eight CPUs, one H200, 200 GiB, 60-minute allocation, `gpu_0003_grpA`; canceled at the worker time fence after 38:55, still compiling; observed resident lower bound 58,935,472 KiB (56.21 GiB). This exceeded the banked 33:44 / 150 GiB canceled floor without memory failure.
- Kill reason for both compile lanes: explicit worker cancellation at the 60-minute node fence; not OOM and not scheduler timeout.
- Requested ULP distribution, byte-identical fraction, H200 pairs/s, projected nine-block fine-build time, and CPU kernel time: unavailable because compilation produced neither device matrix.
- Production source: unchanged.

Full logs are in `scripts/device_kernel_port/work/prepare.log`, `cpu.log`, `gpu.log`, `numpy.log`, `pre-cancel-resource.log`, `scheduler-accounting.log`, and `monitor.log`.
