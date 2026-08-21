NEEDS-HELP: the exact-paired CPU graph exhausted a 96 GiB allocation before compilation completed, the second failed measurement after the banked 15:13 compile timeout.

tried: The unchanged `measure.py` pipeline prepared the 551 by 551 coarse reference in job 1252936 (15:01.63 wall, 537,520 KiB process peak RSS, digest `64b0067358bcc1362f948b0a5ffb2d5185d792e150b670173f677f66da8109f1`). The independent NumPy lane completed in job 1252958 at 964.346460 s for 303,601 pairs (314.825649 pairs/s, 388,136 KiB process peak RSS). The CPU lane used 16 CPUs, 96 GiB and a 50-minute limit; job 1252956 was OOM-killed during XLA compilation at 32:24.87 with 100,437,500 KiB process peak RSS (96,763,360 KiB scheduler peak RSS), before writing a result. This follows the banked 15:13 / 53,009,476 KiB compile timeout. The H200 lane used eight CPUs, one H200, 150 GiB, the `gpu_0003_grpA` reservation and a 50-minute limit; job 1252957 was still compiling at 33:44 when the stop rule was reached, after an observed 46,731,832 KiB resident lower bound, versus the banked 9:19 / 23,502,520 KiB canceled floor. Because neither compiled block exists, the unchanged merge cannot produce the GPU-to-CPU ULP distribution, byte-identical fraction, H200 throughput, fine-build projection, or CPU-lane kernel cost.

options: (1) Authorize one fresh measurement node with at least 160 GiB for the CPU compile and a full 60-minute debug limit; retain 150-200 GiB and a full 60-minute limit for H200. (2) Extend scope to `nova/biot` and reduce the exact-paired graph's compiler memory footprint before measuring again. (3) Bank these compile and NumPy costs as the device-port outcome without a cross-device numerical receipt.

leaning: Option 1. The numerical graph is already committed and the CPU failure is a directly measured allocation limit, while the H200 lane showed no memory failure. A 160 GiB CPU request gives meaningful headroom over the 95.78 GiB process peak, and full 60-minute allocations preserve the existing physics and unchanged receipt pipeline.

cost-if-wrong: Another node spends about 15 minutes regenerating the ignored fixed input/reference, then up to 60 minutes per concurrent compile lane; an undersized CPU allocation repeats the OOM without parity evidence. Choosing compiler refactoring instead changes production code, requires a scope extension, and reopens the exact-section oracle and kernel-suite validation before the receipt can be trusted.

## Quantitative receipt status

- Original cross-device baseline: 1.434% byte-identical; p99.9 1.875e12 ULP.
- Filament-rule baseline retained for comparison only: 1.550% byte-identical; p99.9 6.468e6 ULP.
- Current exact-paired sliver context: source-self relative deviation 2.19e-5; far relative deviation 1.976e-3; maximum absolute deviation 1.49e-9.
- Current NumPy exact-paired stage: 964.346460 s for 303,601 pairs, 314.825649 pairs/s.
- Current CPU compile: greater than 32:24.87; OOM at 100,437,500 KiB process peak RSS under 96 GiB requested memory.
- Current H200 compile: greater than 33:44; observed resident lower bound 46,731,832 KiB under 150 GiB requested memory; canceled only because the two-failure stop rule fired on the CPU lane.
- Requested ULP distribution, byte-identical fraction, H200 pairs/s, projected nine-block fine-build time, and CPU kernel time: unavailable because compilation produced neither device matrix.
- Production source: unchanged.

Full logs are in `scripts/device_kernel_port/work/prepare.log`, `cpu.log`, `gpu.log`, `numpy.log`, and `scheduler-accounting.log`.
