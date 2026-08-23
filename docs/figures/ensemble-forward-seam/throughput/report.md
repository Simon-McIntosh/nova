# Ensemble coupled-window evaluator throughput

Tree: `87dd7358a5c36e04eccc5b27642060e0a84b61c1`. Results SHA-256: `399075ebd1c60ee791ef5baf82abf47c8879cc384dd67b4a0de36a077657bb8b`.

This budget measures the affine coupled-window contract workload with full typed receipt assembly. It isolates the public evaluator and identity/receipt cost; it is not a production equilibrium-physics rate.

Largest measured batch: **8 members/window**. The exact CPU and one-H200-node windows-per-second rows are in `results.tsv`. The execution receipt records the JAX backend and device, so the H200-node host-bound callback workload is not represented as device acceleration.

The H200 row passed exact tree and CUDA device/backend provenance gates.

Each row is the median of nine independently completed calls after one warmup. `member_windows_per_second` counts admitted member windows; `batch_calls_per_second` counts facade calls.
