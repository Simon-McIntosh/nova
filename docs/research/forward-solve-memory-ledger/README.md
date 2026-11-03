# Interrupted-run evidence

`interrupted-run-ledger.json` and `interrupted-run.log` are one CPU-rung run of
`benchmarks/forward_solve_memory_ledger.py` stopped at its wall limit (exit 124,
SIGTERM) on 2026-11-03 on node `98dci4-clu-2068` with `JAX_PLATFORMS=cpu`.

They are the control evidence for the ledger's per-phase persistence. The same
command, run for the same 150 s from revision `a51678ba2` — where the ledger was
written only at the last statement of `main()` — left its output directory
empty. This run leaves a ledger carrying `completed: false`, the nine phases it
reached, the flux-map `memory_analysis()` and StableHLO size, and
`census_inflight: after-warm-execute`, the array census it was inside when it
was stopped.

It is a partial record of an interrupted run, not the study's measurement.

The log opens with the `# command:` line that produced it, so the receipt can be
re-run without reading this file. Re-running writes `ledger-cpu.json` here; the
archived ledger's `output_path` field records the scratch directory this run
actually used.