# Benchmark baselines

Machine: 98dci4-srv-1006 (Xeon Gold 6442Y, login node under shared load) ·
python 3.14 · asv 0.6.6 (`asv run --quick -E existing`) · recorded 2026-07-24.
Timings are single-round `--quick` figures intended as order-of-magnitude
references; re-run on a quiet node before reading small deltas as regressions.

## biotoperate (asv)

| benchmark | time |
|---|---|
| PlasmaEvaluate.time_flux_function | 109 ms |
| PlasmaEvaluate.time_flux_function_ev_only | 109 ms |
| PlasmaEvaluate.time_radial_field | 117 ms |
| PlasmaEvaluate.time_field_magnitude (`bp`) | 107 ms |
| PlasmaOperate.time_load_operators | 4.4 ms |
| PlasmaOperate.time_solve | 963 ms |
| PlasmaTurns.time_update_turns (svd_rank 10/75/200/500/-1) | 229 / 229 / 209 / 290 / 205 ms |

## framespace (asv)

| benchmark | time |
|---|---|
| GetCurrent.time_getattr | 40.4 µs |
| GetCurrent.time_getattr_subspace | 8.8 µs |
| GetCurrent.time_getitem | 35.6 µs |
| GetCurrent.time_loc | 43.1 µs |
| SetCurrent.time_metaframe_data | 2.3 µs |
| SetCurrent.time_subspace | 7.5 µs |

## framesetloc (asv, indexer sweep, µs)

Get item: loc 31.1 / sloc 6.1 / aloc 25.5 / saloc 3.1 (Ic); nturn loc 7.3 / aloc 3.8; x loc 8.0.
Get subitem: loc 44.8 / sloc 11.2 / aloc 29.3 / saloc 6.0 (Ic); nturn loc 31.0 / aloc 7.3; x loc 37.3.
Set item: sloc 28.6 / saloc 18.0 (Ic); nturn loc 24.4 / aloc 20.3; x loc 26.5.
Set subitem: sloc 13.6 / saloc 5.5 (Ic); nturn loc 50.6 / aloc 18.9; x loc 49.8.

## Standalone scripts

| script | result |
|---|---|
| coil_construction.py — construct n=100/500/1000/5000 | 2.12 / 3.29 / 4.73 / 16.65 ms |
| coil_construction.py — read `loc[:, "Ic"]` / `.Ic` (n=5000) | 0.009 / 0.008 ms |
| io_ingest.py — AL full-get vs warm zarr re-read | 981.6 ms vs 6.9 ms (**142×**) |
