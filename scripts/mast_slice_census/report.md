# MAST equilibrium-slice census

The reachable FAIR-MAST level-2 catalog contains **1,341,435 equilibrium slices**. Finishing them in 3,600 s requires **372.621 slices/s aggregate**, or **46.578 slices/s/device** across 8 devices.

## Coverage

The catalog index has 11,573 rows and 11,573 unique shot identifiers, against the asserted 11,573. The local mirror exposes 11,573 numeric shot stores; 11,573 join to the index and 11,378 contain equilibrium slices. The remaining 195 reachable stores contain no `equilibrium` group and therefore contribute zero slices.

There is no catalog or mirror coverage shortfall: every asserted shot was enumerated.

| Campaign | Index shots | Reachable stores | Equilibrium shots | Slices | Aggregate slices/s | Slices/s/device |
|---|---:|---:|---:|---:|---:|---:|
| M5 | 1,960 | 1,960 | 1,937 | 189,372 | 52.603 | 6.575 |
| M6 | 2,697 | 2,697 | 2,684 | 281,827 | 78.285 | 9.786 |
| M7 | 3,514 | 3,514 | 3,484 | 382,886 | 106.357 | 13.295 |
| M8 | 2,009 | 2,009 | 1,888 | 314,180 | 87.272 | 10.909 |
| M9 | 1,101 | 1,101 | 1,093 | 139,015 | 38.615 | 4.827 |
| Unknown | 292 | 292 | 292 | 34,155 | 9.488 | 1.186 |

## Method and provenance

The enumeration reads only `shot_id` and `campaign` from the FAIR-MAST metadata index at `https://mastapp.site/parquet/level2/shots`. It joins those identifiers to the local level-2 mirror and parses only each `equilibrium/time/zarr.json` file. The declared one-dimensional Zarr shape is counted as that shot's equilibrium slices; no plasma-current, validity, or topology filter is applied.

The JSON receipt records hashes of the normalized catalog identity, mirror shot identifiers, and per-shot slice counts so the exact enumeration can be compared with a later catalog snapshot.

## Execution boundary

No equilibrium solve was run (count: 0). No bulk signal data was downloaded for counting. The only network read was the compact shot-catalog metadata; all per-shot reads were local Zarr metadata (11,378 JSON files), not array chunks.
