# Empty exact polygon per cut cell: the clip stage that discards it

Eight cells carry an analytic separatrix cutting a real polygon out of the atomic mesh, yet the exact-mode support reports zero area for them at both fixed states. This report names, per cell and per state, the vertex signs, the level-set crossings on the cell edges, the vertex count emitted by each packing stage of the traced clip, and the first stage that cannot form a polygon.

- revision: `3ecf5bc810200696f2c1e890e016652505472e02`
- case: `weak-rotation-reactor-static`
- requested cells: `-110`
- realised cells: `135`

## Stage chain of the traced clip

| stage | line | what it packs |
| --- | --- | --- |
| `crossing_vertices` | 1135 | crossing points packed from the straddling edges |
| `candidate_vertices` | 1207 | inside vertices, crossings and saddle candidate |
| `deduped_support_vertices` | 1224 | candidates after the duplicate collapse |
| `post_arc_vertices` | 1327 | support vertices after the spline-gap arc expansion |
| `branch_vertices` | 1366 | support vertices split into the two saddle branches |
| `saddle_chain_vertices` | 1904 | saddle-chain support after the chain expansion |

A stage emitting fewer than three vertices cannot form a polygon and its shoelace area is zero, which is what makes `included = area > 0.0` false.

## seed state

boundary level `35.8528226` Wb; profile participation on 135 of 135 cells; exact total area `1.046502e+01` m^2.

| cell | verts | inside boundary | inside curved | straddling edges | participating crossings | blocked by participation | candidate | crossing | deduped | post-arc | branch | area m^2 | included | first empty stage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | 0.000000e+00 | False | traced_polygon_moments (line 1405) |
| 35 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | 0.000000e+00 | False | traced_polygon_moments (line 1405) |
| 75 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | 0.000000e+00 | False | traced_polygon_moments (line 1405) |
| 89 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | 0.000000e+00 | False | traced_polygon_moments (line 1405) |
| 114 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | 0.000000e+00 | False | traced_polygon_moments (line 1405) |
| 126 | 15 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | 0.000000e+00 | False | traced_polygon_moments (line 1405) |
| 128 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | 0.000000e+00 | False | traced_polygon_moments (line 1405) |
| 134 | 13 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | 0.000000e+00 | False | traced_polygon_moments (line 1405) |

### Per-edge crossings

- cell 2: participating crossings on edges []; straddling edges blocked by participation: []; level per vertex [boundary] [-54.59681, -54.55941, -54.55941, -54.55403, -54.55403, -54.6005, -54.6005, -54.6931, -54.6931, -54.81854, -54.81854, -54.86304, -51.18854, -50.43181, -54.40514] / [curved] [-0.71776, -0.72314, -0.72314, -0.73088, -0.73088, -0.7394, -0.7394, -0.74745, -0.74745, -0.75416, -0.75416, -0.75588, -0.65669, -0.64435, -0.71226]
- cell 35: participating crossings on edges []; straddling edges blocked by participation: []; level per vertex [boundary] [-37.13253, -37.16465, -37.16465, -37.53729, -37.53729, -40.29049, -40.29049, -41.10607, -37.22753, -26.3121, -26.46278] / [curved] [-0.48136, -0.48319, -0.48319, -0.49704, -0.49704, -0.5134, -0.5134, -0.53135, -0.47943, -0.33795, -0.34544]
- cell 75: participating crossings on edges []; straddling edges blocked by participation: []; level per vertex [boundary] [-54.86304, -54.95596, -54.95596, -55.07693, -55.07693, -54.67729, -54.67729, -54.74852, -54.74852, -54.7528, -50.6001, -47.44089, -51.18854] / [curved] [-0.75588, -0.75926, -0.75926, -0.7631, -0.7631, -0.76629, -0.76629, -0.76918, -0.76918, -0.77094, -0.63951, -0.60035, -0.65669]
- cell 89: participating crossings on edges []; straddling edges blocked by participation: []; level per vertex [boundary] [-26.63337, -24.56862, -24.56862, -20.02132, -19.75864] / [curved] [-0.33181, -0.30769, -0.30769, -0.25539, -0.24724]
- cell 114: participating crossings on edges []; straddling edges blocked by participation: []; level per vertex [boundary] [-20.02132, -24.56862, -24.56862, -26.63337, -19.75864] / [curved] [-0.25539, -0.30769, -0.30769, -0.33181, -0.24724]
- cell 126: participating crossings on edges []; straddling edges blocked by participation: []; level per vertex [boundary] [-54.86304, -54.81854, -54.81854, -54.6931, -54.6931, -54.6005, -54.6005, -54.55403, -54.55403, -54.55941, -54.55941, -54.59681, -54.40514, -50.43181, -51.18854] / [curved] [-0.75588, -0.75416, -0.75416, -0.74745, -0.74745, -0.7394, -0.7394, -0.73088, -0.73088, -0.72314, -0.72314, -0.71776, -0.71226, -0.64435, -0.65669]
- cell 128: participating crossings on edges []; straddling edges blocked by participation: []; level per vertex [boundary] [-41.10607, -40.29049, -40.29049, -37.53729, -37.53729, -37.16465, -37.16465, -37.13253, -26.46278, -26.1037, -37.22753] / [curved] [-0.53135, -0.5134, -0.5134, -0.49704, -0.49704, -0.48319, -0.48319, -0.48136, -0.34544, -0.33795, -0.47943]
- cell 134: participating crossings on edges []; straddling edges blocked by participation: []; level per vertex [boundary] [-54.7528, -54.74852, -54.74852, -54.67729, -54.67729, -55.07693, -55.07693, -54.95596, -54.95596, -54.86304, -51.18854, -47.55379, -50.6001] / [curved] [-0.77094, -0.76918, -0.76918, -0.76629, -0.76629, -0.7631, -0.7631, -0.75926, -0.75926, -0.75588, -0.65669, -0.60035, -0.63951]

### First empty stage

- cell 2: `traced_polygon_moments` at `nova/equilibrium/separatrix_clip.py:1405` with 0 vertices — the branch shoelace area is zero, so included = area > 0.0 is false at line 1408
- cell 35: `traced_polygon_moments` at `nova/equilibrium/separatrix_clip.py:1405` with 0 vertices — the branch shoelace area is zero, so included = area > 0.0 is false at line 1408
- cell 75: `traced_polygon_moments` at `nova/equilibrium/separatrix_clip.py:1405` with 0 vertices — the branch shoelace area is zero, so included = area > 0.0 is false at line 1408
- cell 89: `traced_polygon_moments` at `nova/equilibrium/separatrix_clip.py:1405` with 0 vertices — the branch shoelace area is zero, so included = area > 0.0 is false at line 1408
- cell 114: `traced_polygon_moments` at `nova/equilibrium/separatrix_clip.py:1405` with 0 vertices — the branch shoelace area is zero, so included = area > 0.0 is false at line 1408
- cell 126: `traced_polygon_moments` at `nova/equilibrium/separatrix_clip.py:1405` with 0 vertices — the branch shoelace area is zero, so included = area > 0.0 is false at line 1408
- cell 128: `traced_polygon_moments` at `nova/equilibrium/separatrix_clip.py:1405` with 0 vertices — the branch shoelace area is zero, so included = area > 0.0 is false at line 1408
- cell 134: `traced_polygon_moments` at `nova/equilibrium/separatrix_clip.py:1405` with 0 vertices — the branch shoelace area is zero, so included = area > 0.0 is false at line 1408

## terminal state

boundary level `10.3951318` Wb; profile participation on 135 of 135 cells; exact total area `2.047412e+01` m^2.

| cell | verts | inside boundary | inside curved | straddling edges | participating crossings | blocked by participation | candidate | crossing | deduped | post-arc | branch | area m^2 | included | first empty stage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 15 | 2 | 2 | 4 | 4 | 0 | 6 | n/a | n/a | n/a | n/a | 0.000000e+00 | False | traced_polygon_moments (line 1405) |
| 35 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | 0.000000e+00 | False | traced_polygon_moments (line 1405) |
| 75 | 13 | 3 | 11 | 4 | 4 | 0 | 15 | n/a | n/a | n/a | n/a | 0.000000e+00 | False | traced_polygon_moments (line 1405) |
| 89 | 5 | 2 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | 0.000000e+00 | False | traced_polygon_moments (line 1405) |
| 114 | 5 | 2 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | 0.000000e+00 | False | traced_polygon_moments (line 1405) |
| 126 | 15 | 2 | 2 | 4 | 4 | 0 | 6 | n/a | n/a | n/a | n/a | 0.000000e+00 | False | traced_polygon_moments (line 1405) |
| 128 | 11 | 0 | 0 | 0 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | 0.000000e+00 | False | traced_polygon_moments (line 1405) |
| 134 | 13 | 3 | 11 | 4 | 4 | 0 | 15 | n/a | n/a | n/a | n/a | 0.000000e+00 | False | traced_polygon_moments (line 1405) |

### Per-edge crossings

- cell 2: participating crossings on edges [10, 11, 12, 13]; straddling edges blocked by participation: []; level per vertex [boundary] [-6.9048, -6.80646, -6.80646, -6.69163, -6.69163, -6.57593, -6.57593, -6.43633, -6.43633, -6.25175, -6.25175, -6.10044, 1.09004, 1.5765, -6.45321] / [curved] [-0.06621, -0.09087, -0.09087, -0.11687, -0.11687, -0.12198, -0.12198, -0.09315, -0.09315, -0.03096, -0.03096, 0.00438, -0.04269, 0.00524, -0.05695]
- cell 35: participating crossings on edges []; straddling edges blocked by participation: []; level per vertex [boundary] [-13.29099, -13.32858, -13.32858, -13.4739, -13.4739, -13.54113, -13.54113, -13.56802, -9.00675, -0.11462, -2.40546] / [curved] [-0.07492, -0.07062, -0.07062, -0.04856, -0.04856, -0.04937, -0.04937, -0.07451, -0.08097, -0.01255, -0.03138]
- cell 75: participating crossings on edges [8, 9, 11, 12]; straddling edges blocked by participation: []; level per vertex [boundary] [-6.10044, -6.00352, -6.00352, -5.67589, -5.67589, -5.05039, -5.05039, -4.76883, -4.76883, -4.47155, 2.37266, 7.02407, 1.09004] / [curved] [0.00438, 0.04772, 0.04772, 0.11374, 0.11374, 0.13533, 0.13533, 0.09036, 0.09036, -0.01235, 0.1018, 0.04057, -0.04269]
- cell 89: participating crossings on edges []; straddling edges blocked by participation: []; level per vertex [boundary] [-0.21536, -0.16463, -0.16463, 0.03303, 3.94514] / [curved] [-0.09925, -0.11367, -0.11367, -0.09729, -0.03995]
- cell 114: participating crossings on edges []; straddling edges blocked by participation: []; level per vertex [boundary] [0.03303, -0.16463, -0.16463, -0.21536, 3.94514] / [curved] [-0.09729, -0.11367, -0.11367, -0.09925, -0.03995]
- cell 126: participating crossings on edges [0, 12, 13, 14]; straddling edges blocked by participation: []; level per vertex [boundary] [-6.10044, -6.25175, -6.25175, -6.43633, -6.43633, -6.57593, -6.57593, -6.69163, -6.69163, -6.80646, -6.80646, -6.9048, -6.45321, 1.5765, 1.09004] / [curved] [0.00438, -0.03096, -0.03096, -0.09315, -0.09315, -0.12198, -0.12198, -0.11687, -0.11687, -0.09087, -0.09087, -0.06621, -0.05695, 0.00524, -0.04269]
- cell 128: participating crossings on edges []; straddling edges blocked by participation: []; level per vertex [boundary] [-13.56802, -13.54113, -13.54113, -13.4739, -13.4739, -13.32858, -13.32858, -13.29099, -2.40546, -0.12024, -9.00675] / [curved] [-0.07451, -0.04937, -0.04937, -0.04856, -0.04856, -0.07062, -0.07062, -0.07492, -0.03138, -0.01255, -0.08097]
- cell 134: participating crossings on edges [0, 9, 10, 12]; straddling edges blocked by participation: []; level per vertex [boundary] [-4.47155, -4.76883, -4.76883, -5.05039, -5.05039, -5.67589, -5.67589, -6.00352, -6.00352, -6.10044, 1.09004, 6.87325, 2.37266] / [curved] [-0.01235, 0.09036, 0.09036, 0.13533, 0.13533, 0.11374, 0.11374, 0.04772, 0.04772, 0.00438, -0.04269, 0.04057, 0.1018]

### First empty stage

- cell 2: `traced_polygon_moments` at `nova/equilibrium/separatrix_clip.py:1405` with 0 vertices — the branch shoelace area is zero, so included = area > 0.0 is false at line 1408
- cell 35: `traced_polygon_moments` at `nova/equilibrium/separatrix_clip.py:1405` with 0 vertices — the branch shoelace area is zero, so included = area > 0.0 is false at line 1408
- cell 75: `traced_polygon_moments` at `nova/equilibrium/separatrix_clip.py:1405` with 0 vertices — the branch shoelace area is zero, so included = area > 0.0 is false at line 1408
- cell 89: `traced_polygon_moments` at `nova/equilibrium/separatrix_clip.py:1405` with 0 vertices — the branch shoelace area is zero, so included = area > 0.0 is false at line 1408
- cell 114: `traced_polygon_moments` at `nova/equilibrium/separatrix_clip.py:1405` with 0 vertices — the branch shoelace area is zero, so included = area > 0.0 is false at line 1408
- cell 126: `traced_polygon_moments` at `nova/equilibrium/separatrix_clip.py:1405` with 0 vertices — the branch shoelace area is zero, so included = area > 0.0 is false at line 1408
- cell 128: `traced_polygon_moments` at `nova/equilibrium/separatrix_clip.py:1405` with 0 vertices — the branch shoelace area is zero, so included = area > 0.0 is false at line 1408
- cell 134: `traced_polygon_moments` at `nova/equilibrium/separatrix_clip.py:1405` with 0 vertices — the branch shoelace area is zero, so included = area > 0.0 is false at line 1408

## Recorder control

The recorder is checked against the production support on every cell: the post-arc recorded count must equal `profile_support.vertex_count`, and the recorded crossing count must equal the participating curved-level edge crossings computed from the vertex signs.

| state | cells where recorded post-arc count equals production vertex count | cells where recorded crossings equal computed crossings |
| --- | --- | --- |
| seed | 0 of 8 | 0 of 8 |
| terminal | 0 of 8 | 0 of 8 |

![empty exact polygon stage funnel](/nova/figures/cut-cell-current-attribution/empty-exact-polygon/empty-exact-polygon.svg)
