# Private-region polygon, oracle, and saddle-wedge audit

Job 1271036 completed the four cached single-null meshes on `all_debug` in
1 minute 55 seconds with exit status 0. The receipt is
[`audit-receipt.json`](audit-receipt.json), and the unrounded per-cell census is
[`wedge-disagreements.csv`](wedge-disagreements.csv).

## Verdict

The production private-region mask is confirmed. A host NumPy breadth-first
search from the magnetic-axis cell over the confined-cell adjacency differs
from the production flood on **zero cells at every rung**. Pointer jumping also
differs from that independent oracle on zero cells at every rung.

The banked saddle-wedge result was mostly a convention error, not separatrix
curvature. The Hessian eigenvectors are the saddle's principal-curvature axes;
they are not the zero-level directions of the saddle quadratic. The local
separatrix tangents are the two linear combinations satisfying
`direction.T @ Hessian @ direction == 0`. Substituting those tangents changes
the disagreement counts from **3, 5, 5, 13** to **3, 0, 0, 0** at 550, 814,
1074, and 2616 cells. The three residual cells at 550 are the genuine
finite-radius curvature limit of a straight-ray test. Consequently the
pointer-jumping component algorithm remains the exact replacement: its banked
cost is **0.020 to 0.047 ms per state**, versus **0.007 to 0.016 ms** for the
inexact wedge shortcut.

## What was wrong in the old renderer

The old panel never read the mesh's polygons. It inferred one global pitch from
the minimum distance in the sparse connectivity table, searched centroid
coordinates for neighbours within 1.05 times that pitch, averaged the neighbour
directions, snapped that unstable average to a 30-degree lattice direction, and
drew a regular hexagon of radius `pitch / sqrt(3)` around the cell centroid.

That construction preserves the centroid to floating-point precision, which
made the error look plausible, but it discards the facts that matter: boundary
cells are clipped by the wall, their areas and vertex counts differ, the carrier
is not one global affine lattice, and a symmetric neighbour set has a near-zero
mean whose angle is not a cell orientation. The result was rotated, grossly
mis-sized, overlapping polygons; where the inferred global pitch found no
neighbour, the renderer silently omitted the cell altogether.

| realised cells | selected | compared | silently omitted | synthetic overlap pairs | true overlap pairs | minimum IoU | median IoU | synthetic / true area range |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 550 | 5 | 5 | none | 6 | 0 | 0.147 | 0.523 | 1.00 to 6.79 |
| 814 | 9 | 7 | 36, 37 | 10 | 0 | 0.00485 | 0.618 | 0.903 to 206.04 |
| 1074 | 12 | 12 | none | 21 | 0 | 0.0156 | 0.894 | 0.966 to 63.99 |
| 2616 | 27 | 0 | 1371, 1373, 1442–1456, 1458–1462, 1464–1466, 1469, 1471 | 0 | 0 | n/a | n/a | n/a |

The corrected renderer aligns `OracleMachine.cell_polygons` against
`machine.node` and draws those vertex arrays directly. At 2616 cells this is
the difference between drawing all 27 selected polygons faithfully and drawing
none of them. The panels use line-only flux contours with no axes or grid,
include the true wall, show analytic and admitted nulls separately, hatch the
production private mask, outline the pointer-jumping mask, outline the banked
wedge disagreements in a third style, and draw both the analytic separatrix
legs and the misused Hessian eigenvector directions.

![Private-region masks and wedge geometry at 1074 cells](/nova/figures/cut-cell-current-attribution/private-region/private-mask-poloidal-1074.svg)

![Private-region masks and wedge geometry at 2616 cells](/nova/figures/cut-cell-current-attribution/private-region/private-mask-poloidal-2616.svg)

## Independent breadth-first oracle

For each rung the oracle independently applies the polarity-specific
separatrix-level inequality, intersects it with the inside-wall mask, constructs
an undirected adjacency from each sparse ring row's explicit centre index, and
uses a host `deque` to visit cells from the one known-present magnetic-axis
seed. Private cells are confined cells not reached by that search.

| realised cells | axis seed | confined | reached | private: BFS / production | BFS != production cells | BFS != production flood cells | BFS != pointer cells |
|---:|---:|---:|---:|---:|---|---|---|
| 550 | 347 | 308 | 305 | 3 / 3 | `[]` | `[]` | `[]` |
| 814 | 603 | 457 | 448 | 9 / 9 | `[]` | `[]` | `[]` |
| 1074 | 766 | 606 | 594 | 12 / 12 | `[]` | `[]` | `[]` |
| 2616 | 1949 | 1523 | 1496 | 27 / 27 | `[]` | `[]` | `[]` |

There are therefore no differing cells to attribute to the production flood's
hysteresis band, saddle-cell bridging rule, or representative-candidate loop.
The positive control also fires at every rung: exactly one confined axis seed is
present and reached, and the BFS visits a non-empty component before an absence
is accepted.

## Convention candidates

The banked implementation is reproduced bit-for-bit by the audit. Each
candidate changes one convention while retaining the same admitted X-point,
confined mask, wall mask, and cell-centroid flux values.

| realised cells | banked eigenvector pair | reverse eigenvector signs | reverse eigenvector order | reverse flux inequality | use zero-level tangent pair |
|---:|---:|---:|---:|---:|---:|
| 550 | 3 | 3 | 3 | 3 | 3 |
| 814 | 5 | 5 | 5 | 9 | 0 |
| 1074 | 5 | 5 | 5 | 12 | 0 |
| 2616 | 13 | 13 | 13 | 27 | 0 |

Reversing either eigenvector sign or their order changes **zero mask entries**:
the probe-derived side signs reverse or permute with the directions, making the
test invariant. Reversing the flux-side inequality is not the fix; from 814
cells upward its disagreement is exactly the full production-private count.
The pair itself was wrong. For example, at 1074 cells the Hessian eigenvalues
are -0.08425 and +0.04309; the eigenvectors evaluate to those nonzero quadratic
values, while both constructed tangent directions evaluate to less than
`7e-18`, numerically zero.

## Banked wedge disagreements in the eigenvector frame

`u- / pitch` and `u+ / pitch` are centroid coordinates along the negative- and
positive-curvature Hessian eigenvectors. `delta psi / span` is the centroid
flux above the admitted X-point level divided by the axis-to-X-point span.
The two side columns say whether the banked test placed the centroid on the
private side of each misidentified “leg.” `Oracle -> wedge` exposes false
negatives and false positives. `Curve gap / pitch` is the nearest vertical
separation, at that cell's radius, between the correct straight zero-level
tangent and the analytic separatrix branch.

| cells | cell | u- / pitch | u+ / pitch | delta psi / span | side 1 | side 2 | Oracle -> wedge | curve gap / pitch |
|---:|---:|---:|---:|---:|---|---|---|---:|
| 550 | 1 | 0.609 | 2.137 | 0.09458 | private | outside | true -> false | 0.02744 |
| 550 | 2 | -0.149 | 1.772 | 0.08281 | private | private | false -> true | 0.00634 |
| 550 | 5 | -0.0666 | 0.775 | 0.01441 | private | private | false -> true | 0.00634 |
| 814 | 31 | 0.0426 | 2.918 | 0.13917 | private | outside | true -> false | 0.00803 |
| 814 | 32 | 0.984 | 2.601 | 0.06678 | private | outside | true -> false | 0.01775 |
| 814 | 34 | 0.137 | 2.055 | 0.06330 | private | outside | true -> false | 0.00903 |
| 814 | 37 | 0.226 | 1.006 | 0.01246 | private | outside | true -> false | 0.00903 |
| 814 | 47 | 1.610 | 2.743 | 0.01907 | private | outside | true -> false | 0.01949 |
| 1074 | 0 | 1.143 | 3.260 | 0.09521 | private | outside | true -> false | 0.02193 |
| 1074 | 2 | 0.426 | 3.057 | 0.11579 | private | outside | true -> false | 0.01238 |
| 1074 | 4 | 1.349 | 2.652 | 0.03508 | private | outside | true -> false | 0.02219 |
| 1074 | 5 | 0.514 | 2.072 | 0.04520 | private | outside | true -> false | 0.01147 |
| 1074 | 8 | 0.599 | 1.058 | 0.00463 | private | outside | true -> false | 0.01147 |
| 2616 | 1447 | 1.411 | 7.243 | 0.08006 | private | outside | true -> false | 0.000576 |
| 2616 | 1451 | 2.462 | 9.248 | 0.12181 | private | outside | true -> false | 0.01619 |
| 2616 | 1452 | 4.011 | 8.643 | 0.05993 | private | outside | true -> false | 0.03112 |
| 2616 | 1453 | 1.259 | 8.900 | 0.13174 | private | outside | true -> false | 0.000217 |
| 2616 | 1454 | 2.848 | 8.242 | 0.08123 | private | outside | true -> false | 0.02692 |
| 2616 | 1455 | 4.434 | 7.497 | 0.01713 | private | outside | true -> false | 0.02782 |
| 2616 | 1456 | 2.996 | 6.498 | 0.03497 | private | outside | true -> false | 0.02683 |
| 2616 | 1458 | 1.558 | 5.497 | 0.03977 | private | outside | true -> false | 0.000576 |
| 2616 | 1460 | 0.120 | 4.497 | 0.03253 | private | outside | true -> false | 0.02609 |
| 2616 | 1464 | 0.267 | 2.751 | 0.01148 | private | outside | true -> false | 0.02609 |
| 2616 | 1466 | 0.414 | 1.005 | 0.00107 | private | outside | true -> false | 0.02609 |
| 2616 | 1469 | 3.144 | 4.752 | 0.00192 | private | outside | true -> false | 0.02683 |
| 2616 | 1471 | 1.705 | 3.751 | 0.01221 | private | outside | true -> false | 0.000576 |

The banked pair's geometric failure is visible in the side columns: every
false negative from 814 cells upward is accepted by the first eigenvector axis
and rejected by the second. The disagreement therefore clusters on one side of
one supposed leg exactly because that “leg” is a principal axis, not because
the physical separatrix bends preferentially on that side.

## Residual curvature effect

After the tangent convention is corrected, the wedge is exact at every rung
from 814 through 2616 cells even though the analytic branches are curved. At
the banked disagreement radii, the correct straight tangents differ from the
analytic branches by the following amounts:

| realised cells | cells sampled | median curve gap / pitch | maximum curve gap / pitch | corrected wedge differences |
|---:|---:|---:|---:|---:|
| 550 | 3 | 0.00634 | 0.02744 | 3 |
| 814 | 5 | 0.00903 | 0.01949 | 0 |
| 1074 | 5 | 0.01238 | 0.02219 | 0 |
| 2616 | 13 | 0.02609 | 0.03112 | 0 |

Thus curvature is real and measurable, but it does not explain the banked 5,
5, and 13-cell errors: separations up to 0.031 pitch do not change a membership there
once the correct tangent pair is used. Only the coarsest mesh remains too coarse
for the straight local tangent approximation, leaving three genuine curvature
differences. A wedge shortcut would therefore need an explicit coarse-mesh
fallback and still buys only about 0.013 to 0.031 ms over exact pointer jumping;
the topological pointer-jumping mask is exact without that exception.
