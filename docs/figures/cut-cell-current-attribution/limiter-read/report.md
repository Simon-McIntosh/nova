# Analytic limiter read resolution

The certificate adapter calls `operator.read` at `/home/ITER/mcintos/Code/nova/benchmarks/solovev_certificate.py:1680`; the public read begins at `/home/ITER/mcintos/Code/nova/nova/equilibrium/forward_operator.py:2001` and selects `_fixed_design_read` at line 2006. Because these hex carriers have moment geometry but no tensor axes, the lazy class property takes the moment-geometry branch at `/home/ITER/mcintos/Code/nova/nova/equilibrium/forward_operator.py:1926` and performs another fixed-design read at line 1929. The tensor-spline connectivity limiter is unavailable on these rows.

The contact is selected at `/home/ITER/mcintos/Code/nova/nova/equilibrium/topology.py:496` (the extremal node); three wall-node values are fit by `traced_quadratic_wall` at line 519 and their coordinates are interpolated by `wall_coordinate` at line 524. That operation returns a point between nodes, but the measured position error remains first order in wall panel length. It is wall-panel-limited, not the global tensor-spline restriction and derivative-root polish already available on raster reads.

| Case | Cells | Wall nodes | Class | Panel / pitch | Contact error [m] | Level error / span | Warm read [s] |
|---|---:|---:|---|---:|---:|---:|---:|
| weak-rotation-reactor-static | 342 | 121 | limited | 0.761192 | 0.101901 | 0.000803533 | 0.174995 |
| moderate-rotation-conventional-static | 341 | 121 | limited | 0.754388 | 0.0284941 | 0.000814265 | 0.228288 |
| diverted-single-null | 550 | 121 | diverted | 0.691463 | 0.00804901 | 0.00164014 | 0.454385 |
| weak-rotation-reactor-static | 1072 | 121 | limited | 1.38974 | 0.101901 | 0.000803763 | 1.9828 |
| moderate-rotation-conventional-static | 1070 | 121 | limited | 1.37732 | 0.0284941 | 0.000814475 | 2.02005 |
| diverted-single-null | 1074 | 121 | diverted | 0.977877 | 0.00804901 | 0.00163962 | 1.90825 |
| weak-rotation-reactor-static | 342 | 241 | limited | 0.382253 | 0.0512087 | 0.000202449 | 0.176413 |
| moderate-rotation-conventional-static | 341 | 241 | limited | 0.378838 | 0.014319 | 0.000205152 | 0.227959 |
| diverted-single-null | 550 | 241 | diverted | 0.347828 | 0.00128231 | 0.000895472 | 0.878735 |
| weak-rotation-reactor-static | 1072 | 241 | limited | 0.697895 | 0.0512087 | 0.000202507 | 3.18285 |
| moderate-rotation-conventional-static | 1070 | 241 | limited | 0.691661 | 0.014319 | 0.000205205 | 2.9243 |
| diverted-single-null | 1074 | 241 | diverted | 0.491903 | 0.00128231 | 0.000895187 | 1.92271 |
| weak-rotation-reactor-static | 342 | 481 | limited | 0.191534 | 0.0256638 | 5.08165e-05 | 0.182198 |
| moderate-rotation-conventional-static | 341 | 481 | limited | 0.189823 | 0.00717606 | 5.14947e-05 | 0.247618 |
| diverted-single-null | 550 | 481 | diverted | 0.174354 | 0.00190812 | 9.85601e-05 | 0.714914 |
| weak-rotation-reactor-static | 1072 | 481 | limited | 0.349691 | 0.0256638 | 5.0831e-05 | 2.71058 |
| moderate-rotation-conventional-static | 1071 | 481 | limited | 0.346568 | 0.00717606 | 5.1508e-05 | 1.99977 |
| diverted-single-null | 1074 | 481 | diverted | 0.246573 | 0.00190812 | 9.85288e-05 | 1.90852 |
| weak-rotation-reactor-static | 342 | 961 | limited | 0.095868 | 0.012846 | 1.27302e-05 | 0.176622 |
| moderate-rotation-conventional-static | 341 | 961 | limited | 0.0950118 | 0.00359198 | 1.29001e-05 | 0.232317 |
| diverted-single-null | 550 | 961 | diverted | 0.0872804 | 0.000462695 | 5.61672e-05 | 0.701094 |
| weak-rotation-reactor-static | 1072 | 961 | limited | 0.17503 | 0.012846 | 1.27338e-05 | 3.11142 |
| moderate-rotation-conventional-static | 1071 | 961 | limited | 0.173467 | 0.00359198 | 1.29035e-05 | 2.83521 |
| diverted-single-null | 1074 | 961 | diverted | 0.123433 | 0.000462695 | 5.61493e-05 | 1.8661 |

## Wall resolution against plasma pitch

Panel-to-pitch entries use the selected outboard panel. The nominal 2500-cell row uses about 2600 realised cells, with pitch scaled from the realised thousand-cell carrier.

| Case | Cell scale | Realised cells | 121 | 241 | 481 | 961 | Minimum odd wall count |
|---|---:|---:|---:|---:|---:|---:|---:|
| weak-rotation-reactor-static | 300 | 342 | 0.7612 | 0.3823 | 0.1915 | 0.0959 | 93 |
| weak-rotation-reactor-static | 1000 | 1072 | 1.3897 | 0.6979 | 0.3497 | 0.1750 | 169 |
| weak-rotation-reactor-static | 2500 | 2600 | 2.1643 | 1.0871 | 0.5447 | 0.2726 | 263 |
| moderate-rotation-conventional-static | 300 | 341 | 0.7544 | 0.3788 | 0.1898 | 0.0950 | 93 |
| moderate-rotation-conventional-static | 1000 | 1070 | 1.3773 | 0.6917 | 0.3466 | 0.1735 | 167 |
| moderate-rotation-conventional-static | 2500 | 2600 | 2.1470 | 1.0784 | 0.5403 | 0.2705 | 261 |
| diverted-single-null | 500 | 550 | 0.6915 | 0.3478 | 0.1744 | 0.0873 | 85 |
| diverted-single-null | 1000 | 1074 | 0.9779 | 0.4919 | 0.2466 | 0.1234 | 119 |
| diverted-single-null | 2500 | 2600 | 1.5215 | 0.7655 | 0.3837 | 0.1921 | 185 |

## Measured wall order

- weak-rotation-reactor-static at 300 requested cells: position order 0.9999; level order 2.0003.
- weak-rotation-reactor-static at 1000 requested cells: position order 0.9999; level order 2.0003.
- moderate-rotation-conventional-static at 300 requested cells: position order 0.9999; level order 2.0003.
- moderate-rotation-conventional-static at 1000 requested cells: position order 0.9999; level order 2.0003.

## Why the 121-node level looked exact

Yes: `limiter_contour` samples angles as `2*pi*(arange(points)+0.5)/points`. Every requested odd count places index `(points-1)/2` exactly at angle pi, the authored smooth outboard tangency. Across weak and moderate rows and all four counts, the maximum node displacement is 7.61e-08 m and the maximum analytic node flux magnitude is 0 Wb. The 2.7e-9 Wb weak-121 read is therefore a lucky sampling identity, not evidence that the piecewise wall contact is position-accurate. The wall polygon's adjacent chord enters the analytic plasma; its extremum is one panel away in position, producing the measured first-order position and second-order level ladders.

## Private-flux wall shadow

All 8 single-null controls passed: the injected private-region node won the unmasked limited read and was rejected by the masked read. `wall_height_shadow_mask` uses the qualified-saddle height band and intersects it with the connectivity-private wall mask. The selecting exclusion is therefore height-based over connectivity-qualified private nodes, not a connectivity-only shadow; each row records whether either side fell back to connectivity alone.

| Cells | Wall nodes | Private below X | Shadowed below X | Lower branch | Selected instead [R, Z] m |
|---:|---:|---:|---:|---|---|
| 550 | 121 | 5 | 5 | qualified_height_band | [1.355258, -1.135439] |
| 1074 | 121 | 5 | 5 | qualified_height_band | [1.355258, -1.135439] |
| 550 | 241 | 9 | 9 | qualified_height_band | [1.379131, -1.161739] |
| 1074 | 241 | 9 | 9 | qualified_height_band | [1.379131, -1.161739] |
| 550 | 481 | 17 | 17 | qualified_height_band | [1.393762, -1.173420] |
| 1074 | 481 | 18 | 18 | qualified_height_band | [1.393762, -1.173420] |
| 550 | 961 | 35 | 35 | qualified_height_band | [1.396527, -1.175401] |
| 1074 | 961 | 35 | 35 | qualified_height_band | [1.401748, -1.178781] |

## Why the single-null saddle moves with wall count

- 500 requested cells — published boundary/X-point errors [wall nodes:m] 121:0.00804901, 241:0.00128231, 481:0.00190812, 961:0.000462695; 4 distinct carrier node identities. The raw fixed-design saddle candidate errors are 121:0.00119929, 241:0.00120222, 481:0.00120222, 961:0.00120165. wall sampling changes the cached carrier mesh before the read; containment admits the selected candidate, no shadow is supplied, and one finite candidate leaves no dedupe choice.
- 1000 requested cells — published boundary/X-point errors [wall nodes:m] 121:0.00804901, 241:0.00128231, 481:0.00190812, 961:0.000462695; 4 distinct carrier node identities. The raw fixed-design saddle candidate errors are 121:0.00151624, 241:0.00151152, 481:0.00151269, 961:0.00151444. wall sampling changes the cached carrier mesh before the read; containment admits the selected candidate, no shadow is supplied, and one finite candidate leaves no dedupe choice.
This is not a shadow effect: the analytic public read receives no prior wall mask. It is not containment or dedupe selection either: the selected candidate remains contained and each row has one finite X candidate. The coupling enters in `cached_machine`: inserting the differently sampled wall rebuilds the plasma carrier and its null-fit stencils, so a flux-only saddle moves with what should have been wall-only resolution.

## Limited-row map floor

Weak 1000 exact-clip RMS floor: 0.136087741 of span at 121 wall nodes and 0.137163529 at 481, an absolute movement of 0.00108. The floor therefore does not move materially with wall resolution.

## Figures

- `/nova/figures/cut-cell-current-attribution/limiter-read/contact-error-vs-wall-resolution.svg` — all contact errors against local wall-panel-to-cell-pitch ratio.
- `/nova/figures/cut-cell-current-attribution/limiter-read/single-null-contact-shadow.png` — analytic single-null 1000 field, wall, analytic nulls, selected wall contact and excluded private-wall nodes.
