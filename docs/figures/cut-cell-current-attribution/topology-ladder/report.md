# Production topology-read resolution ladder

The benchmark supplies the closed-form single-null flux directly to the production fixed-design hex read; no nonlinear solve or persisted terminal state is in the path.

The smallest measured rung that admits the analytic saddle is **200 requested cells (233 realised)**, but this is not a resolution threshold. The read misses at 132 and 340 realised cells, admits at 233, and admits every measured rung from 382 realised cells upward. The non-monotone 233-admitted / 340-missed pair shows that admission depends on where the analytic saddle falls relative to the rotated six-centroid ring: it is a positional sign-pattern effect, not a scalar cell-count floor.

## Resolution census

| requested | realised | pitch (m) | X candidates | admitted | class | saddle error (m) | error / pitch | level error / span | miss |
|---:|---:|---:|---:|:---:|:---:|---:|---:|---:|:---|
| 110 | 132 | 0.15145387 | 0 | no | limited | — | — | — | sign_change_census |
| 200 | 233 | 0.1123212 | 1 | yes | diverted | 0.010067034155547064 | 0.08962719690715878 | 0.0018889123367933985 | — |
| 300 | 340 | 0.091709874 | 0 | no | limited | — | — | — | sign_change_census |
| 342 | 382 | 0.085894167 | 1 | yes | diverted | 0.0051293797294255095 | 0.05971743951714121 | 4.8032081434796e-05 | — |
| 400 | 449 | 0.079423081 | 1 | yes | diverted | 0.0015863576208060243 | 0.01997350897577911 | 0.00034784366018761705 | — |
| 500 | 550 | 0.071038163 | 1 | yes | diverted | 0.0011992892842294948 | 0.01688232394909023 | 3.968691382223275e-05 | — |
| 750 | 814 | 0.058002417 | 1 | yes | diverted | 0.0006808918419851411 | 0.011739025245631966 | 0.00014640968007650657 | — |
| 1000 | 1074 | 0.050231567 | 1 | yes | diverted | 0.0015162433043237832 | 0.03018506880856973 | 0.00016331794996161592 | — |
| 2500 | 2616 | 0.031769232 | 1 | yes | diverted | 0.00040351512399440986 | 0.01270144393240582 | 2.6751296641653768e-05 | — |

## Miss mechanism

- 110 requested / 132 realised: `sign_change_census`; the eligible centroid ring nearest the analytic X-point is ring 1 (central cell 17), whose pattern `000110` produces two cyclic sign changes rather than the four required to seed a saddle. No X candidate enters containment or final admission.
- 300 requested / 340 realised: `sign_change_census`; the eligible centroid ring nearest the analytic X-point is ring 2 (central cell 21), whose pattern `000010` also produces two cyclic sign changes rather than four. No X candidate enters containment or final admission. The finer 340-cell miss beside the coarser 233-cell admission is the direct positional control.

## H200 read cost

| requested | realised | one state median (ms) | batch 16 median (ms) | batched per state (ms) |
|---:|---:|---:|---:|---:|
| 200 | 233 | 3.97159 | 6.12193 | 0.38262 |
| 342 | 382 | 5.71361 | 12.1025 | 0.756404 |
| 400 | 449 | 6.77691 | 15.7293 | 0.98308 |
| 500 | 550 | 7.88241 | 22.2802 | 1.39251 |
| 750 | 814 | 15.5439 | 43.6447 | 2.72779 |
| 1000 | 1074 | 20.6262 | 78.1672 | 4.88545 |
| 2500 | 2616 | 54.1771 | 480.093 | 30.0058 |

Across admitted rungs, the median vmapped batch cost is **1.39251 ms per state**. The cost stays below 1 ms per state through 449 realised cells (`0.98308 ms`), then rises to `1.39251 ms` at 550, `4.88545 ms` at 1074 and `30.0058 ms` at 2616 realised cells. Thus the production ring read is batchable on the H200, but it does not meet the 1 ms target at the intended 1000- to 2500-cell resolutions.

## Figures

- [Admitted saddle error against realised plasma cells](/nova/figures/cut-cell-current-attribution/topology-ladder/admitted-saddle-error.svg) plots position error in pitch for admitted rows and marks unadmitted rows separately below zero so an absent saddle cannot read as zero error.
- [Unadmitted 132-cell saddle region](/nova/figures/cut-cell-current-attribution/topology-ladder/unadmitted-cells-110.svg) shows the analytic line contours, wall, analytic X-point, surrounding hex cells and the nearest eligible rotated centroid ring that produced two sign changes.
- [Unadmitted 340-cell saddle region](/nova/figures/cut-cell-current-attribution/topology-ladder/unadmitted-cells-300.svg) shows the corresponding finer mesh with the same two-sign-change failure, establishing that refinement alone does not determine admission.

## Build evidence

- 200 requested / 233 realised: cache hit `True`, build 0 s, load 0.138324 s, store 0 s.
- 400 requested / 449 realised: cache hit `False`, build 89.6459 s, load 0.00068889 s, store 0.511324 s.

The 500-cell positive control was required to retain at least one X candidate and admit the saddle; this protects a uniform or empty instrument from being reported as a resolution threshold.
