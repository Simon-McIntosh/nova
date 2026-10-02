# Frozen Newton fixed-point RCA

## Terminal requalification

The route recorded 3 partition reads and 1 re-freezes. The terminal read ran; its partition difference was 0 cells (0 labels, 0 support inclusions, and 0 residual-shadow entries).

## Residual at the frozen terminal state

The frozen map residual is 2.318404626579e-05; the live-map residual is 1.282728063088e-11. It decomposes into a grid component (686 rows) reaching 2.832667433950e-11 maximum absolute (l2 4.671694765915e-10), and a wall/sample component (0 rows) reaching 0.000000000000e+00 maximum absolute (l2 0.000000000000e+00).

| entry | live map − state | frozen map − state | live − frozen |
| --- | ---: | ---: | ---: |
| grid cell 287 at R=0.97583333, Z=-0.00000000 | -2.832667433950e-11 | 4.862579033427e-05 | -4.862581866094e-05 |
| grid cell 288 at R=0.97583333, Z=0.03500000 | -2.820321753916e-11 | 4.832214949912e-05 | -4.832217770234e-05 |
| grid cell 286 at R=0.97583333, Z=-0.03500000 | -2.820321753916e-11 | 4.832214949912e-05 | -4.832217770234e-05 |
| grid cell 262 at R=0.94166667, Z=-0.00000000 | -2.809441568274e-11 | 4.644057143555e-05 | -4.644059952996e-05 |
| grid cell 261 at R=0.94166667, Z=-0.03500000 | -2.793543174562e-11 | 4.614908945433e-05 | -4.614911738976e-05 |
| grid cell 263 at R=0.94166667, Z=0.03500000 | -2.793498765641e-11 | 4.614908945433e-05 | -4.614911738932e-05 |
| grid cell 312 at R=1.01000000, Z=-0.00000000 | -2.789946051962e-11 | 5.019292308628e-05 | -5.019295098574e-05 |
| grid cell 313 at R=1.01000000, Z=0.03500000 | -2.781597174817e-11 | 4.987955991176e-05 | -4.987958772773e-05 |
| grid cell 311 at R=1.01000000, Z=-0.03500000 | -2.781552765896e-11 | 4.987955991176e-05 | -4.987958772729e-05 |
| grid cell 285 at R=0.97583333, Z=-0.07000000 | -2.780597974095e-11 | 4.741898569427e-05 | -4.741901350025e-05 |
| grid cell 289 at R=0.97583333, Z=0.07000000 | -2.780575769634e-11 | 4.741898569449e-05 | -4.741901350025e-05 |
| grid cell 314 at R=1.01000000, Z=0.07000000 | -2.752997829703e-11 | 4.894807144451e-05 | -4.894809897449e-05 |

## What remained frozen

The terminal label comparison is not a complete map comparison. The frozen partition also retains topology coordinates and fluxes, residual domain masking, and clipped support geometry/moments. The operator geometry is static rather than copied into the partition. There is no net-current normalisation scalar in this absolute-current fixture.

| retained quantity | changed entries | maximum absolute warm-to-terminal difference | warm value | terminal value |
| --- | ---: | ---: | --- | --- |
| partition.label | 0 | 0.0 | {'minimum': 0, 'maximum': 2, 'l2_norm': 15.811388300841896} | {'minimum': 0, 'maximum': 2, 'l2_norm': 15.811388300841896} |
| partition.topology.axis | 2 | 4.4253046915798677e-07 | {'minimum': 1.309578655507218e-14, 'maximum': 1.0138129839144978, 'l2_norm': 1.0138129839144978} | {'minimum': -3.5138385897413717e-15, 'maximum': 1.0138125413840287, 'l2_norm': 1.0138125413840287} |
| partition.topology.axis_flux | 1 | 8.806365402236338e-06 | 2.0784671880496055 | 2.0784759944150077 |
| partition.topology.boundary | 2 | 5.352783467185551e-14 | {'minimum': 5.477242616454956e-14, 'maximum': 1.1973905409717343, 'l2_norm': 1.1973905409717343} | {'minimum': 1.2445914926940496e-15, 'maximum': 1.1973905409717365, 'l2_norm': 1.1973905409717365} |
| partition.topology.boundary_flux | 1 | 6.6430230281078195e-06 | 1.5405680766331673 | 1.5405747196561954 |
| partition.topology.x_point | 2 | nan | {'minimum': nan, 'maximum': nan, 'l2_norm': nan} | {'minimum': nan, 'maximum': nan, 'l2_norm': nan} |
| partition.topology.x_point_flux | 1 | nan | nan | nan |
| partition.topology.wall_point | 2 | 5.352783467185551e-14 | {'minimum': 5.477242616454956e-14, 'maximum': 1.1973905409717343, 'l2_norm': 1.1973905409717343} | {'minimum': 1.2445914926940496e-15, 'maximum': 1.1973905409717365, 'l2_norm': 1.1973905409717365} |
| partition.topology.wall_point_flux | 1 | 6.6430230281078195e-06 | 1.5405680766331673 | 1.5405747196561954 |
| partition.topology.diverted | 0 | None | False | False |
| partition.topology.wall_unit_index | 0 | 0.0 | 0 | 0 |
| partition.profile_support | n/a | n/a | not present in this fixture | not present in this fixture |
| partition.residual_shadow | 0 | None | {'minimum': False, 'maximum': False} | {'minimum': False, 'maximum': False} |

The operator grid is shared static geometry, not a per-solve partition copy: its coordinate array has 0 warm-to-terminal changes. Its physical-node count is 686. Net-current normalisation is not applicable: absolute-current fixture.

## Route termination

The terminal route converged=True with reason CONVERGED after 2 active-set trips, attempting 1 and accepting 1 Newton promotions. Its per-trip live residuals were [2.3728714793315085e-06, 2.3728714793315085e-06, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] and its per-trip mask differences were [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]. The retry route converged=True with reason CONVERGED and accepted 0 Newton promotions.

## Falsifiable mechanism check

Two measurements refute the frozen-partition hypothesis. At the terminal state the frozen map output differs from the live map output by at most 5.019295098574e-05 absolute over the ranked cells, so the frozen and live maps are the same map to machine precision there. Second, a second frozen route from the terminal state, re-reading the partition (2 reads, 0 re-freeze) and taking 8 more Newton steps, returns the identical live residual 1.282728063088183e-11 and the identical CONVERGED reason with 0 accepted promotion. A partition refresh therefore does not move the terminus.

Caveat: that second solve re-freezes too, so it tests a repeated frozen pass, not a live-read pass. What it establishes is narrow and sufficient: the residual is reproduced bit-for-bit, so it is a property of the route's terminus, not of a partition that went stale between reads.

## Recommended route

The terminal state has a live residual of 1.282728063088e-11 after a direct live Newton fallback, so it meets the fixed-point tolerance without accepting the frozen objective. The frozen map remains distinct at this state; the route records two frozen passes, three live partition reads, and one live-read Newton step. A terminal refresh preserves the live residual at 1.282728063088e-11. The initial frozen route segment took 205.71 s and the refresh segment took 37.73 s.
