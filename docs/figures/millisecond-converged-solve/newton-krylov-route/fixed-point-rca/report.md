# Frozen Newton fixed-point RCA

## Terminal requalification

The route recorded 3 partition reads and 1 re-freezes. The terminal read ran; its partition difference was 0 cells (0 labels, 0 support inclusions, and 0 residual-shadow entries).

## Residual at the frozen terminal state

The frozen map residual is 2.372871476315e-06; the live-map residual is 2.372871479533e-06. It decomposes into a grid component (686 rows) reaching 5.240042604937e-06 maximum absolute (l2 8.592331853079e-05), and a wall/sample component (0 rows) reaching 0.000000000000e+00 maximum absolute (l2 0.000000000000e+00).

| entry | live map − state | frozen map − state | live − frozen |
| --- | ---: | ---: | ---: |
| grid cell 262 at R=0.94166667, Z=-0.00000000 | 5.240042604937e-06 | 5.240042597832e-06 | 7.105427357601e-15 |
| grid cell 237 at R=0.90750000, Z=-0.00000000 | 5.228089124509e-06 | 5.228089118292e-06 | 6.217248937901e-15 |
| grid cell 263 at R=0.94166667, Z=0.03500000 | 5.224780396995e-06 | 5.224780390334e-06 | 6.661338147751e-15 |
| grid cell 261 at R=0.94166667, Z=-0.03500000 | 5.224780394997e-06 | 5.224780388113e-06 | 6.883382752676e-15 |
| grid cell 238 at R=0.90750000, Z=0.03500000 | 5.201704374347e-06 | 5.201704367686e-06 | 6.661338147751e-15 |
| grid cell 236 at R=0.90750000, Z=-0.03500000 | 5.201704372126e-06 | 5.201704365687e-06 | 6.439293542826e-15 |
| grid cell 264 at R=0.94166667, Z=0.07000000 | 5.171468403731e-06 | 5.171468396847e-06 | 6.883382752676e-15 |
| grid cell 260 at R=0.94166667, Z=-0.07000000 | 5.171468399290e-06 | 5.171468392628e-06 | 6.661338147751e-15 |
| grid cell 239 at R=0.90750000, Z=0.07000000 | 5.117900187201e-06 | 5.117900180984e-06 | 6.217248937901e-15 |
| grid cell 235 at R=0.90750000, Z=-0.07000000 | 5.117900183649e-06 | 5.117900177432e-06 | 6.217248937901e-15 |
| grid cell 287 at R=0.97583333, Z=-0.00000000 | 5.084532932909e-06 | 5.084532925803e-06 | 7.105427357601e-15 |
| grid cell 288 at R=0.97583333, Z=0.03500000 | 5.082562870573e-06 | 5.082562863468e-06 | 7.105427357601e-15 |

## What remained frozen

The terminal label comparison is not a complete map comparison. The frozen partition also retains topology coordinates and fluxes, residual domain masking, and clipped support geometry/moments. The operator geometry is static rather than copied into the partition. There is no net-current normalisation scalar in this absolute-current fixture.

| retained quantity | changed entries | maximum absolute warm-to-terminal difference | warm value | terminal value |
| --- | ---: | ---: | --- | --- |
| partition.label | 0 | 0.0 | {'minimum': 0, 'maximum': 2, 'l2_norm': 15.811388300841896} | {'minimum': 0, 'maximum': 2, 'l2_norm': 15.811388300841896} |
| partition.topology.axis | 1 | 8.464390640260603e-16 | {'minimum': 1.2249347491045237e-14, 'maximum': 1.0138129839144976, 'l2_norm': 1.0138129839144976} | {'minimum': 1.3095786555071297e-14, 'maximum': 1.0138129839144976, 'l2_norm': 1.0138129839144976} |
| partition.topology.axis_flux | 1 | 8.881784197001252e-16 | 2.0784671880496064 | 2.0784671880496055 |
| partition.topology.boundary | 1 | 9.637783703870368e-16 | {'minimum': 5.382598193751481e-14, 'maximum': 1.1973905409717345, 'l2_norm': 1.1973905409717345} | {'minimum': 5.286220356712777e-14, 'maximum': 1.1973905409717345, 'l2_norm': 1.1973905409717345} |
| partition.topology.boundary_flux | 1 | 1.3322676295501878e-15 | 1.5405680766331675 | 1.5405680766331662 |
| partition.topology.x_point | 2 | nan | {'minimum': nan, 'maximum': nan, 'l2_norm': nan} | {'minimum': nan, 'maximum': nan, 'l2_norm': nan} |
| partition.topology.x_point_flux | 1 | nan | nan | nan |
| partition.topology.wall_point | 1 | 9.637783703870368e-16 | {'minimum': 5.382598193751481e-14, 'maximum': 1.1973905409717345, 'l2_norm': 1.1973905409717345} | {'minimum': 5.286220356712777e-14, 'maximum': 1.1973905409717345, 'l2_norm': 1.1973905409717345} |
| partition.topology.wall_point_flux | 1 | 1.3322676295501878e-15 | 1.5405680766331675 | 1.5405680766331662 |
| partition.topology.diverted | 0 | None | False | False |
| partition.topology.wall_unit_index | 0 | 0.0 | 0 | 0 |
| partition.profile_support | n/a | n/a | not present in this fixture | not present in this fixture |
| partition.residual_shadow | 0 | None | {'minimum': False, 'maximum': False} | {'minimum': False, 'maximum': False} |

The operator grid is shared static geometry, not a per-solve partition copy: its coordinate array has 0 warm-to-terminal changes. Its physical-node count is 686. Net-current normalisation is not applicable: absolute-current fixture.

## Route termination

The terminal route converged=False with reason ACTIVE_SET_SETTLED after 2 active-set trips, attempting 1 and accepting 1 Newton promotions. Its per-trip live residuals were [2.372871479532645e-06, 2.372871479532645e-06, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan] and its per-trip mask differences were [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]. The retry route converged=False with reason ACTIVE_SET_SETTLED and accepted 1 Newton promotions.

## Falsifiable mechanism check

Two measurements refute the frozen-partition hypothesis. At the terminal state the frozen map output differs from the live map output by at most 7.105427357601e-15 absolute over the ranked cells, so the frozen and live maps are the same map to machine precision there. Second, a second frozen route from the terminal state, re-reading the partition (3 reads, 1 re-freeze) and taking 8 more Newton steps, returns the identical live residual 2.372871479532645e-06 and the identical ACTIVE_SET_SETTLED reason with 1 accepted promotion. A partition refresh therefore does not move the terminus.

Caveat: that second solve re-freezes too, so it tests a repeated frozen pass, not a live-read pass. What it establishes is narrow and sufficient: the residual is reproduced bit-for-bit, so it is a property of the route's terminus, not of a partition that went stale between reads.

## Recommended route

The measured terminus is a settlement, not a stale partition. The outer loop stops after two active-set trips on an unchanged mask with no accepted promotion, retaining a state whose live relative-sup residual is 2.372871479533e-06 while the inner Newton local step residual is 7.969573083161e-13. The route holds the active set fixed within each pass, and once the mask stops changing the settle test ends the loop on the retained state without consulting the live residual. Keep the locked design (live reads through warm-up, one partition per Newton pass) and gate the settle: admit settlement only when the reconciled live relative-sup residual is at or below tolerance, and otherwise continue the local Newton trajectory, which the design already preserves across an unchanged mask. This node's re-frozen refresh does not improve the residual (2.372871479533e-06 to 2.372871479533e-06), so the fix is the residual gate rather than the refresh. Each bounded pass costs 12.04 s against the base route's 142.10 s.

