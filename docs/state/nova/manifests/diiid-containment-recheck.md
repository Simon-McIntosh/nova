node: diiid-containment-recheck
status: complete
commits: f7a2dad87c7f2bf4d16889a3e4e6b2f58295500c
changed_paths:
  - benchmarks/diiid_containment_recheck.py
  - docs/figures/plateau-input-attribution/diiid-containment-recheck.json
tests: `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync python benchmarks/diiid_containment_recheck.py` — PASS, 10/10 terminals measured; `UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -p no:cacheprovider tests/test_connectivity_boundary.py tests/test_topology_boundary.py` — PASS, 25/25 in 223.89 s (13 connectivity-boundary and 12 topology-boundary); Ruff check and format — PASS.
test_logs:
  - /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/topology-and-throughput/diiid-containment-recheck/diiid-containment-recheck.log
  - /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/topology-and-throughput/diiid-containment-recheck/diiid-containment-recheck-tests.log
  - /home/ITER/mcintos/Code/.reckon-worktrees/nova-a0f1e0938fc2/topology-and-throughput/diiid-containment-recheck/diiid-containment-recheck-ruff.log
artifacts: `docs/figures/plateau-input-attribution/diiid-containment-recheck.json` — exact before/after delta for 10/10 terminals against input SHA-256 `52ee8f321b13615e50efe3f7d5a446ba7ebfdd0a53a31c4908bf972215e7432e`; 4/10 pre-repair selections are outside the true governed DIII-D wall polygon, 3/10 active-surface selections change after commit `5acfe07b`, and 0/10 achieved classes flip. The banked receipt and all prior figures remained untouched.
evidence_inputs: |
  The containment test is the production fixed-shape ray crossing with boundary inclusion against the governed 84-point DIII-D wall polygon (`wall_coordinate_sha256=a45135511161237ad38db8e6515b66bf79471b9eb719779281a37dbda9bfffd8`), not a bounding box. The after selector exactly applies commit `5acfe07b`: finite typed candidates are masked against each terminal's active topology-surface polygon before normalized-level argmin, and the wall-shadow band is recomputed from the surviving candidates. Thus the pseudo-wall arm deliberately retains candidates that lie inside its rectangle even when outside the physical vessel; the separate `pre_repair_selection_inside_diiid_wall` verdict always uses the true physical wall.

  Before/after terminal table (`*` marks an active-surface selection change; coordinates are `(R,Z)` in m, `u` is normalized level, all wall verdicts are ADMITTED, and full precision is in the JSON artifact):

  | changed | arm | shot:frame | inside DIII-D wall before | selected X before | u before | margin before | class before | selected X after | u after | margin after | class after |
  |---|---|---|---|---|---:|---:|---|---|---:|---:|---|
  |  | physical_ring | 00000c4a7b:179 | yes | (1.276448, +0.706195) | 0.446446486 | -0.332475247 | limited | (1.276448, +0.706195) | 0.446446486 | -0.332475247 | limited |
  |  | physical_ring | 0003ff34e7:44 | yes | (1.306297, +0.725951) | 0.448272704 | -0.342839638 | limited | (1.306297, +0.725951) | 0.448272704 | -0.342839638 | limited |
  | * | physical_ring | 001554e054:144 | NO | (1.313460, -1.545350) | 0.249910078 | -0.178349740 | limited | (1.117412, +0.427935) | 0.660757778 | -0.589197440 | limited |
  | * | physical_ring | 002495e835:146 | NO | (0.992419, +1.522250) | 0.426432918 | -0.350993659 | limited | (1.172024, +0.614542) | 0.547174367 | -0.471735108 | limited |
  | * | physical_ring | 0040ca9bdc:137 | NO | (0.993538, +1.521277) | 0.423493782 | -0.336829723 | limited | (1.171408, +0.660033) | 0.537483430 | -0.450819372 | limited |
  |  | pseudo_wall | 00000c4a7b:179 | yes | (1.231688, +0.292319) | 0.583913109 | -0.529573464 | limited | (1.231688, +0.292319) | 0.583913109 | -0.529573464 | limited |
  |  | pseudo_wall | 0003ff34e7:44 | yes | (1.262244, +0.504127) | 0.468521541 | -0.410667786 | limited | (1.262244, +0.504127) | 0.468521541 | -0.410667786 | limited |
  |  | pseudo_wall | 001554e054:144 | yes | (1.146199, -0.450202) | 0.474596912 | -0.421414919 | limited | (1.146199, -0.450202) | 0.474596912 | -0.421414919 | limited |
  |  | pseudo_wall | 002495e835:146 | yes | (1.112819, -0.523022) | 0.503599666 | -0.449919515 | limited | (1.112819, -0.523022) | 0.503599666 | -0.449919515 | limited |
  |  | pseudo_wall | 0040ca9bdc:137 | NO | (0.977341, +1.523773) | 0.425659137 | -0.374403090 | limited | (0.977341, +1.523773) | 0.425659137 | -0.374403090 | limited |

  The class result is stable: limited remains 10/10 before and 10/10 after, so achieved-class flips are 0/10. Wall-shadow verdicts also remain ADMITTED on 10/10. The three changed physical-ring selections move to higher normalized saddle levels and more negative limited margins; the fourth out-of-vessel selection belongs to the pseudo-wall arm and remains selected because it is inside that arm's active rectangle.

  The banked diagnostic still says `selected_typed_saddle_not_connectivity_reachable` on 10/10 terminals with connectivity support on 0/10, and the selector repair does not change the separately measured connectivity candidate count. Numerically that label remains 10/10 after. It DOES NOT survive as evidence for the reachability hypothesis: 4/10 selections used to make the cohort claim are outside the actual DIII-D vessel, including 3/5 physical-ring terminals whose selected saddle changes under the repair. Per the claim-at-risk criterion, withdraw the 10-of-10 cohort pattern as mechanistic evidence rather than using its unchanged count to support connectivity routing.
follow_ons: Orchestrator writeback must withdraw the DIII-D 10/10 pattern as evidence and adjudicate whether the pseudo-wall arm should continue permitting saddles outside the physical vessel before any receipt or figure regeneration. This node re-banked neither `margin-frame-remeasure.json` nor any figure, as required.
blockers: none
