node: typed-saddle-containment
status: complete
commits: 5acfe07bd589534d70dd88ce98516889ff5f7db0
changed_paths:
  - nova/equilibrium/connectivity_boundary.py
  - tests/test_connectivity_boundary.py
  - docs/figures/primary-xpoint-evidence/typed-saddle-containment.json
tests: `JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest tests/test_connectivity_boundary.py tests/test_jax_topology.py` — PASS, 23/23 in 117.94 s: 13/13 connectivity-boundary tests and 10/10 JAX-topology tests. Ruff check and format passed on both changed Python paths; JSON parsing, `git diff --check`, and a 12/12-row source validation passed.
test_logs:
  - /tmp/typed-saddle-containment-topology-tests-final.log
  - /tmp/typed-saddle-containment-measurement.log
  - /tmp/typed-saddle-containment-fresh.json
  - /tmp/typed-saddle-containment-commit.log
artifacts: `docs/figures/primary-xpoint-evidence/typed-saddle-containment.json` — 6 references, 12 arm rows, 8 changed and 4 unchanged; every selected coordinate, normalized level, wall-shadow verdict, and class margin validates exactly against the old pinned bank and the fresh post-change temporary measurement. `docs/figures/dual-branch-selection/pinned-branch-contrast.json` and all pre-existing figures remained untouched.
evidence_inputs: |
  Containment belongs in the exact-classification branch of `_read_ingredients`, immediately before normalized-level reduction. That is the shared hard-read, smooth-read, and diagnostic seam where the fixed typed-candidate table and the actual `operator.wall.coordinate` samples (passed as `wall_r` / `wall_z`) first coexist. The separate connectivity-local detector already masks with `inside_limiter`, but its candidate table never feeds the exact class selector. Filtering there would therefore leave the defect intact. The implementation performs a JAX ray-crossing polygon test with boundary inclusion, combines that mask with finite-candidate validity, and retains the original table shape; no data-dependent compaction or count enters tracing. The regression uses a diamond wall and an offending point that lies inside the bounding box but outside the polygon, then checks direct JIT execution and `vmap` batch shape.

  Before/after frozen-reference table (`*` marks every changed row; `+inf` is serialized as a null numeric margin plus `positive_infinity`):

  | changed | reference | arm | selected X before (R,Z) m | level before | effective band before m | shadow before | margin before | selected X after (R,Z) m | level after | effective band after m | shadow after | margin after |
  |---|---|---|---|---:|---|---|---:|---|---:|---|---|---:|
  |  | 21978/35 | pure | (0.567656, -1.233831) | 0.442528 | [-1.233831, +1.244849] | SHADOWED | +inf | (0.567656, -1.233831) | 0.442528 | [-1.233831, +1.244849] | SHADOWED | +inf |
  |  | 21978/35 | mixed | (0.574698, +1.263986) | 0.435802 | [-1.212692, +1.263986] | SHADOWED | +inf | (0.574698, +1.263986) | 0.435802 | [-1.212692, +1.263986] | SHADOWED | +inf |
  | * | 21983/35 | pure | (0.450010, -1.817467) | 0.351140 | [-1.817467, +1.249132] | ADMITTED | +0.007233 | (0.544466, +1.249132) | 0.447113 | [-1.223072, +1.249132] | SHADOWED | +inf |
  | * | 21983/35 | mixed | (0.462520, +1.803693) | 0.348190 | [-1.819262, +1.803693] | ADMITTED | +0.007733 | (0.552602, +1.274043) | 0.434986 | [-1.193450, +1.274043] | SHADOWED | +inf |
  | * | 21985/51 | pure | (0.367141, +1.908566) | 0.007860 | [-1.890180, +1.908566] | ADMITTED | +0.019711 | (0.555917, -1.117865) | 0.230218 | [-1.117865, +inf] | ADMITTED | -0.202646 |
  | * | 21985/51 | mixed | (0.365555, +1.907338) | 0.008626 | [-1.891358, +1.907338] | ADMITTED | +0.019070 | (0.542022, -1.097248) | 0.240553 | [-1.097248, +inf] | ADMITTED | -0.213159 |
  | * | 21986/46 | pure | (0.347017, +1.920913) | 0.304306 | [-1.934963, +1.920913] | ADMITTED | +0.004897 | (0.622334, +1.183521) | 0.455304 | [-1.081548, +1.183521] | SHADOWED | +inf |
  | * | 21986/46 | mixed | (0.378182, -1.899386) | 0.027879 | [-2.420115, +inf] | ADMITTED | -0.006570 | (0.606511, +1.154073) | 0.473036 | [-1.104212, +1.154073] | SHADOWED | +inf |
  | * | 21989/55 | pure | (0.286298, +1.940481) | 0.263688 | [-1.108763, +1.940481] | ADMITTED | -0.003395 | (0.582068, +1.144963) | 0.425705 | [-1.108763, +1.144963] | SHADOWED | +inf |
  | * | 21989/55 | mixed | (0.285735, +1.939570) | 0.261805 | [-1.925470, +1.939570] | ADMITTED | -0.003633 | (0.584555, +1.149699) | 0.420890 | [-1.102258, +1.149699] | SHADOWED | +inf |
  |  | 22086/43 | pure | (0.587417, +1.212319) | 0.436389 | [-1.189639, +1.212319] | SHADOWED | +inf | (0.587417, +1.212319) | 0.436389 | [-1.189639, +1.212319] | SHADOWED | +inf |
  |  | 22086/43 | mixed | (0.587075, +1.222555) | 0.430979 | [-1.177931, +1.222555] | SHADOWED | +inf | (0.587075, +1.222555) | 0.430979 | [-1.177931, +1.222555] | SHADOWED | +inf |

  Plain containment verdict by reference: 21978/35 was in-vessel in both arms all along; 21983/35 was outside-vessel in both arms; 21985/51 was outside-vessel in both arms; 21986/46 was outside-vessel in both arms; 21989/55 was outside-vessel in both arms; 22086/43 was in-vessel in both arms all along. Thus all three previously unchecked references fail the old-containment assumption. The pure-arm fork-distribution margins called out in the node change as follows: 21985/51 `+0.0197108 -> -0.202646`, 21986/46 `+0.00489713 -> +inf`, and 21989/55 `-0.00339536 -> +inf`. The last reference is no longer a finite negative boundary case; 21985/51 instead becomes a much more negative finite case. The four already-shadowed terminals (both arms of 21978/35 and 22086/43) retain their in-vessel selections at |Z| = 1.21 to 1.26 m and are unchanged.
follow_ons: Orchestrator adjudication is required before regenerating `pinned-branch-contrast.json` or revising any fork-arming distribution/characterisation that consumed the prior finite margins. No regeneration was performed in this node.
blockers: none
