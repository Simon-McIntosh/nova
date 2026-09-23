# Exact support attribution — blocked

The H200 evaluation refused a non-simple support polygon in cell 9 of the diverted-single-null case at 110 requested cells:

```
AssertionError: invalid exact polygons: [9]
```

The attribution has not been produced. No equilibrium solve or production repair ran.

The archive coordinate, analytic-state, finite-moment, physical-moment and support-fraction checks passed before the refusal. In particular, the recomputed exact support fraction matched the archived **0.9541772190883415** within relative tolerance **1e-9**. The explicit 35.4 mm centroid control was not reached. Neither chord mode nor the other case/rung pairs ran. No missing-current split or panel is claimed.

The first preflight failed while sampling the analytic boundary at 11,521 points: `RuntimeError: zero-flux lobe is not star-shaped at angle 4.450549`. Its traceback is preserved in [boundary-preflight.log](boundary-preflight.log). The driver now refines the established 2,881-point contour by normal projection onto analytic zero flux, keeping the saddle endpoint fixed. The second job passed the projection residual bound (1e-12 Wb) and the boundary-current refinement bound (1e-5 relative L1) before the polygon refusal. The production sampler is unchanged.

| Job | Source revision | Result | Scheduler elapsed |
|---|---|---|---:|
| 1276895 | b2fd0f22e98b51c68b56f157289b83369b6ee4ec | Boundary-sampling refusal, exit 1 | 16 s |
| 1276896 | 918f4edb457a28e466be7be79c7570efb4a9eaec | Non-simple polygon refusal, exit 1 | 28 s |

Both jobs ran on the H200 lane with one GPU, 8 cores, 64 GB and double precision. The first log line identifies the source revision, worktree and command. Read the [evaluation log](job.log) and [machine-readable blocker record](report.json).

The driver computes true minus booked as geometry minus integration, where integration is booked minus the analytic current on the identical polygon. It retains the archived target separately, because the diverted target itself integrates over traced fixture supports. Concave-polygon and hole-subtraction controls passed locally; scoped Ruff checks passed.

**Decision needed:** the returned chain is not a valid simple polygon. Signed-winding integration, geometric repair, and an upstream clip investigation define different experiments. The next step is to persist the refused vertices and cell geometry, inspect the intersection, and choose the intended support semantics before resuming. No arbitrary repair has been applied. The failed process did not serialize those vertices or the intermediate integrals, so they must be captured on a resumed run.

The reviewed chord receipt also carries a caveat: at diverted 110 cells its fraction is 1.003530216161609, a 0.353% excess. That satisfies a signed missing-current criterion but fails absolute agreement within 1e-3; a resumed attribution must retain both readings.
