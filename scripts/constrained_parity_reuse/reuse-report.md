# Reuse map — constrained-arm parity questions (followup f-efp-constrained-arm-parity)

Scope: read-only survey. No file outside this report was changed.

## (1) Zero-solve residual of the constrained forward map at a stored reference flux map

**Capability:** evaluate the residual of nova's constrained forward map AT a stored
reference own flux map with zero Newton solves, via the shared `jax.linearize`
one-application pattern already used by the banked
`mast-dina-composition-diff` / `boundary-imbalance-attribution` receipts.

| module:symbol | verdict | reason |
| --- | --- | --- |
| `benchmarks/efit_forward_parity_slice.py:_composition_case_receipt` (L2336) | REUSE DIRECTLY | Calls `jax.linearize(operator.flux_map(), state)` at L2340 — exactly the one-application pattern: builds `(image, tangent)` at a fixed state with zero Newton promotions. Generalizes to the constrained arm by swapping the state for the arm's stored reference seed. |
| `benchmarks/efit_forward_parity_slice.py:_boundary_attribution_receipt` (L2881) | EXTEND IN PLACE | Consumes the same one-application image/tangent to decompose the update by conductor circuit group; the decomposition logic is generic over which case dict it is handed (`_mast_composition_case`, `_dina_composition_case`, or a new constrained-arm case), so a third case builder is the only new code needed. |
| `benchmarks/efit_forward_parity_slice.py:_mast_case_from_selection` (L2438) | REUSE DIRECTLY | Already builds the operator/state/reference-flux triple for a MAST row (including the constrained-arm's own `22086/43` row) — the exact case dict shape `_composition_case_receipt` expects. |
| `docs/figures/efit-forward-parity/mast-dina-composition-diff.json` | REUSE DIRECTLY | Banked receipt — the reference numbers (update sup/RMS, boundary imbalance, Picard eigenvalue) to diff a new constrained-arm one-application receipt against. |
| `docs/figures/efit-forward-parity/boundary-imbalance-attribution.json` | REUSE DIRECTLY | Banked per-circuit-group attribution receipt, same reuse role as above. |
| `nova/equilibrium/fixed_point.py` | PARTIAL | Contains `jax.linearize` usage for the solver's own Newton-Krylov linearization step (not a labeled zero-solve one-application entry point); the map object it linearizes is the same `operator.flux_map()` surface `_composition_case_receipt` uses, so it corroborates the pattern but is not itself the reusable receipt-building call site. |

**Verdict:** the machinery for a zero-solve residual-at-reference measurement on the
constrained arm already exists as a generalizable pattern in one file; no new
kernel is needed, only a case builder analogous to `_mast_composition_case`
pointed at the constrained arm's converged/stalled rows.

## (2) Intermediate point sets and matching rule behind the LCFS / axis / X-point metrics

**Capability:** expose the point sets and matching rule that make axis and
X-point agree to 0.0222 m / 0.0206 m on `22086/43` while LCFS disagrees by
0.4469 m, so the discrepancy pattern is interpretable.

| module:symbol | verdict | reason |
| --- | --- | --- |
| `nova/imas/parity_tolerances.py:ScorecardField`, `MetricTolerance`, `registered_tolerances`, `scorecard_verdicts` (L20-284) | UNFIT for point-set internals | Holds only the tolerance thresholds and pass/fail verdict logic (`_common_tolerances`, `validate_scorecard_fields`); it does not compute or expose the point sets or the matching rule itself — the requested internals are not here. |
| `benchmarks/efit_forward_parity_slice.py:_contour` (L481) | REUSE DIRECTLY | Builds the LCFS point set: extracts the longest closed unit-normalized flux contour via `contour_generator(...).lines(1.0)`, filtering to finite polylines of length ≥ 4 and picking the longest by arclength. This is the exact intermediate point set behind the LCFS metric. |
| `benchmarks/efit_forward_parity_slice.py:_symmetric_mean_distance` (L495) | REUSE DIRECTLY | The matching rule for LCFS: a symmetric mean nearest-neighbour distance between two point sets via `scipy.spatial.cKDTree` (`0.5*(query(left→right).mean() + query(right→left).mean())`) — no ordering or arclength correspondence is imposed, which is why a globally-shifted or size-mismatched contour degrades this metric much faster than a single-point axis/X-point comparison. |
| `nova/equilibrium/stencil_nulls.py:xpoint_candidates` (L1721), `magnetic_axis_subgrid` (L1763) | REUSE DIRECTLY | The single-point candidate finders behind the axis and X-point metrics — subgrid-refined stationary points on the flux surface, not a contour or nearest-neighbour match, which explains why they can be accurate while the LCFS contour (a shape match, not a point match) is not. |
| `nova/equilibrium/topology.py:x_point`, `x_point_data`, `x_point_index` (L153-166) | REUSE DIRECTLY | The topology-labeling layer that selects which candidate X-point is reported (polarity- and psi-ordering aware), sitting between `xpoint_candidates` and the scorecard row. |

**Verdict:** the LCFS metric is a *shape*-matching metric (longest contour, symmetric
nearest-neighbour distance, no correspondence constraint) while axis/X-point are
*point*-matching metrics (subgrid-refined stationary points). A recovered root
that is accurate near the two stationary points but has a differently-shaped or
offset separatrix will show exactly the observed pattern (axis/X-point tight,
LCFS loose) — this is a representability question about the contour shape, not
necessarily a reference-completeness question. All internals needed to
interpret the `22086/43` numbers already exist and are named above; no new
code is required to answer question 2, only a diagnostic script that calls
`_contour` on both the recovered and reference maps and inspects the two point
sets directly (e.g. plotting them, or computing per-point nearest-neighbour
distance instead of the aggregate mean).

## (3) The same-shot warm-neighbour offset ladder (commit 818722b3)

**Capability:** name the exact functions in the DIII-D warm-neighbour ladder,
state which parts are machine-agnostic vs. DIII-D-corpus-specific, and whether
MAST's efm reader exposes a per-shot walkable time sequence.

| module:symbol | machine-scope | verdict | reason |
| --- | --- | --- | --- |
| `benchmarks/diiid_constrained_cold_start.py:NEIGHBOUR_FRAME_OFFSETS` (L67), `_neighbour_candidates` (L376) | machine-agnostic | REUSE DIRECTLY | A symmetric geometric offset ladder `(-1,1,-2,2,-4,4,-8,8,-16,16,-32,32)` applied to an integer frame index bounded by `[0, count)` — no DIII-D-specific data touched. |
| `benchmarks/diiid_constrained_cold_start.py:_find_warm_source` (L385) | machine-agnostic | REUSE DIRECTLY | Walks `_neighbour_candidates`, calling `_solve_public_seam` on each candidate frame until one converges, returning the first qualified warm source — generic over any object shaped like `PreparedFrame`. |
| `benchmarks/diiid_constrained_cold_start.py:_solve_public_seam` (L222), `solve_frame` (L444), `ROUTE_NAMES` (L70) | machine-agnostic | REUSE DIRECTLY | Drives `ForwardProfile.solve_branch` (the same public seam named in area 4) with a declared target current and a warm-start state; nothing here reads a DIII-D column name. |
| `benchmarks/diiid_constrained_cold_start.py:PreparedFrame`, `prepare_frame` (L75-166), `_fixed_wiring_adapter` (L167) | DIII-D-corpus-specific | UNFIT for direct reuse, EXTEND IN PLACE as a pattern | Reads DIII-D-only columns (`_CURRENT_COLUMNS`, `_GEOMETRY_COLUMNS`, `_LABEL_COLUMNS` from `benchmarks.diiid_forward_gs_match`) and DIII-D circuit/geometry description (`nova.imas.diiid_current`, `nova.imas.diiid_description`). A MAST equivalent needs its own `PreparedFrame`-shaped builder, not this one. |
| `benchmarks/efit_forward_parity_slice.py:_mast_case_from_selection` (L2438) | MAST-specific reader | REUSE DIRECTLY as the MAST analogue of `prepare_frame` | Opens `zarr.open_group(store / f"{shot}.zarr")["efm"]` and indexes `group["time"][row]`, `group["magnetic_axis_r"/"z"][row]`, `group["plasma_current_c"][row]` by an integer `row` — **this confirms the MAST efm reader already exposes a per-shot, per-row-indexable time sequence** structurally equivalent to DIII-D's frame index, so the offset-ladder walk in `_neighbour_candidates`/`_find_warm_source` is directly portable once a `PreparedFrame`-shaped wrapper around this zarr group exists. |

**Verdict:** the ladder mechanism itself (offsets, candidate walk, public-seam
solve) is fully machine-agnostic and reusable without modification. Only the
per-machine frame-preparation layer (`PreparedFrame`/`prepare_frame` for
DIII-D) is corpus-specific, and MAST already has the structural equivalent
(`group["time"][row]`-indexable zarr groups) needed to write a MAST
`prepare_frame`. No blocker exists to applying the ladder to the five stalled
frozen-six references.

## (4) The frozen-six selection, qualification and passive-inclusive current policy (`benchmarks/efit_forward_parity_slice.py`)

| module:symbol | verdict | reason |
| --- | --- | --- |
| `select_slices_by_shot` (L234), `_qualification` (L182), `select_slice` (L207) | REUSE DIRECTLY | The frozen-six row selection and pre-solve qualification (declared-boundary/map-saddle offset and stored-LCFS-contour discrepancy bounds) — already the exact selection the constrained arm consumes via `run_current_constrained`. |
| `_passive_inclusive_case` (L2715), `_digest_prescribed_response_inputs` (L2562), `_stored_circuit_fields` (L2596) | REUSE DIRECTLY | The passive-inclusive, 101-circuit prescribed-current policy (13 active + 88 passive/vessel through one exact-kernel response matrix) that the constrained arm's `_passive_inclusive_solve` (L3307) builds on. |
| `run_current_constrained` (L3765) | REUSE DIRECTLY | The constrained-arm driver itself — calls `profile.operator.prescribed_current_field`, then `_passive_inclusive_solve(..., target_current=target_current)`, which reaches the public seam `ForwardProfile.solve_branch(target_current=...)` — this is the named public entry point the constrained arm drives. |
| `_pinned_metrics` (L922), `_metric_qualification` (L3471) | REUSE DIRECTLY | The metric computation (flux map, axis, LCFS, X-point, plasma current, poloidal beta, internal inductance) and per-metric tolerance qualification consumed by the constrained-arm scorecard rows. |
| `docs/figures/current-constrained-forward-solve/mast-constrained/current-constrained-frozen-six-scorecard.json` | REUSE DIRECTLY | The constrained-arm's own banked scorecard — the receipt both parity-lane questions (2) and (3) must diff against when new runs are made. |

**Verdict:** `ForwardProfile.solve_branch` with a `target_current=` keyword is
the public seam the constrained arm drives; the frozen-six selection,
qualification and passive-inclusive current policy the constrained arm
consumes are all already implemented in this one file and require no new
machinery for either parity-lane question.

## (5) Search of imas-ambix and reckon for existing (1)/(2) capability

Searched both directions (`jax.linearize`, `symmetric_mean_distance`,
`parity_tolerances`, `LCFS_DISTANCE`, and capability-shaped terms
`one-application`, `forward residual`, `constrained parity`) across
`~/Code/imas-ambix` and `~/Code/reckon`.

| path | verdict | reason |
| --- | --- | --- |
| `~/Code/imas-ambix/scripts/tau_map_accelerator_probe.py:make_nk_runner` | UNFIT | Uses `jax.linearize` for its own Jacobian-free Newton-Krylov step (`newton_step`) inside a Newton-Krylov accelerator probe, not a zero-solve residual-at-reference measurement — different purpose (in-loop linearization vs. a single fixed-state diagnostic), and the map it linearizes is ambix's own psi-map surface, not nova's `ForwardOperator`. |
| `~/Code/imas-ambix/imas_ambix/camdyn/reveal_oracle.py` | UNFIT | Matched only on generic capability terms ("forward", "residual"); this is a camera-dynamics filament-placement floor diagnostic — a wholly different physics domain (imaging, not equilibrium reconstruction) with no equilibrium-metric or flux-map content. |
| `~/Code/reckon` (both search patterns) | UNFIT (no matches) | No hits for `jax.linearize`, `symmetric_mean_distance`, or `parity_tolerances` anywhere in the reckon tree — reckon owns plan/crew infrastructure only, as expected per the repo's coupled-repository boundary (`AGENTS.md`: reckon supplies docs/state tooling, not equilibrium capability). |

**Verdict:** no duplication and no reusable capability for (1) or (2) exists on
either side of the ambix or reckon seam. Both capabilities should stay owned
here.

## Path verification

```
$ for p in \
    benchmarks/efit_forward_parity_slice.py \
    docs/figures/efit-forward-parity/mast-dina-composition-diff.json \
    docs/figures/efit-forward-parity/boundary-imbalance-attribution.json \
    nova/imas/parity_tolerances.py \
    nova/equilibrium/stencil_nulls.py \
    nova/equilibrium/topology.py \
    benchmarks/diiid_constrained_cold_start.py \
    docs/figures/current-constrained-forward-solve/mast-constrained/current-constrained-frozen-six-scorecard.json; do
  test -e "$p" && echo "OK $p" || echo "MISSING $p"
done
OK benchmarks/efit_forward_parity_slice.py
OK docs/figures/efit-forward-parity/mast-dina-composition-diff.json
OK docs/figures/efit-forward-parity/boundary-imbalance-attribution.json
OK nova/imas/parity_tolerances.py
OK nova/equilibrium/stencil_nulls.py
OK nova/equilibrium/topology.py
OK benchmarks/diiid_constrained_cold_start.py
OK docs/figures/current-constrained-forward-solve/mast-constrained/current-constrained-frozen-six-scorecard.json
```

8/8 cited paths verified present.
