# Thin interfaces to shared infrastructure

Node `replan-thin-interfaces`, plan section `nova:nova-portfolio-replan` §2.
Scope: find where nova reimplements, or couples tightly to, shared machinery, and name
the refactor that reduces each coupling to a thin interface, with a one-line done-when.

**Tree measured:** worktree `.reckon-worktrees/nova-a0f1e0938fc2/s21-nova/replan-thin-interfaces`
at `a71934910d2a0835dd9f6ab35576eef5c2f44754` (2026-10-01). Read-only: no source edits,
no GPU, no SLURM jobs. Every count below is the stated bounded grep, run from the worktree
root over the named subtrees listed in the command — no whole-tree walk. Sizes are the
quoted output of the file commands shown. A line count measures lines carrying the
matched literal (including comments and docstrings); it bounds the refactor's surface,
not its difficulty.

Areas follow plan §2. Section 6 cross-references the refactors the handoff named
(plan followup `f-replan-004`). One of them — the solve clip mode — sits outside the
five shared-infrastructure areas and is written up there.

## 1. reckon crew and plan layer (scripts, docs/state, receipts)

### 1.1 The committed crew-state mirror

`docs/state/nova/` commits reckon's crew state into the product repo:

| path | size / count (`ls -l`, `find docs/state/nova/manifests -type f \| wc -l`) |
| --- | --- |
| docs/state/nova/crew.json | 15,277,373 B |
| docs/state/nova/index.json | 92,999 B |
| docs/state/nova/review.html | 73,201 B |
| docs/state/nova/timeline.html | 43,614 B |
| docs/state/nova/project.json | 608 B |
| docs/state/nova/manifests/ | 39 files, 149 KB |

It is load-bearing: plan followups cite it by line — `docs/plans/figure-and-solver-audit.html:230`
cites `docs/state/nova/crew.json:8673`, line 235 cites `:201511`,
`docs/plans/millisecond-converged-solve.html:278` cites `:214737`, and
`docs/plans/discrete-operator-analytic-error.html:427` cites
`docs/state/nova/manifests/recovery-regression-attribution.md`. A line number inside a
15 MB machine-generated file is not a durable citation.

- **Refactor.** Serve crew state from the crew data directory instead of committing it
  into the product repo; keep the citation surface at (node, run id, field) so a reader
  resolves a row without a line number, and retire or redirect the four line-cited
  references above when the mirror goes.
- **Done-when.** `docs/state/nova/` holds no crew runtime state, and every plan citation
  resolves by node/run/field rather than by a `crew.json` line number.
- **Owner.** unowned — no live followup names `docs/state`.

### 1.2 Committed receipts carry absolute worktree paths, and the cut-cell family is 263 MB (handoff refactor 3)

Measured:

- `grep -rn 'reckon-worktrees' docs/figures/ --include='*.json' -l | wc -l` → **413** JSON
  files under `docs/figures/` carry a worktree path.
- `grep -rnE '/\.reckon-worktrees/|/\.cache/reckon-worktrees' nova/ tests/ benchmarks/ scripts/ | wc -l`
  → **177 matching lines across 63 files**. Source (not log) carriers: `nova/scripts/render_misfit_figures.py`,
  `benchmarks/pure_arm_comparison_figure.py`, `benchmarks/backend_divergence_forensics.py`,
  `benchmarks/constraint_centroid_receipt.py`, `scripts/run_amplitude_census.sh`.
- Per-file samples (`grep -c reckon-worktrees <file>`): `docs/figures/cut-cell-current-attribution/exact-support-floor/report.json` 2,
  `docs/figures/cut-cell-current-attribution/dual-stencil/receipt.json` 1.

Sizes, `find docs/figures -name '*.json' -size +5M -printf '%s %p\n' | sort -rn`:

| receipt | bytes (MiB) |
| --- | --- |
| cut-cell-current-attribution/exact-support-floor/report.json | 74,967,608 (71.5) |
| null-identification-authority/held-tip-diagnosis/held-tip-trip-diagnosis.json | 32,118,539 (30.6) |
| cut-cell-current-attribution/dual-stencil/receipt.json | 25,602,195 (24.4) |
| cut-cell-current-attribution/exact-support-floor/weak-rotation-reactor-static-cells-2500.json | 17,356,992 (16.6) |
| cut-cell-current-attribution/exact-support-floor/diverted-single-null-cells-2500.json | 16,389,436 (15.6) |
| null-identification-authority/held-tip-diagnosis/containment.json | 13,420,079 (12.8) |
| null-identification-authority/held-tip-diagnosis/integrated.json | 13,417,670 (12.8) |
| gs-absolute-accuracy/solovev-certificate-production-route.json | 12,709,774 (12.1) |
| cut-cell-current-attribution/spline-read/receipt.json | 10,436,734 (10.0) |
| mast-catalog-gpu-solve/mast-catalog-throughput-inputs.json | 10,019,705 (9.6) |
| jax-dissolution/fieldnull_candidate_audit.json | 9,705,521 (9.3) |
| cut-cell-current-attribution/exact-moment-stages/report.json | 9,533,684 (9.1) |
| polish-support-performance/candidate-{3,5,6}/trip-census-candidate.json | ~6.7 MB each |

The whole cut-cell family: `find docs/figures/cut-cell-current-attribution -name '*.json' -printf '%s\n' | awk '{s+=$1} END {printf "%.1f MB across %d files\n", s/1048576, NR}'`
→ **263.0 MB across 558 files**. The handoff's "59 to 71 MB" claim is confirmed by the
71.5 MiB `report.json`; the family total is larger than any single file.

- **Refactor.** Each receipt keeps a per-row summary plus an npz payload, and every path
  it stores is repo-relative (the shape named by `f-cca-slim-committed-attribution-receipts`).
  An audit re-opened from a fresh clone then resolves its inputs without the producing worktree.
- **Done-when.** Every committed receipt resolves its inputs by repo-relative path and
  re-running its audit from a fresh clone reproduces its rows; the stated grep over
  `docs/figures` returns zero files.
- **Owner.** cut-cell-current-attribution `f-cca-slim-committed-attribution-receipts`
  (open; `docs/plans/cut-cell-current-attribution.html:1179`) for the cut-cell family;
  forward-solve-api `f-fsa-repair-coil-edit-latency-numbers` (open;
  `docs/plans/forward-solve-api.html:147`) for the `figure`-field repair; the
  null-identification-authority, gs-absolute-accuracy, jax-dissolution, mast-catalog-gpu-solve
  and polish-support-performance families are unowned.

### 1.3 Benchmarks read the crew run/report directories by absolute path

`grep -rEc '/\.config/reckon' nova/ benchmarks/ tests/ scripts/ --include='*.py' --include='*.sh' | grep -v ':0$'` →
**49 files**. Named defaults: `benchmarks/efit_reproduction_gate.py:49`
(`/home/ITER/mcintos/.config/reckon/crew/runs/`), `benchmarks/production_solve_host_profile.py:56,70`
(`…/crew/reports/nova/millisecond/`), `benchmarks/exact_clip_moment_floor.py:64,70,74`
(`…/crew/reports/nova/s19-local/exact-gauss`, `…/s19-review/`), `benchmarks/trip_quantum_profile.py:48,52`
(`…/crew/reports/nova/attribution/`), and 45 further files at 1–2 lines each, including
`tests/test_coil_edit_persistence_guard.py`, `tests/test_batched_operator_boundary.py`,
`tests/test_stationary_point_admission.py`, `scripts/run_amplitude_census.sh`,
`scripts/labeller_batch/shard.py`.

- **Refactor.** One resolver for receipt roots (environment override, repo-relative cache
  default); a benchmark takes its input directory as an explicit argument instead of
  reaching into the coordinator's private run/report tree it does not own.
- **Done-when.** The stated grep returns zero lines at the merged head.
- **Owner.** unowned.

### 1.4 Four crew runtime defects, and a plan write that resolved to the main checkout (handoff refactor 4)

Reported in the plan at `docs/plans/nova-portfolio-replan.html:106`: supervisors that
outlive their worker; codex review manifests being rejected; a review sweep that ignores
the pause; orphaned followers holding the lock. The fifth defect — a worker's plan write
resolving to the main checkout instead of its worktree — is documented in
`docs/plans/plasma-cell-read-fidelity.html:370,385`, where the seed-policy record landed
on `main` at `edbd943f7b5e01a707bf99e0ddab1f2e167ea18d` ("docs(plans): record the
seed-policy measurement at the reduced rung", 2026-09-23) with the worker's own worktree
holding no record.

- **Refactor.** Fix upstream in reckon; nova keeps no workaround.
- **Done-when.** Each defect carries a landed upstream fix or a recorded won't-fix, and a
  dispatched worker's plan write cannot resolve outside its own worktree (the write
  refuses rather than falling back to the main checkout).
- **Owner.** upstream (reckon): the four are tracked in reckon's crew plans matched by
  keyword — `a-dispatch-returns-without-its-worker`, `a-review-dies-where-nobody-is-looking`,
  `crew-runtime-positive-controls`, `a-crew-process-ends-with-its-owner`. No nova-side
  followup; nothing in nova's own scripts invokes the reckon CLI
  (`grep -rn 'reckon ' scripts/ --include='*.sh'` → 0 lines).

## 2. Local model lane

`grep -rniE 'lane\.json|imas-ambix/lane|v4\.1-flash|deepseek|ollama|vllm' nova/ tests/ benchmarks/ scripts/`
→ **0 lines**. Positive control: `grep -rnE 'localhost|https?://'` over the same subtrees
returns hits (e.g. `nova/imas/mast_seed_parameters.py:49`), so the instrument sees content
where content exists. The only loopback literals in the tree are nova's own playable
server (`tests/test_playable_session.py:707,733,759`, `scripts/playable_server/payload.sh:60`),
a nova-owned HTTP surface — not the model lane.

- **Refactor.** None; handoff item 5 confirmed for the model lane.
- **Done-when.** The stated grep remains empty at the merged head.
- **Owner.** none (verified absent).

## 3. Fleet and SLURM placement

### 3.1 The H200 lane carries its own submission vocabulary

`grep -rEc 'betelgeuse|gpu_0003_grpA|--mem=|--cpus-per-task|--gpus=|--gres=|sbatch|srun|TMPDIR' scripts/h200_test_lane 2>/dev/null` ;
the lane directory holds 1,210 lines total:

| file | coupled lines | total lines |
| --- | --- | --- |
| scripts/h200_test_lane/run.sh | 11 | 192 |
| scripts/h200_test_lane/prewarm.sh | 9 | 244 |
| scripts/h200_test_lane/cache_guard.py | 0 | 367 |
| scripts/h200_test_lane/cache_reuse_summary.py | 0 | 247 |
| scripts/h200_test_lane/prewarm_row.py | 0 | 92 |
| scripts/h200_test_lane/publish_prewarm_pin.py | 0 | 68 |

Coupled lines named: `run.sh:5` (`PYTHON=/home/ITER/mcintos/Code/nova/.venv/bin/python`),
`:9` (`DEFAULT_PINNED_ROOT=/work/projects/imas_gpu/sophelio/jax-cache/nova-prewarm`),
`:28` (`export TMPDIR=/tmp`), `:78` (`srun --ntasks=1 --cpus-per-task=7`), `:145-161`
(`sbatch --partition=betelgeuse --reservation=gpu_0003_grpA --cpus-per-task=7
--gpus=h200:1 --mem=64G --time=01:00:00`). No `titan` or `all_debug` token appears
anywhere in the lane directory, so the fallback rungs of the fleet hierarchy cannot be
taken from it.

- **Refactor** (handoff refactor 2). One nova lane launcher generalising `run.sh`, taking
  a script, a rung preference (H200, then titan, then all_debug) and mem/cores, with the
  rules built in: a real submission rather than `--test-only`, never `--mem=0`,
  `TMPDIR=/tmp`, no `uv` on compute nodes.
- **Done-when.** One file carries the partition/reservation literals, and briefs cite the
  launcher instead of a hand-written sbatch recipe.
- **Owner.** forward-solver-route-integrity `f-fsri-one-nova-lane-launcher` (open;
  `docs/plans/forward-solver-route-integrity.html:512`).

### 3.2 Every other heavy caller writes its own payload

`grep -rEc 'sbatch|srun' scripts/ benchmarks/` (nonzero): `benchmarks/traced_current_bitwise.py` 6,
`benchmarks/raster_flux_target_receipt.py` 6, `benchmarks/coil_edit_latency.py` 6,
`scripts/regenerate_solovev_certificate.sh` 4, `scripts/h200_test_lane/run.sh` 3,
`benchmarks/traced_vector_identity_receipt.py` 3, `benchmarks/cpu_solve_discriminator.py` 3,
`scripts/playable_server/run.sh` 2, `scripts/h200_test_lane/prewarm.sh` 2,
`scripts/geometry_service_benchmark/bench.sbatch` 2, and 1 each in
`scripts/window_gpu_benchmark/bench.sbatch`, `scripts/ensemble_window_throughput/bench.sbatch`,
`scripts/playable_server/payload.sh`, `scripts/labeller_parallel/run.sh`,
`scripts/labeller_batch/run.sh`, `benchmarks/topology_batch_probe.py`,
`benchmarks/poloidal_convergence_atlas.py`, `benchmarks/playable_keyframe_receipt.py`,
`benchmarks/kernel_cost_sweep.sh`, `benchmarks/forward_labeller_throughput.py` — **21 files**
(20 code payloads; `scripts/geometry_service_benchmark/report.md` carries the literal in
prose).
Nothing under `nova/` or `tests/` submits: `grep -rn 'sbatch\|srun' nova/ tests/ --include='*.py'`
returns only `nova/thermalhydralic/naka/database.py:101-102`, where `srun` is a shot-label
variable, not a submission.

Lane literals inline (`grep -rn -E -- '--partition=|--reservation=|--mem=|--gres=' scripts/`):
`betelgeuse` + `gpu_0003_grpA` in `regenerate_solovev_certificate.sh:183-187`,
`playable_server/run.sh:20-21,145-151`, `labeller_parallel/run.sh:102-106`,
`labeller_batch/run.sh:133-137`, `window_gpu_benchmark/bench.sbatch:3-9`,
`ensemble_window_throughput/bench.sbatch:3-9`, `geometry_service_benchmark/bench.sbatch:3-9`;
`all_debug` in `run_amplitude_census.sh:2-5`. `--mem=0` never appears:
`grep -rn -- '--mem=0' nova/ tests/ benchmarks/ scripts/ | wc -l` → **0** — the rule holds
today and the launcher must keep it true.

- **Refactor.** Every payload becomes a caller of the one launcher (3.1), or takes its
  partition/reservation/mem through the launcher's declared configuration.
- **Done-when.** As 3.1: one file carries the literals; the stated grep over `scripts/`
  returns only launcher configuration.
- **Owner.** `f-fsri-one-nova-lane-launcher` (same as 3.1).

### 3.3 Partition literals pinned in tests and briefs

`grep -rEc 'betelgeuse|gpu_0003_grpA|all_debug' tests/` → 5 files:
`tests/test_certificate_baseline_revision.py` 1, `tests/test_solve_program_size.py` 2,
`tests/test_playable_server.py` 2 (asserts `PLAYABLE_PARTITION == "betelgeuse"`,
`PLAYABLE_RESERVATION == "gpu_0003_grpA"` against `scripts/playable_server/run.sh:20-21`),
`tests/test_playable_session.py` 1, `tests/test_strict_exit_incidence.py` 2
(`SLURM_JOB_PARTITION == "all_debug"`). Briefs carry hand-written recipes: root
`AGENTS.md` (the compute hierarchy and its sbatch recipe), `nova/equilibrium/AGENTS.md:217`
(one `all_debug` allocation), `nova/biot/AGENTS.md:55-72` (titan/betelgeuse guidance and
`/work/projects/imas_gpu` mount notes).

- **Refactor.** Tests assert against the launcher's declared defaults rather than pinning
  strings independently; briefs cite the launcher instead of restating a recipe.
- **Done-when.** A rung change edits one file, not a test assertion and a brief.
- **Owner.** `f-fsri-one-nova-lane-launcher` for the briefs; the test defaults follow
  their launcher (no separate followup).

## 4. imas-python data access

### 4.1 DD access goes through imas-python; the constructor is called from 15 modules

`grep -rEc 'imas\.DBEntry|import imas' nova/ | grep -v ':0$'` → **15 files** reference the
library directly: `nova/imas/db_entry.py` 3 (line 15 subclasses `imas.DBEntry`),
`mast_solve_input_ids.py` 4, `dataset.py` 4, `mast_geometry.py` / `magnetics.py` /
`diiid_machine_ids.py` 3 each, `diiid_description.py` 2, `metadata.py` 1,
`diiid_passive.py` 1 — and six modules outside `nova/imas/`:
`nova/scripts/diiid_machine_artifact.py` 5, `nova/datachain/ccfe_uda.py` 3,
`nova/media/sources/diiid_efit.py` 2, `nova/io/egress.py` 2,
`nova/equilibrium/steering_frames.py` 2, `nova/field/errorfield.py` 1 (import only).
No local Data Dictionary loader or storage reimplementation was found under `nova/`;
reads and writes go through `imas.DBEntry` with an explicit `dd_version` at the call
site. The handoff's "no reimplementation" holds; what remains is constructor fan-out.

- **Refactor (light).** Construct through `nova.imas.DBEntry` (or one factory) so the
  dd_version/URI policy lives in a single adapter, and direct `import imas` appears only
  under `nova/imas/`. No behaviour change intended.
- **Done-when.** `grep -rn 'import imas' nova/ --include='*.py'` returns only files under
  `nova/imas/`.
- **Owner.** unowned.

### 4.2 One direct h5py read of an IMAS-written file

`grep -rn 'h5py' nova/ --include='*.py'` → only `nova/imas/hdf5_read.py`: line 1
`import h5py`, line 16 `h5py.File(f"{database.ids_path}/equilibrium.h5")` — a read of the
imas-python backend's on-disk layout behind `nova.imas.Database`, bypassing `DBEntry`.
The file is tracked (`git ls-files nova/imas/hdf5_read.py`), 13 lines, with no callers
found in the stated greps. The other `h5py` hits in named subtrees are reads of nova-owned
stores (`benchmarks/poloidal_convergence_atlas.py:743-749` opens the writer-replay steering
store), which is not an IMAS read.

- **Refactor.** Delete the module or rewrite its read through `Database().get_ids(...)`.
- **Done-when.** `grep -rn 'h5py' nova/` returns only nova-owned stores (or nothing).
- **Owner.** unowned.

### 4.3 IMASDB path convention

`nova/imas/connect.py:117` composes `/home/ITER/{username}/public/imasdb/{machine}/`.
Counted in area 5 (5.1); it is the one imas-adjacent hard-coded path.

## 5. Hard-coded GPFS paths

`grep -rEc '/work/projects|/home/ITER' nova/ benchmarks/ tests/ --include='*.py' --include='*.sh'`
→ **119 files, 177 lines** in source files: nova/ 10 files / 12 lines, benchmarks/ 97 files /
148 lines, tests/ 12 files / 17 lines.

### 5.1 Product code under nova/

| file | coupled lines | literal |
| --- | --- | --- |
| nova/catalog/mast_geometry.py | 2 | `DEFAULT_LEVEL1_ROOT` / `DEFAULT_LEVEL2_ROOT` under `/work/projects/imas_gpu/mast/level{1,2}/shots` |
| nova/scripts/render_misfit_figures.py | 2 | `/home/ITER/mcintos/.cache/nova-mast/...` and a `.cache/reckon-worktrees` path |
| nova/imas/mast_vacuum_cohort.py | 1 | `SHOT_STORE = /work/projects/imas_gpu/mast/level1/shots` |
| nova/imas/mast_fitted_parameters.py | 1 | same level-1 store |
| nova/imas/diiid_current.py | 1 | `/home/ITER/tribolp/Public/imasdb/DIII-D/200000.nc` |
| nova/imas/diiid_machine_ids.py | 1 | same DIII-D netCDF |
| nova/imas/connect.py | 1 | public imasdb path convention |
| nova/media/sources/nova_labels.py | 1 | `/work/projects/imas_gpu/sophelio/labeller_sessions/76906a29` |
| nova/media/sources/diiid_efit.py | 1 | the DIII-D netCDF |
| nova/scripts/identify_source_cocos.py | 1 | the MAST level-1 store |

`nova/catalog/mast_geometry.json` additionally carries 2 matching lines under
`grep -rEc ... nova/ --include='*.json'`.

- **Refactor.** One resolver module (e.g. `nova.paths`) owns the data roots — the current
  literals become documented defaults, overridable by environment or config — and every
  caller imports from it rather than restating the path.
- **Done-when.** `grep -rnE '/work/projects|/home/ITER' nova/` returns lines only in the
  resolver module (and docs).
- **Owner.** unowned.

### 5.2 Evidence tooling

benchmarks/: 97 files, 148 lines; top counts `benchmarks/diiid_poloidal_figures.py` 5,
`benchmarks/diiid_vacuum_composition_audit.py` 4, and 3 each in `trip_quantum_profile.py`,
`split_fit_measured_maps.py`, `diiid_negative_tail_attribution.py`, `diiid_current_polarity_audit.py`,
`axis_definition_referents.py`; the remainder at 1–2 lines each (the stated grep enumerates
them). tests/: 12 files, 17 lines — `tests/test_mast_catalog_geometry.py` 4,
`tests/test_wall_units.py` 2, `tests/test_batched_operator_boundary.py` 2, and 1 each in
`test_steering_frames.py`, `test_stationary_point_admission.py`, `test_media.py`,
`test_machine_artifact_concurrency.py`, `test_forward_support_clip_mode.py`,
`test_exact_clip_memory.py`, `test_efit_referee.py`, `test_coil_edit_persistence_guard.py`,
`test_chain_factory.py`.

- **Refactor.** Import roots from the resolver, or take a data root as an explicit
  argument; fixtures that must run against the shared corpus name the root once.
- **Done-when.** The 119/177 figure falls to the product resolver plus per-call arguments.
- **Owner.** unowned.

## 6. Cross-reference: the refactors named at the handoff

Plan followup `f-replan-004` (the thin-interfaces followup) names five refactors. Coverage:

| # | Refactor (as named) | Detailed in | Owning followup |
| --- | --- | --- | --- |
| 1 | Clip mode becomes a solve-request and receipt field | §6.1 | forward-solve-api `f-fsapi-clip-mode-is-a-request-field` (open; `docs/plans/forward-solve-api.html:182`) |
| 2 | One nova lane launcher | §3.1, §3.2, §3.3 | forward-solver-route-integrity `f-fsri-one-nova-lane-launcher` (open) |
| 3 | Receipts: per-row summaries plus npz, repo-relative paths | §1.2 | cut-cell-current-attribution `f-cca-slim-committed-attribution-receipts` (open); forward-solve-api `f-fsa-repair-coil-edit-latency-numbers` (open) |
| 4 | Four reckon crew defects (plus the plan-write target) | §1.4 | upstream reckon crew plans (named in the row) |
| 5 | imas-python and the local lane: no nova-specific coupling found | §4.1, §2 — confirmed, with the 13-line exception in §4.2 | unowned (the exception) |

### 6.1 Clip mode is a module global, not a request field

`grep -rEc '_SUPPORT_CLIP_MODE|support_clip_mode' nova/ tests/ benchmarks/ scripts/` →
**44 files**; top: `benchmarks/centroid_constrained_fixture_receipt.py` 27,
`benchmarks/solovev_certificate.py` 16, `nova/equilibrium/forward_operator.py` 15,
`benchmarks/oracle_start_newton_probe.py` 13, `tests/test_forward_support_clip_mode.py` 12,
`benchmarks/exact_clip_memory_scaling.py` 10, `tests/test_exact_clip_memory.py` 9. The
production pair: `nova/equilibrium/forward_operator.py:107` defines the process-global
`_SUPPORT_CLIP_MODE = "chord"` with `set_support_clip_mode`/`support_clip_mode` accessors
(lines 123–132), and the solve path reads it (`nova/equilibrium/forward.py:124,914`
branches on `support_clip_mode() != "exact"`). Tests and benchmarks mutate the global, so
the clip mode is an implicit process-wide request channel rather than an argument.

- **Refactor.** Make the mode an explicit field on the solve request and carry it onto the
  receipt; exact mode then gets a production caller before any cutover.
- **Done-when.** A solve cannot change clip behaviour without a request/receipt
  difference, and no module-global clip mode remains.
- **Owner.** forward-solve-api `f-fsapi-clip-mode-is-a-request-field` (open).

Note: the concurrent node `cca-chord-membership-applied-flag` edits
`nova/equilibrium/clip_quadrature.py`; this report is read-only and did not touch it.