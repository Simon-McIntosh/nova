NEEDS-HELP: The banked margin-receipt egress is outside this node's exclusive write fence.

tried: Read the live plan and traced the ten-terminal banked artifact back to `benchmarks/diiid_forward_gs_match.py`; `_solve_surface_frame` manually constructs each `margin_graded` record and serializes only `terminal_class_margin`, while none of the five authorized files participates in that JSON construction.
options: (1) extend this node's scope to `benchmarks/diiid_forward_gs_match.py` and its focused receipt test or artifact output; (2) dispatch a separate receipt-egress node owning that benchmark while this node implements the two in-fence test closures and core diagnostic payload; (3) narrow the requirement to core diagnostic availability, explicitly waiving the statement that regenerated banked receipts carry the fields.
leaning: Option 1, because the benchmark is the sole authoritative egress and a small additive record update can be regenerated against one banked terminal without changing solver behavior.
cost-if-wrong: If only core diagnostics are added, the authoritative banked receipt remains unchanged and the stated serialization measure is falsely reported complete; if the benchmark is edited without a scope grant, this node violates the exclusive-write contract and risks collision with unannounced work.

node: xpoint-spatial-evidence
status: blocked
commits: none
changed_paths: none in the worker worktree; this manifest only
tests: not run because the exact-scope stop condition was reached before implementation
test_logs: none
artifacts: `/home/ITER/mcintos/Code/nova/docs/state/nova/manifests/xpoint-spatial-evidence.md` — scope-blocker trace identifying the sole receipt egress
evidence_inputs: The live plan requires terminal receipts to carry selected X coordinate, flux operand, and admitted candidate diagnostics with zero numeric drift in existing entries. The banked artifact is `docs/figures/plateau-input-attribution/margin-frame-remeasure.json`. Its records are manually assembled in `benchmarks/diiid_forward_gs_match.py::_solve_surface_frame`; the existing continuous-margin branch adds `terminal_class_margin` and aggregate selected-margin distributions but no coordinate, raw X flux, normalized X operand, or candidate table. `nova/equilibrium/connectivity_boundary.py` computes candidate data, but `ForwardFluxOperator._connectivity_class_margin` returns only the scalar margin and the fenced files cannot alter the benchmark's JSON record. The worktree was clean at discovery; no source edits were made.
follow_ons: After scope resolution, implement the batched `Topology.read_batch` primary-X coordinate-and-flux parity on double-null and near-tangency fixtures; add receipt diagnostics at the authoritative benchmark egress; make `test_jit_vmap_grad_safe_and_fixed_shape` call `configure_dtypes()`; run isolated and focused checks plus one-terminal receipt regeneration.
blockers: Required write access to `benchmarks/diiid_forward_gs_match.py` (and, if persisted, the selected receipt artifact path or a temporary-output authorization) is absent from the exclusive scope.
