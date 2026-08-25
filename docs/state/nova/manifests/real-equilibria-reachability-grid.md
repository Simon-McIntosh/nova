NEEDS-HELP: The real-field run cannot seed the pre-saddle axis component because the nearest grid cell to the continuous axis lies outside the inward-offset confined mask.
tried: Implemented all 12 persisted-carrier MAST terminal reconstructions, one regenerated DIII-D terminal, and the banked ITER hybrid EQDSK; the first run exhausted login-GPU memory, and the preallocation-disabled rerun reached classification before raising `RuntimeError: magnetic-axis seed is absent from pre-saddle region` at `_classify` line 277.
options: (1) seed from the nearest confined cell to the continuous axis, matching the landed private-flux adjudication; (2) bilinearly evaluate the continuous-axis flux and reduce the inward offset only enough to retain its nearest cell; (3) use the topology reader's connected-core seed/mask directly if it exposes the exact pre-saddle state.
leaning: Option 1, because the landed real-equilibrium adjudication already uses nearest-confined-cell seeding and records the axis-to-cell distance, while preserving the exact component definition and avoiding a changed flux threshold.
cost-if-wrong: The generated grid and JSON must be rerun; no banked source receipt or production code has been changed, and the only uncommitted implementation path is the fenced generator.

node: real-equilibria-reachability-grid
status: blocked
commits: none
changed_paths: docs/figures/primary-xpoint-evidence/real_equilibria_reachability.py
tests: `ruff format` completed; `ruff check` still requires a final post-edit run; generation failed after reaching the first real MAST pre-saddle classification
test_logs: /home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T140745454321-real-equilibria-reachability-grid/generate.log; /home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T140745454321-real-equilibria-reachability-grid/generate-preallocate-disabled.log
artifacts: none; the failed run produced neither PNG nor JSON
evidence_inputs: The available ITER equilibrium is `iterhybrid_cocos17.eqdsk`, whose own EQDSK limiter contour supplies a compatible wall; the default `iter_md` wall at 116000/2 was opened through imas-python at its written DD 3.37.0 but is geometrically incompatible with this equilibrium. The generator is designed to reconstruct 12/12 MAST arms from the persisted response carrier, regenerate 1/10 DIII-D terminals, and force MAST 21983/35 plus 21989/55 into a topology-spanning panel selection. GPU preallocation must be disabled on the login-node GPU (`XLA_PYTHON_CLIENT_PREALLOCATE=false`) because the default allocator exhausted available memory.
follow_ons: Apply nearest-confined-cell axis seeding, record axis-to-cell distance per panel, then rerun once with GPU preallocation disabled; no scope change is required.
blockers: Dispatch stop rule triggered after two executions of the generator required different fixes; the remaining exact unmet condition is a valid positive component label for the continuous magnetic-axis seed at the pre-saddle level.
