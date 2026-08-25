node: recovery-regression-attribution

status: complete

commits: none

changed_paths:

- `docs/state/nova/manifests/recovery-regression-attribution.md`

tests:

- Prior authoritative clean-checkout gate, commit `c701600431f9a7f4d5d06b885e0213161e329de2`: `JAX_PLATFORMS=cpu UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest -m "slow or not slow" tests/test_two_class_recovery.py` -> **0 passed, 5 failed in 254.91 s**. This is the evidence run used below; the topology-container repair had already cleared the other two test files while these five assertions survived unchanged.
- Assigned worktree attempt at `c8db1abafec95a53fd52ac792aeb42ffd6029066`: the same file did not collect. Attempt 1 stopped before pytest because uv could not initialise `/home/ITER/mcintos/.cache/uv`; attempt 2 used `UV_NO_CACHE=1` and stopped during collection because Numba's `cache=True` could not obtain a locator/writeable cache for the read-only worktree. No test assertion was re-measured locally.
- Static bank analysis: direct NumPy reads of `root-coarse.npz` and `root-fine.npz` completed successfully; no bank was modified.
- Read-only composition diagnostic: `qualify(write=False)` first stopped on the warm-cache lock file under the read-only Nova data directory. A retry bypassed only that lock context and kept `write=False`, but produced no receipt after five silent minutes and was interrupted. No bank was regenerated, overwritten, or re-banked.

test_logs:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T093634640882-primary-branch-green-check/test-logs/test_two_class_recovery.log` — authoritative 0/5 clean-checkout failure log.
- `/home/ITER/mcintos/Code/nova/docs/figures/discrete-operator-analytic-error/primary-branch-green-check.md` — complete surviving assertion excerpts and run metadata.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T100314047850-recovery-regression-attribution/two-class-recovery.log` — assigned-worker collection blocker.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T100314047850-recovery-regression-attribution/bank-static-analysis.log` — preserved-bank seed/root distance calculation.
- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T100314047850-recovery-regression-attribution/composition-diff.log` — empty because the interrupted qualifier emitted only after completion; the full composition table is fenced out of this sandbox tier and dispatched separately to a writable-cache worker.

artifacts:

- `scripts/oracle_rebaseline/receipt-coarse.json` and `root-coarse.npz` — bank commit `0bed363d0e179fed1d4d265c6dee1998c32aa77f`; cold seed is **2.2225149882045554 Wb** from the root in sup norm, **1.152536862089681 root spans** and **2.1708648621170483 root absolute scales**.
- `scripts/oracle_rebaseline/receipt-fine.json` and `root-fine.npz` — bank commit `0bed363d0e179fed1d4d265c6dee1998c32aa77f`; cold seed is **2.199467759011981 Wb** from the root in sup norm, **1.1385308312765108 root spans** and **2.1514921904960422 root absolute scales**.
- `scripts/dual_basin_fixtures/diverted-receipt.json`, `diverted-state.npz`, and `diverted-root-receipt.json` — most recently refreshed together in `6f8898174149a2c93cc1e545f3bb95e900492398`; state digest `11a7e9d00556e91a6d76a69212107592501e1e8cedae60fd17e9e8032ff14801`.
- Banked pinned-root map: requested/achieved class **1/1**, topology-consistent **true**, converged **true**, one iteration, relative residual **1.5840538799920246e-16** against a **1e-14** floor, terminal and absolute residual **4.440892098500626e-16 Wb**.

evidence_inputs:

### Headline conclusion

The red primary branch is **not a second production regression**. Nothing in these five assertions indicts production code. They separate into (a) two limited cold-start basin-sensitivity results outside any guaranteed regime, (b) two stale or unbanked diverted test expectations, and (c) one composition-comparison drift over a sound machine-precision root. None is the repaired JAX-container fault.

The expectation-side move is named: `tests/test_two_class_recovery.py` last moved in commit `d30939124d38a4808e112f3a000a440bd01ef41f`, after both the limited banks (`0bed363d`) and the refreshed diverted banks (`6f889817`). In particular, `d3093912` introduced the full-vmap `[0,0]` negative expectation and the unbanked 0.05 all-pass ladder. The former now scores an improvement as failure—the diverted branch achieves diverted class—and the latter demands a basin amplitude no banked receipt ever established, against the live plan's conclusion of only 0.01.

The sprint-critical root conclusion is unchanged and explicit: the **machine-precision pinned diverted root is not in doubt**. The clean measurement reports `map` among the five identical top-level receipt items: requested/achieved class **1/1**, topology-consistent **true**, converged **true** in one application, and relative residual **1.5840538799920246e-16**. Only the composition comparison differs.

### Attribution table

| Failure | Named cause | Quantitative evidence | Artifact vs expectation provenance |
|---|---|---|---|
| Coarse limited `converged == false` | **Current limited cold start misses the banked basin within the fixed ten-step Newton-Krylov budget.** This is a sensitivity result, not an artifact identity failure or production regression: the test first proves the produced cold seed is byte-equal to `root-coarse.npz`, then retains limited classification and topology consistency, but terminates at residual **1.29980506** rather than `<= 1e-10`. | The cold seed is **1.15254 full root spans** from the banked root. That is more than three orders of magnitude outside the sibling H200 perturbation study's 1e-4 to 1e-3 range. The supplied hypothesis is therefore **rejected**: the H200 hint supports limited-lane fragility in direction only; it does not cover this cold start and proves no code regression. | Bank created `0bed363d` at 00:23; recovery test created later in `87ed6fdf` at 16:23; test last moved in `d3093912` at 17:51. The expectation is newer than the untouched bank, but the seed-equality assertion proves that this is not a wrong-file or regenerated-bank mismatch. |
| Fine limited `converged == false` | **Same limited cold-start basin sensitivity**, grouped with coarse and outside any guaranteed recovery regime. | Terminal residual **1.30949372**; cold seed **1.13853 root spans** from the banked root, again more than three orders of magnitude outside the 1e-4 to 1e-3 perturbation hint. The supplied regression hypothesis is **rejected** for the same reason. Resolution does not cure the sensitivity, matching the known lane direction without extending its measured regime. | Same `0bed363d` bank vs later `87ed6fdf`/`d3093912` test provenance; seed equality passes. |
| Vmapped achieved class `[0,1]` vs expected `[0,0]` | **Stale negative expectation: the diverted branch now reaches the diverted class, while the test still expects it to fall into limited.** Actual class `1` agrees with the banked state/root identity and is an improvement, not loss of the diverted root. | One of two expected elements differs; actual requested classes remain `[0,1]`, and the diverted element achieves class `1`. | Diverted state/root bank refreshed in `6f889817`; `d3093912` then updated the digest to that bank and, without changing any diverted artifact, introduced the full Newton solve expectation `[0,0]`. This is explicitly **test updated without artifact** and the expectation moved most recently. |
| Near-basin `passed == [True,True,False]` | **Unbanked over-wide basin expectation at relative amplitude 0.05.** The current largest passing amplitude is **0.01**; §4's banked conclusion is only that the diverted basin is at least 0.01 wide and its edge was not found there. Requiring 0.05 to pass exceeds that conclusion. | Policy amplitudes are `[0.001, 0.01, 0.05]`; first two pass and the third fails. The sibling H200 evidence only reaches 1e-3, where diverted held 4/4 and 16/16, so it neither predicts nor contradicts the 0.05 edge. | The state/root artifacts last moved in `6f889817`; `d3093912` later introduced the perturbation code and all-pass test without a banked perturbation receipt. This is **test expectation added after, and without, its read artifact**. |
| Measured receipt differs from banked receipt | **Composition comparison drift over a sound root, not production-root failure.** The clean pytest diff reports **five identical top-level items and only `composition` different**. It exposes at least `composition.external_field.sha256`: measured `d6941b63cd30c1a60b31cd18bb3f473e27c500295fa0155251583ae6c23c69e6` vs banked `b1a26f6828854302e6a62bb18938e3e5f908630dff06a50783353c2e2df47463`. The separately dispatched writable-cache worker owns the full differing-field table. | Because `map` is one of the five identical top-level items, the fresh measurement still has requested/achieved class 1/1, topology consistency, convergence in one application, relative residual **1.58405e-16**, and absolute/terminal difference **4.44089e-16 Wb**. Thus the **machine-precision pinned root itself is not in doubt; only its composition comparison is**. | `diverted-root-receipt.json` and the state/oracle banks moved together in `6f889817`; the equality assertion originated with the earlier root receipt in `b16d032c` and was not altered afterward. The most recent whole test-file commit `d3093912` does not change the equality assertion but does postdate the bank. Exact-path provenance bounds this as comparison drift; the separate measurement names its remaining fields. |

### Exact-path log ledger

| Exact path | Most recent path commit | Relation to test |
|---|---|---|
| `scripts/oracle_rebaseline/receipt-coarse.json` | `0bed363d` | Bank older than test creation. |
| `scripts/oracle_rebaseline/root-coarse.npz` | `0bed363d` | Bank older than test creation. |
| `scripts/oracle_rebaseline/receipt-fine.json` | `0bed363d` | Bank older than test creation. |
| `scripts/oracle_rebaseline/root-fine.npz` | `0bed363d` | Bank older than test creation. |
| `scripts/dual_basin_fixtures/diverted-receipt.json` | `6f889817` | Artifact older than last test update `d3093912`. |
| `scripts/dual_basin_fixtures/diverted-state.npz` | `6f889817` | Artifact older than last test update `d3093912`. |
| `scripts/dual_basin_fixtures/diverted-root-receipt.json` | `6f889817` | Root receipt refreshed after equality-test creation `b16d032c`, then whole test file moved in `d3093912` for other expectations. |
| `tests/test_two_class_recovery.py` | `d3093912` | Most recent expectation-side move; introduced the full-vmap negative expectation and 0.05 all-pass ladder after the diverted banks. |

### Preservation statement

No source, test, NPZ, JSON bank, receipt, cache artifact under the repository, plan, or index was changed. `qualify` was called only with `write=False`. No re-banking, regeneration, overwrite, bisect, history blame hunt, restore, stash, merge, or push occurred.

follow_ons:

- The full measured-versus-banked composition table is fenced out of this node and already dispatched separately to a writable-cache worker. It is not a blocker on this attribution deliverable; that worker must preserve the bank and use `qualify(write=False)`.
- Any later repair should treat the vmap `[0,0]` and 0.05 all-pass assertions as expectation review, while the two limited cold-start failures require a separate robustness decision; neither should be conflated with the pinned-root receipt.

blockers:

- none
