node: receipt-composition-delta

status: complete

commits:

- `33012360113fc6eca26a1fe369485d6fa35dcb0c`

changed_paths:

- `docs/figures/discrete-operator-analytic-error/receipt-composition-delta.md`

tests:

- `JAX_PLATFORMS=cpu XDG_CACHE_HOME=<writable-run-cache> NUMBA_CACHE_DIR=<writable-run-cache> UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync python <qualify-and-leaf-comparison-driver>` calling `qualify(write=False)` -> **PASS**, `EXIT_MARKER=0` in about 60 seconds; all 26 composition scalar leaves printed side by side.
- Preservation check: `git diff --exit-code -- scripts tests` -> **PASS**, no source, test, script, NPZ bank, or JSON receipt changed.
- Pre-stage path, label, deliverable-id, changelog-prose, and `git diff --check` checks -> **PASS**.

test_logs:

- `/home/ITER/mcintos/.config/reckon/crew/runs/r-20260825T102051495285-receipt-composition-delta/qualify-composition.log` — full `qualify(write=False)` output, all 26 measured/banked leaves, complete composition JSON objects, top-level equality, and `EXIT_MARKER=0`.

artifacts:

- `docs/figures/discrete-operator-analytic-error/receipt-composition-delta.md` — 26 composition leaves enumerated; 16 identical and 10 differing, comprising 2 raw-value SHA-256 digests and 8 numeric leaves representing 4 pinned/unpinned quantities. Largest coordinate delta `2.220446049250313e-16 m`; largest flux delta `8.673617379884035e-19 Wb`; composition closure unchanged at `4.440892098500626e-16 Wb`.

evidence_inputs:

- Verdict: drift is not confined to digests or provenance. It reaches roundoff-scale physical anchor values: axis radius `+2.220446049250313e-16 m`, axis height `+4.163336342344337e-17 m`, axis flux `-8.673617379884035e-19 Wb`, and boundary flux `+3.231430693775156e-19 Wb`, each repeated identically for pinned and unpinned anchors.
- `composition.external_field.sha256` measured `d6941b63cd30c1a60b31cd18bb3f473e27c500295fa0155251583ae6c23c69e6` versus banked `b1a26f6828854302e6a62bb18938e3e5f908630dff06a50783353c2e2df47463`.
- `composition.source_forcing.sha256` also differs: measured `7e3a16ff57c5ba6d85e3e3a6bf8de0d17e097529f2200c4f9553e87173c61f7c` versus banked `8a6bd7a20414c141ae7c8d2d7b0fe51451bfb61c9cc32fc2321b9b642d08bb89`.
- Both current and banking code compute these digests as SHA-256 over `np.ascontiguousarray(values).tobytes()`. The external digest covers evaluated `operator.external()` values and the source digest covers the repeated internal source image. Cache/file metadata is excluded, so the differing digests prove the arrays are not bitwise identical; metadata-only movement is excluded. The bank does not retain the old arrays, so no pointwise historical array-delta magnitude is recoverable without regeneration, which was forbidden and not attempted.
- Every stored composition summary remains bounded at roundoff: external and source maximum absolute fluxes are exactly unchanged (`2.8331924006294433 Wb` and `0.14172801675859822 Wb`), reconstruction and repeat differences remain exactly `0.0 Wb`, pinned-versus-unpinned distances and flux differences remain exactly zero, and closure remains `4.440892098500626e-16 Wb`.
- The other five top-level receipt items are exactly identical. The pinned root remains requested/achieved class `1/1`, topology-consistent, converged in one application, relative residual `1.5840538799920246e-16`, and absolute/terminal difference `4.440892098500626e-16 Wb`. This composition drift does not reopen the root verdict.
- `qualify` was invoked only with `write=False`; no bank was re-banked, regenerated, or overwritten.

follow_ons: none

blockers: none
