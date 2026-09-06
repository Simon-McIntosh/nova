# Equilibrium forward-model guidance

This guidance applies when constructing equilibrium machines, running forward
maps, or measuring their fixed points. Repository-wide development and git
rules remain in the root `AGENTS.md`.

## Run a forward solve through the request-receipt seam

The only supported way to run a forward solve with production defaults and a
receipt is a typed request through the public seam. A bare call to a private
fixed-point kernel, or a solver-control keyword threaded past the seam, is a
routing-around: it resolves outside the declared defaults, so its verdict is
not comparable with any production receipt.

- Build a `ForwardSolveRequest` and pass it to `ForwardProfile.solve`; the
  return is a `ForwardSolveReceipt` carrying the terminal state, its
  qualification and termination reason, the per-trip residual and mask
  histories, the compilation-cache hit, wall seconds, and the resolved
  defaults that actually ran. The request, receipt, policy, seed, and
  resolved-defaults types — `ForwardSolveRequest`, `ForwardSolveReceipt`,
  `ForwardSolvePolicy`, `ExplicitSolveSeed`, `ResolvedForwardSolveDefaults` —
  live in `nova/equilibrium/solve_request.py` and are re-exported under
  `nova.equilibrium`. Passing a request together with any solve keyword is
  rejected, so the defaults cannot be half-overridden through the seam.
- The declared-defaults table is `FORWARD_SOLVE_DEFAULTS` in the same module,
  keyed by the installed Nova package version. Read it through
  `declared_forward_solve_policy()` and build a request from those defaults
  with `ForwardSolveRequest.from_defaults(...)`. A deliberate deviation (a
  bank row pinned to a different budget, a stricter qualification floor) is
  expressed through `policy_overrides` or `resolve_forward_solve_policy(...)`
  and is recorded on the receipt under `resolved_defaults.deviations` — never
  passed to the seam as a raw solver keyword.
- Production routes and benchmarks construct one typed request per solve,
  leave every default that is not a deliberate deviation at its declared
  value, and write the resolved-defaults block into any JSON they emit. After
  touching a route, run the default-wiring tests:
  `tests/test_route_default_wiring.py` (each production route receipt resolves
  every declared default and no production route calls a private solver
  kernel) and `tests/test_default_wiring.py` (every production entry point
  resolves the defaults on and its launcher leaves them to the public seam).

## Keep the two reference lanes separate

### Closed-form analytic oracle

- The analytic lane is `scripts/analytic_oracle_fixtures/`. It selects the
  `moderate-rotation-conventional` member from
  `tests/rotating_equilibrium_references.py` and expresses its rotating source
  with `nova.equilibrium.rotation`.
- Its flux, profiles, current density, axis, separatrix, and exterior boundary
  supply are evaluated from that closed form. It has no stored map, IMAS read,
  or archive dependency.
- The exterior is the independently evaluated analytic total field minus the
  exact analytic-density plasma image on the same authored supports. This is
  what makes the closed-form state an accuracy oracle for the composed map.

### Stored DINA reproduction

- The stored-reference lane is
  `tests/test_equilibrium_forward_reference.py`. It reads DINA pulse/run
  `135011/7`, data-dictionary version `3.39.0`, time-slice index `353`, including
  the stored map, profiles, machine geometry, driven coils, and passive loops.
- This lane measures Nova's reproduction of that stored case. It is not an
  analytic oracle: the case's declared boundary constant and stored field are
  not assumed to be mutually exact.
- Never import an anchor, state, profile constant, cache identity, or acceptance
  floor from one lane into the other. Name every result and cache by its actual
  provenance.

## Build fixture machines through their semantic caches

- Standalone consumers request the lane's `cached_machine`; they do not call a
  direct machine builder. Only the cache-owned, advisory-locked miss path may
  construct geometry or evaluate a Green operator.
- The stored-reference identity includes its pulse/run/time locator, all
  reference scalars and array-content hashes, conductor geometry and currents,
  requested discretisation, precision, route policies, and cache schema. The
  analytic identity carries the closed-form constants, discretisation, wall
  sampling, precision, routes, and its distinct analytic schema. A changed
  semantic input must select a changed key.
- Store and restore native dtypes, shapes, and bytes. A warm hit is accepted
  only after descriptor, schema, payload inventory, digest, and semantic key
  validate. Publication and recovery of an interrupted group stay under the
  same lock.
- The measured stored-reference requests are expensive enough to make bypasses
  defects: coarse is `1,246.880145 s` cold and `0.265939 s` warm; fine is
  `3,335.325393 s` cold and `0.817662 s` warm. All `31` persisted arrays matched
  bitwise in both measurements. Expect warm loads for repeated work.
- A worktree consumes the shared root environment read-only:
  `UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync`.
  Never run `uv sync`, `uv venv`, or `pip` from a fixture worker. A missing
  capability is an environment-owner blocker, not permission to provision the
  shared environment.

## Preserve the serialized root banks

- Stored DINA reproduction roots are
  `scripts/root_gate_attribution/coarse-terminal-root.npz` and
  `scripts/root_gate_attribution/fine-terminal-root.npz`.
- Closed-form oracle roots are `scripts/oracle_rebaseline/root-coarse.npz` and
  `scripts/oracle_rebaseline/root-fine.npz`; their receipts retain the oracle,
  seed, terminal state, residual trajectory, locally normalized flux, topology,
  and current observations.
- Treat these files as banked evidence. Load and validate them before reuse; do
  not overwrite a bank while changing a fixture, solver budget, seed, or gauge.
  Bank a separately named measurement when the semantic experiment changes.

## Keep flux gauges and normalized anchors local

- The Biot-Savart composition sets Nova's poloidal-flux gauge. Do not re-zero a
  Nova state at a saddle or combine a Nova-gauge amplitude with a stored
  reference constant.
- `psi_norm` is gauge-safe only when its axis and boundary anchors are read from
  the same field being normalized. Normalize an oracle field with oracle-field
  anchors and a solved field with solved-field anchors.
- Compare raw flux amplitudes only when the states share a declared gauge, such
  as the same exact exterior supply. Record the gauge and anchor provenance in
  every receipt that reports a flux deviation.
- Pin `JAX_PLATFORMS=cpu` for every CPU-lane build, solve, test, or benchmark.
  Login and compute hosts may expose CUDA, and an implicit backend change is a
  different numerical measurement.
