# Agent Guidelines

> **TL;DR**: Use `uv run` for Python commands, `ruff` for linting, conventional commits with single quotes, and `pytest` for testing. No backward compatibility constraints.

## Critical Rules

### One Environment per Repository

The binding policy is `~/.agents/AGENTS.md` "Development Environment". For
nova: the one environment is the root `.venv` at `~/Code/nova/.venv`; syncing it
is ordinary work (`uv sync`, or plain `uv run`, which syncs first), and bringing
a stale one up to date is the agent's job. Dependency changes go through
`uv add` / `uv remove` so they land in `pyproject.toml` + `uv.lock` — never
`pip install`. What is banned is a *second* environment: a duplicate worktree
environment is ~70k files / ~1.8 GiB on GPFS.

In a worktree, reuse the main checkout's environment. `PYTHONPATH="$PWD"` makes
the worktree's own code shadow the main checkout's editable install, and
`--no-sync` keeps an incidental sync from mutating an environment the main
checkout and concurrent workers share:

```bash
UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv PYTHONPATH="$PWD" \
  uv run --no-sync pytest <targets>
```

### An exact sync strips the extras (binding)

**No worker ever runs `uv sync`, and every `uv run` in a worktree carries
`--no-sync`.** A plain `uv sync`, or a `uv run` without `--no-sync`, resolves
the lock's *default* dependency set into whatever environment
`UV_PROJECT_ENVIRONMENT` names and removes everything outside that set. Pointed
at the shared root `.venv` it does two things at once: it repoints the editable
install (`.venv/lib/python3.14/site-packages/_editable_impl_nova_stella.pth`)
at the caller's worktree, so every process without `PYTHONPATH` imports that
worktree's code; and it uninstalls every extra the default set does not carry:
`gpu` (the CUDA plugin and the `nvidia-*` libraries), `schema` (linkml, whose
`jsonschema` the DIII-D H200 gate imports), and the rest.

Measured 2026-09-08: four worker repoints in one morning (three shadow workers
and one implement worker, on three different lanes) and one strip. The strip
came from a worker "repairing" the pointer with a bare `uv sync` from the main
checkout; two hours later the H200 lane reported no CUDA backend (29 collected,
26 errors) and a second gate stopped on a missing `jsonschema`. Neither symptom
named the cause. `uv sync --all-extras` is not the prior state either: it would
add a hundred packages the environment never carried.

- **After every worker landing, the coordinator checks two things** from the
  main checkout:

  ```bash
  cat .venv/lib/python3.14/site-packages/_editable_impl_nova_stella.pth   # /home/ITER/mcintos/Code/nova
  ls .venv/lib/python3.14/site-packages | grep -c jax_cuda12_plugin          # 2
  ```

- **Repair with an inexact sync of every extra, from the main checkout only:**

  ```bash
  uv sync --inexact --all-extras    # adds what is missing, removes nothing, repoints the editable
  .venv/bin/python -c "import jsonschema, vedo, linkml_runtime"
  ```

  `--inexact` is the load-bearing flag: without it the sync is exact and
  strips again. Dry-run first (`--dry-run`); the expected delta is one
  uninstall and one install of `nova-stella` plus whatever extras' packages
  are absent. A GPU device only shows on a compute node, so verify CUDA
  through the next H200 lane's log header, not from the login node.

- **Workers repair nothing in the shared environment.** A worker that finds the
  pointer wrong or a package missing reports it as a blocker and stops; the
  brief says so in one line: "every uv command carries `--no-sync` and nothing
  runs `uv sync`".

- **Inherited variables collide with the explicit flag.** Three workers died
  before pytest because an inherited `UV_NO_SYNC` or `UV_RUN_RECURSION_DEPTH`
  duplicated their explicit `--no-sync`. A lane script therefore begins with
  `unset UV_NO_SYNC UV_RUN_RECURSION_DEPTH` and then passes `--no-sync`
  explicitly, so the recipe does not depend on the launcher's environment.

- **On any SLURM compute node, do not invoke `uv` at all.** The wrapper there
  injects `--no-sync` itself, so a payload that also passes the flag dies before
  Python starts and the job log is empty. This is not an H200 quirk: two workers
  lost an allocation to it on `all_debug` in one wave, each reporting only that
  the allocation exited before pytest. Run the shared environment's interpreter
  directly and set the rest through the environment:

  ```bash
  srun --partition=all_debug --time=00:59:00 --cpus-per-task=4 --mem=64G \
    bash -lc 'export TMPDIR=/tmp JAX_PLATFORMS=cpu PYTHONPATH=<worktree>; \
      /home/ITER/mcintos/Code/nova/.venv/bin/python -m pytest \
        -p no:cacheprovider <targets>' > <log> 2>&1; echo EXIT=$?
  ```

  `PYTHONPATH` is what makes the worktree's code shadow the editable install,
  which is the job `--directory` would otherwise do. The `uv run --no-sync`
  recipe above is for login-node work in a worktree, and only there.

### Pre-commit Hooks Require Virtual Environment

The pre-commit hook runs checks through `.venv/bin/python3`, so it needs the
environment reachable at that path:

```bash
# Main checkout: uv run resolves and syncs the root .venv (recommended)
uv run git commit -m 'type: description'

# Worktree: link the main checkout's environment once (never build a second
# one), and pass --no-sync because peers share that environment
ln -s ~/Code/nova/.venv .venv
uv run --no-sync git commit -m 'type: description'
```

**Why**: Pre-commit hooks fail with "pre-commit not found" if the venv is not accessible.

## Quick Reference

| Task                    | Command                                             |
| ----------------------- | --------------------------------------------------- |
| Commit changes          | `uv run git commit -m 'type: description'`          |
| Run Python              | `uv run python <script>`                            |
| Run tests               | `uv run pytest`                                     |
| Run tests with coverage | `uv run pytest --cov=nova`                          |
| Lint/format             | `uv run ruff check --fix . && uv run ruff format .` |
| Run tests in a worktree | `UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest` |
| Add dependency          | `uv add <package>`                                  |
| Add dev dependency      | `uv add --dev <package>`                            |
| Launch Spyder           | `uv run spyder &`                                   |
| Run Bokeh app           | `uv run bokeh serve apps/pulsedesign`               |

## Project Overview

**nova** is a Python package for electromagnetic analysis and plasma equilibrium reconstruction:

- **Core Function**: Generation and interpretation of magnetically confined plasma equilibria
- **Data Model**: Extends pandas with `FrameSpace` types for plasma filaments, coils, passive structures, and circuits
- **Computation**: Grid-free electromagnetic calculations using Biot-Savart integrals
- **Data Storage**: Multi-dimensional data via xarray, cached as netCDF files

### Key Modules

| Module          | Purpose                                                                                                                                                                                                                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `nova.imas`     | IMAS data model interface (IDS access, database connections). Locating a machine description, opening its static IDSs at the manifest DD pin, and auditing receipt staleness: [`nova/imas/AGENTS.md`](nova/imas/AGENTS.md).      |
| `nova.assembly` | ITER TF coil assembly analysis (Monte Carlo, fiducials, deformations). See [nova/assembly/README.md](nova/assembly/README.md) for detailed documentation on coordinate systems, gap calculations, and installed sector layout. |
| `nova.biot`     | Biot-Savart integral calculations. Section routes and FrameSpace access: [`nova/biot/AGENTS.md`](nova/biot/AGENTS.md).                                                                                                         |
| `nova.equilibrium` | Forward equilibrium construction and solves. The forward-solve request-receipt seam, reference-lane, cache, gauge, and CPU-measurement guidance: [`nova/equilibrium/AGENTS.md`](nova/equilibrium/AGENTS.md).                     |
| `nova.transport` | Coupled forward transport over the native and TORAX current-diffusion rungs, with equilibrium coupling. The forward-solve seam and coupling guidance: [`nova/transport/AGENTS.md`](nova/transport/AGENTS.md).                       |
| `nova.frame`    | FrameSpace types for electromagnetic components                                                                                                                                                                                |
| `nova.database` | netCDF storage, filepath management                                                                                                                                                                                            |
| `nova.media`    | Presentation figures and animations: one imas-ink style, a three-view layout, poloidal painters, pulse-fixed trace scales, a Pillow GIF compositor, and per-archive read adapters. Publication rules below.                     |

### Data Caching

Nova caches expensive calculations as netCDF files. Key classes:

- `nova.database.netcdf.netCDF`: xarray dataset storage with groups
- `nova.database.filepath.FilePath`: Cross-platform path management with fsspec/appdirs

### Publishing a figure so the reckon server lists it

The reckon mount for nova is `~/Code/nova/docs`, so a file at
`docs/figures/<topic>/<name>` is served at the project-absolute URL
`/nova/figures/<topic>/<name>`. Use project-absolute `src` in authored HTML; a
relative `src` 404s in the SPA. Both `*.png` and `*.gif` are gitignored, so
publishing needs `git add -f`.

**The Figures index counts PNG and SVG only — a GIF is served but never
listed.** Measured 2026-09-07: the tab read `509 in nova` while
`docs/figures` held exactly 409 PNG + 100 SVG = 509, with a GIF present on
disk and uncounted; one added PNG took it to 510. So an animation that should
be findable by browsing needs a still companion beside it.
`nova.media.gif.write_contact_sheet` builds one from the frames already
rendered, and `nova.media.gif.animate` writes it when passed `contact_sheet`.
Name figures with hyphens to match the served convention
(`mast-21978-efit-pulse.gif` beside `mast-21978-efit-pulse-frames.png`).

Rendering many frames is heavy work: run it on a `*_debug` partition, never on
the login node, and never point `--output` at `/run/user`, which is node-local
tmpfs — slurmstepd then cancels the job in zero seconds with an empty log.

## Coupled Repositories

Prior-art scouts (reckon-ship §1b) search these repos in both directions
before authoring new machinery here, and their sessions search nova likewise:

- **imas-ambix** (`~/Code/imas-ambix`) — the flux-function seam, challenge
  corpus and COCOS conventions; ambix consumes Nova's forwards and conditions
  their inputs.
- **reckon** (`~/Code/reckon`) — plan/crew infrastructure this repo's
  docs/state tooling is built on.

## Agent Workflows

### Committing Changes

**Step-by-step procedure:**

```bash
# 1. Check current state
git status

# 2. Lint PATH-SCOPED over the files you touched — never repo-wide: a
#    repo-wide fixer rewrites pre-existing unformatted files outside your
#    scope, and [tool.ruff] excludes tests/ so test files must be named
#    explicitly to be linted at all
uv run ruff check --fix <file1> <file2> ...
uv run ruff format <file1> <file2> ...

# 3. Stage specific files (NEVER use git add -A)
git add <file1> <file2> ...

# 4. Commit with conventional format (use uv run to ensure pre-commit runs).
#    In the SHARED PRIMARY CHECKOUT, always pathspec-limit the commit: the
#    index is shared, so a bare `git commit` sweeps whatever a concurrent
#    session has staged - path-scoped `git add` alone does not protect you,
#    and sweeping a peer's half-written plan edit publishes it live (the
#    docs server serves this working tree). Private worktrees need no
#    pathspec.
uv run git commit -m 'type: brief description

Detailed body explaining what changed and why.' -- <file1> <file2> ...

# 5. Fix pre-commit errors and repeat steps 3-4 until clean

# 6. Push
git push
```

**Commit message format:**

| Type       | Purpose            |
| ---------- | ------------------ |
| `feat`     | New feature        |
| `fix`      | Bug fix            |
| `refactor` | Code restructuring |
| `docs`     | Documentation      |
| `test`     | Test changes       |
| `chore`    | Maintenance        |

**Breaking changes**: Add `BREAKING CHANGE:` footer in the body (not `type!:` suffix).

**Body required**: every commit carries a body stating what changed and why —
the subject-only conventional grammar is the format of the first line, not of
the whole message. Verify before push:
`git log -1 --format=%b | grep -q . || echo 'BODYLESS — amend'`.

### Testing

Marker policy: `slow` marks the curated heavy numerical and integration tests—numerical sweeps, JAX JIT/vmap/grad kernels, signal-store round-trips, structural solves, reference-equilibrium reproductions, and pulse-scale integration—that the path-free default `pytest` lane excludes with `-m 'not slow'`; use `pytest -m "slow or not slow"` to select both fast and slow tests. An explicitly named test path or node ID clears only that inherited default filter and collects the named tests unless the command supplies its own `-m` or `--markexpr`, which is preserved. Any otherwise-successful session that collects zero tests is changed to pytest's nonzero `NO_TESTS_COLLECTED` exit status.

#### CPU and H200 test lanes

Use the default CPU lane for fast tests and for bit-identity contracts that pin
CPU x64 arithmetic. Pin it explicitly with `JAX_PLATFORMS=cpu`; an implicit
backend change is a different numerical measurement. Run a test file through
the H200 lane when its measured CPU wall exceeds ten minutes. The launcher uses
one reserved H200, records the selected JAX platforms in the log header, and
reuses the persistent compilation cache:

```bash
scripts/h200_test_lane/run.sh \
  --log /absolute/path/to/caller-named.log --wait -- \
  tests/test_equilibrium_forward_solve.py -vv
```

The launcher submits to `betelgeuse` under `gpu_0003_grpA`, runs pytest from
the repository's shared environment without invoking uv on the compute node,
and appends both the pytest wall time and exit status to the named log. Preserve
GPU failures from CPU-specific identity assertions as device-qualified evidence;
do not weaken those CPU contracts to make the H200 lane green.

The H200 node is shared and its reservation is a **core** budget (30 cores, no
cards): every submission states an explicit `--mem` (never `--mem=0`, which
SLURM reads as the whole 1.5 TB and leaves the job pending on `Resources`
while blocking the queue behind it) and sizes `--cpus-per-task` against the
cores the serving jobs already hold (`squeue -w 98dci4-gpu-0003 -o '%C %b %m %j'`).
A single-card measurement job is 8 cores, `--gres=gpu:1`, `--mem=128G`. The
node's live budget and the serving footprint are kept in imas-ambix
`imas_ambix/agent/AGENTS.md`.

```bash
# In a worktree, reuse the main checkout's environment (see One Environment
# per Repository above); --no-sync because peers share that environment:
# UV_PROJECT_ENVIRONMENT=~/Code/nova/.venv PYTHONPATH="$PWD" uv run --no-sync pytest

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=nova

# Run specific test
uv run pytest tests/path/to/test.py::test_function

# Run with verbose output
uv run pytest -v
```

### Working in Worktrees

Use worktrees only when the user has already authorised a non-primary branch or
when an orchestrated worker needs an isolated checkout. Do not create a topic
branch or move commits between branches with cherry-pick.

A worktree never builds its own environment: run everything through the main
checkout's `.venv` with `UV_PROJECT_ENVIRONMENT` + `--no-sync`, and link
`.venv` (`ln -s ~/Code/nova/.venv .venv`) only so the pre-commit hook resolves.
When the work genuinely changes dependencies, make the change durable in
`pyproject.toml` first, then drop `--no-sync` deliberately and say so in the
report so the orchestrator can serialize it against other workers.

Commit and push from the worktree on its assigned branch:

```bash
# Lint and format (path-scoped over the files you touched)
uv run --no-sync ruff check --fix <files>
uv run --no-sync ruff format <files>

# Stage, commit, and publish
git add <file1> <file2> ...
uv run --no-sync git commit -m 'type: description'
git pull --no-rebase origin <assigned-branch>
git push origin <assigned-branch>
```

Remove a worktree only after its worktree is clean and every required commit is
reachable from its published branch.

### Branch policy

- **`main`** — the primary branch for all code and all Reckon plans.
- **`legacy/v1`** + tag **`v1-assembly-baseline`** — the frozen goldens-green
  baseline, branch-protected (no force-push, no deletion).
- **`develop`** — historical integration branch retained for traceability; do
  not start new work there.

Keep the primary checkout on `main`. The Reckon server serves this checkout's
working tree, so commit and push plan edits in the same session.

## IMAS Data Access (imas-python)

Nova reads and writes IMAS data through imas-python only — the global
guidelines carry the binding rules (never h5py on IMAS data; open with the
written DD version; check `homogeneous_time`; `float()` before formatting
`IDSFloat0D`). Nova-scoped depth:

- **DD version lookup**: for test data, take `dd_version` per dataset from
  `tests/data-manifest.json`; never guess. When unknown, read
  `ids_properties.version_put.data_dictionary` from an opened IDS.
- **`homogeneous_time == 1`**: the time base is `ids.time` at IDS level;
  per-signal `.time` arrays are empty by design — an empty per-signal time is
  NOT missing data.
- **`DBEntry` lifecycle**: the constructor opens the pulse; a subsequent
  `.open()` raises "already open".
- **DDv3 `time_slice` iteration** may raise `int() argument must be a
  string…` — use index-based access (`for i in range(len(eq.time_slice))`).
- **`IDSNumericArray`**: convert with `np.asarray(...)` (or `.value`); len,
  shape, and indexing work directly.
- **Assembling a pulse from standalone `*.h5` files**: `master.h5` needs the
  `HDF5_BACKEND_VERSION` attribute (numpy bytes `b'1.0'`) and h5py
  `ExternalLink`s targeting `'/{ids_name}'` inside each file (data sits under
  a root-level group named after the IDS).

## Rules

### DO

- Use `uv run` for all Python commands
- Use single quotes for commit messages
- Stage files individually (`git add <file>`)
- Use modern Python 3.10+ syntax: `list[str]`, `X | Y`
- Use `dataclasses` for data classes
- Use exception chaining with `from`

### DON'T

- Don't prefix commands with `cd /path &&`
- Don't manually activate venv (`.venv/bin/activate`) unless necessary
- Don't use `git add -A`
- Don't use `type!:` suffix for breaking changes
- Don't use double quotes with special characters in commits
- Don't use `List[str]`, `Union[X, Y]`, or `isinstance(e, (X, Y))`
- Don't use "new", "refactored", "enhanced" in names

### Windows-Specific

On Windows, avoid f-string escaping issues in `uv run python -c "..."` commands:

```python
# Wrong - causes syntax errors on Windows
print(f'Gap: mean={r["key"]:.3f}')

# Correct - use % formatting or separate variable
print('Gap: mean=%.3f' % r['key'])
```

The backslash escapes in f-strings within `-c` commands cause parsing failures.

Use `git status` instead of `get_changed_files` tool (not available on Windows).

## Project Structure

```
nova/
├── ansys/          # ANSYS integration (excluded from package)
├── assembly/       # ITER TF coil assembly analysis
├── biot/           # Biot-Savart calculations
├── control/        # Control systems
├── database/       # netCDF storage, filepath management
├── datachain/      # Data pipeline utilities
├── dataset/        # Dataset management
├── frame/          # FrameSpace types
├── geometry/       # Geometric primitives
├── graphics/       # Visualization utilities
├── imas/           # IMAS data model interface
├── jax/            # JAX-based optimizations
├── limits/         # Operational limits
├── linalg/         # Linear algebra utilities
├── utilities/      # General utilities
└── xarray/         # xarray extensions

tests/              # Mirror source structure
apps/               # Bokeh applications
input/              # Input data files
data/               # Reference data
```

## Code Style

### Type Annotations

```python
# Correct
def process(items: list[str]) -> dict[str, int]: ...
if isinstance(e, ValueError | TypeError): ...

# Wrong
def process(items: List[str]) -> Dict[str, int]: ...
if isinstance(e, (ValueError, TypeError)): ...
```

### Error Handling

```python
# Correct - chain exceptions
try:
    operation()
except IOError as e:
    raise ProcessingError("failed to process") from e

# Wrong - loses context
except IOError:
    raise ProcessingError("failed to process")
```

## Philosophy

This is a **research project** focused on ITER assembly support:

- Prioritize correctness over backward compatibility
- Write code as if it's always been this way
- Avoid legacy naming patterns
