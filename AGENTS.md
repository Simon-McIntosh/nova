# Agent Guidelines

> **TL;DR**: Use `uv run` for Python commands, `ruff` for linting, conventional commits with single quotes, and `pytest` for testing. No backward compatibility constraints.

## Critical Rules

### Pre-commit Hooks Require Virtual Environment

The pre-commit hook uses `.venv/bin/python3` to run checks. In worktrees, you must:

```bash
# Option 1: Use uv run for git commit (recommended)
uv run git commit -m 'type: description'

# Option 2: Activate venv first
source .venv/bin/activate
git commit -m 'type: description'
```

**Why**: Pre-commit hooks fail with "pre-commit not found" if the venv is not active or accessible.

## Quick Reference

| Task                    | Command                                             |
| ----------------------- | --------------------------------------------------- |
| Commit changes          | `uv run git commit -m 'type: description'`          |
| Run Python              | `uv run python <script>`                            |
| Run tests               | `uv run pytest`                                     |
| Run tests with coverage | `uv run pytest --cov=nova`                          |
| Lint/format             | `uv run ruff check --fix . && uv run ruff format .` |
| Sync dependencies       | `uv sync --extra test`                              |
| Sync all extras         | `uv sync --all-extras`                              |
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
| `nova.imas`     | IMAS data model interface (IDS access, database connections)                                                                                                                                                                   |
| `nova.assembly` | ITER TF coil assembly analysis (Monte Carlo, fiducials, deformations). See [nova/assembly/README.md](nova/assembly/README.md) for detailed documentation on coordinate systems, gap calculations, and installed sector layout. |
| `nova.biot`     | Biot-Savart integral calculations                                                                                                                                                                                              |
| `nova.frame`    | FrameSpace types for electromagnetic components                                                                                                                                                                                |
| `nova.database` | netCDF storage, filepath management                                                                                                                                                                                            |

### Data Caching

Nova caches expensive calculations as netCDF files. Key classes:

- `nova.database.netcdf.netCDF`: xarray dataset storage with groups
- `nova.database.filepath.FilePath`: Cross-platform path management with fsspec/appdirs

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

# 4. Commit with conventional format (use uv run to ensure pre-commit runs)
uv run git commit -m 'type: brief description

Detailed body explaining what changed and why.'

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

```bash
# Sync dependencies first (required in worktrees)
uv sync --extra test

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

Commit and push from the worktree on its assigned branch:

```bash
# Lint and format
uv run ruff check --fix .
uv run ruff format .

# Stage, commit, and publish
git add <file1> <file2> ...
uv run git commit -m 'type: description'
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
