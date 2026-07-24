"""Two-tier replay: re-run a recorded fit and re-check it against its golden.

A *run record* (``data/Assembly/run_records/<fit_id>.yaml``) binds a fit's
outputs to the canonical-unit inputs, the fit configuration, the environment,
and the golden it produced.

Tier-1 replay resolves a record's input digests against the units available
here, re-runs the recorded entry point with the *current* code, and compares
the fresh output to the committed golden at the tolerance classes the record
pinned -- proving the fit still reproduces what the record says it did.

Tier-2 replay reconstructs the record's *historical* environment -- checks out
its git revision into a throwaway worktree, restores the recorded dependency
closure from the matching ``uv.lock`` into an isolated venv, and re-runs the
entry point under that interpreter -- then compares against the golden at the
same tolerance classes. It surfaces drift the current environment would mask.
Tier-2 is expensive (a full dependency sync per record) and gated opt-in behind
:data:`TIER2_ENV_FLAG`. When the historical environment cannot be rebuilt (the
sha is unreachable, the lock will not restore, or the entry point will not run)
the outcome degrades *visibly* -- reproduced by current code, historical env
unreconstructable, with the precise stage that failed -- never a silent pass.
Both tiers share the golden-comparison core (:func:`_compare_to_golden`).

The pilot records are emitted by :func:`write_pilot_record`, invoked from the
sector golden generator so the first records describe real, freshly captured
fits rather than hand-written stubs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from nova.assembly.provenance import FitConfig, RunRecord, capture_environment

from . import _canonical, _environment, _fixtures, _manifest, _registry, _tolerance

RUN_RECORDS_DIR = _environment.repo_root() / "data" / "Assembly" / "run_records"
_SHA256_PREFIX = "sha256:"

# Opt-in gate for Tier-2: rebuilding a historical environment is a full
# dependency sync, too costly for the default suite. Set to ``1`` to run it.
TIER2_ENV_FLAG = "NOVA_TIER2_REPLAY"

# Per-stage subprocess budgets (seconds), overridable so a compute node can
# grant more time than a login node should spend.
_SYNC_TIMEOUT = int(os.environ.get("NOVA_TIER2_SYNC_TIMEOUT", "1800"))
_RUN_TIMEOUT = int(os.environ.get("NOVA_TIER2_RUN_TIMEOUT", "900"))

# The stages of a historical rebuild, in order; a degradation names the stage
# it failed at so the reason distinguishes "sha gone" from "lock won't restore"
# from "entry point won't run".
STAGE_RESOLVE_SHA = "resolve-sha"
STAGE_VERIFY_LOCK = "verify-lock"
STAGE_SYNC_DEPS = "sync-deps"
STAGE_RUN_ENTRYPOINT = "run-entrypoint"
STAGE_COMPARE = "compare"

# Random seed the ILIS plane PCA pins; recorded so the fit configuration is
# self-describing even though the fitter reads it from source.
_PCA_RANDOM_STATE = 2025


def _prefixed(bare_hex: str) -> str:
    """Return a bare hex digest tagged as ``sha256:<hex>``."""
    return f"{_SHA256_PREFIX}{bare_hex}"


def _array_digest(arrays: dict, key: str) -> str:
    """Return the ``sha256:<hex>`` content digest of a single canonical array."""
    return _prefixed(
        _canonical.sha256_bytes(_canonical.to_npz_bytes({key: arrays[key]}))
    )


def run_records() -> list[tuple[Path, RunRecord]]:
    """Return ``(path, record)`` for every run record, sorted by filename.

    The ``*.fitconfig.yaml`` sidecars written beside each record are excluded;
    they are fit configurations, not records.
    """
    if not RUN_RECORDS_DIR.is_dir():
        return []
    return [
        (path, RunRecord.load(path))
        for path in sorted(RUN_RECORDS_DIR.glob("*.yaml"))
        if not path.name.endswith(".fitconfig.yaml")
    ]


def _available_digests() -> set[str]:
    """Return the content digests of every canonical unit available here."""
    return set(_fixtures.canonical_units().values())


def unresolved_inputs(record: RunRecord) -> list[str]:
    """Return the logical names of record inputs absent from the local corpus."""
    available = _available_digests()
    return sorted(
        name for name, digest in record.input_digests.items() if digest not in available
    )


@dataclass
class ReplayResult:
    """Outcome of replaying one run record against its golden."""

    fit_id: str
    passed: bool
    checked: int
    failures: list[str]


def _load_golden(fit_id: str) -> dict:
    """Return the committed golden arrays for a fit id."""
    artifact = _manifest.GOLDENS_DIR / f"{fit_id}.npz"
    return _canonical.load_npz(artifact.read_bytes())


def _compare_to_golden(record: RunRecord, golden: dict, candidate: dict) -> list[str]:
    """Compare a freshly produced output against the record's golden.

    Shared by both tiers: Tier-1 passes a candidate produced in-process by the
    current code, Tier-2 passes one produced by the historical interpreter.
    Verifies the golden still matches the digest the record pinned, then checks
    every recorded output array at its tolerance class. Returns a list of
    human-readable failure strings (empty when the candidate reproduces).
    """
    failures: list[str] = []
    for output in record.outputs:
        key = output["name"]
        tol = output["tolerance_class"]
        if key not in golden:
            failures.append(f"[{key}] recorded output absent from golden")
            continue
        if key not in candidate:
            failures.append(f"[{key}] run did not produce this output")
            continue
        # Integrity: the golden must still match the digest the record pinned.
        if _array_digest(golden, key) != output["digest"]:
            failures.append(f"[{key}] golden drifted from the digest the record pinned")
        result = _tolerance.compare(candidate[key], golden[key], tol)
        if not result.passed:
            failures.append(f"[{key}] ({tol}) {result.detail}")
    return failures


def replay(record: RunRecord) -> ReplayResult:
    """Re-run a record's fit and compare it to the committed golden.

    Assumes the record's inputs resolve (see :func:`unresolved_inputs`) and its
    entry point is runnable here. Verifies the golden still matches the digest
    the record pinned, then re-runs the entry point and compares every output
    array to the golden at the record's tolerance class.
    """
    entries = {entry.id: entry for entry in _registry.registry()}
    entry = entries[record.fit_id]

    golden = _load_golden(record.fit_id)
    candidate = _canonical.canonicalize(entry.run())
    failures = _compare_to_golden(record, golden, candidate)

    return ReplayResult(
        fit_id=record.fit_id,
        passed=not failures,
        checked=len(record.outputs),
        failures=failures,
    )


# Driver run inside the reconstructed interpreter: it imports the *historical*
# harness from the checked-out worktree (added to ``sys.path``), runs the named
# entry point, canonicalizes its result with the worktree's own canonicaliser,
# and writes the .npz where the parent process can read it back. Kept as a
# standalone script so it executes under the worktree venv, not this process.
_DRIVER_SOURCE = """\
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

worktree, fit_id, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, worktree)

from tests.characterization import _canonical, _registry

entries = {entry.id: entry for entry in _registry.registry()}
entry = entries[fit_id]
arrays = _canonical.canonicalize(entry.run())
with open(out_path, "wb") as handle:
    handle.write(_canonical.to_npz_bytes(arrays))
"""


@dataclass
class HistoricalReplayResult:
    """Outcome of replaying a record under its reconstructed environment.

    ``outcome`` is one of:

    * ``"passed"`` -- the historical environment rebuilt and its run matched
      the golden at the record's tolerance classes.
    * ``"degraded"`` -- the current code reproduces the golden, but the
      historical environment could not be reconstructed (or, for a record
      captured from a dirty tree, its committed sha diverged); ``stage`` and
      ``reason`` say precisely why. This is a *visible* skip, not a pass.
    * ``"drift"`` -- the historical environment rebuilt from a clean record but
      its run diverged from the golden, or the current code itself fails to
      reproduce; a genuine finding that must fail.
    """

    fit_id: str
    outcome: str
    reproduced_by_current_code: bool
    reconstructed: bool
    stage: str | None
    reason: str | None
    checked: int
    failures: list[str] = field(default_factory=list)
    elapsed: dict[str, float] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    @property
    def degraded(self) -> bool:
        return self.outcome == "degraded"

    def message(self) -> str:
        """Return the human-readable one-line summary of the outcome."""
        if self.outcome == "passed":
            timing = ", ".join(f"{k} {v:.1f}s" for k, v in self.elapsed.items())
            return f"historical environment rebuilt and matched golden ({timing})"
        if self.outcome == "degraded":
            return (
                "reproduced by current code; historical env unreconstructable: "
                f"[{self.stage}] {self.reason}"
            )
        return f"[{self.stage}] {self.reason}\n" + "\n".join(self.failures)


def _run_subprocess(cmd, cwd, timeout, env=None):
    """Run a subprocess capturing output; return ``(ok, detail)``.

    ``ok`` is True on a zero exit within ``timeout``. ``detail`` carries a
    trimmed stderr/stdout tail (on failure) or a timeout note, so the caller
    can build a precise degradation reason.
    """
    full_env = os.environ.copy()
    full_env.update(env or {})
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            env=full_env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    if proc.returncode == 0:
        return True, ""
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
    return False, f"exit {proc.returncode}: " + " | ".join(tail)


def _single_thread_env() -> dict:
    """Return the single-threaded-BLAS environment the goldens were made under."""
    return {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def replay_historical(
    record: RunRecord, *, workdir: Path | None = None
) -> HistoricalReplayResult:
    """Replay a record under its reconstructed historical environment.

    Establishes the current-code baseline first (so a degradation can honestly
    say "reproduced by current code"), then rebuilds the record's environment
    in stages -- checkout, lock verify, dependency sync, entry-point run,
    compare -- degrading visibly with the failing stage if any step cannot be
    completed. Assumes the record's inputs resolve and its entry point is
    runnable here (the caller gates on :func:`unresolved_inputs` and the
    entry-point skip probe, exactly as Tier-1 does).

    A record captured from a dirty working tree (``code_dirty``) can never be
    reconstructed exactly -- the uncommitted changes are not recoverable from
    the committed sha -- so a divergence there degrades (with that caveat)
    rather than being reported as drift.
    """
    root = _environment.repo_root()
    baseline = replay(record)
    if not baseline.passed:
        return HistoricalReplayResult(
            fit_id=record.fit_id,
            outcome="drift",
            reproduced_by_current_code=False,
            reconstructed=False,
            stage=STAGE_COMPARE,
            reason=(
                "current code does not reproduce the golden (Tier-1 failed); "
                "resolve Tier-1 before attributing historical drift"
            ),
            checked=baseline.checked,
            failures=baseline.failures,
        )

    # Same-sha fast path: if HEAD is the recorded commit and both the record
    # and the working tree are clean, the historical environment *is* the
    # current one; the Tier-1 result already covers it.
    current = capture_environment(root)
    if (
        not record.code_dirty
        and not current["code_dirty"]
        and record.code_git_sha == current["code_git_sha"]
    ):
        return HistoricalReplayResult(
            fit_id=record.fit_id,
            outcome="passed",
            reproduced_by_current_code=True,
            reconstructed=True,
            stage=None,
            reason="HEAD is the recorded commit on a clean tree; historical == current",
            checked=baseline.checked,
        )

    def degraded(stage, reason, elapsed):
        # A record captured from a dirty tree carries uncommitted changes that
        # the committed sha cannot restore; when the rebuild fails at sync or
        # run, that is the likely root cause, so name it.
        if record.code_dirty and stage in (STAGE_SYNC_DEPS, STAGE_RUN_ENTRYPOINT):
            reason = (
                f"{reason} -- note the record was captured from a DIRTY tree "
                f"(code_dirty=true), so the committed sha {record.code_git_sha[:12]} "
                f"may lack uncommitted changes that were live at run time"
            )
        return HistoricalReplayResult(
            fit_id=record.fit_id,
            outcome="degraded",
            reproduced_by_current_code=True,
            reconstructed=False,
            stage=stage,
            reason=reason,
            checked=baseline.checked,
            elapsed=elapsed,
        )

    elapsed: dict[str, float] = {}
    owns_workdir = workdir is None
    workdir = (
        Path(workdir)
        if workdir is not None
        else Path(tempfile.mkdtemp(prefix="nova-tier2-"))
    )
    worktree = workdir / "worktree"

    try:
        # Stage 1: check out the recorded revision into a throwaway worktree.
        t0 = time.monotonic()
        ok, detail = _run_subprocess(
            ["git", "worktree", "add", "--detach", str(worktree), record.code_git_sha],
            cwd=root,
            timeout=300,
        )
        elapsed[STAGE_RESOLVE_SHA] = time.monotonic() - t0
        if not ok:
            return degraded(
                STAGE_RESOLVE_SHA,
                f"git sha {record.code_git_sha[:12]} unreachable ({detail})",
                elapsed,
            )

        # Stage 2: the worktree's uv.lock must match the recorded closure.
        from nova.assembly.provenance import digest as _digest

        lock_path = worktree / "uv.lock"
        if not lock_path.exists():
            return degraded(
                STAGE_VERIFY_LOCK,
                f"uv.lock absent at {record.code_git_sha[:12]}",
                elapsed,
            )
        worktree_lock = _digest.digest_file(lock_path)
        if worktree_lock != record.uv_lock_sha256:
            return degraded(
                STAGE_VERIFY_LOCK,
                (
                    f"worktree uv.lock {worktree_lock[:19]}... != recorded "
                    f"{record.uv_lock_sha256[:19]}..."
                ),
                elapsed,
            )

        # Stage 3: restore the recorded dependency closure into an isolated venv.
        t0 = time.monotonic()
        ok, detail = _run_subprocess(
            ["uv", "sync", "--frozen"],
            cwd=worktree,
            timeout=_SYNC_TIMEOUT,
            env={"UV_PROJECT_ENVIRONMENT": str(worktree / ".venv")},
        )
        elapsed[STAGE_SYNC_DEPS] = time.monotonic() - t0
        if not ok:
            hint = (
                "; exceeds a login-node budget -- rerun on a compute node"
                if "timed out" in detail
                else " (missing wheels or interpreter)"
            )
            return degraded(
                STAGE_SYNC_DEPS,
                f"uv sync --frozen failed{hint}: {detail}",
                elapsed,
            )

        venv_python = worktree / ".venv" / "bin" / "python"
        if not venv_python.exists():
            return degraded(
                STAGE_SYNC_DEPS,
                f"venv interpreter missing after sync ({venv_python})",
                elapsed,
            )

        # Stage 4: run the entry point under the reconstructed interpreter.
        driver_path = workdir / "driver.py"
        driver_path.write_text(_DRIVER_SOURCE)
        out_npz = workdir / f"{record.fit_id}.npz"
        t0 = time.monotonic()
        ok, detail = _run_subprocess(
            [
                str(venv_python),
                str(driver_path),
                str(worktree),
                record.fit_id,
                str(out_npz),
            ],
            cwd=worktree,
            timeout=_RUN_TIMEOUT,
            env=_single_thread_env(),
        )
        elapsed[STAGE_RUN_ENTRYPOINT] = time.monotonic() - t0
        if not ok or not out_npz.exists():
            return degraded(
                STAGE_RUN_ENTRYPOINT,
                f"entry point {record.fit_id!r} did not run under the "
                f"rebuilt env: {detail}",
                elapsed,
            )

        # Stage 5: compare the historical run against the golden the record pinned.
        golden = _load_golden(record.fit_id)
        candidate = _canonical.load_npz(out_npz.read_bytes())
        failures = _compare_to_golden(record, golden, candidate)
        if not failures:
            return HistoricalReplayResult(
                fit_id=record.fit_id,
                outcome="passed",
                reproduced_by_current_code=True,
                reconstructed=True,
                stage=None,
                reason=None,
                checked=baseline.checked,
                elapsed=elapsed,
            )
        if record.code_dirty:
            # The recorded run had uncommitted changes; the committed sha is not
            # the exact code that made the golden, so a divergence is expected
            # and degrades rather than being reported as drift.
            return HistoricalReplayResult(
                fit_id=record.fit_id,
                outcome="degraded",
                reproduced_by_current_code=True,
                reconstructed=True,
                stage=STAGE_COMPARE,
                reason=(
                    "historical run diverged from golden, but the record was "
                    f"captured from a DIRTY tree (code_dirty=true): the committed "
                    f"sha {record.code_git_sha[:12]} is not the exact code that "
                    f"produced the golden -- {failures[0]}"
                ),
                checked=baseline.checked,
                failures=failures,
                elapsed=elapsed,
            )
        return HistoricalReplayResult(
            fit_id=record.fit_id,
            outcome="drift",
            reproduced_by_current_code=True,
            reconstructed=True,
            stage=STAGE_COMPARE,
            reason=(
                "historical environment rebuilt cleanly but its run diverged "
                "from the golden -- the current environment masks this drift"
            ),
            checked=baseline.checked,
            failures=failures,
            elapsed=elapsed,
        )
    finally:
        # Always tear the worktree down: remove the registration, then prune,
        # then drop the temp dir we own.
        _run_subprocess(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=root,
            timeout=120,
        )
        _run_subprocess(["git", "worktree", "prune"], cwd=root, timeout=60)
        if owns_workdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)


def _environment_snapshot() -> dict:
    """Return the software-environment mapping stored in a run record."""
    return {
        "packages": _environment.package_versions(),
        "blas_single_thread": _environment.threads_pinned(),
    }


def _fit_config_for(lane: str, entry) -> FitConfig:
    """Return the fit configuration describing a lane's actual parameters."""
    extra = {
        "entry_point": lane,
        "callable": entry.callable,
    }
    if lane == "sector.fit.ssat":
        extra.update(
            {
                "phase": "SSAT BR",
                "sectors": {"7": [8, 9]},
                "private": False,
                "augment": True,
                "version": "latest",
            }
        )
    return FitConfig(random_state=_PCA_RANDOM_STATE, extra=extra)


def write_pilot_record(lane: str, entry) -> Path:
    """Write the immutable run record for a freshly captured golden.

    Parameters
    ----------
    lane
        Entry-point id, used as the ``fit_id`` and the record filename.
    entry
        The registry :class:`EntryPoint` that produced the golden.
    """
    root = _environment.repo_root()
    arrays = _canonical.load_npz((_manifest.GOLDENS_DIR / f"{lane}.npz").read_bytes())

    input_digests = {
        Path(rel).stem: _prefixed(_canonical.sha256_file(root / rel))
        for rel in entry.inputs
    }

    outputs = [
        {
            "name": key,
            "digest": _array_digest(arrays, key),
            "tolerance_class": entry.tolerance_for(key),
        }
        for key in sorted(arrays)
    ]

    fit_config = _fit_config_for(lane, entry)
    env = capture_environment(root)
    record = RunRecord(
        fit_id=lane,
        code_git_sha=env["code_git_sha"],
        code_dirty=env["code_dirty"],
        uv_lock_sha256=env["uv_lock_sha256"],
        input_digests=input_digests,
        fit_config_sha256=fit_config.sha256,
        outputs=outputs,
        env=_environment_snapshot(),
        operator="characterization-harness",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    RUN_RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUN_RECORDS_DIR / f"{lane}.yaml"
    record.write(path)
    # Persist the fit configuration beside the record so it is inspectable and
    # its digest is verifiable.
    fit_config.write(RUN_RECORDS_DIR / f"{lane}.fitconfig.yaml")
    return path
