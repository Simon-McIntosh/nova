"""Tier-1 replay: re-run a recorded fit with current code and re-check it.

A *run record* (``data/Assembly/run_records/<fit_id>.yaml``) binds a fit's
outputs to the canonical-unit inputs, the fit configuration, the environment,
and the golden it produced. Tier-1 replay resolves a record's input digests
against the units available here, re-runs the recorded entry point with the
*current* code, and compares the fresh output to the committed golden at the
tolerance classes the record pinned -- proving the fit still reproduces what
the record says it did.

Tier-2 replay (reconstructing the record's *historical* environment -- its git
revision and dependency lock -- and re-running under that) is future work; a
marker test names it without implementing it.

The pilot records are emitted by :func:`write_pilot_record`, invoked from the
sector golden generator so the first records describe real, freshly captured
fits rather than hand-written stubs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from nova.assembly.provenance import FitConfig, RunRecord, capture_environment

from . import _canonical, _environment, _fixtures, _manifest, _registry, _tolerance

RUN_RECORDS_DIR = _environment.repo_root() / "data" / "Assembly" / "run_records"
_SHA256_PREFIX = "sha256:"

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


def replay(record: RunRecord) -> ReplayResult:
    """Re-run a record's fit and compare it to the committed golden.

    Assumes the record's inputs resolve (see :func:`unresolved_inputs`) and its
    entry point is runnable here. Verifies the golden still matches the digest
    the record pinned, then re-runs the entry point and compares every output
    array to the golden at the record's tolerance class.
    """
    entries = {entry.id: entry for entry in _registry.registry()}
    entry = entries[record.fit_id]

    artifact = _manifest.GOLDENS_DIR / f"{record.fit_id}.npz"
    golden = _canonical.load_npz(artifact.read_bytes())

    failures: list[str] = []
    candidate = _canonical.canonicalize(entry.run())

    for output in record.outputs:
        key = output["name"]
        tol = output["tolerance_class"]
        if key not in golden:
            failures.append(f"[{key}] recorded output absent from golden")
            continue
        if key not in candidate:
            failures.append(f"[{key}] live run did not produce this output")
            continue
        # Integrity: the golden must still match the digest the record pinned.
        if _array_digest(golden, key) != output["digest"]:
            failures.append(f"[{key}] golden drifted from the digest the record pinned")
        result = _tolerance.compare(candidate[key], golden[key], tol)
        if not result.passed:
            failures.append(f"[{key}] ({tol}) {result.detail}")

    return ReplayResult(
        fit_id=record.fit_id,
        passed=not failures,
        checked=len(record.outputs),
        failures=failures,
    )


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
