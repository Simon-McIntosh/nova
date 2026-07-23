"""Component lane -- each entry point re-run and compared against its golden.

This is the protection gate: every registered fitting/metrology entry point is
re-run on its recorded inputs and every canonical output array is compared to
the committed golden at that array's tolerance class. A sha256 fingerprint
match is reported for information; the pass/fail decision is the tolerance
comparison, so the gate survives BLAS/dependency bumps without re-baselining.

Entry points gated by a missing optional dependency (ANSYS, pyvista drift)
degrade to a skip -- visible, never a silent pass.
"""

from __future__ import annotations

import pytest

from . import _canonical, _manifest, _registry, _tolerance

ENTRIES = _registry.registry()


def _ids(entries):
    return [entry.id for entry in entries]


@pytest.mark.parametrize("entry", ENTRIES, ids=_ids(ENTRIES))
def test_entry_point_matches_golden(entry, manifest, goldens_dir):
    reason = entry.skip_reason()
    if reason is not None:
        pytest.skip(reason)
    if entry.id not in manifest.entries:
        pytest.skip(f"no golden recorded for {entry.id}")

    record = manifest.entries[entry.id]
    artifact = goldens_dir / f"{entry.id}.npz"
    if not artifact.exists():
        pytest.skip(f"golden artifact missing: {artifact.name}")

    golden = _canonical.load_npz(artifact.read_bytes())
    candidate = _canonical.canonicalize(entry.run())

    missing = set(golden) - set(candidate)
    extra = set(candidate) - set(golden)
    assert not missing, (
        f"{entry.id}: candidate is missing golden arrays {sorted(missing)}"
    )
    assert not extra, (
        f"{entry.id}: candidate produced unexpected arrays {sorted(extra)}"
    )

    failures = []
    for key in sorted(golden):
        tol = entry.tolerance_for(key)
        result = _tolerance.compare(candidate[key], golden[key], tol)
        if not result.passed:
            failures.append(f"  [{key}] ({tol}) {result.detail}")

    # Fingerprint is a change detector only; report it, do not gate on it.
    payload = _canonical.to_npz_bytes({k: candidate[k] for k in sorted(candidate)})
    fingerprint_matches = _canonical.sha256_bytes(payload) == record.artifact_sha256
    if not fingerprint_matches and not failures:
        # Within tolerance but bits moved -- acceptable, but worth surfacing.
        print(
            f"note: {entry.id} within tolerance but fingerprint changed "
            "(expected after a BLAS/dependency bump)"
        )

    assert not failures, f"{entry.id} exceeded tolerance:\n" + "\n".join(failures)


def test_manifest_env_lock_present():
    """The manifest must record the environment it was generated under."""
    if not _manifest.manifest_exists():
        pytest.skip("goldens manifest not generated yet")
    manifest = _manifest.Manifest.load()
    assert manifest.env_lock, "manifest is missing its environment-lock hash"
    assert "numpy" in manifest.package_versions
