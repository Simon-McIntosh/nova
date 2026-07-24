"""Additively capture the sector-fit golden and its run record.

Unlike :mod:`generate_goldens`, which rewrites every golden, this generator
touches only the sector-module-backed lanes that this environment can now run
from the in-repo canonical corpus. It captures each runnable lane's canonical
output as a ``.npz`` golden, folds a manifest entry in beside the existing
records without disturbing them, and emits an immutable run record binding the
fit to the canonical unit digests, the fit configuration, the environment, and
the golden it produced.

Run under the pinned numeric environment::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \\
        python -m tests.characterization.generate_sector_goldens
"""

from __future__ import annotations

import sys

import numpy as np

from . import _canonical, _environment, _fixtures, _manifest, _registry, _replay

# Lanes this generator is responsible for (sector-module backed).
_TARGET_LANES = ("sector.fit.ssat",)


def _input_records(root, inputs) -> list[dict[str, str]]:
    records = []
    for rel in inputs:
        path = root / rel
        records.append(
            {
                "path": rel,
                "sha256": _canonical.sha256_file(path) if path.exists() else "missing",
            }
        )
    return records


def generate() -> None:
    """Capture goldens + run records for the runnable sector lanes."""
    _environment.require_pinned_threads()
    _fixtures.ensure_sector_cache()
    root = _environment.repo_root()

    manifest = (
        _manifest.Manifest.load()
        if _manifest.manifest_exists()
        else _manifest.fresh_manifest()
    )
    entries = {entry.id: entry for entry in _registry.registry()}

    for lane in _TARGET_LANES:
        entry = entries[lane]
        reason = entry.skip_reason()
        if reason is not None:
            print(f"skip  {lane}: {reason}")
            continue

        arrays = _canonical.canonicalize(entry.run())
        if not arrays:
            raise RuntimeError(f"{lane} produced no numeric output to characterize")

        payload = _canonical.to_npz_bytes(arrays)
        artifact_name = f"{lane}.npz"
        (_manifest.GOLDENS_DIR / artifact_name).write_bytes(payload)
        artifact_sha = _canonical.sha256_bytes(payload)

        array_meta = {
            key: {
                "shape": list(np.asarray(value).shape),
                "tolerance": entry.tolerance_for(key),
            }
            for key, value in sorted(arrays.items())
        }
        manifest.entries[lane] = _manifest.ManifestEntry(
            callable=entry.callable,
            inputs=_input_records(root, entry.inputs),
            artifact=f"goldens/{artifact_name}",
            artifact_sha256=artifact_sha,
            arrays=array_meta,
        )
        print(f"wrote {lane}: {len(arrays)} array(s) -> {artifact_name}")

        record_path = _replay.write_pilot_record(lane, entry)
        print(f"record {lane}: {record_path.name}")

    manifest.dump()
    print(f"\nmanifest: {len(manifest.entries)} entries total")


if __name__ == "__main__":
    generate()
    sys.exit(0)
