"""Generate (or regenerate) the canonical goldens and their manifest.

Run under the pinned numeric environment::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg \\
        python -m tests.characterization.generate_goldens

For each registered entry point this runs the operation, canonicalizes the
result to a sorted ``.npz``, writes it under ``goldens/``, records the input
digests, output fingerprint and per-array tolerance classes in the manifest,
and stamps the environment-lock hash. Entry points gated by a missing optional
dependency are recorded as skipped with their reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from . import _canonical, _environment, _manifest, _registry


def _input_records(root: Path, inputs) -> list[dict[str, str]]:
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


def generate() -> _manifest.Manifest:
    """Generate all goldens and return the written manifest."""
    _environment.require_pinned_threads()
    root = _environment.repo_root()
    manifest = _manifest.fresh_manifest()
    _manifest.GOLDENS_DIR.mkdir(parents=True, exist_ok=True)

    skipped: dict[str, str] = {}
    for entry in _registry.registry():
        reason = entry.skip_reason()
        if reason is not None:
            skipped[entry.id] = reason
            print(f"skip  {entry.id}: {reason}")
            continue

        result = entry.run()
        arrays = _canonical.canonicalize(result)
        if not arrays:
            raise RuntimeError(f"{entry.id} produced no numeric output to characterize")

        payload = _canonical.to_npz_bytes(arrays)
        artifact_name = f"{entry.id}.npz"
        (_manifest.GOLDENS_DIR / artifact_name).write_bytes(payload)

        array_meta = {
            key: {
                "shape": list(np.asarray(value).shape),
                "tolerance": entry.tolerance_for(key),
            }
            for key, value in sorted(arrays.items())
        }
        manifest.entries[entry.id] = _manifest.ManifestEntry(
            callable=entry.callable,
            inputs=_input_records(root, entry.inputs),
            artifact=f"goldens/{artifact_name}",
            artifact_sha256=_canonical.sha256_bytes(payload),
            arrays=array_meta,
        )
        print(f"wrote {entry.id}: {len(arrays)} array(s) -> {artifact_name}")

    manifest.dump()
    if skipped:
        (_manifest.GOLDENS_DIR / "skipped.txt").write_text(
            "\n".join(f"{k}\t{v}" for k, v in sorted(skipped.items())) + "\n"
        )
    print(f"\nmanifest: {len(manifest.entries)} entries, {len(skipped)} skipped")
    print(f"env_lock: {manifest.env_lock}")
    return manifest


if __name__ == "__main__":
    generate()
    sys.exit(0)
