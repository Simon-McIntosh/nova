"""Author and publish one local MAST machine-description artifact revision.

The command writes no registry and contacts no remote: it authors the seeded DD
4.1.1 set from the packaged registry, round-trips it through the dictionary pin,
and publishes it into a content-addressed local cache.

The cache location has to be chosen, not defaulted. Publication completes with an
atomic no-clobber directory rename so a reader never sees a half-written object,
and several parallel filesystems reject that operation, which would make any
shared default fail on some machines and silently work on others.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.machine_evidence import FieldEvidence
from nova.imas.mast_geometry import REPRESENTATIVE_SHOT, publish_provisional_artifact


def _report(cache_directory: Path, shot: int) -> dict[str, object]:
    """Publish one revision and summarize what a consumer can rely on."""

    artifact = publish_provisional_artifact(cache_directory, shot=shot)
    manifest = artifact.manifest
    ledger = manifest.evidence
    return {
        "artifact_digest": artifact.digest,
        "semantic_identity": manifest.semantic_identity(),
        "directory": str(artifact.directory),
        "dd_version": manifest.dd_version,
        "physical_digest": manifest.physical_digest,
        "registry_digest": manifest.registry_digest,
        "evidence_digest": ledger.digest,
        "oci_tag": manifest.oci.tag,
        "complete": manifest.complete,
        "files": [artifact_file.name for artifact_file in manifest.files],
        "shot_ranges": [
            {
                "first_shot": shot_range.first_shot,
                "last_shot": shot_range.last_shot,
                "evidence": shot_range.evidence,
            }
            for shot_range in manifest.shot_ranges
        ],
        "evidence_states": ledger.state_counts(),
        "unresolved_fields": list(ledger.paths_with_state(FieldEvidence.UNRESOLVED)),
        "forward_model_blockers": list(manifest.forward_model_blockers()),
        "unresolved_gaps": list(manifest.unresolved_gaps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a provisional MAST machine-description artifact."
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shot", type=int, default=REPRESENTATIVE_SHOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    registry = MachineGeometryRegistry.default()
    if not any(shot_range.contains(args.shot) for shot_range in registry.ranges):
        raise SystemExit(f"shot {args.shot} is outside the MAST geometry registry")
    text = json.dumps(_report(args.cache, args.shot), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
