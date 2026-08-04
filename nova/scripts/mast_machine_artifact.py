"""Author and publish one local MAST machine-description artifact revision.

The command writes no registry and contacts no remote: it authors a DD 4.1.1 set
from the packaged registry, round-trips it through the dictionary pin, and
publishes it into a content-addressed local cache.

Two revisions can be authored and the command will not choose between them.  The
seeded revision carries only what public sources license, and its turn counts,
probe axes and circuit connections are unresolved; the refined revision carries
what the vacuum cohort measured on top of that, and it is the one a forward
operator can be built from.  The seeded revision is superseded, so publishing it
by accident would hand a consumer a description whose largest flux source has no
usable weight -- which is why ``--revision`` is required rather than defaulted.

The cache location has to be chosen for a different reason.  Publication
completes with an atomic no-clobber directory rename so a reader never sees a
half-written object, and several parallel filesystems reject that operation,
which would make any shared default fail on some machines and silently work on
others.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from nova.catalog.mast_geometry import MachineGeometryRegistry
from nova.imas.machine_evidence import FieldEvidence
from nova.imas.mast_artifact import VerifiedMachineArtifact
from nova.imas.mast_geometry import (
    REPRESENTATIVE_SHOT,
    publish_provisional_artifact,
    publish_refined_artifact,
)

SEEDED = "seeded"
"""What public sources license, before any shot has been read."""

REFINED = "refined"
"""The seeded description plus everything the vacuum cohort measured."""

_PUBLISHERS: dict[str, Callable[..., VerifiedMachineArtifact]] = {
    SEEDED: publish_provisional_artifact,
    REFINED: publish_refined_artifact,
}


def _report(revision: str, cache_directory: Path, shot: int) -> dict[str, object]:
    """Publish one revision and summarize what a consumer can rely on."""

    artifact = _PUBLISHERS[revision](cache_directory, shot=shot)
    manifest = artifact.manifest
    ledger = manifest.evidence
    drives = manifest.drive_map
    return {
        "revision": revision,
        "artifact_digest": artifact.digest,
        "semantic_identity": manifest.semantic_identity(),
        "directory": str(artifact.directory),
        "dd_version": manifest.dd_version,
        "physical_digest": manifest.physical_digest,
        "registry_digest": manifest.registry_digest,
        "evidence_digest": ledger.digest,
        "drive_digest": drives.digest,
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
        "driven_columns": len(manifest.driven_columns()),
        "channel_drive": [
            {
                "channel": drive.channel,
                "container": drive.container,
                "conductor": drive.conductor,
                "elements": list(drive.elements),
                "circuit": drive.circuit,
                "ampere_turns_per_ampere": drive.ampere_turns_per_ampere,
                "evidence": str(drive.evidence),
                "path": drive.path,
            }
            for drive in drives.drives
        ],
    }


def main() -> None:
    """Publish one named revision into a chosen content-addressed cache."""

    parser = argparse.ArgumentParser(
        description="Publish a MAST machine-description artifact revision."
    )
    parser.add_argument(
        "--revision",
        required=True,
        choices=sorted(_PUBLISHERS),
        help=(
            "which description to author: 'seeded' carries public sources only and "
            "is superseded; 'refined' carries the vacuum cohort's measurements"
        ),
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--shot", type=int, default=REPRESENTATIVE_SHOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    registry = MachineGeometryRegistry.default()
    if not any(shot_range.contains(args.shot) for shot_range in registry.ranges):
        raise SystemExit(f"shot {args.shot} is outside the MAST geometry registry")
    report = _report(args.revision, args.cache, args.shot)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
