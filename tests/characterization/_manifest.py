"""Read and write the goldens manifest.

The manifest is the record: for each characterized entry point it names the
recorded inputs and their digests, the callable that was run, the serialized
canonical output and its sha256 fingerprint, and the tolerance class of every
array in that output. It also stamps the environment-lock hash under which the
goldens were generated.

Layout (``goldens/manifest.json``)::

    {
      "env_lock": "<sha256>",
      "package_versions": {"numpy": "1.26.4", ...},
      "entries": {
        "<entry-point-id>": {
          "callable": "module:qualname",
          "inputs": [{"path": "data/Assembly/...", "sha256": "..."}],
          "artifact": "goldens/<id>.npz",
          "artifact_sha256": "...",
          "arrays": {"<array-key>": {"shape": [...], "tolerance": "length_mm"}}
        }
      }
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import _environment

GOLDENS_DIR = Path(__file__).resolve().parent / "goldens"
MANIFEST_PATH = GOLDENS_DIR / "manifest.json"


@dataclass
class ManifestEntry:
    """One characterized entry point's record."""

    callable: str
    inputs: list[dict[str, str]]
    artifact: str
    artifact_sha256: str
    arrays: dict[str, dict]


@dataclass
class Manifest:
    """The full goldens manifest."""

    env_lock: str
    package_versions: dict[str, str]
    entries: dict[str, ManifestEntry] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = MANIFEST_PATH) -> "Manifest":
        raw = json.loads(Path(path).read_text())
        entries = {
            key: ManifestEntry(**value) for key, value in raw.get("entries", {}).items()
        }
        return cls(
            env_lock=raw.get("env_lock", ""),
            package_versions=raw.get("package_versions", {}),
            entries=entries,
        )

    def dump(self, path: Path = MANIFEST_PATH) -> None:
        payload = {
            "env_lock": self.env_lock,
            "package_versions": self.package_versions,
            "entries": {
                key: {
                    "callable": entry.callable,
                    "inputs": entry.inputs,
                    "artifact": entry.artifact,
                    "artifact_sha256": entry.artifact_sha256,
                    "arrays": entry.arrays,
                }
                for key, entry in sorted(self.entries.items())
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def fresh_manifest() -> Manifest:
    """Return an empty manifest stamped with the current environment."""
    return Manifest(
        env_lock=_environment.env_lock(),
        package_versions=_environment.package_versions(),
    )


def manifest_exists() -> bool:
    return MANIFEST_PATH.exists()
