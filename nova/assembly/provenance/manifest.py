"""The preservation manifest: a content inventory of a corpus tree.

A manifest is a YAML document that records, for every file beneath a source
root, its content digest, byte size, modification time, and the metadata parsed
from its name. It exists so a corpus of measured metrology can be verified bit
for bit against a known-good state at any later date.

The manifest is deterministic: building it twice against an unchanged tree
yields byte-identical canonical YAML. To achieve that, the document contains no
generation timestamp -- *when* the manifest was produced is recorded by git
when it is committed, not baked into the content (which would defeat content
addressing). File modification times, by contrast, are *data* about the
sources and are preserved as UTC ISO-8601 strings.
"""

import os
from pathlib import Path

from nova.assembly.provenance import digest, sourcemeta, yamlio

# Identifies the document shape so a future reader can dispatch on it.
SCHEMA_ID = "nova.assembly.provenance/manifest/1"

GENERATOR = "nova.assembly.provenance"


def build_manifest(
    root: os.PathLike | str,
    corpus_name: str,
    source_root_description: str = "",
) -> dict:
    """Build a preservation manifest for a corpus tree.

    Parameters
    ----------
    root : os.PathLike or str
        Directory whose files are inventoried.
    corpus_name : str
        Short name identifying the corpus.
    source_root_description : str, optional
        Human-readable description of where the source root lives.

    Returns
    -------
    dict
        Manifest document with a fixed header and a sorted ``files`` list.
    """
    tree = digest.digest_tree(root)
    files = []
    for relpath, entry in tree.items():
        meta = sourcemeta.parse_filename(relpath)
        files.append(
            {
                "path": relpath,
                "digest": entry["digest"],
                "size": entry["size"],
                "mtime": entry["mtime"],
                "metadata": meta.to_dict(),
            }
        )
    files.sort(key=lambda item: item["path"])
    return {
        "schema": SCHEMA_ID,
        "generator": GENERATOR,
        "corpus": corpus_name,
        "source_root": source_root_description,
        "files": files,
    }


def write_manifest(manifest: dict, path: os.PathLike | str) -> None:
    """Write a manifest to a file as canonical YAML.

    Parameters
    ----------
    manifest : dict
        Manifest document as produced by :func:`build_manifest`.
    path : os.PathLike or str
        Destination file.
    """
    yamlio.dump_yaml(manifest, path)


def load_manifest(path: os.PathLike | str) -> dict:
    """Load a manifest document from a YAML file.

    Parameters
    ----------
    path : os.PathLike or str
        Source file.

    Returns
    -------
    dict
        The decoded manifest document.
    """
    return yamlio.load_yaml(path)


def verify(root: os.PathLike | str, manifest: dict) -> dict:
    """Compare a tree against a manifest by re-digesting its files.

    Parameters
    ----------
    root : os.PathLike or str
        Directory to check.
    manifest : dict
        Reference manifest.

    Returns
    -------
    dict
        Report with sorted ``missing`` (in manifest, absent on disk),
        ``changed`` (digest differs), and ``new`` (on disk, absent from
        manifest) relative-path lists.
    """
    root = Path(root)
    recorded = {entry["path"]: entry["digest"] for entry in manifest["files"]}
    current = {
        relpath: entry["digest"] for relpath, entry in digest.digest_tree(root).items()
    }

    missing = [path for path in recorded if path not in current]
    changed = [
        path for path in recorded if path in current and current[path] != recorded[path]
    ]
    new = [path for path in current if path not in recorded]

    return {
        "missing": sorted(missing),
        "changed": sorted(changed),
        "new": sorted(new),
    }
