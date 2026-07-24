"""Provenance and preservation of recorded assembly metrology.

This package is the provenance substrate for the coil-metrology corpus. It
keeps measured ITER coil-alignment data verifiable and reproducible:

* :mod:`~nova.assembly.provenance.digest` -- streaming sha256 of files, byte
  strings, and directory trees, formatted ``"sha256:<hex>"``.
* :mod:`~nova.assembly.provenance.yamlio` -- deterministic, round-trippable
  YAML serialization used for every stored document.
* :mod:`~nova.assembly.provenance.sourcemeta` -- structured metadata parsed
  from corpus filenames.
* :mod:`~nova.assembly.provenance.manifest` -- the preservation manifest that
  content-addresses a corpus tree and verifies it against a known state.
* :mod:`~nova.assembly.provenance.fitconfig` -- schema-validated declarative
  fit configuration.
* :mod:`~nova.assembly.provenance.runrecord` -- the immutable run record that
  binds a fit's outputs to the code, inputs, and environment that produced it.

The substrate depends only on the standard library, PyYAML, and packaging; it
deliberately does not import spreadsheet or dataframe libraries so that the
workbook transcoder can depend on it without pulling in those dependencies.
"""

from nova.assembly.provenance import (
    digest,
    fitconfig,
    manifest,
    runrecord,
    sourcemeta,
    yamlio,
)
from nova.assembly.provenance.digest import (
    digest_bytes,
    digest_file,
    digest_tree,
    mtime_iso,
)
from nova.assembly.provenance.fitconfig import FitConfig
from nova.assembly.provenance.manifest import (
    build_manifest,
    load_manifest,
    verify,
    write_manifest,
)
from nova.assembly.provenance.runrecord import RunRecord, capture_environment
from nova.assembly.provenance.sourcemeta import SourceMeta, parse_filename
from nova.assembly.provenance.yamlio import (
    canonical_yaml_bytes,
    dump_yaml,
    load_yaml,
)

__all__ = [
    "digest",
    "yamlio",
    "sourcemeta",
    "manifest",
    "fitconfig",
    "runrecord",
    "digest_bytes",
    "digest_file",
    "digest_tree",
    "mtime_iso",
    "canonical_yaml_bytes",
    "dump_yaml",
    "load_yaml",
    "SourceMeta",
    "parse_filename",
    "build_manifest",
    "write_manifest",
    "load_manifest",
    "verify",
    "FitConfig",
    "RunRecord",
    "capture_environment",
]
