"""Git-safe canonical form for the coil-metrology corpus.

This package transcodes measured coil-alignment workbooks into a
deterministic, content-addressable canonical form and back. The canonical
form of one workbook is a long-format CSV (one row per atomic measured value)
paired with a YAML sidecar recording provenance and the layout needed to
rebuild the workbook. Both are byte-stable, so a sha256 over the CSV is a
meaningful content address, and the round trip
workbook -> canonical -> workbook preserves every numeric value exactly.

Public entry points:

* :func:`~nova.assembly.canonical.transcode.ingest_workbook` -- workbook to
  :class:`~nova.assembly.canonical.transcode.CanonicalUnit`.
* :func:`~nova.assembly.canonical.transcode.write_unit` /
  :func:`~nova.assembly.canonical.transcode.read_unit` -- persist and reload.
* :func:`~nova.assembly.canonical.transcode.egress_workbook` -- rebuild a
  workbook from a unit.
* :func:`~nova.assembly.canonical.transcode.ingest_tree` -- sweep a directory.
"""

from nova.assembly.canonical import longformat, sidecar, transcode, workbook
from nova.assembly.canonical.longformat import (
    Row,
    csv_bytes_to_rows,
    normalize_feature,
    normalize_point_group,
    rows_from_parse,
    rows_to_csv_bytes,
)
from nova.assembly.canonical.transcode import (
    CanonicalUnit,
    egress_workbook,
    ingest_tree,
    ingest_workbook,
    read_unit,
    write_unit,
)
from nova.assembly.canonical.workbook import parse_workbook

__all__ = [
    "workbook",
    "longformat",
    "sidecar",
    "transcode",
    "parse_workbook",
    "Row",
    "rows_from_parse",
    "rows_to_csv_bytes",
    "csv_bytes_to_rows",
    "normalize_point_group",
    "normalize_feature",
    "CanonicalUnit",
    "ingest_workbook",
    "write_unit",
    "read_unit",
    "egress_workbook",
    "ingest_tree",
]
