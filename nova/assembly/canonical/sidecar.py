"""Canonical sidecar: provenance and the layout needed to rebuild a workbook.

Each canonical unit pairs its long-format CSV with a YAML sidecar. The CSV
holds the measured values; the sidecar holds everything else needed to trust
and to reverse the transcode:

* the source file's identity -- name, sha256, size, and modification time;
* the metadata parsed from the source filename -- sector, IDM document,
  version, private flag, and any revision prefix;
* the per-sheet block layout -- header positions, block widths, and transform
  positions -- so a workbook can be rebuilt at the recorded coordinates;
* the free text preserved from the ``Metadata`` sheet;
* the anomalies noted during parsing;
* the sha256 of the canonical CSV, binding the two files together.

The sidecar serializes through the shared deterministic YAML writer, so an
unchanged input re-emits byte-identically.
"""

from __future__ import annotations

from nova.assembly.provenance import yamlio

# Identifier for the canonical layout this module reads and writes. Bump the
# trailing number only on an incompatible change to the sidecar shape.
SCHEMA_ID = "nova-coil-metrology-canonical/1"


def _block_layout(block, transform):
    """Return the serializable layout of one coil block.

    Parameters
    ----------
    block : workbook.CoilBlock
        Parsed coil block.
    transform : workbook.Transform or None
        Transform tied to the block, if any.

    Returns
    -------
    dict
        Header position, width, uncertainty flag, and transform position.
    """
    layout = {
        "coil": block.coil,
        "header_row": block.header_row,
        "header_col": block.header_col,
        "width": block.width,
        "has_uncertainty": block.has_uncertainty,
        "transform": None,
    }
    if transform is not None:
        layout["transform"] = {
            "label_row": transform.label_row,
            "label_col": transform.label_col,
            "populated": transform.populated,
        }
    return layout


def _sheet_layout(sheet):
    """Return the serializable layout of one sheet.

    Parameters
    ----------
    sheet : workbook.SheetParse
        Parsed sheet.

    Returns
    -------
    dict
        Sheet name and its ordered block layouts, plus any transform blocks
        not tied to a coil block.
    """
    by_col: dict[int, list] = {}
    for transform in sheet.transforms:
        by_col.setdefault(transform.header_col, []).append(transform)

    used: set[int] = set()
    blocks = []
    for block in sheet.blocks:
        candidates = by_col.get(block.header_col, [])
        transform = candidates[0] if candidates else None
        if transform is not None:
            used.add(id(transform))
        blocks.append(_block_layout(block, transform))

    layout = {"name": sheet.name, "blocks": blocks}
    extra = [
        {
            "label_row": transform.label_row,
            "label_col": transform.label_col,
            "header_col": transform.header_col,
            "populated": transform.populated,
        }
        for transform in sheet.transforms
        if id(transform) not in used
    ]
    if extra:
        layout["unmatched_transforms"] = extra
    return layout


def collect_anomalies(parse) -> list[str]:
    """Aggregate per-sheet and workbook anomalies into one list.

    Parameters
    ----------
    parse : workbook.WorkbookParse
        Parsed workbook.

    Returns
    -------
    list of str
        Anomalies, each sheet anomaly prefixed with its sheet name.
    """
    anomalies = []
    for sheet in parse.sheets:
        anomalies.extend(f"{sheet.name}: {note}" for note in sheet.anomalies)
    anomalies.extend(parse.anomalies)
    return anomalies


def build_sidecar(parse, *, source, filename_meta, csv_sha256) -> dict:
    """Assemble the sidecar mapping for a parsed workbook.

    Parameters
    ----------
    parse : workbook.WorkbookParse
        Parsed workbook structure.
    source : dict
        Source-file identity: ``filename``, ``sha256``, ``size``, ``mtime``.
    filename_meta : dict
        Metadata parsed from the filename.
    csv_sha256 : str
        Digest of the canonical CSV, formatted ``"sha256:<hex>"``.

    Returns
    -------
    dict
        A YAML-serializable sidecar mapping.
    """
    return {
        "schema": SCHEMA_ID,
        "source": dict(source),
        "filename_meta": dict(filename_meta),
        "sheets": [_sheet_layout(sheet) for sheet in parse.sheets],
        "metadata_text": [
            {"row": row, "col": col, "text": text}
            for row, col, text in parse.metadata_text
        ],
        "anomalies": collect_anomalies(parse),
        "csv_sha256": csv_sha256,
    }


def emit_sidecar(sidecar) -> bytes:
    """Serialize a sidecar mapping to deterministic canonical YAML bytes.

    Parameters
    ----------
    sidecar : dict
        Sidecar mapping.

    Returns
    -------
    bytes
        Canonical YAML.
    """
    return yamlio.canonical_yaml_bytes(sidecar)


def load_sidecar_bytes(data):
    """Parse a sidecar from canonical YAML bytes.

    Parameters
    ----------
    data : bytes or str
        Sidecar YAML.

    Returns
    -------
    dict
        The parsed sidecar mapping.
    """
    return yamlio.loads(data)


def write_sidecar(sidecar, path) -> None:
    """Write a sidecar mapping to a file as canonical YAML.

    Parameters
    ----------
    sidecar : dict
        Sidecar mapping.
    path : os.PathLike or str
        Destination file.
    """
    yamlio.dump_yaml(sidecar, path)


def load_sidecar(path):
    """Read and parse a sidecar file.

    Parameters
    ----------
    path : os.PathLike or str
        Sidecar file.

    Returns
    -------
    dict
        The parsed sidecar mapping.
    """
    return yamlio.load_yaml(path)
