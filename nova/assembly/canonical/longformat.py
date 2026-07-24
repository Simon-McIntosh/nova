"""Long-format canonical rows and their deterministic CSV encoding.

The canonical body of a workbook is one CSV row per atomic measured value:
each fiducial coordinate and each rigid-body transform component becomes its
own row. This maximally diffable shape means a changed measurement touches a
single line. Serialization is deterministic -- stable row order, ``repr``
float formatting for shortest exact round-trip, LF line endings, minimal
quoting -- so the bytes are reproducible and a digest over them is a
meaningful content address.

Point-group and feature labels are normalized (folding the corpus's typos and
spacing) while the raw label is retained alongside, so grouping is reliable
without discarding provenance.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import io

from nova.assembly.canonical import workbook

RECORD_FIDUCIAL = "fiducial"
RECORD_TRANSFORM = "transform"

UNITS_POSITION = "mm"

COLUMNS = (
    "coil",
    "phase",
    "record_kind",
    "point_group",
    "point_group_raw",
    "feature",
    "feature_raw",
    "axis",
    "value",
    "uncertainty",
    "units",
    "is_formula",
)


def normalize_point_group(raw: str | None) -> str:
    """Fold a point-group label to its canonical spelling.

    Parameters
    ----------
    raw : str or None
        Label as written in the sheet.

    Returns
    -------
    str
        Canonical label. Known typos and case/spacing variants of
        ``"Measured"`` and ``"Best Fitted"`` are folded; every other label is
        returned with surrounding whitespace stripped.
    """
    if raw is None:
        return ""
    stripped = str(raw).strip()
    lowered = stripped.lower()
    if lowered in ("mesured", "measured"):
        return "Measured"
    if lowered in ("best fitted", "best fit"):
        return "Best Fitted"
    return stripped


def normalize_feature(raw: str | None) -> str:
    """Fold a fiducial feature name to its canonical spelling.

    Parameters
    ----------
    raw : str or None
        Feature name as written.

    Returns
    -------
    str
        Canonical name. The prime marker on ``"F'"`` is dropped so it joins
        with ``"F"``; other names are returned with whitespace stripped.
    """
    if raw is None:
        return ""
    stripped = str(raw).strip()
    if stripped == "F'":
        return "F"
    return stripped


@dataclass(frozen=True)
class Row:
    """One atomic canonical measurement row.

    Parameters
    ----------
    coil : int or None
        Coil number, or ``None`` when the source block omitted it.
    phase : str
        Assembly phase (sheet name, verbatim).
    record_kind : str
        ``"fiducial"`` or ``"transform"``.
    point_group, point_group_raw : str
        Normalized and raw point-group label (empty for transforms).
    feature, feature_raw : str
        Normalized and raw feature name (empty for transforms).
    axis : str
        ``x``/``y``/``z`` for fiducials; ``dx``/``dy``/``dz``/``rx``/``ry``/
        ``rz`` for transforms.
    value : float or None
        Measured value, or ``None`` when absent.
    uncertainty : float or None
        Two-sigma uncertainty, or ``None`` when absent.
    units : str
        ``"mm"`` for positions and translations, ``"deg"`` for rotations.
    is_formula : bool
        ``True`` when the source value or uncertainty cell was a formula.
    """

    coil: int | None
    phase: str
    record_kind: str
    point_group: str
    point_group_raw: str
    feature: str
    feature_raw: str
    axis: str
    value: float | None
    uncertainty: float | None
    units: str
    is_formula: bool


def _sheet_rows(sheet):
    """Yield canonical rows for one parsed sheet in deterministic order."""
    transforms_by_col: dict[int, list] = {}
    for transform in sheet.transforms:
        transforms_by_col.setdefault(transform.header_col, []).append(transform)
    consumed: set[int] = set()

    for block in sheet.blocks:
        for index, transform in enumerate(transforms_by_col.get(block.header_col, [])):
            consumed.add(id(transform))
            yield from _transform_rows(transform, sheet.name)
        yield from _fiducial_rows(block, sheet.name)

    for transform in sheet.transforms:
        if id(transform) not in consumed:
            yield from _transform_rows(transform, sheet.name)


def _transform_rows(transform, phase):
    """Yield rows for a transform block (none when it is empty)."""
    if not transform.populated:
        return
    for axis in workbook.TRANSFORM_AXES:
        cell = transform.components[axis]
        yield Row(
            coil=transform.coil,
            phase=phase,
            record_kind=RECORD_TRANSFORM,
            point_group="",
            point_group_raw="",
            feature="",
            feature_raw="",
            axis=axis,
            value=cell.value,
            uncertainty=None,
            units=workbook.TRANSFORM_UNITS[axis],
            is_formula=cell.is_formula,
        )


def _fiducial_rows(block, phase):
    """Yield x/y/z rows for each fiducial in a coil block."""
    for fiducial in block.fiducials:
        point_group = normalize_point_group(fiducial.point_group_raw)
        feature = normalize_feature(fiducial.name_raw)
        for axis in workbook.COORD_AXES:
            coord = fiducial.coords[axis]
            uncert = fiducial.uncerts[axis]
            yield Row(
                coil=block.coil,
                phase=phase,
                record_kind=RECORD_FIDUCIAL,
                point_group=point_group,
                point_group_raw=fiducial.point_group_raw,
                feature=feature,
                feature_raw=fiducial.name_raw,
                axis=axis,
                value=coord.value,
                uncertainty=uncert.value,
                units=UNITS_POSITION,
                is_formula=coord.is_formula or uncert.is_formula,
            )


def rows_from_parse(parse) -> list[Row]:
    """Build canonical rows from a parsed workbook.

    Parameters
    ----------
    parse : workbook.WorkbookParse
        Parsed workbook structure.

    Returns
    -------
    list of Row
        Rows in canonical order: sheet order, then block order left to right,
        transform components before fiducials, then top-to-bottom rows with
        axes in ``x``, ``y``, ``z`` order.
    """
    rows: list[Row] = []
    for sheet in parse.sheets:
        rows.extend(_sheet_rows(sheet))
    return rows


def _format_value(value: float | None) -> str:
    """Format a float for exact round-trip, or empty string for ``None``."""
    if value is None:
        return ""
    return repr(float(value))


def _format_coil(coil: int | None) -> str:
    """Format a coil number, or empty string for ``None``."""
    return "" if coil is None else str(coil)


def _record(row: Row) -> list[str]:
    """Render a row as a list of CSV field strings."""
    return [
        _format_coil(row.coil),
        row.phase,
        row.record_kind,
        row.point_group,
        row.point_group_raw,
        row.feature,
        row.feature_raw,
        row.axis,
        _format_value(row.value),
        _format_value(row.uncertainty),
        row.units,
        "true" if row.is_formula else "false",
    ]


def rows_to_csv_bytes(rows) -> bytes:
    """Serialize rows to deterministic canonical CSV bytes.

    Parameters
    ----------
    rows : iterable of Row
        Rows to serialize.

    Returns
    -------
    bytes
        UTF-8 CSV with a header line, LF line endings, and minimal quoting.
    """
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(COLUMNS)
    for row in rows:
        writer.writerow(_record(row))
    return buffer.getvalue().encode("utf-8")


def _parse_value(text: str) -> float | None:
    """Parse a formatted value field back to a float or ``None``."""
    if text == "":
        return None
    return float(text)


def _parse_coil(text: str) -> int | None:
    """Parse a coil field back to an int or ``None``."""
    if text == "":
        return None
    return int(text)


def csv_bytes_to_rows(data: bytes) -> list[Row]:
    """Parse canonical CSV bytes back into rows.

    Parameters
    ----------
    data : bytes
        Canonical CSV produced by :func:`rows_to_csv_bytes`.

    Returns
    -------
    list of Row
        The parsed rows.

    Raises
    ------
    ValueError
        If the header line does not match the canonical column set.
    """
    text = data.decode("utf-8")
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader, None)
    if header is None or tuple(header) != COLUMNS:
        raise ValueError("CSV header does not match the canonical schema")
    rows: list[Row] = []
    for record in reader:
        rows.append(
            Row(
                coil=_parse_coil(record[0]),
                phase=record[1],
                record_kind=record[2],
                point_group=record[3],
                point_group_raw=record[4],
                feature=record[5],
                feature_raw=record[6],
                axis=record[7],
                value=_parse_value(record[8]),
                uncertainty=_parse_value(record[9]),
                units=record[10],
                is_formula=record[11] == "true",
            )
        )
    return rows
