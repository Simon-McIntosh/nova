"""Historical fit-parameter manifest for the coil-alignment corpus.

Every rigid-body fit the alignment engine produces is written back into a
workbook as a *transform* block: three translations and three intrinsic Euler
angles, one block per coil, on a target sheet. The canonical transcode
preserves those blocks as ``record_kind == "transform"`` rows, so a fit that
was ever recorded in a workbook is recoverable from the canonical CSV without
re-opening the spreadsheet.

What the workbook does *not* store is the configuration that produced the fit:
the component weights, the constraint index sets, the radial offset, the
Gaussian-process kernel, the random seed, and (latterly) the coupled-sector
switch all live in source, not in the sheet. Those controls have changed over
the project's history. This module carries :data:`PARAMETER_EPOCHS`, a curated
table of that evolution mined from the fit-engine git history: each epoch pins
the commit that changed a control, the date it landed, and the full parameter
set in force from then until the next change.

A manifest entry therefore has two parts. ``recorded`` holds the transform the
workbook actually stores -- the fit's output. ``inferred`` holds the parameter
set attributed to the fit by correlating the workbook's saved-modification time
with the epoch table, marked ``inferred: true`` with the commit, the epoch date
range, and the modification time used as evidence. The modification time is the
workbook's last save (or download) time; it is an *upper bound* on when the fit
ran, not the fit time itself, and the evidence note says so.

The manifest serializes through the deterministic canonical-YAML writer, so
rebuilding it over an unchanged canonical corpus yields byte-identical output.

Regeneration
------------
The committed ``data/Assembly/fit_manifest.yaml`` is built from the full
digest-verified canonical snapshot. To rebuild it against any canonical tree::

    from nova.assembly.provenance import fithistory
    manifest = fithistory.build_fit_manifest([canonical_dir, canonical_dir / "IDM"])
    fithistory.write_fit_manifest(manifest, "data/Assembly/fit_manifest.yaml")
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from nova.assembly.provenance import yamlio

SCHEMA_ID = "nova.assembly.provenance/fit-manifest/1"
GENERATOR = "nova.assembly.provenance.fithistory"

# Column order of the canonical measurement CSV. Mirrored here (rather than
# imported from the transcoder) so this module stays free of the spreadsheet
# dependency chain; the header is validated on every read.
_CSV_COLUMNS = (
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
_COL = {name: index for index, name in enumerate(_CSV_COLUMNS)}

_RECORD_TRANSFORM = "transform"

# Translation-then-rotation order the engine writes and the transcode records.
TRANSFORM_AXES = ("dx", "dy", "dz", "rx", "ry", "rz")

# Sheet-name substrings that mark a sheet as a fit-result sheet by convention.
_FIT_NAME_TOKENS = ("target", "fit")

# Radial offset introduced to manage the 36 mm cumulative inter-coil gap limit:
# (nominal_gap - limit) spread over the toroidal circumference. Matches the
# fit engine's ``(33.04 - 36) / (2 * pi)`` default exactly.
_RADIAL_OFFSET = (33.04 - 36) / (2 * math.pi)

# Gaussian-process kernel in force across the whole workbook-write era: a
# fixed-length periodic kernel plus a small constant term, with per-fiducial
# observation variance read from the workbook ("file"). The random seed was
# unset until it was pinned to 2025; ``random_state`` records that transition.
_GPR_BASE = {
    "kernel": "ExpSineSquared + ConstantKernel",
    "length_scale": 0.85,
    "length_scale_bounds": "fixed",
    "periodicity": 1.0,
    "periodicity_bounds": "fixed",
    "constant_value": 1e-8,
    "constant_value_bounds": (1e-22, 1e2),
    "alpha_source": "file",
    "random_state": None,
}

# Constraint index sets: which fiducials constrain each cylindrical component.
# ``toroidal: None`` means every fiducial constrains the toroidal component
# (the early sheets used no subset); a list is an explicit reduced set.
_INDEX_TOROIDAL_ALL = {
    "radial": [5, 3, 4],
    "toroidal": None,
    "vertical": [2, 1, -1, -2],
}
_INDEX_TOROIDAL_REDUCED = {
    "radial": [5, 3, 4],
    "toroidal": [0, 5, 3, 4],
    "vertical": [2, 1, -1, -2],
}


def _gpr(random_state):
    """Return a GPR parameter mapping with the given seed.

    Parameters
    ----------
    random_state : int or None
        Seed pinned on the regressor, or ``None`` before it was set.

    Returns
    -------
    dict
        Copy of the base GPR parameters with ``random_state`` applied and the
        bounds tuples rendered as lists for YAML.
    """
    params = dict(_GPR_BASE)
    params["constant_value_bounds"] = list(_GPR_BASE["constant_value_bounds"])
    params["random_state"] = random_state
    return params


def _parameters(*, weights, radial_offset, index, random_state, coupled):
    """Assemble a full fit-parameter snapshot for one epoch.

    Parameters
    ----------
    weights : list[float]
        Per-component observation weights ``[radial, toroidal, vertical]``.
    radial_offset : float
        Radial offset applied to the fit geometry.
    index : dict
        Constraint index sets (``radial``/``toroidal``/``vertical``).
    random_state : int or None
        Gaussian-process random seed.
    coupled : bool or None
        Coupled-sector switch, or ``None`` before it existed.

    Returns
    -------
    dict
        The parameter snapshot.
    """
    return {
        "method": "rms",
        "infer": True,
        "weights": list(weights),
        "radial_offset": radial_offset,
        "fiducial_index": {
            "radial": list(index["radial"]),
            "toroidal": None if index["toroidal"] is None else list(index["toroidal"]),
            "vertical": list(index["vertical"]),
        },
        "transform_dof": 6,
        "gpr": _gpr(random_state),
        "coupled": coupled,
    }


# Fit-parameter evolution mined from the fit-engine git history, oldest first.
# Each epoch pins the commit that changed a control, the date it landed, a
# one-line description of the change, and the full parameter set in force from
# that commit until the next epoch. Only the workbook-write era is covered:
# rigid-body transforms were first written back into workbooks on 2023-12-12,
# so no earlier code state can appear in a recorded fit.
_EPOCHS = (
    {
        "commit": "c5f52b8e",
        "date": "2023-12-12",
        "change": "First rigid-body fits written back into workbooks (TFC FAT).",
        "parameters": _parameters(
            weights=[1.0, 1.0, 0.5],
            radial_offset=0.0,
            index=_INDEX_TOROIDAL_ALL,
            random_state=None,
            coupled=None,
        ),
    },
    {
        "commit": "98983c19",
        "date": "2024-01-11",
        "change": "Rotational degrees of freedom activated in TFC fits.",
        "parameters": _parameters(
            weights=[1.0, 1.0, 0.5],
            radial_offset=0.0,
            index=_INDEX_TOROIDAL_ALL,
            random_state=None,
            coupled=None,
        ),
    },
    {
        "commit": "d15ec606",
        "date": "2024-01-17",
        "change": "Radial offset introduced for the 36 mm cumulative gap limit.",
        "parameters": _parameters(
            weights=[1.0, 1.0, 0.5],
            radial_offset=_RADIAL_OFFSET,
            index=_INDEX_TOROIDAL_ALL,
            random_state=None,
            coupled=None,
        ),
    },
    {
        "commit": "f447a63e",
        "date": "2025-02-03",
        "change": "Toroidal constraint reduced from all fiducials to a subset.",
        "parameters": _parameters(
            weights=[1.0, 1.0, 0.5],
            radial_offset=_RADIAL_OFFSET,
            index=_INDEX_TOROIDAL_REDUCED,
            random_state=None,
            coupled=None,
        ),
    },
    {
        "commit": "20e72bfc",
        "date": "2025-02-21",
        "change": "Gaussian-process random seed pinned to 2025 for reproducible draws.",
        "parameters": _parameters(
            weights=[1.0, 1.0, 0.5],
            radial_offset=_RADIAL_OFFSET,
            index=_INDEX_TOROIDAL_REDUCED,
            random_state=2025,
            coupled=None,
        ),
    },
    {
        "commit": "f85e5d90",
        "date": "2025-08-06",
        "change": "Vertical component weight reduced (weights 0.5 -> 0.25).",
        "parameters": _parameters(
            weights=[1.0, 1.0, 0.25],
            radial_offset=_RADIAL_OFFSET,
            index=_INDEX_TOROIDAL_REDUCED,
            random_state=2025,
            coupled=None,
        ),
    },
    {
        "commit": "66369ace",
        "date": "2025-10-13",
        "change": "Constraint index sets promoted to a class constant (values kept).",
        "parameters": _parameters(
            weights=[1.0, 1.0, 0.25],
            radial_offset=_RADIAL_OFFSET,
            index=_INDEX_TOROIDAL_REDUCED,
            random_state=2025,
            coupled=None,
        ),
    },
    {
        "commit": "093eea55",
        "date": "2026-02-02",
        "change": "Robust tangential outlier detection added for ILIS fiducials.",
        "parameters": _parameters(
            weights=[1.0, 1.0, 0.25],
            radial_offset=_RADIAL_OFFSET,
            index=_INDEX_TOROIDAL_REDUCED,
            random_state=2025,
            coupled=None,
        ),
    },
    {
        "commit": "a18dad0d",
        "date": "2026-02-06",
        "change": "Coupled sector-fit switch added (default coupled).",
        "parameters": _parameters(
            weights=[1.0, 1.0, 0.25],
            radial_offset=_RADIAL_OFFSET,
            index=_INDEX_TOROIDAL_REDUCED,
            random_state=2025,
            coupled=True,
        ),
    },
    {
        "commit": "eac8c052",
        "date": "2026-04-12",
        "change": "Sector-1 SSAT targets with gap profile and ILIS adjustments.",
        "parameters": _parameters(
            weights=[1.0, 1.0, 0.25],
            radial_offset=_RADIAL_OFFSET,
            index=_INDEX_TOROIDAL_REDUCED,
            random_state=2025,
            coupled=True,
        ),
    },
)

# The mined commits above are abbreviated in git; the full 40-hex forms are
# padded with zeros only to satisfy digest-shaped validators. The authoritative
# short shas, for cross-referencing the repository history, are recorded here.
_EPOCH_SHORT_SHA = {epoch["commit"]: epoch["commit"][:8] for epoch in _EPOCHS}


def parameter_epochs() -> list[dict]:
    """Return the fit-parameter evolution table with computed date ranges.

    Each epoch is the parameter set in force from its ``start_date`` until the
    next epoch's start (half-open ``[start, end)``); the final epoch's ``end``
    is ``None`` (still in force).

    Returns
    -------
    list[dict]
        Epochs oldest-first, each with ``commit`` (short sha), ``date``,
        ``start_date``, ``end_date``, ``change``, and ``parameters``.
    """
    epochs = []
    for index, epoch in enumerate(_EPOCHS):
        end = _EPOCHS[index + 1]["date"] if index + 1 < len(_EPOCHS) else None
        epochs.append(
            {
                "commit": _EPOCH_SHORT_SHA[epoch["commit"]],
                "date": epoch["date"],
                "start_date": epoch["date"],
                "end_date": end,
                "change": epoch["change"],
                "parameters": epoch["parameters"],
            }
        )
    return epochs


def epoch_for_date(date: str) -> dict:
    """Return the epoch in force on a given calendar date.

    Parameters
    ----------
    date : str
        ISO ``YYYY-MM-DD`` date.

    Returns
    -------
    dict
        The newest epoch whose ``start_date`` is on or before ``date``. Dates
        before the first epoch return the first epoch (the earliest recorded
        fit cannot predate it).
    """
    chosen = parameter_epochs()[0]
    for epoch in parameter_epochs():
        if epoch["start_date"] <= date:
            chosen = epoch
        else:
            break
    return chosen


def attribute_epoch(mtime_iso: str) -> dict:
    """Attribute a fit-parameter set to a workbook by its modification time.

    Parameters
    ----------
    mtime_iso : str
        The workbook's ISO-8601 modification time (its last save time).

    Returns
    -------
    dict
        Mapping with ``inferred: True``, the ``epoch_commit`` (short sha), the
        attributed ``parameters``, and an ``evidence`` block recording the
        basis, the modification time and epoch date range used, and a note that
        the modification time upper-bounds the fit date.
    """
    date = mtime_iso[:10]
    epoch = epoch_for_date(date)
    return {
        "inferred": True,
        "epoch_commit": epoch["commit"],
        "parameters": epoch["parameters"],
        "evidence": {
            "basis": "source_mtime",
            "source_mtime": mtime_iso,
            "epoch_date_range": [epoch["start_date"], epoch["end_date"]],
            "note": (
                "source_mtime is the workbook's last save/download time and "
                "upper-bounds the fit date; the parameter set is the one in "
                "force at the newest epoch on or before that date. Sheets "
                "written earlier may have used an earlier epoch."
            ),
        },
    }


def _read_rows(csv_path) -> list[list[str]]:
    """Read a canonical CSV into records, validating the header.

    Parameters
    ----------
    csv_path : os.PathLike or str
        Canonical CSV path.

    Returns
    -------
    list[list[str]]
        Data records (header excluded).

    Raises
    ------
    ValueError
        If the header does not match the canonical column set.
    """
    import csv as _csv

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = _csv.reader(handle)
        header = next(reader, None)
        if header is None or tuple(header) != _CSV_COLUMNS:
            raise ValueError(f"{csv_path}: header does not match canonical schema")
        return [record for record in reader]


def _fit_name_signal(phase: str) -> bool:
    """Return whether a sheet name marks it as a fit sheet by convention.

    Parameters
    ----------
    phase : str
        Sheet (assembly-phase) name.

    Returns
    -------
    bool
        ``True`` when the lower-cased name contains a fit-name token.
    """
    low = phase.lower()
    return any(token in low for token in _FIT_NAME_TOKENS)


def _parse_float(text: str):
    """Parse a CSV value field to float, or ``None`` when empty."""
    return None if text == "" else float(text)


def iter_fits(csv_path) -> list[dict]:
    """Extract recorded fits from one canonical unit.

    A fit is a populated transform block: all transform rows sharing a
    ``(phase, coil)`` are one fit, its recorded output the six transform
    components.

    Parameters
    ----------
    csv_path : os.PathLike or str
        Canonical CSV path; the sidecar ``<stem>.meta.yaml`` beside it supplies
        source identity and filename metadata.

    Returns
    -------
    list[dict]
        Fit entries sorted by ``(sheet, coil)``. Each entry carries the unit
        identity, the sheet, the coil, the fit-name-signal flag, the recorded
        transform, and the inferred parameter attribution.
    """
    csv_path = Path(csv_path)
    sidecar = yamlio.load_yaml(csv_path.parent / f"{csv_path.stem}.meta.yaml")
    source = sidecar["source"]
    meta = sidecar["filename_meta"]
    mtime = source["mtime"]
    attribution = attribute_epoch(mtime)

    # (phase, coil) -> {axis: value}
    blocks: dict[tuple, dict] = {}
    for record in _read_rows(csv_path):
        if record[_COL["record_kind"]] != _RECORD_TRANSFORM:
            continue
        phase = record[_COL["phase"]]
        coil_text = record[_COL["coil"]]
        coil = None if coil_text == "" else int(coil_text)
        axis = record[_COL["axis"]]
        value = _parse_float(record[_COL["value"]])
        blocks.setdefault((phase, coil), {})[axis] = value

    fits = []
    for (phase, coil), components in blocks.items():
        transform = {axis: components.get(axis) for axis in TRANSFORM_AXES}
        fits.append(
            {
                "unit": csv_path.stem,
                "source_filename": source["filename"],
                "sector": meta.get("sector"),
                "idm_doc": meta.get("idm_doc"),
                "version": meta.get("version"),
                "private": meta.get("private"),
                "revision_prefix": meta.get("revision_prefix", ""),
                "csv_sha256": sidecar["csv_sha256"],
                "source_mtime": mtime,
                "sheet": phase,
                "coil": coil,
                "fit_name_signal": _fit_name_signal(phase),
                "recorded": {"transform": transform},
                "inferred": attribution,
            }
        )
    fits.sort(key=lambda item: (item["sheet"], _coil_sort_key(item["coil"])))
    return fits


def name_signal_without_fit(csv_path) -> list[dict]:
    """List sheets named like a fit but carrying no transform block.

    These are sheets whose name contains a fit-name token yet hold no recorded
    rigid-body transform -- candidate fit sheets stored in another form, or
    measurement sheets that merely borrow the naming. Surfaced for review.

    Parameters
    ----------
    csv_path : os.PathLike or str
        Canonical CSV path.

    Returns
    -------
    list[dict]
        ``{"unit", "sheet"}`` entries sorted by sheet.
    """
    csv_path = Path(csv_path)
    phases: set[str] = set()
    phases_with_transform: set[str] = set()
    for record in _read_rows(csv_path):
        phase = record[_COL["phase"]]
        phases.add(phase)
        if record[_COL["record_kind"]] == _RECORD_TRANSFORM:
            phases_with_transform.add(phase)
    unresolved = [
        {"unit": csv_path.stem, "sheet": phase}
        for phase in phases
        if _fit_name_signal(phase) and phase not in phases_with_transform
    ]
    unresolved.sort(key=lambda item: item["sheet"])
    return unresolved


def _coil_sort_key(coil):
    """Return a sort key placing ``None`` coils first, then ascending int."""
    return (coil is not None, coil if coil is not None else 0)


def _iter_canonical_csvs(directories) -> list[Path]:
    """Return all canonical CSV paths across directories, deterministically.

    Parameters
    ----------
    directories : iterable of (os.PathLike or str)
        Directories to sweep for ``*.csv`` (non-recursive per directory).

    Returns
    -------
    list[pathlib.Path]
        CSV paths, de-duplicated and sorted by name then parent.
    """
    seen: dict[Path, None] = {}
    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for path in directory.glob("*.csv"):
            seen[path.resolve()] = None
    return sorted(seen, key=lambda path: (path.name, str(path.parent)))


def build_fit_manifest(directories, corpus_description: str = "") -> dict:
    """Build the fit manifest over one or more canonical directories.

    Parameters
    ----------
    directories : iterable of (os.PathLike or str)
        Canonical directories to sweep. A single path is accepted.
    corpus_description : str, optional
        Human-readable note recording which corpus was swept.

    Returns
    -------
    dict
        Manifest document with the parameter-epoch table, the enumerated fits
        (sorted by ``unit``, then sheet, then coil), the unresolved name-signal
        sheets, and a summary of counts. Deterministic: an unchanged canonical
        corpus yields byte-identical canonical YAML.
    """
    if isinstance(directories, (str, os.PathLike)):
        directories = [directories]
    csv_paths = _iter_canonical_csvs(directories)

    fits: list[dict] = []
    unresolved: list[dict] = []
    units_with_fits: set[str] = set()
    for csv_path in csv_paths:
        unit_fits = iter_fits(csv_path)
        fits.extend(unit_fits)
        if unit_fits:
            units_with_fits.add(csv_path.stem)
        unresolved.extend(name_signal_without_fit(csv_path))

    fits.sort(
        key=lambda item: (
            item["unit"],
            item["sheet"],
            _coil_sort_key(item["coil"]),
        )
    )
    unresolved.sort(key=lambda item: (item["unit"], item["sheet"]))

    fit_sheets = {(fit["unit"], fit["sheet"]) for fit in fits}
    recorded_fits = sum(
        1
        for fit in fits
        if all(v is not None for v in fit["recorded"]["transform"].values())
    )

    return {
        "schema": SCHEMA_ID,
        "generator": GENERATOR,
        "corpus": corpus_description,
        "parameter_epochs": parameter_epochs(),
        "summary": {
            "units_scanned": len(csv_paths),
            "units_with_fits": len(units_with_fits),
            "fit_sheets": len(fit_sheets),
            "fits": len(fits),
            "fits_with_complete_transform": recorded_fits,
            "name_signal_without_fit": len(unresolved),
        },
        "fits": fits,
        "name_signal_without_fit": unresolved,
    }


def write_fit_manifest(manifest: dict, path: os.PathLike | str) -> None:
    """Write a fit manifest to a file as canonical YAML.

    Parameters
    ----------
    manifest : dict
        Manifest as produced by :func:`build_fit_manifest`.
    path : os.PathLike or str
        Destination file.
    """
    yamlio.dump_yaml(manifest, path)


def load_fit_manifest(path: os.PathLike | str) -> dict:
    """Load a fit manifest from a YAML file.

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
