"""Inventory the populated IMAS leaves in the DIII-D netCDF entry.

The backend's filled-path list is the completeness authority.  Concrete IDS
nodes supply the shapes and dtypes, and the run fails if those two views do not
agree.  No reconstruction or diagnostic value is fitted or promoted to an
input by this census.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import imas
import numpy as np

DEFAULT_ENTRY = Path("/home/ITER/tribolp/Public/imasdb/DIII-D/200000.nc")
DEFAULT_OUTPUT = Path("docs/figures/diiid-forward-onboarding/netcdf-census")
IDS_NAMES = ("equilibrium", "magnetics", "pf_active", "tf", "wall")
DD_VERSION = "3.41.0"

MACHINE_DESCRIPTION = "MACHINE DESCRIPTION"
NOT_ADMISSIBLE = "NOT ADMISSIBLE"
FIREWALLED = "FIREWALLED"
CLASS_REASONS = {
    MACHINE_DESCRIPTION: (
        "Machine and diagnostic geometry, identifiers, outlines, turns, wall "
        "contour, and declared actuator waveforms are admissible entry inputs."
    ),
    NOT_ADMISSIBLE: (
        "Magnetics signal-channel samples, errors, validity and time bases are "
        "measurements; the magnetics-free entry fits no measurement."
    ),
    FIREWALLED: (
        "Equilibrium leaves, including reconstructed constraint blocks, are EFIT "
        "products and may be audited but never used as labels or fitting targets."
    ),
}

MAGNETICS_SIGNAL_PREFIXES = (
    "b_field_pol_probe/field/",
    "diamagnetic_flux/",
    "flux_loop/flux/",
    "ip/",
)


def classify_path(ids_name: str, path: str) -> tuple[str, str]:
    """Return the sole admissibility class and its stable reason."""
    if ids_name == "equilibrium":
        category = FIREWALLED
    elif ids_name == "magnetics" and path.startswith(MAGNETICS_SIGNAL_PREFIXES):
        category = NOT_ADMISSIBLE
    else:
        category = MACHINE_DESCRIPTION
    return category, CLASS_REASONS[category]


def _walk_leaves(node: Any):
    if hasattr(node, "iter_nonempty_"):
        for child in node.iter_nonempty_():
            yield from _walk_leaves(child)
        return
    if node.__class__.__name__ == "IDSStructArray":
        for item in node:
            yield from _walk_leaves(item)
        return
    yield node


def _dtype_and_shape(value: Any) -> tuple[str, tuple[int, ...]]:
    array = np.asarray(value)
    dtype = "str" if array.dtype.kind == "U" else str(array.dtype)
    return dtype, tuple(int(size) for size in array.shape)


def _time_lengths(fields: list[dict[str, Any]], homogeneous_length: int) -> None:
    local: dict[str, set[int]] = defaultdict(set)
    for field in fields:
        if field["path"].endswith("/time") or field["path"] == "time":
            parent = field["path"].rsplit("/", 1)[0]
            for shape in field["instance_shapes"]:
                if shape:
                    local[parent].add(shape[0])

    for field in fields:
        dynamic = field.pop("_dynamic")
        field["time_dependent"] = dynamic
        if not dynamic:
            field["time_base_lengths"] = []
            continue
        if homogeneous_length:
            field["time_base_lengths"] = [homogeneous_length]
            continue
        parent = field["path"].rsplit("/", 1)[0]
        field["time_base_lengths"] = sorted(local.get(parent, set()))


def census_ids(entry: Any, ids_name: str) -> tuple[dict[str, Any], Any]:
    """Census one IDS and return its receipt plus the loaded IDS."""
    backend_paths = set(entry.list_filled_paths(ids_name, autoconvert=False))
    ids = entry.get(ids_name, autoconvert=False)
    aggregates: dict[str, dict[str, Any]] = {}
    for node in _walk_leaves(ids):
        path = str(node.metadata.path)
        if path not in backend_paths:
            continue
        dtype, shape = _dtype_and_shape(node.value)
        record = aggregates.setdefault(
            path,
            {
                "dtypes": set(),
                "instance_shapes": set(),
                "instances": 0,
                "dynamic": False,
            },
        )
        record["dtypes"].add(dtype)
        record["instance_shapes"].add(shape)
        record["instances"] += 1
        record["dynamic"] = record["dynamic"] or str(node.metadata.type).endswith(
            "DYNAMIC"
        )

    missing = sorted(backend_paths - set(aggregates))
    if missing:
        raise RuntimeError(
            f"{ids_name}: backend-populated leaves not traversed: {missing}"
        )

    fields = []
    for path in sorted(backend_paths):
        aggregate = aggregates[path]
        category, reason = classify_path(ids_name, path)
        shapes = sorted(aggregate["instance_shapes"])
        dtypes = sorted(aggregate["dtypes"])
        if len(dtypes) != 1:
            raise RuntimeError(f"{ids_name}/{path}: mixed dtypes {dtypes}")
        fields.append(
            {
                "path": path,
                "shape": [aggregate["instances"], *shapes[0]]
                if aggregate["instances"] > 1 and len(shapes) == 1
                else list(shapes[0]),
                "instance_shapes": [list(shape) for shape in shapes],
                "populated_instances": aggregate["instances"],
                "dtype": dtypes[0],
                "admissibility": category,
                "reason": reason,
                "_dynamic": aggregate["dynamic"],
            }
        )

    homogeneous_time = int(ids.ids_properties.homogeneous_time)
    ids_time = np.asarray(ids.time)
    _time_lengths(fields, int(ids_time.size) if homogeneous_time == 1 else 0)
    return (
        {
            "dd_version": str(ids.ids_properties.version_put.data_dictionary),
            "homogeneous_time": homogeneous_time,
            "filled_leaf_count": len(fields),
            "fields": fields,
            "unclassified": [],
        },
        ids,
    )


def _constraint_counts(equilibrium: Any) -> dict[str, int]:
    first = equilibrium.time_slice[0].constraints
    return {
        "bpol_probe": len(first.bpol_probe),
        "flux_loop": len(first.flux_loop),
        "pf_current": len(first.pf_current),
        "mse_polarisation_angle": len(first.mse_polarisation_angle),
    }


def build_census(entry_path: Path) -> dict[str, Any]:
    """Read the file with its written DD and return a complete receipt."""
    ids_receipts: dict[str, Any] = {}
    loaded: dict[str, Any] = {}
    with imas.DBEntry(entry_path, "r", dd_version=DD_VERSION) as entry:
        populated_ids = tuple(
            name
            for name in entry.factory.ids_names()
            if entry.list_all_occurrences(name)
        )
        if populated_ids != IDS_NAMES:
            raise RuntimeError(
                f"expected exactly {IDS_NAMES}, found populated IDSs {populated_ids}"
            )
        for ids_name in IDS_NAMES:
            ids_receipts[ids_name], loaded[ids_name] = census_ids(entry, ids_name)

        equilibrium_time = np.asarray(loaded["equilibrium"].time, dtype=float)
        pf_lengths = sorted(
            {
                int(np.asarray(loaded["pf_active"].coil[index].current.time).size)
                for index in range(len(loaded["pf_active"].coil))
            }
        )
        constraints = _constraint_counts(loaded["equilibrium"])
        magnetics = loaded["magnetics"]
        overview = {
            "equilibrium_slice_count": len(loaded["equilibrium"].time_slice),
            "equilibrium_time_span_s": [
                float(equilibrium_time[0]),
                float(equilibrium_time[-1]),
            ],
            "pf_active_coil_count": len(loaded["pf_active"].coil),
            "pf_active_current_time_base_lengths": pf_lengths,
            "equilibrium_constraint_channels": constraints,
            "magnetics_signal_channels": {
                "bpol_probe": len(magnetics.b_field_pol_probe),
                "flux_loop": len(magnetics.flux_loop),
                "ip": len(magnetics.ip),
                "diamagnetic_flux": len(magnetics.diamagnetic_flux),
            },
            "case_reading": "physical shot-derived reconstruction",
            "case_reading_evidence": (
                "The entry combines 340 EFIT slices over 0.10-7.22 s with "
                "independently sampled diagnostic and actuator streams (44 flux loops, "
                "76 bpol probes, 24 PF currents and 69 MSE constraints; raw time bases "
                "reach 480256 "
                "samples). That content is characteristic of a physical discharge and "
                "its reconstruction, not a geometry-only synthetic reference case."
            ),
        }

    totals = {category: 0 for category in CLASS_REASONS}
    for ids_receipt in ids_receipts.values():
        for field in ids_receipt["fields"]:
            totals[field["admissibility"]] += 1
    return {
        "source": str(entry_path),
        "backend": "imas-python netCDF",
        "configured_written_dd_version": DD_VERSION,
        "populated_ids": list(IDS_NAMES),
        "shape_semantics": (
            "shape prepends populated concrete-instance count when a DD leaf repeats "
            "through an array of structures; instance_shapes retains each leaf value "
            "shape before that aggregation"
        ),
        "classification_reasons": CLASS_REASONS,
        "overview": overview,
        "ids": ids_receipts,
        "class_totals": totals,
        "total_populated_leaf_paths": sum(
            receipt["filled_leaf_count"] for receipt in ids_receipts.values()
        ),
        "unclassified": [],
        "fitting_performed": False,
    }


def summary_html(census: dict[str, Any]) -> str:
    """Render a compact semantic table suitable for direct HTML inclusion."""
    categories = tuple(CLASS_REASONS)
    rows = []
    for ids_name, receipt in census["ids"].items():
        counts = {category: 0 for category in categories}
        for field in receipt["fields"]:
            counts[field["admissibility"]] += 1
        cells = "".join(f"<td>{counts[category]}</td>" for category in categories)
        rows.append(
            f'<tr><th scope="row">{html.escape(ids_name)}</th>'
            f"<td>{receipt['dd_version']}</td><td>{receipt['homogeneous_time']}</td>"
            f"<td>{receipt['filled_leaf_count']}</td>{cells}<td>0</td></tr>"
        )
    headings = "".join(f'<th scope="col">{html.escape(c)}</th>' for c in categories)
    return (
        "<table>\n<caption>DIII-D 200000.nc populated-leaf census</caption>\n"
        f'<thead><tr><th scope="col">IDS</th><th scope="col">DD</th>'
        f'<th scope="col">homogeneous_time</th><th scope="col">Leaves</th>'
        f'{headings}<th scope="col">Unclassified</th></tr></thead>\n<tbody>\n'
        + "\n".join(rows)
        + "\n</tbody>\n</table>\n"
    )


def write_outputs(census: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "diiid_netcdf_census.json"
    html_path = output_dir / "diiid_netcdf_census_summary.html"
    json_path.write_text(json.dumps(census, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(summary_html(census), encoding="utf-8")
    return json_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", type=Path, default=DEFAULT_ENTRY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    census = build_census(args.entry)
    json_path, html_path = write_outputs(census, args.output_dir)
    overview = census["overview"]
    print(f"wrote {json_path} and {html_path}")
    print(
        f"{census['total_populated_leaf_paths']} leaves; "
        f"{overview['equilibrium_slice_count']} equilibrium slices; "
        f"PF current time bases {overview['pf_active_current_time_base_lengths']}; "
        f"classes {census['class_totals']}"
    )


if __name__ == "__main__":
    main()
