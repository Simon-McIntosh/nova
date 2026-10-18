"""Write nova's own decisions back out as version-2 boundary trace records.

The sweep in :mod:`crosscheck` reads imas-efit records and records nova's
decision beside imas-efit's. This module turns that around: it takes the rows
the sweep produced, rebuilds each source record, replaces the decision fields
with nova's own, and writes the result through the record writer. The point is
round-tripping nova's decision through the same codec imas-efit reads, so a
later consumer can open a nova-decided state with the Fortran reader.

Provenance is in the file name and in the emitted record's own status fields:
the grid, wall and psi of an export are the source record's, unchanged, and only
the axis level and position, the refined saddle level and the fed-back level are
nova's. The verdict is remapped from nova's admitted class onto the record's own
encoding (2 diverted, 1 limited) so the written record is self-consistent.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path


from scripts.prototypes.efit_field_swap.crosscheck import resolve_trace
from scripts.prototypes.efit_field_swap.trace_record import (
    read_trace_record,
    write_trace_record,
)

DIVERTED = 2
LIMITED = 1


def export_row(row, pairs, out_dir):
    """Write one sweep row as a version-2 record and return its path."""
    pair = row["case"].split(":")[0]
    side = row["case"].split(":")[1]
    paths = next(parts for parts in pairs if parts[0] == pair)
    source = resolve_trace(paths[1] if side == "A" else paths[2])
    if row["nova_label"] not in ("diverted", "limited"):
        raise ValueError(f"{row['case']} has no nova decision ({row['nova_label']})")
    record = read_trace_record(source)
    diverted = row["nova_label"] == "diverted"
    exported = replace(
        record,
        verdict=DIVERTED if diverted else LIMITED,
        axis_psi=float(row["nova_axis_psi"]),
        axis_r=float(row["nova_axis_r"]),
        axis_z=float(row["nova_axis_z"]),
        saddle_refined=float(row["nova_saddle_psi"]),
        fed_back_psi=float(row["nova_boundary_psi"]),
    )
    stem = source.stem
    target = Path(out_dir) / f"{stem}-nova-{row['nova_label']}.txt"
    write_trace_record(target, exported)
    return target


def main(rows_path, pairs_path, out_dir):
    payload = json.loads(Path(rows_path).read_text())
    pairs = [
        line.split()
        for line in Path(pairs_path).read_text().splitlines()
        if line.strip()
    ]
    wanted = {}
    for row in payload["rows"]:
        label = row["nova_label"]
        if label in ("limited", "diverted") and label not in wanted:
            wanted[label] = row
    written = []
    for _, row in sorted(wanted.items()):
        path = export_row(row, pairs, out_dir)
        written.append((row["case"], row["nova_label"], str(path)))
    for case, label, path in written:
        print("exported", case, label, path)
    print("EXPORTED", len(written))
    return written


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:4])
