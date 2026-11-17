"""Render the iron contribution tables from the seam's measurement.

Reads the uniform-80 mm ``field.npz`` and prints the HTML tables the evidence fragment
carries: the iron contribution at every in-vessel poloidal probe (both field
components, the air-core value and the share) and at every flux loop (psi).

Usage: table.py probes | loops
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DEFAULT_FIELD = Path(
    "/work/projects/imas_gpu/sophelio/evidence/iron-core-magnetisation/"
    "field-source/west-62751-uniform-0.08-field.npz"
)


def rows(data):
    probes = data["probes"]
    contribution = data["frame_probe"]
    air = data["direct_probe"]
    for index, ((r, z), (dr, dz), (br, bz)) in enumerate(
        zip(probes, contribution, air, strict=True), start=1
    ):
        yield index, r, z, dr, dz, br, bz


def probes_table(path=None):
    data = np.load(path or DEFAULT_FIELD)
    out = [
        "<table><thead><tr><th>Probe</th><th>R [m]</th><th>Z [m]</th>"
        "<th>dBr [mT]</th><th>dBz [mT]</th><th>|dB| [mT]</th>"
        "<th>|B| air [mT]</th><th>share [%]</th></tr></thead><tbody>"
    ]
    for index, r, z, dr, dz, br, bz in rows(data):
        delta = np.hypot(dr, dz)
        air = np.hypot(br, bz)
        out.append(
            f"<tr><td>BPOL_{index:03d}</td><th>{r:.4f}</th><th>{z:.4f}</th>"
            f"<td>{dr * 1e3:.4f}</td><td>{dz * 1e3:.4f}</td><td>{delta * 1e3:.4f}</td>"
            f"<td>{air * 1e3:.3f}</td><td>{delta / air * 100:.3f}</td></tr>"
        )
    out.append("</tbody></table>")
    print("\n".join(out))


def loops_table(path=None):
    data = np.load(path or DEFAULT_FIELD)
    loops = data["loops"]
    contribution = data["frame_loop"]
    out = [
        "<table><thead><tr><th>Flux loop</th><th>R [m]</th><th>Z [m]</th>"
        "<th>dpsi [mWb/rad]</th></tr></thead><tbody>"
    ]
    for index, ((r, z), dpsi) in enumerate(
        zip(loops, contribution, strict=True), start=1
    ):
        out.append(
            f"<tr><td>FLOOP_{index:03d}</td><th>{r:.4f}</th><th>{z:.4f}</th>"
            f"<td>{float(dpsi) * 1e3:.5f}</td></tr>"
        )
    out.append("</tbody></table>")
    print("\n".join(out))


def main():
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    match sys.argv[1:2]:
        case ["probes"]:
            probes_table(path)
        case ["loops"]:
            loops_table(path)
        case _:
            raise SystemExit("usage: table.py probes | loops")


if __name__ == "__main__":
    main()