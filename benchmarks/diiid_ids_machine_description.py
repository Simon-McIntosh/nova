"""Publish a receipt for the DIII-D IMAS machine description.

The source is a legacy MDSplus entry and is read only through the official IMAS
Python access layer.  The extraction subprocess exists because that entry uses
the legacy pulse layout, while Nova's project environment uses the current
imas-python API.  No MDSplus file is opened or interpreted directly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


DATABASE = "DIII-D"
USER = "hoeneno"
RUN = 0
MACHINE_SHOT = 1000
CORROBORATING_SHOT = 133221
SOURCE_COCOS = 11
TARGET_COCOS = 17
LEGACY_IMAS_MODULE = "IMAS/3.39.0-4.11.10-foss-2023b"


def _plain(value: Any) -> Any:
    """Return an Access Layer scalar without changing its numerical value."""
    return value.value if hasattr(value, "value") else value


def _legacy_entry(shot: int) -> dict[str, Any]:
    """Read one legacy MDSplus entry with the official IMAS Python API."""
    import imas

    entry = imas.DBEntry(
        imas.imasdef.MDSPLUS_BACKEND,
        DATABASE,
        shot,
        RUN,
        USER,
        "3",
    )
    entry.open()
    try:
        wall = entry.get("wall")
        active = entry.get("pf_active")
        passive = entry.get("pf_passive")
        tf = entry.get("tf")
        thomson = entry.get("thomson_scattering")

        descriptions = wall.description_2d
        contour = None
        if len(descriptions):
            description = descriptions[0]
            units = description.vessel.unit
            kind = "vessel"
            if not len(units):
                units = description.limiter.unit
                kind = "limiter"
            if len(units):
                outline = units[0].outline
                contour = {
                    "kind": kind,
                    "r": [float(value) for value in outline.r],
                    "z": [float(value) for value in outline.z],
                }

        sections = []
        for index, coil in enumerate(active.coil):
            for element_index, element in enumerate(coil.element):
                geometry = element.geometry
                geometry_type = int(_plain(geometry.geometry_type))
                name = str(
                    _plain(coil.identifier) or _plain(coil.name) or f"coil_{index}"
                )
                if len(coil.element) > 1:
                    name = f"{name}_{element_index}"
                if geometry_type == 1:
                    section = geometry.outline
                    record = {
                        "geometry_type": geometry_type,
                        "name": name,
                        "r": [float(value) for value in section.r],
                        "z": [float(value) for value in section.z],
                    }
                elif geometry_type == 2:
                    section = geometry.rectangle
                    record = {
                        "geometry_type": geometry_type,
                        "name": name,
                        "r": float(_plain(section.r)),
                        "z": float(_plain(section.z)),
                        "width": float(_plain(section.width)),
                        "height": float(_plain(section.height)),
                    }
                elif geometry_type == 3:
                    section = geometry.oblique
                    record = {
                        "geometry_type": geometry_type,
                        "name": name,
                        "r": float(_plain(section.r)),
                        "z": float(_plain(section.z)),
                        "length_alpha": float(_plain(section.length_alpha)),
                        "length_beta": float(_plain(section.length_beta)),
                        "alpha": float(_plain(section.alpha)),
                        "beta": float(_plain(section.beta)),
                    }
                else:
                    record = {"geometry_type": geometry_type, "name": name}
                sections.append(record)

        channels = []
        for index, channel in enumerate(thomson.channel):
            position = channel.position
            channels.append(
                {
                    "name": str(
                        _plain(channel.identifier)
                        or _plain(channel.name)
                        or f"channel_{index}"
                    ),
                    "position": [
                        float(_plain(position.r)),
                        float(_plain(position.z)),
                        float(_plain(position.phi)),
                    ],
                    "start": None,
                    "end": None,
                }
            )

        versions = {}
        for name, ids in (
            ("wall", wall),
            ("pf_active", active),
            ("pf_passive", passive),
            ("tf", tf),
            ("thomson_scattering", thomson),
        ):
            version = str(_plain(ids.ids_properties.version_put.data_dictionary))
            if version:
                versions[name] = version
        return {
            "shot": shot,
            "run": RUN,
            "database": DATABASE,
            "user": USER,
            "dd_versions": versions,
            "contour": contour,
            "pf_active": sections,
            "pf_passive_loop_count": len(passive.loop),
            "tf_coil_count": len(tf.coil),
            "thomson_scattering": channels,
        }
    finally:
        entry.close()


def read_entries() -> list[dict[str, Any]]:
    """Read both named entries in an isolated legacy-AL subprocess."""
    command = (
        "module purge >/dev/null 2>&1; "
        f"module load {LEGACY_IMAS_MODULE}; "
        f"python {Path(__file__).resolve()} --extract-only"
    )
    process = subprocess.run(
        ["bash", "-lc", command],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(process.stdout)


def _section_metrics(section: Any) -> dict[str, Any]:
    name = section.section.name
    data = section.section.data
    if name == "rectangle":
        return {
            "geometry_class": name,
            "r_m": data["r"],
            "z_m": data["z"],
            "width_m": data["width"],
            "height_m": data["height"],
            "skew_rad": 0.0,
        }
    r = list(data["r"])
    z = list(data["z"])
    skew = math.atan2(z[1] - z[0], r[1] - r[0]) if len(r) > 1 else None
    return {
        "geometry_class": name,
        "r_m": 0.5 * (min(r) + max(r)),
        "z_m": 0.5 * (min(z) + max(z)),
        "width_m": max(r) - min(r),
        "height_m": max(z) - min(z),
        "skew_rad": skew,
    }


def build_receipt(entries: list[dict[str, Any]]) -> tuple[dict[str, Any], Any]:
    """Route the machine entry through Nova and classify every quantity."""
    from nova.imas.machine import StaticMachineDescription
    from nova.io.cocos import (
        B0_LIKE,
        DODPSI_LIKE,
        IP_LIKE,
        PSI_LIKE,
        Q_LIKE,
        convention_transform,
    )

    source = next(entry for entry in entries if entry["shot"] == MACHINE_SHOT)
    machine = StaticMachineDescription.from_record(source)
    contour = machine.contour
    contour_receipt = {
        "status": "read_after_named_change",
        "change": "accept DDv3 limiter outline when vessel units are absent",
        "kind": contour.kind if contour else None,
        "vertex_count": len(contour.r) if contour else 0,
        "r_extent_m": [min(contour.r), max(contour.r)] if contour else None,
        "z_extent_m": [min(contour.z), max(contour.z)] if contour else None,
    }
    coils = [
        {"name": section.name, **_section_metrics(section)}
        for section in machine.active_sections
    ]
    endpoints_present = sum(
        sightline.start is not None and sightline.end is not None
        for sightline in machine.sightlines
    )
    transform = convention_transform(source=SOURCE_COCOS, target=TARGET_COCOS)
    receipt = {
        "measurement": "DIII-D IMAS static machine-description read",
        "source_entries": entries,
        "selected_entry": {"shot": MACHINE_SHOT, "run": RUN},
        "dd_version_policy": (
            "read from ids_properties/version_put/data_dictionary for each "
            "non-empty IDS"
        ),
        "cocos": {
            "source_index": SOURCE_COCOS,
            "source": "IMAS Data Dictionary coordinate convention",
            "target_index": TARGET_COCOS,
            "factors": {
                PSI_LIKE: transform.factor(PSI_LIKE),
                IP_LIKE: transform.factor(IP_LIKE),
                B0_LIKE: transform.factor(B0_LIKE),
                Q_LIKE: transform.factor(Q_LIKE),
                DODPSI_LIKE: transform.factor(DODPSI_LIKE),
            },
        },
        "quantities": {
            "wall_or_limiter": contour_receipt,
            "pf_active": {
                "status": "read_after_named_change",
                "change": (
                    "route legacy scalar and outline records through the tabular "
                    "geometry reader"
                ),
                "coil_count": len(coils),
                "coils": coils,
            },
            "pf_passive": {
                "status": "cannot_reach",
                "loop_count": machine.passive_loop_count,
                "reason": "the selected entry carries no pf_passive loops",
            },
            "tf": {
                "status": "cannot_reach",
                "coil_count": machine.toroidal_coil_count,
                "reason": (
                    "the selected entry carries no tf coils or static tf parameters"
                ),
            },
            "thomson_scattering": {
                "status": (
                    "cannot_reach" if not endpoints_present else "read_unmodified"
                ),
                "chord_count": len(machine.sightlines),
                "endpoint_pair_count": endpoints_present,
                "reason": (
                    "DD 3.28 channels carry one position each and no line-of-sight "
                    "endpoint fields"
                    if not endpoints_present
                    else None
                ),
                "channels": [
                    {
                        "name": sightline.name,
                        "position_r_z_phi": sightline.position,
                        "start_r_z_phi": sightline.start,
                        "end_r_z_phi": sightline.end,
                    }
                    for sightline in machine.sightlines
                ],
            },
        },
    }
    return receipt, machine


def write_figures(machine: Any, output: Path) -> list[Path]:
    """Draw carried geometry and make absent geometry visible as absence."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch

    output.mkdir(parents=True, exist_ok=True)
    paths = []

    def base_axes():
        figure, axes = plt.subplots(figsize=(6.2, 6.4), constrained_layout=True)
        contour = machine.contour
        if contour:
            axes.plot(
                contour.r,
                contour.z,
                color="black",
                linewidth=1.6,
                label=contour.kind,
            )
        axes.set_aspect("equal")
        axes.set_xlabel("R [m]")
        axes.set_ylabel("Z [m]")
        return figure, axes

    figure, axes = base_axes()
    for section in machine.active_sections:
        polygon = section.section.poly
        axes.add_patch(
            PolygonPatch(
                list(zip(*polygon.exterior.xy, strict=True)),
                fill=False,
                edgecolor="tab:blue",
                linewidth=1.3,
            )
        )
        centre = polygon.centroid
        axes.text(centre.x, centre.y, section.name, fontsize=7)
    axes.set_title("Limiter and active-coil cross-sections")
    path = output / "limiter_pf_active.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    figure, axes = base_axes()
    positions = [sightline.position for sightline in machine.sightlines]
    if positions:
        axes.scatter(
            [position[0] for position in positions],
            [position[1] for position in positions],
            s=20,
            color="tab:orange",
            label="reported scattering position",
        )
    axes.text(
        0.02,
        0.02,
        "No line-of-sight endpoints are stored; no chord is fabricated.",
        transform=axes.transAxes,
        fontsize=8,
    )
    axes.set_title("Thomson geometry carried by the IDS")
    path = output / "thomson_geometry.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)

    figure, axes = base_axes()
    axes.text(
        0.5,
        0.5,
        f"pf_passive loops stored: {machine.passive_loop_count}",
        ha="center",
        va="center",
        transform=axes.transAxes,
    )
    axes.set_title("Passive-structure availability")
    path = output / "passive_structure.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/diiid-forward-onboarding/ids-description"),
    )
    args = parser.parse_args()
    if args.extract_only:
        json.dump(
            [_legacy_entry(MACHINE_SHOT), _legacy_entry(CORROBORATING_SHOT)],
            sys.stdout,
        )
        return
    entries = read_entries()
    receipt, machine = build_receipt(entries)
    figures = write_figures(machine, args.output)
    receipt["figures"] = [str(path) for path in figures]
    receipt_path = args.output / "ids_description_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"receipt": str(receipt_path), "figures": len(figures)}))


if __name__ == "__main__":
    main()
