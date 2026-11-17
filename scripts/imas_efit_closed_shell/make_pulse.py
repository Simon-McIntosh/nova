"""Write the closed-shell shielding case as an IMAS pulse.

The case is the analytic control of iron-core-magnetisation section 1: a small
toroidal-current coil inside a linear permeable spherical shell (inner radius
a, outer radius b, relative permeability mu_r) whose exterior poloidal flux is
reduced by the exact factor

    S = ((2 mu + 1)(mu + 2) - 2 (mu - 1)^2 (a/b)^3) / (9 mu),

which is not restated here: it is
``scripts.iron_linear_magnetisation.controls.exact_shielding``, the shared
closed-shell control, called with this case's ratio and mu_r.

Everything imas-efit needs arrives through IDSs, written with imas-python at
the DD version imas-efit reads (4.1.0, its build default and the version its
reader takes from ids_properties/version_put/data_dictionary):

  pf_active  one coil, geometry_type=2 rectangle, one-turn source circuit
  iron_core  the shell's poloidal half-annulus as one polygon outline plus a
             flat B-H table (constant mu_r), omitted for the air-core control
  wall       the plasma limiter box
  magnetics  four poloidal field probes plus Ip
  tf         b_field_phi_vacuum_r = 0 (Bvac * R)

Usage:
  make_pulse.py <outdir> --ratio 0.9 [--arc-vertices 20] [--no-iron]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import imas
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.iron_linear_magnetisation.controls import exact_shielding  # noqa: E402

DD_VERSION = "4.1.0"

MU_R = 1000.0
B_OUTER = 1.0
COIL_R, COIL_DR, COIL_DZ, COIL_I = 0.03, 0.02, 0.02, 1.0
PLASMA_IP = 100.0
PLASMA_R0, PLASMA_A = 0.09, 0.04
LIMITER_BOX = (0.02, 0.30, -0.12, 0.12)
PROBES = (
    (2.0, 0.0, 0.0),
    (0.0, 2.0, 0.5 * math.pi),
    (3.0, 1.0, 0.0),
    (3.0, 1.0, 0.5 * math.pi),
)


def shell_outline(a: float, b: float, n_arc: int) -> tuple[list[float], list[float]]:
    """Outer arc (0,b) -> (0,-b), then inner arc (0,-a) -> (0,a).

    The polygon closes from (0,a) back to (0,b), so the two axis segments are
    the implicit closing edges and the traversal is the one imas-efit's
    IronBoundary requires: its inwards normal n and storage direction l
    satisfy (n x l)_phi > 0.
    """
    r = [0.0] * (2 * (n_arc + 1))
    z = [0.0] * (2 * (n_arc + 1))
    for i in range(n_arc + 1):
        theta = 0.5 * math.pi - math.pi * i / n_arc
        r[i] = b * math.cos(theta)
        z[i] = b * math.sin(theta)
    for j in range(n_arc + 1):
        theta = -0.5 * math.pi + math.pi * j / n_arc
        k = n_arc + 1 + j
        r[k] = a * math.cos(theta)
        z[k] = a * math.sin(theta)
    return r, z


def build(
    ratio: float,
    n_arc: int,
    with_iron: bool,
    current_scale: float = 1.0,
    plasma_scale: float = 1.0,
):
    a = B_OUTER * ratio
    coil_current = COIL_I * current_scale
    plasma_current = PLASMA_IP * current_scale * plasma_scale
    factory = imas.IDSFactory(DD_VERSION)
    ids = factory.new("pf_active")
    ids.ids_properties.homogeneous_time = 1
    ids.ids_properties.comment = f"closed-shell control a/b={ratio} coil source"
    ids.time = [0.0]
    ids.coil.resize(1)
    coil = ids.coil[0]
    coil.name = "DIP"
    coil.element.resize(1)
    el = coil.element[0]
    el.geometry.geometry_type = 2
    el.geometry.rectangle.r = COIL_R
    el.geometry.rectangle.z = 0.0
    el.geometry.rectangle.width = COIL_DR
    el.geometry.rectangle.height = COIL_DZ
    el.turns_with_sign = 1.0
    coil.current.data = [coil_current]
    coil.current.time = [0.0]
    coil.current.data_error_upper = [1.0e-6]
    # One circuit carrying the coil.  A supply element sits in the same node
    # rows, so the coil-pair search finds no pair and the solver registers the
    # coil as its own single-coil circuit (positive polarity).
    ids.supply.resize(1)
    ids.supply[0].name = "DIP_SUPPLY"
    ids.supply[0].current.data = [coil_current]
    ids.supply[0].current.time = [0.0]
    ids.supply[0].current.data_error_upper = [1.0e-6]
    ids.circuit.resize(1)
    circuit = ids.circuit[0]
    circuit.name = "DIP_CIRCUIT"
    # rows = nodes, columns = [supply, coil]
    circuit.connections = np.array([[1, 1], [-1, -1]], dtype=np.int32)

    magnetics = factory.new("magnetics")
    magnetics.ids_properties.homogeneous_time = 1
    magnetics.ids_properties.comment = "closed-shell control field points"
    magnetics.time = [0.0]
    magnetics.b_field_pol_probe.resize(len(PROBES))
    for i, (r, z, angle) in enumerate(PROBES):
        probe = magnetics.b_field_pol_probe[i]
        probe.name = f"shell_probe_{i + 1}"
        probe.position.r = r
        probe.position.z = z
        probe.position.phi = 0.0
        probe.poloidal_angle = angle
        probe.area = 1.0e-4
        probe.length = 0.01
        probe.turns = 1
        probe.field.data = [0.0]
        probe.field.time = [0.0]
        probe.field.data_error_upper = [10.0]
    magnetics.ip.resize(1)
    magnetics.ip[0].data = [plasma_current]
    magnetics.ip[0].time = [0.0]
    magnetics.ip[0].data_error_upper = [1.0]

    wall = factory.new("wall")
    wall.ids_properties.homogeneous_time = 1
    wall.description_2d.resize(1)
    unit = wall.description_2d[0].limiter.unit
    unit.resize(1)
    r0, r1, z0, z1 = LIMITER_BOX
    unit[0].name = "shell interior"
    unit[0].outline.r = [r0, r1, r1, r0, r0]
    unit[0].outline.z = [z0, z0, z1, z1, z0]

    tf = factory.new("tf")
    tf.ids_properties.homogeneous_time = 1
    tf.time = [0.0]
    tf.b_field_phi_vacuum_r.data = [0.0]
    tf.b_field_phi_vacuum_r.time = [0.0]

    iron = None
    if with_iron:
        iron = factory.new("iron_core")
        iron.ids_properties.homogeneous_time = 1
        iron.segment.resize(1)
        segment = iron.segment[0]
        segment.name = "spherical_shell"
        segment.geometry.geometry_type = 1
        outline_r, outline_z = shell_outline(a, B_OUTER, n_arc)
        segment.geometry.outline.r = outline_r
        segment.geometry.outline.z = outline_z
        # Flat B-H table: constant mu_r, the linear-mu reference S assumes.
        segment.b_field = [0.0, 1.0e3]
        segment.permeability_relative = [MU_R, MU_R]

    return ids, magnetics, wall, tf, iron


def write_pulse(
    outdir: Path,
    ratio: float,
    n_arc: int,
    with_iron: bool,
    current_scale: float = 1.0,
    plasma_scale: float = 1.0,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    uri = f"imas:hdf5?path={outdir}"
    entry = imas.DBEntry(uri, "w", dd_version=DD_VERSION)
    pf, magnetics, wall, tf, iron = build(
        ratio, n_arc, with_iron, current_scale, plasma_scale
    )
    entry.put(pf)
    entry.put(magnetics)
    entry.put(wall)
    if iron is not None:
        entry.put(iron)
    entry.put(tf)
    entry.close()
    return dict(
        ratio=ratio,
        a=B_OUTER * ratio,
        b=B_OUTER,
        n_arc=n_arc,
        vertices=2 * (n_arc + 1),
        with_iron=with_iron,
        exact_S=exact_shielding(ratio, MU_R),
        mu_r=MU_R,
        uri=uri,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir")
    parser.add_argument("--ratio", type=float, default=0.9)
    parser.add_argument("--arc-vertices", type=int, default=20)
    parser.add_argument("--no-iron", action="store_true")
    parser.add_argument(
        "--current-scale",
        type=float,
        default=1.0,
        help="multiply the coil and plasma currents; the case is linear in the "
        "currents and the flux scale only matters for solver conditioning",
    )
    parser.add_argument(
        "--plasma-scale",
        type=float,
        default=1.0,
        help="multiply the plasma current alone, to make the coil the dominant "
        "source of the flux at the probes",
    )
    args = parser.parse_args()
    out = Path(args.outdir)
    case = write_pulse(
        out,
        args.ratio,
        args.arc_vertices,
        not args.no_iron,
        args.current_scale,
        args.plasma_scale,
    )
    (out.parent / (out.name + ".json")).write_text(json.dumps(case, indent=1))
    print(f"wrote {out} exact S={case['exact_S']:.6f}")


if __name__ == "__main__":
    main()
