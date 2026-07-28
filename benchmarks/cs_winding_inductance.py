"""Does the discrete winding explain the self-inductance a smeared coil misses?

A poloidal field coil ships as a uniform current density over its gross
rectangular outline.  The hardware is not that: an ITER central solenoid module
is wound from 554 turns of cable-in-conduit conductor, each turn a cable space
inside a steel jacket, with turn insulation between and ground insulation
around the pack.  The current occupies a FRACTION of the outline and sits in
discrete sub-regions, and concentrating current into smaller sub-conductors
RAISES self-inductance -- a smaller geometric mean distance inside each turn is
more internal flux -- so the smeared outline is the lowest self-inductance a
given outline can carry.

The machine description sits ABOVE the smeared continuum on every self term,
which is the direction the winding predicts.  This benchmark asks whether the
winding accounts for the SIZE of that offset, and it asks it as a LADDER so the
effect can be attributed rather than merely observed:

    continuum   uniform current density over the gross outline (what ships)
    pitch       one solid square conductor per lattice site, at the turn pitch
    jacket      one solid square conductor per site, at the jacket outline
    cable       one disc per site, of the cable-space diameter
    annulus     one annulus per site, cable space around its central channel
    inert       the annulus with the steel jacket present and carrying nothing

Each rung is a strictly smaller current-carrying region inside the same
outline, so the ladder separates "the current is discrete" from "the current is
concentrated".  Every rung is reduced the same way the shipped operator reduces
the continuum -- sum of the pairwise flux linkage over all element pairs,
weighted by turns on both sides -- so the numbers are directly comparable to
each other and to the machine description.

The conductor dimensions are INPUTS, not facts: the cable-space fraction is
exactly what sets the effect size, so the ladder is swept over a plausible
range and reported as a band.

Stages, each writing its own JSON beside the figures:

    python benchmarks/cs_winding_inductance.py ladder
    python benchmarks/cs_winding_inductance.py sensitivity
    python benchmarks/cs_winding_inductance.py device --device gpu
    python benchmarks/cs_winding_inductance.py figures

Run the ladder and the device stage on a compute node; the coupling matrix of a
three-module winding is a few million pairs and a login node will not hold it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
import pathlib
import time

import numpy as np

from nova.biot.circle import Circle
from nova.frame.coilset import CoilSet
from nova.frame.turn import Turn

FIGURES = (
    pathlib.Path(__file__).resolve().parents[1] / "docs" / "figures" / "cs-winding"
)

# Reduced self- and mutual inductance [H] from the machine description, for the
# three-module case the shipped operator is gated against.  This is the ORACLE
# here: it is the built machine, and the smeared element is the thing on test.
MACHINE_DESCRIPTION = np.array(
    [
        [7.076e-01, 1.348e-01, 6.021e-02],
        [1.348e-01, 7.954e-01, 2.471e-01],
        [6.021e-02, 2.471e-01, 7.954e-01],
    ]
)


@dataclass(frozen=True)
class Module:
    """A wound coil: its gross outline, its turn count, and its winding."""

    name: str
    part: str
    radius: float  # outline centre, major radius [m]
    height: float  # outline centre, elevation [m]
    width: float  # outline radial extent [m]
    thickness: float  # outline vertical extent [m]
    nturn: float  # electrical turn count, may be fractional

    @property
    def area(self) -> float:
        """Return the gross cross-sectional area of the outline."""
        return self.width * self.thickness


# The geometry the shipped gate uses.  Two central solenoid modules and the
# upper poloidal field coil: the solenoid gap is 4.7e-04 and the field coil's is
# nearly ten times that, so one conductor model has to account for both or it is
# not the explanation.
MODULES = (
    Module("PF1", "PF", 3.9431, 7.5641, 0.9590, 0.9841, 248.64),
    Module("CS3U", "CS", 1.722, 5.313, 0.719, 2.075, 554),
    Module("CS2U", "CS", 1.722, 3.188, 0.719, 2.075, 554),
)


@dataclass(frozen=True)
class Conductor:
    """Cable-in-conduit dimensions, in metres.

    Every one of these is an input to be varied, not a fact to be asserted.
    ``jacket`` is the outside of the square steel conduit, ``cable`` the
    diameter of the cable space inside it, and ``channel`` the diameter of the
    central cooling spiral the strands wrap around -- so the current-carrying
    region of the annulus rung is the cable space minus that channel.
    """

    jacket: float = 0.049
    cable: float = 0.033
    channel: float = 0.010

    @property
    def cable_area(self) -> float:
        """Return the area of the cable space."""
        return 0.25 * math.pi * self.cable**2

    @property
    def annulus_area(self) -> float:
        """Return the cable space outside its central channel."""
        return 0.25 * math.pi * (self.cable**2 - self.channel**2)

    @property
    def skin_fraction(self) -> float:
        """Return the annulus wall as a fraction of the outer radius."""
        return 1.0 - self.channel / self.cable


@dataclass(frozen=True)
class Lattice:
    """A rectangular winding lattice inside a module outline."""

    module: Module
    n_radial: int
    n_vertical: int

    @property
    def sites(self) -> int:
        """Return the number of lattice sites."""
        return self.n_radial * self.n_vertical

    @property
    def pitch_radial(self) -> float:
        """Return the radial site spacing."""
        return self.module.width / self.n_radial

    @property
    def pitch_vertical(self) -> float:
        """Return the vertical site spacing."""
        return self.module.thickness / self.n_vertical

    @property
    def pitch(self) -> float:
        """Return the isotropic pitch equivalent to the site cell."""
        return math.sqrt(self.pitch_radial * self.pitch_vertical)

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the (radius, height) of every site, cell-centred."""
        radius = self.module.radius + self.pitch_radial * (
            np.arange(self.n_radial) - 0.5 * (self.n_radial - 1)
        )
        height = self.module.height + self.pitch_vertical * (
            np.arange(self.n_vertical) - 0.5 * (self.n_vertical - 1)
        )
        grid_r, grid_z = np.meshgrid(radius, height, indexing="ij")
        return grid_r.ravel(), grid_z.ravel()


def derive_lattice(module: Module) -> Lattice:
    """Return the lattice the outline and the turn count imply.

    The lattice is DERIVED, not asserted: among every rectangular arrangement
    with at least as many sites as turns, take the one whose site cell is
    closest to square, breaking ties towards fewer spare sites.  A conductor is
    square, so a square cell is the arrangement that a real winding tends to.
    """
    turns = math.ceil(module.nturn)
    best, best_key = None, None
    for n_radial in range(1, 4 * int(math.isqrt(turns)) + 2):
        n_vertical = math.ceil(turns / n_radial)
        trial = Lattice(module, n_radial, n_vertical)
        anisotropy = abs(math.log(trial.pitch_radial / trial.pitch_vertical))
        key = (round(anisotropy, 6), trial.sites - turns)
        if best_key is None or key < best_key:
            best, best_key = trial, key
    return best


def site_turns(lattice: Lattice, policy: str) -> np.ndarray:
    """Return the turn count carried by each lattice site.

    A lattice large enough to hold the turns is usually larger than it needs to
    be -- 560 sites for 554 solenoid turns -- and the shortfall is joints and
    transitions the outline does not resolve.  Where the spare sites go is a
    modelling choice, so it is a parameter:

    ``smear``    every site carries the same fractional turn.  Keeps the
                 current distribution symmetric about the outline centre.
    ``corner``   spare sites are dropped from the outline corners outwards,
                 the rest carrying a whole turn each and one site the
                 fractional remainder.  A joint occupies a corner.
    ``edge``     spare sites are dropped from the top of the innermost column,
                 which is where a solenoid module's helium and current leads
                 leave the pack.
    """
    radius, height = lattice.centres()
    turns = np.full(lattice.sites, lattice.module.nturn / lattice.sites)
    if policy == "smear":
        return turns
    spare = lattice.sites - math.ceil(lattice.module.nturn)
    if policy == "corner":
        centre_r = radius.mean()
        centre_z = height.mean()
        rank = np.argsort(
            -(
                ((radius - centre_r) / lattice.pitch_radial) ** 2
                + ((height - centre_z) / lattice.pitch_vertical) ** 2
            )
        )
    elif policy == "edge":
        rank = np.lexsort((-height, radius))
    else:
        raise ValueError(f"unknown turn placement policy {policy!r}")
    keep = np.setdiff1d(np.arange(lattice.sites), rank[:spare])
    turns = np.zeros(lattice.sites)
    turns[keep] = 1.0
    turns[keep[-1]] = lattice.module.nturn - (len(keep) - 1)
    return turns


# What each rung puts at a lattice site: the frame section family, and how its
# principal dimension and skin fraction come from the conductor.  ``current``
# is False for a region that is present but carries nothing.
RUNGS = ("continuum", "pitch", "jacket", "cable", "annulus", "inert")

# The skin fraction a SOLID section carries.  A solid family scales its outline
# by ``dl`` alone and ignores this second dimension -- but only if it is
# non-zero: a zero collapses the section to an empty polygon, which surfaces
# either as a zero-area frame or as a GEOS error on the empty centroid.
SOLID = 1.0


def rung_sections(rung: str, lattice: Lattice, conductor: Conductor) -> list[dict]:
    """Return the conductor regions one lattice site holds on a given rung."""
    if rung == "pitch":
        return [
            {"section": "square", "dl": lattice.pitch, "dt": SOLID, "current": True}
        ]
    if rung == "jacket":
        return [
            {"section": "square", "dl": conductor.jacket, "dt": SOLID, "current": True}
        ]
    if rung == "cable":
        return [
            {"section": "disc", "dl": conductor.cable, "dt": SOLID, "current": True}
        ]
    if rung == "annulus":
        return [
            {
                "section": "skin",
                "dl": conductor.cable,
                "dt": conductor.skin_fraction,
                "current": True,
            }
        ]
    if rung == "inert":
        return [
            {
                "section": "skin",
                "dl": conductor.cable,
                "dt": conductor.skin_fraction,
                "current": True,
            },
            {
                "section": "skin",
                "dl": conductor.jacket,
                "dt": 1.0 - conductor.cable / conductor.jacket,
                "current": False,
            },
        ]
    raise ValueError(f"rung {rung!r} carries no lattice")


def continuum_coilset(modules, delta: float) -> CoilSet:
    """Return the shipped model: uniform current density over each outline."""
    coilset = CoilSet(dcoil=delta)
    for module in modules:
        coilset.coil.insert(
            module.radius,
            module.height,
            module.width,
            module.thickness,
            nturn=module.nturn,
            name=module.name,
            part=module.part,
        )
    return coilset


def winding_coilset(modules, rung: str, conductor: Conductor, policy: str) -> CoilSet:
    """Return a coilset whose every turn is its own conductor region.

    One frame per module, one subframe element per turn, so the reduction that
    sums a coil's elements is the same one the continuum uses -- the comparison
    is like-for-like in everything but where the current sits.
    """
    coilset = CoilSet(dcoil=0.25)
    for module in modules:
        lattice = derive_lattice(module)
        radius, height = lattice.centres()
        turns = site_turns(lattice, policy)
        live = turns != 0.0
        for index, region in enumerate(rung_sections(rung, lattice, conductor)):
            turn = Turn(*coilset.frames, turn=region["section"])
            suffix = "" if region["current"] else "j"
            turn.insert(
                radius[live],
                height[live],
                region["dl"],
                region["dt"],
                nturn=turns[live] if region["current"] else 0.0 * turns[live],
                name=f"{module.name}{suffix}" if index else module.name,
                part=module.part if region["current"] else f"{module.part}j",
                active=region["current"],
                delta=0,
                segment="circle",
            )
    return coilset


def reduced_inductance(coilset: CoilSet) -> np.ndarray:
    """Return the coil-by-coil reduced inductance matrix [H]."""
    biot = Circle(
        coilset.subframe, coilset.subframe, turns=[True, True], reduce=[True, True]
    )
    return np.asarray(biot.compute("Psi")[0])


def active_inductance(coilset: CoilSet, modules) -> np.ndarray:
    """Return the reduced inductance of the current-carrying frames only."""
    matrix = reduced_inductance(coilset)
    names = list(coilset.frame.index)
    index = [names.index(module.name) for module in modules]
    return matrix[np.ix_(index, index)]


def measure_ladder(modules, conductor: Conductor, policy: str, rungs) -> dict:
    """Return the ladder: one reduced inductance matrix per rung."""
    result = {}
    for rung in rungs:
        start = time.perf_counter()
        if rung == "continuum":
            coilset = continuum_coilset(modules, 0.25)
        else:
            coilset = winding_coilset(modules, rung, conductor, policy)
        matrix = active_inductance(coilset, modules)
        result[rung] = {
            "matrix": matrix.tolist(),
            "elements": int(len(coilset.subframe)),
            "seconds": time.perf_counter() - start,
        }
    return result


def continuum_limit(modules, deltas) -> dict:
    """Return the smeared self-inductance against mesh size.

    The continuum baseline has to be a converged number rather than whatever
    the shipped mesh happens to give, or the ladder measures the mesh.
    """
    result = {}
    for delta in deltas:
        coilset = continuum_coilset(modules, delta)
        matrix = reduced_inductance(coilset)
        result[str(delta)] = {
            "matrix": matrix.tolist(),
            "elements": int(len(coilset.subframe)),
        }
    return result


def cable_sensitivity(modules, conductor: Conductor, policy: str, diameters) -> dict:
    """Return the ladder's cable rungs against cable-space diameter.

    The cable-space fraction is the input that sets the effect size, so the
    answer is a band over a plausible range, never a point.
    """
    result = {}
    for diameter in diameters:
        trial = replace(conductor, cable=float(diameter))
        entry = {}
        for rung in ("cable", "annulus"):
            coilset = winding_coilset(modules, rung, trial, policy)
            entry[rung] = active_inductance(coilset, modules).tolist()
        entry["cable_area"] = trial.cable_area
        entry["annulus_area"] = trial.annulus_area
        result[f"{diameter:.4f}"] = entry
    return result


def placement_spread(modules, conductor: Conductor, rung: str) -> dict:
    """Return the ladder rung under each way of placing turns on sites."""
    return {
        policy: active_inductance(
            winding_coilset(modules, rung, conductor, policy), modules
        ).tolist()
        for policy in ("smear", "corner", "edge")
    }


def turn_polygons(modules, conductor: Conductor, policy: str):
    """Return one square polygon per turn, with the turn centres as targets.

    The tiled operator couples uniform-current POLYGON sections to target
    points, so a winding is a natural batch for it: every turn is a four-corner
    section and every turn centre is a target, which makes a three-module pack
    a few million independent pairs of identical shape.
    """
    sections, target_r, target_z = [], [], []
    for module in modules:
        lattice = derive_lattice(module)
        radius, height = lattice.centres()
        turns = site_turns(lattice, policy)
        half = 0.5 * conductor.jacket
        for site_r, site_z, count in zip(radius, height, turns):
            if count == 0.0:
                continue
            sections.append(
                np.array(
                    [
                        [site_r - half, site_z - half],
                        [site_r + half, site_z - half],
                        [site_r + half, site_z + half],
                        [site_r - half, site_z + half],
                    ]
                )
            )
            target_r.append(site_r)
            target_z.append(site_z)
    return sections, np.array(target_r), np.array(target_z)


def _write(name: str, payload: dict) -> pathlib.Path:
    """Write a stage result beside the figures and return its path."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    path.write_text(json.dumps(payload, indent=1))
    return path


def stage_ladder(args) -> None:
    """Measure the ladder, the continuum limit and the placement spread."""
    conductor = Conductor(args.jacket, args.cable, args.channel)
    payload = {
        "conductor": conductor.__dict__,
        "modules": [module.__dict__ for module in MODULES],
        "lattice": {
            module.name: {
                "n_radial": derive_lattice(module).n_radial,
                "n_vertical": derive_lattice(module).n_vertical,
                "pitch_radial": derive_lattice(module).pitch_radial,
                "pitch_vertical": derive_lattice(module).pitch_vertical,
                "sites": derive_lattice(module).sites,
                "nturn": module.nturn,
            }
            for module in MODULES
        },
        "continuum_limit": continuum_limit(MODULES, args.deltas),
        "ladder": measure_ladder(MODULES, conductor, args.policy, RUNGS),
        "placement": placement_spread(MODULES, conductor, "cable"),
        "machine_description": MACHINE_DESCRIPTION.tolist(),
        "policy": args.policy,
    }
    print(f"wrote {_write('ladder.json', payload)}")
    report_ladder(payload)


def report_ladder(payload: dict) -> None:
    """Print the ladder table: every rung against the continuum and the oracle."""
    names = [module["name"] for module in payload["modules"]]
    machine = np.array(payload["machine_description"])
    finest = payload["continuum_limit"][
        min(payload["continuum_limit"], key=lambda key: float(key))
    ]
    continuum = np.array(finest["matrix"])
    print(f"\n{'rung':<11}" + "".join(f"{name:>34}" for name in names))
    print(f"{'':<11}" + "".join(f"{'self  -continuum   -machine':>34}" for _ in names))
    for rung, entry in payload["ladder"].items():
        matrix = np.array(entry["matrix"])
        row = f"{rung:<11}"
        for i in range(len(names)):
            row += (
                f"{matrix[i, i]:>14.6f}"
                f"{matrix[i, i] - continuum[i, i]:>+11.2e}"
                f"{matrix[i, i] - machine[i, i]:>+11.2e}"
            )
        print(row + f"   [{entry['elements']} elements]")
    gap = np.diag(machine) - np.diag(continuum)
    print("\ngap the winding has to close (machine - continuum):")
    for name, value in zip(names, gap):
        print(f"  {name:<6}{value:+.4e}")


def stage_sensitivity(args) -> None:
    """Sweep the cable-space diameter and report the ladder as a band."""
    conductor = Conductor(args.jacket, args.cable, args.channel)
    diameters = np.linspace(args.cable_min, args.cable_max, args.cable_steps)
    payload = {
        "conductor": conductor.__dict__,
        "diameters": diameters.tolist(),
        "sweep": cable_sensitivity(MODULES, conductor, args.policy, diameters),
        "machine_description": MACHINE_DESCRIPTION.tolist(),
        "policy": args.policy,
    }
    print(f"wrote {_write('sensitivity.json', payload)}")
    report_sensitivity(payload)


def report_sensitivity(payload: dict) -> None:
    """Print the swept rungs against the machine description."""
    machine = np.array(payload["machine_description"])
    names = [module.name for module in MODULES]
    print(f"\n{'cable [mm]':<12}{'rung':<9}" + "".join(f"{name:>26}" for name in names))
    for key, entry in payload["sweep"].items():
        for rung in ("cable", "annulus"):
            matrix = np.array(entry[rung])
            row = f"{1e3 * float(key):<12.1f}{rung:<9}"
            for i in range(len(names)):
                row += f"{matrix[i, i]:>14.6f}{matrix[i, i] - machine[i, i]:>+12.2e}"
            print(row)


def stage_device(args) -> None:
    """Time the tiled polygon operator over the winding, host against device."""
    import os

    os.environ.setdefault("JAX_PLATFORMS", "cuda" if args.device == "gpu" else "cpu")
    from nova.biot.polygon import pad_batch
    from nova.biot.tiledassembly import (
        TilePlan,
        compilation_cache,
        forget_evaluators,
        tile_coupling,
        tile_evaluator,
    )

    conductor = Conductor(args.jacket, args.cable, args.channel)
    sections, target_r, target_z = turn_polygons(MODULES, conductor, args.policy)
    edge, weight, norm = pad_batch(sections)
    pairs = len(target_r) * len(sections)
    print(f"{len(target_r)} targets x {len(sections)} sections = {pairs} pairs")

    cache = None if args.no_cache else compilation_cache()
    payload = {
        "device": args.device,
        "kernel": args.kernel,
        "targets": len(target_r),
        "sections": len(sections),
        "pairs": pairs,
        "compilation_cache": str(cache),
        "tiles": {},
    }
    for tile in args.tiles:
        plan = TilePlan(tile, tile, 16, 16, 48)
        rows = slice(0, min(tile, len(target_r)))
        columns = slice(0, min(tile, len(sections)))
        block = (
            target_r[rows],
            target_z[rows],
            edge[:, :, columns],
            weight[:, columns],
            norm[columns],
        )
        tile_pairs = (rows.stop - rows.start) * (columns.stop - columns.start)

        host_start = time.perf_counter()
        host = tile_coupling(*block)
        host_seconds = time.perf_counter() - host_start

        forget_evaluators()
        cold_start = time.perf_counter()
        evaluate = tile_evaluator(plan, batched=args.batched, kernel=args.kernel)
        first = evaluate(*block)
        first = tuple(np.asarray(component) for component in first)
        cold_seconds = time.perf_counter() - cold_start

        warm_start = time.perf_counter()
        for _ in range(args.repeats):
            warm = evaluate(*block)
            warm[0].block_until_ready()
        warm_seconds = (time.perf_counter() - warm_start) / args.repeats

        worst = max(
            float(np.max(np.abs(np.asarray(got) - want)))
            for got, want in zip(first, host)
        )
        scale = max(float(np.max(np.abs(component))) for component in host)
        payload["tiles"][str(tile)] = {
            "tile_pairs": tile_pairs,
            "host_us_per_pair": 1e6 * host_seconds / tile_pairs,
            "cold_seconds": cold_seconds,
            "warm_us_per_pair": 1e6 * warm_seconds / tile_pairs,
            "worst_absolute": worst,
            "worst_relative": worst / scale,
        }
        entry = payload["tiles"][str(tile)]
        print(
            f"tile {tile:>4} pairs {tile_pairs:>7}"
            f" host {entry['host_us_per_pair']:>8.2f} us/pair"
            f" cold {cold_seconds:>7.1f} s"
            f" warm {entry['warm_us_per_pair']:>8.3f} us/pair"
            f" agreement {worst:.2e} ({entry['worst_relative']:.1e} relative)"
        )
    print(f"wrote {_write(f'device-{args.device}-{args.kernel}.json', payload)}")


def stage_figures(args) -> None:
    """Draw the lattice, the ladder and the throughput curve."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle as CirclePatch, Rectangle, Wedge

    ladder = json.loads((FIGURES / "ladder.json").read_text())
    conductor = Conductor(**ladder["conductor"])
    machine = np.array(ladder["machine_description"])
    names = [module["name"] for module in ladder["modules"]]
    finest = ladder["continuum_limit"][
        min(ladder["continuum_limit"], key=lambda key: float(key))
    ]
    continuum = np.array(finest["matrix"])

    module = MODULES[1]
    lattice = derive_lattice(module)
    radius, height = lattice.centres()
    turns = site_turns(lattice, ladder["policy"])
    figure, axes = plt.subplots(1, len(RUNGS), figsize=(3.0 * len(RUNGS), 8.4))
    for axis, rung in zip(axes, RUNGS):
        axis.add_patch(
            Rectangle(
                (
                    module.radius - module.width / 2,
                    module.height - module.thickness / 2,
                ),
                module.width,
                module.thickness,
                fill=False,
                edgecolor="0.4",
                lw=1.0,
            )
        )
        if rung == "continuum":
            axis.add_patch(
                Rectangle(
                    (
                        module.radius - module.width / 2,
                        module.height - module.thickness / 2,
                    ),
                    module.width,
                    module.thickness,
                    facecolor="C0",
                    alpha=0.55,
                )
            )
        else:
            for region in rung_sections(rung, lattice, conductor):
                face = "C0" if region["current"] else "0.75"
                for site_r, site_z, count in zip(radius, height, turns):
                    if count == 0.0:
                        continue
                    if region["section"] == "square":
                        axis.add_patch(
                            Rectangle(
                                (
                                    site_r - region["dl"] / 2,
                                    site_z - region["dl"] / 2,
                                ),
                                region["dl"],
                                region["dl"],
                                facecolor=face,
                                edgecolor="none",
                            )
                        )
                    elif region["section"] == "disc":
                        axis.add_patch(
                            CirclePatch(
                                (site_r, site_z), region["dl"] / 2, facecolor=face
                            )
                        )
                    else:
                        axis.add_patch(
                            Wedge(
                                (site_r, site_z),
                                region["dl"] / 2,
                                0,
                                360,
                                width=region["dl"] * region["dt"] / 2,
                                facecolor=face,
                            )
                        )
        matrix = np.array(ladder["ladder"][rung]["matrix"])
        axis.set_title(f"{rung}\n{matrix[1, 1]:.6f} H", fontsize=9)
        axis.set_xlim(
            module.radius - 0.62 * module.width, module.radius + 0.62 * module.width
        )
        axis.set_ylim(
            module.height - 0.56 * module.thickness,
            module.height + 0.56 * module.thickness,
        )
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"{module.name} winding ladder, {lattice.n_radial}x{lattice.n_vertical} lattice"
        f" at {1e3 * lattice.pitch:.1f} mm pitch, {module.nturn:g} turns"
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "lattice.png", dpi=130)
    plt.close(figure)

    figure, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names), 4.4))
    sensitivity_path = FIGURES / "sensitivity.json"
    sensitivity = (
        json.loads(sensitivity_path.read_text()) if sensitivity_path.exists() else None
    )
    for index, (axis, name) in enumerate(zip(axes, names)):
        values = [
            np.array(ladder["ladder"][rung]["matrix"])[index, index] for rung in RUNGS
        ]
        axis.plot(range(len(RUNGS)), values, "o-", color="C0", label="ladder")
        axis.axhline(continuum[index, index], color="C7", ls=":", label="continuum")
        axis.axhline(machine[index, index], color="C3", ls="--", label="machine")
        if sensitivity is not None:
            band = [
                np.array(entry["cable"])[index, index]
                for entry in sensitivity["sweep"].values()
            ]
            axis.axhspan(
                min(band), max(band), color="C0", alpha=0.15, label="cable sweep"
            )
        axis.set_xticks(range(len(RUNGS)))
        axis.set_xticklabels(RUNGS, rotation=35, ha="right")
        axis.set_title(name)
        axis.set_ylabel("reduced self-inductance [H]")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle("self-inductance ladder against the machine description")
    figure.tight_layout()
    figure.savefig(FIGURES / "ladder.png", dpi=130)
    plt.close(figure)

    device = sorted(FIGURES.glob("device-*.json"))
    if device:
        figure, axis = plt.subplots(figsize=(6.4, 4.4))
        for path in device:
            payload = json.loads(path.read_text())
            tiles = sorted(payload["tiles"], key=int)
            axis.plot(
                [int(tile) for tile in tiles],
                [payload["tiles"][tile]["warm_us_per_pair"] for tile in tiles],
                "o-",
                label=f"{payload['device']} {payload['kernel']} warm",
            )
            axis.plot(
                [int(tile) for tile in tiles],
                [payload["tiles"][tile]["host_us_per_pair"] for tile in tiles],
                "s--",
                label=f"{payload['device']} {payload['kernel']} host",
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("tile width [sections]")
        axis.set_ylabel("us per pair")
        axis.set_title("tiled polygon operator over the winding")
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(FIGURES / "throughput.png", dpi=130)
        plt.close(figure)
    print(f"wrote figures to {FIGURES}")


def parse_args(argv=None):
    """Return the parsed command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jacket", type=float, default=0.049)
    parser.add_argument("--cable", type=float, default=0.033)
    parser.add_argument("--channel", type=float, default=0.010)
    parser.add_argument(
        "--policy", default="smear", choices=("smear", "corner", "edge")
    )
    stages = parser.add_subparsers(dest="stage", required=True)

    ladder = stages.add_parser("ladder")
    ladder.add_argument("--deltas", type=float, nargs="+", default=[0.25, 0.12, 0.08])
    ladder.set_defaults(run=stage_ladder)

    sensitivity = stages.add_parser("sensitivity")
    sensitivity.add_argument("--cable-min", type=float, default=0.026)
    sensitivity.add_argument("--cable-max", type=float, default=0.042)
    sensitivity.add_argument("--cable-steps", type=int, default=9)
    sensitivity.set_defaults(run=stage_sensitivity)

    device = stages.add_parser("device")
    device.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    device.add_argument(
        "--kernel", default="quadrature", choices=("quadrature", "closed")
    )
    device.add_argument("--tiles", type=int, nargs="+", default=[16, 32, 64, 128])
    device.add_argument("--repeats", type=int, default=3)
    device.add_argument("--batched", action="store_true", default=True)
    device.add_argument("--scan", dest="batched", action="store_false")
    device.add_argument("--no-cache", action="store_true")
    device.set_defaults(run=stage_device)

    figures = stages.add_parser("figures")
    figures.set_defaults(run=stage_figures)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    arguments.run(arguments)
