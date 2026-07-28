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


def turn_count_equivalent(modules, delta: float, step: float = 0.005) -> dict:
    """Return how many turns each coil's gap to the machine description is worth.

    A pack's inductance goes as the square of its turn count, so a discrepancy
    can always be restated as a turn count -- and that is the honest scale to
    judge a competing explanation on.  The derivative is MEASURED by rebuilding
    the continuum at a perturbed turn count rather than assumed to be exactly
    quadratic, because the reduction is over elements whose own turn shares move
    with it.
    """
    base = np.diag(reduced_inductance(continuum_coilset(modules, delta)))
    perturbed = np.diag(
        reduced_inductance(
            continuum_coilset(
                [
                    replace(module, nturn=module.nturn * (1 + step))
                    for module in modules
                ],
                delta,
            )
        )
    )
    counts = np.array([module.nturn for module in modules])
    slope = (perturbed - base) / (step * counts)
    gap = np.diag(MACHINE_DESCRIPTION) - base
    return {
        "self": base.tolist(),
        "henry_per_turn": slope.tolist(),
        "turns_equivalent": (gap / slope).tolist(),
        "exponent": (counts * slope / base).tolist(),
    }


def cable_sensitivity(modules, conductor: Conductor, policy: str, diameters) -> dict:
    """Return the ladder's cable rungs against cable-space diameter.

    The cable-space fraction is the input that sets the effect size, so the
    answer is a band over a plausible range, never a point.
    """
    result = {}
    for diameter in diameters:
        trial = replace(conductor, cable=float(diameter))
        entry = {}
        # A channel wider than the cable space is not a conductor, so the
        # annulus rung drops out at the small-diameter end of a sweep.
        rungs = ("cable",) if trial.channel >= trial.cable else ("cable", "annulus")
        for rung in rungs:
            coilset = winding_coilset(modules, rung, trial, policy)
            entry[rung] = active_inductance(coilset, modules).tolist()
        entry["cable_area"] = trial.cable_area
        entry["annulus_area"] = trial.annulus_area
        result[f"{diameter:.4f}"] = entry
    return result


def placement_spread(modules, conductor: Conductor, rungs) -> dict:
    """Return each rung under each way of placing turns on lattice sites.

    Two rungs, not one, because the placement moves two things at once.  Where
    the spare sites go changes the current DISTRIBUTION inside the outline, and
    that shows up on the pitch rung, whose conductors fill their cells and
    carry no concentration effect at all.  Anything left after subtracting the
    pitch rung at the same placement is CONCENTRATION.  Quoting a rung's offset
    from the continuum without that subtraction confuses the two.
    """
    return {
        policy: {
            rung: active_inductance(
                winding_coilset(modules, rung, conductor, policy), modules
            ).tolist()
            for rung in rungs
        }
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
        "placement": placement_spread(MODULES, conductor, ("pitch", "cable")),
        "turn_count": turn_count_equivalent(MODULES, min(args.deltas)),
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
    counts = payload["turn_count"]
    print(
        "\nthe same gap restated as a turn count, from a measured dL/dN"
        " (exponent is the local power of N):"
    )
    for index, name in enumerate(names):
        print(
            f"  {name:<6}dL/dN {counts['henry_per_turn'][index]:.3e} H/turn"
            f"   exponent {counts['exponent'][index]:.3f}"
            f"   gap = {counts['turns_equivalent'][index]:+.3f} turns"
        )
    report_placement(payload)


def report_placement(payload: dict) -> None:
    """Print the placement spread, split into distribution and concentration."""
    names = [module["name"] for module in payload["modules"]]
    finest = payload["continuum_limit"][
        min(payload["continuum_limit"], key=lambda key: float(key))
    ]
    continuum = np.array(finest["matrix"])
    print(
        "\nturn placement: distribution is the pitch rung against the continuum,"
        "\nconcentration is the cable rung against the pitch rung at the same"
        " placement\n"
    )
    print(f"{'placement':<11}" + "".join(f"{name:>30}" for name in names))
    print(f"{'':<11}" + "".join(f"{'distribution  concentration':>30}" for _ in names))
    for policy, rungs in payload["placement"].items():
        pitch = np.array(rungs["pitch"])
        cable = np.array(rungs["cable"])
        row = f"{policy:<11}"
        for i in range(len(names)):
            row += (
                f"{pitch[i, i] - continuum[i, i]:>+15.2e}"
                f"{cable[i, i] - pitch[i, i]:>+15.2e}"
            )
        print(row)


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
    print(f"wrote {_write(f'sensitivity{args.label}.json', payload)}")
    report_sensitivity(payload)


def report_sensitivity(payload: dict) -> None:
    """Print the swept rungs against the machine description."""
    machine = np.array(payload["machine_description"])
    names = [module.name for module in MODULES]
    print(f"\n{'cable [mm]':<12}{'rung':<9}" + "".join(f"{name:>26}" for name in names))
    for key, entry in payload["sweep"].items():
        for rung in ("cable", "annulus"):
            if rung not in entry:
                continue
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
        "mapping": "vmap" if args.batched else "scan",
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

        # The evaluator converts its result on the way out, so the call has
        # already blocked on the device by the time it returns -- there is no
        # asynchronous dispatch left to wait for.
        warm_start = time.perf_counter()
        for _ in range(args.repeats):
            evaluate(*block)
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
    payload["build"] = {
        str(tile): full_build(sections, target_r, target_z, args, tile, print_row=True)
        for tile in args.build_tile
    }
    name = f"device-{args.device}-{args.kernel}{args.label}.json"
    print(f"wrote {_write(name, payload)}")


def full_build(sections, target_r, target_z, args, tile, *, print_row=False) -> dict:
    """Assemble the whole winding operator on the host and on the device.

    A tile is a microbenchmark; the operator is the workload.  Two central
    solenoid modules and a poloidal field coil wound turn by turn is a couple of
    million pairs, which is the regime the tiled path exists for, so the two
    backends are timed over the SAME complete build and compared over every
    pair of one tile of the result rather than over a tile evaluated on its own.
    """
    import shutil
    import tempfile

    from nova.biot.tiledassembly import COMPONENTS, TilePlan, assemble

    import zarr

    plan = TilePlan(tile, tile, 16, 16, 48)
    root = pathlib.Path(tempfile.mkdtemp(prefix="cs-winding-"))
    result = {
        "tile": tile,
        "workers": args.workers,
        "tiles": plan.tile_count(len(target_r), len(sections)),
    }
    try:
        stores = {}
        for backend, kwargs in (
            ("numpy", {"workers": args.workers}),
            ("jax", {"batched": args.batched, "kernel": args.kernel}),
        ):
            path = root / backend
            start = time.perf_counter()
            assemble(
                path, target_r, target_z, sections, plan=plan, backend=backend, **kwargs
            )
            seconds = time.perf_counter() - start
            stores[backend] = zarr.open_group(str(path), mode="r")
            pairs = len(target_r) * len(sections)
            result[backend] = {
                "seconds": seconds,
                "us_per_pair": 1e6 * seconds / pairs,
            }
            if print_row:
                print(
                    f"build tile {tile:>4} {backend:<6} {seconds:>8.2f} s"
                    f" {result[backend]['us_per_pair']:>8.3f} us/pair"
                )
        worst, scale = 0.0, 0.0
        for name in COMPONENTS:
            got = np.asarray(stores["jax"][name][:])
            want = np.asarray(stores["numpy"][name][:])
            worst = max(worst, float(np.max(np.abs(got - want))))
            scale = max(scale, float(np.max(np.abs(want))))
        result["worst_absolute"] = worst
        result["worst_relative"] = worst / scale
        if print_row:
            print(f"build agreement over every pair {worst:.3e} ({worst / scale:.2e})")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return result


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

    def draw_rung(axis, rung, window=None):
        """Fill one axis with the current-carrying regions of one rung.

        ``window`` is a (half-width, half-height) view about the module centre;
        sites outside it are skipped rather than drawn and clipped, which is
        what keeps a close-up panel from carrying the whole pack's geometry.
        """
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
            return
        for region in rung_sections(rung, lattice, conductor):
            face = "C0" if region["current"] else "0.75"
            for site_r, site_z, count in zip(radius, height, turns):
                if count == 0.0:
                    continue
                if window is not None and (
                    abs(site_r - module.radius) > window[0] + region["dl"]
                    or abs(site_z - module.height) > window[1] + region["dl"]
                ):
                    continue
                if region["section"] == "square":
                    axis.add_patch(
                        Rectangle(
                            (site_r - region["dl"] / 2, site_z - region["dl"] / 2),
                            region["dl"],
                            region["dl"],
                            facecolor=face,
                            edgecolor="none",
                        )
                    )
                elif region["section"] == "disc":
                    axis.add_patch(
                        CirclePatch((site_r, site_z), region["dl"] / 2, facecolor=face)
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

    # Two rows because the two things worth seeing are at different scales: the
    # pack fills the outline, and what changes between rungs is the conductor.
    figure, axes = plt.subplots(
        2,
        len(RUNGS),
        figsize=(2.6 * len(RUNGS), 9.6),
        gridspec_kw={"height_ratios": [3, 1]},
    )
    for column, rung in enumerate(RUNGS):
        matrix = np.array(ladder["ladder"][rung]["matrix"])
        offset = matrix[1, 1] - continuum[1, 1]
        windows = (
            (0.62 * module.width, 0.56 * module.thickness),
            (1.6 * lattice.pitch_radial, 1.6 * lattice.pitch_vertical),
        )
        for row, (half_r, half_z) in enumerate(windows):
            axis = axes[row, column]
            axis.set_rasterization_zorder(2 if row == 0 else None)
            draw_rung(axis, rung, None if row == 0 else (half_r, half_z))
            axis.set_xlim(module.radius - half_r, module.radius + half_r)
            axis.set_ylim(module.height - half_z, module.height + half_z)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
        axes[0, column].set_title(
            f"{rung}\n{matrix[1, 1]:.6f} H\n{offset:+.2e} on the continuum", fontsize=9
        )
    axes[1, 0].set_xlabel("detail, a few turns", fontsize=8)
    figure.suptitle(
        f"{module.name} winding ladder, {lattice.n_radial}x{lattice.n_vertical} lattice"
        f" at {1e3 * lattice.pitch_radial:.2f} x {1e3 * lattice.pitch_vertical:.2f} mm"
        f" pitch, {module.nturn:g} turns"
        f"\njacket {1e3 * conductor.jacket:.0f} mm, cable space"
        f" {1e3 * conductor.cable:.0f} mm, channel {1e3 * conductor.channel:.0f} mm",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(FIGURES / "lattice.svg", dpi=200)
    plt.close(figure)

    # Everything is plotted as an OFFSET from the converged continuum, because
    # the quantity under study is four decimal places into the value itself.
    sweep_paths = sorted(FIGURES.glob("sensitivity*.json"))
    sweeps = [json.loads(path.read_text()) for path in sweep_paths]
    rungs = [rung for rung in RUNGS if rung != "continuum"]
    figure, axes = plt.subplots(
        2, len(names), figsize=(4.6 * len(names), 7.6), squeeze=False
    )
    for index, name in enumerate(names):
        axis = axes[0, index]
        offsets = [
            np.array(ladder["ladder"][rung]["matrix"])[index, index]
            - continuum[index, index]
            for rung in rungs
        ]
        gap = machine[index, index] - continuum[index, index]
        axis.axhline(0.0, color="C7", ls=":", label="converged continuum")
        axis.axhline(gap, color="C3", ls="--", label="machine description")
        axis.plot(range(len(rungs)), offsets, "o-", color="C0", label="winding ladder")
        turns_equivalent = ladder["turn_count"]["turns_equivalent"][index]
        axis.annotate(
            f"gap {gap:+.2e} H\n= {turns_equivalent:+.3f} turns",
            (0.02, gap),
            xycoords=("axes fraction", "data"),
            textcoords="offset points",
            xytext=(0, -4),
            fontsize=8,
            color="C3",
            va="top",
        )
        axis.set_xticks(range(len(rungs)))
        axis.set_xticklabels(rungs, rotation=30, ha="right")
        axis.set_title(f"{name}: what the winding buys")
        axis.set_ylabel("offset from the continuum [H]")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

        axis = axes[1, index]
        if not sweeps:
            continue
        for rung, style in (("cable", "o-"), ("annulus", "s-")):
            points = sorted(
                (1e3 * float(key), np.array(entry[rung])[index, index])
                for source in sweeps
                for key, entry in source["sweep"].items()
                if rung in entry
            )
            axis.plot(
                [point[0] for point in points],
                [point[1] - continuum[index, index] for point in points],
                style,
                label=rung,
            )
        axis.axhline(0.0, color="C7", ls=":")
        axis.axhline(gap, color="C3", ls="--", label="machine description")
        axis.axvspan(32.0, 35.0, color="0.5", alpha=0.15, label="plausible cable space")
        axis.set_xscale("log")
        axis.set_xticks([2, 5, 10, 20, 30, 40])
        axis.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axis.set_xlabel("cable-space diameter [mm]")
        axis.set_ylabel("offset from the continuum [H]")
        axis.set_title(f"{name}: conductor sensitivity")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(
        "does the discrete winding close the gap to the machine description?"
    )
    figure.tight_layout()
    figure.savefig(FIGURES / "ladder.svg")
    plt.close(figure)

    device = sorted(FIGURES.glob("device-*.json"))
    if device:
        figure, axis = plt.subplots(figsize=(6.4, 4.4))
        for path in device:
            payload = json.loads(path.read_text())
            run = path.stem.removeprefix("device-")
            tiles = sorted(payload["tiles"], key=int)
            axis.plot(
                [int(tile) for tile in tiles],
                [payload["tiles"][tile]["warm_us_per_pair"] for tile in tiles],
                "o-",
                label=f"{run} one tile",
            )
            build = payload.get("build", {})
            if not build:
                continue
            widths = sorted((width for width in build if width.isdigit()), key=int)
            if not widths:
                continue
            axis.plot(
                [int(width) for width in widths],
                [build[width]["jax"]["us_per_pair"] for width in widths],
                "^-",
                label=f"{run} whole operator",
            )
            axis.plot(
                [int(width) for width in widths],
                [build[width]["numpy"]["us_per_pair"] for width in widths],
                "s--",
                label=f"host pool x{build[widths[0]]['workers']} beside {run}",
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("tile width [sections]")
        axis.set_ylabel("us per pair")
        axis.set_title(
            "tiled polygon operator over the winding\n"
            "one tile (circles) against the whole 1.89M-pair build (triangles)"
        )
        axis.grid(alpha=0.3, which="both")
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(FIGURES / "throughput.svg")
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
    sensitivity.add_argument("--label", default="")
    sensitivity.set_defaults(run=stage_sensitivity)

    device = stages.add_parser("device")
    device.add_argument("--device", default="cpu", choices=("cpu", "gpu"))
    device.add_argument(
        "--kernel", default="quadrature", choices=("quadrature", "closed")
    )
    device.add_argument("--tiles", type=int, nargs="+", default=[16, 32, 64, 128])
    device.add_argument("--repeats", type=int, default=3)
    device.add_argument("--build-tile", type=int, nargs="+", default=[128])
    device.add_argument("--workers", type=int, default=8)
    # Distinguishes runs of one kernel that differ in something the filename
    # would otherwise collapse -- a cold compile against a cache-served one.
    device.add_argument("--label", default="")
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
