"""Carry a solved iron magnetisation in a CoilSet as one passive frame.

The affine material solve returns cell magnetisation; the field it produces is
the field of the cell's bound currents. Inserting those currents as a passive
frame is what makes every existing target set -- grid, point, probes, the
forward map -- include the iron without a change to any solve path: the frame's
contribution is a source column set like any other, so ``Solve`` builds it once
per target set and the operator's ``matrix @ Ic`` carries it thereafter.

``Ic`` here is a bound-current amplitude, not an ampere-turn: one sheet current
per poloidal metre on an edge atom, or the azimuthal volume current times the
cell area. The material solve writes it; nothing else does.
"""

from __future__ import annotations

import numpy as np

from nova.biot import magnetisation as iron
from nova.biot.solve import IRON_COLUMN, IRON_SEGMENT

IRON_PART = "iron"


def insert_iron(coilset, solution, label: str = "iron"):
    """Insert a solved magnetisation into ``coilset`` as one passive frame.

    ``solution`` is a :class:`nova.biot.magnetisation.MagnetisationSolution` or
    an already-built :class:`nova.biot.magnetisation.BoundCurrents`. Returns the
    :class:`~nova.biot.magnetisation.BoundCurrents` the frame carries, so the
    material solve can rewrite ``coilset.sloc[label, "Ic"]`` and re-solve.
    """
    source = (
        solution
        if isinstance(solution, iron.BoundCurrents)
        else solution.bound_currents()
    )
    if label in np.asarray(coilset.frame.index):
        raise ValueError(f"frame {label!r} already exists in this coilset")
    count = source.unknowns
    for metaframe, owner in (
        (coilset.frame.metaframe, coilset),
        (coilset.subframe.metaframe, coilset),
    ):
        for attr in metaframe.additional:
            if attr not in metaframe.default:
                # a frame reloaded from its stored cache keeps the attribute in
                # ``additional`` without carrying its default, and the rows this
                # call appends are filled from that table
                metaframe.default[attr] = getattr(owner, attr, "")
    centre = np.mean(source.atoms.reshape(-1, 2), axis=0)
    before = np.asarray(coilset.saloc["Ic"], dtype=np.float64).copy()
    coilset.frame.insert(
        x=centre[0],
        y=0.0,
        z=centre[1],
        r=centre[0],
        label=label,
        part=IRON_PART,
        active=False,
        segment=IRON_SEGMENT,
    )
    reference = len(coilset.subframe) + np.arange(count)
    coilset.subframe.insert(
        x=np.full(count, centre[0]),
        y=np.zeros(count),
        z=np.full(count, centre[1]),
        r=np.full(count, centre[0]),
        frame=np.full(count, label, dtype=object),
        label=label,
        part=IRON_PART,
        active=np.zeros(count, dtype=bool),
        plasma=np.zeros(count, dtype=bool),
        nturn=np.ones(count),
        Ic=source.currents.copy(),
        ref=reference,
        link=np.full(count, "", dtype=object),
        segment=IRON_SEGMENT,
    )
    column = np.empty(len(coilset.subframe), dtype=object)
    column[:] = None
    column[reference] = source
    coilset.subframe[IRON_COLUMN] = column
    # the frame grew, so every locator already built over the old source index
    # is stale: rebuild them lazily against the new subspace
    from nova.frame.framesetloc import FrameSetLoc

    holders = [
        value
        for value in (coilset, *vars(coilset).values())
        if isinstance(value, FrameSetLoc)
    ]
    for holder in holders:
        for name in ("aloc", "saloc", "aloc_hash"):
            holder.__dict__.pop(name, None)
    current = np.zeros(len(coilset.subframe.subspace.index))
    current[: len(before)] = before
    current[iron_positions(coilset, label)] = source.currents
    coilset.saloc["Ic"][:] = current
    return source


def iron_positions(coilset, label: str = "iron") -> np.ndarray:
    """Return the subspace positions of the iron frame's independent rows."""
    mask = np.asarray(coilset.subframe["part"], dtype=object) == IRON_PART
    rows = np.flatnonzero(mask)
    labels = np.asarray(coilset.subframe.index, dtype=object)[rows]
    if "frame" in coilset.subframe:
        member = np.asarray(coilset.subframe["frame"], dtype=object)[rows]
        labels = labels[np.asarray(member) == label]
    positions = coilset.subframe.subspace.index.get_indexer(labels)
    if np.any(positions < 0):
        raise ValueError(f"iron frame {label!r} has rows outside the subspace")
    return positions


def set_iron_currents(coilset, source, label: str = "iron"):
    """Write the material solve's bound currents onto the passive frame."""
    values = np.asarray(coilset.saloc["Ic"], dtype=np.asarray(source.currents).dtype)
    values[iron_positions(coilset, label)] = source.currents
    return source
