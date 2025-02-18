"""Manage deferred import of biot methods."""

__all__ = [
    "Field",
    "Force",
    "Gap",
    "Grid",
    "HexGrid",
    "Inductance",
    "LevelSet",
    "Loop",
    "Overlap",
    "Plasma",
    "PlasmaGap",
    "PlasmaGrid",
    "PlasmaWall",
    "Point",
]

from nova.utilities.importmanager import ImportManager

imp = ImportManager(package="nova.biot")

if imp.defer:
    Field = imp.load(".field", "Field")
    Force = imp.load(".force", "Force")
    Grid = imp.load(".grid", "Grid")
    HexGrid = imp.load(".hexgrid", "HexGrid")
    Inductance = imp.load(".inductance", "Inductance")
    LevelSet = imp.load(".levelset", "LevelSet")
    Loop = imp.load(".loop", "Loop")
    Overlap = imp.load(".overlap", "Overlap")
    Plasma = imp.load(".plasma", "Plasma")
    PlasmaGap = imp.load(".plasmagap", "PlasmaGap")
    PlasmaGrid = imp.load(".plasmagrid", "PlasmaGrid")
    PlasmaWall = imp.load(".plasmawall", "PlasmaWall")
    Point = imp.load(".point", "Point")
else:
    from nova.biot.field import Field
    from nova.biot.force import Force
    from nova.biot.grid import Grid
    from nova.biot.hexgrid import HexGrid
    from nova.biot.inductance import Inductance
    from nova.biot.levelset import LevelSet
    from nova.biot.loop import Loop
    from nova.biot.overlap import Overlap
    from nova.biot.plasma import Plasma
    from nova.biot.plasmagap import PlasmaGap
    from nova.biot.plasmagrid import PlasmaGrid
    from nova.biot.plasmawall import PlasmaWall
    from nova.biot.point import Point
