"""Manage deferred import of biot methods."""
__all__ = [
           'Field',
           'Force',
           'Gap',
           'Grid',
           'HexGrid',
           'Inductance',
           'LevelSet',
           'Loop',
           'Plasma',
           'PlasmaGrid',
           'PlasmaWall',
           'Point',
           ]

from nova import ImportManager

imp = ImportManager(package='nova.biot')

if imp.defer:
    Field = imp.load('.field', 'Field')
    Force = imp.load('.force', 'Force')
    Gap = imp.load('.gap', 'Gap')
    Grid = imp.load('.grid', 'Grid')
    HexGrid = imp.load('.hexgrid', 'HexGrid')
    Inductance = imp.load('.inductance', 'Inductance')
    LevelSet = imp.load('.levelset', 'LevelSet')
    Loop = imp.load('.loop', 'Loop')
    Plasma = imp.load('.plasma', 'Plasma')
    PlasmaGrid = imp.load('.plasmagrid', 'PlasmaGrid')
    PlasmaWall = imp.load('.plasmawall', 'PlasmaWall')
    Point = imp.load('.point', 'Point')
else:
    from nova.biot.field import Field
    from nova.biot.force import Force
    from nova.biot.gap import Gap
    from nova.biot.grid import Grid
    from nova.biot.hexgrid import HexGrid
    from nova.biot.inductance import Inductance
    from nova.biot.levelset import LevelSet
    from nova.biot.loop import Loop
    from nova.biot.plasma import Plasma
    from nova.biot.plasmagrid import PlasmaGrid
    from nova.biot.plasmawall import PlasmaWall
    from nova.biot.point import Point
