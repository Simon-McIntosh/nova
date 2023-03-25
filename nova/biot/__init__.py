"""Manage deferred import of biot methods."""
__all__ = [
           'Field',
           'Force',
           'Gap',
           'Grid',
           'Inductance',
           'KDTree',
           'LevelSet',
           'Loop',
           'Plasma',
           'PlasmaGrid',
           'Point',
           'Wall',
           ]

from nova import ImportManager

imp = ImportManager(package='nova.biot')

if imp.defer:
    Gap = imp.load('.gap', 'Gap')
    Grid = imp.load('.grid', 'Grid')
    Inductance = imp.load('.inductance', 'Inductance')
    KDTree = imp.load('.kdtree', 'KDTree')
    Loop = imp.load('.loop', 'Loop')
    Plasma = imp.load('.plasma', 'Plasma')
    PlasmaGrid = imp.load('.plasmagrid', 'PlasmaGrid')
    Point = imp.load('.point', 'Point')
    Field = imp.load('.field', 'Field')
    Force = imp.load('.force', 'Force')
    LevelSet = imp.load('.levelset', 'LevelSet')
    Wall = imp.load('.wall', 'Wall')
else:
    from nova.biot.gap import Gap
    from nova.biot.grid import Grid
    from nova.biot.inductance import Inductance
    from nova.biot.kdtree import KDTree
    from nova.biot.loop import Loop
    from nova.biot.plasma import Plasma
    from nova.biot.plasmagrid import PlasmaGrid
    from nova.biot.point import Point
    from nova.biot.field import Field
    from nova.biot.levelset import LevelSet
    from nova.biot.force import Force
    from nova.biot.wall import Wall
