"""Manage deferred import of biot methods."""
__all__ = [
           'Field',
           'Force',
           'Gap',
           'Grid',
           'Inductance',
           'LevelSet',
           'Loop',
           'Plasma',
           'PlasmaGrid',
           'Point',
           'Select',
           'Wall',
           ]

from nova import ImportManager

imp = ImportManager(package='nova.biot')

if imp.defer:
    Field = imp.load('.field', 'Field')
    Force = imp.load('.force', 'Force')
    Gap = imp.load('.gap', 'Gap')
    Grid = imp.load('.grid', 'Grid')
    Inductance = imp.load('.inductance', 'Inductance')
    LevelSet = imp.load('.levelset', 'LevelSet')
    Loop = imp.load('.loop', 'Loop')
    Plasma = imp.load('.plasma', 'Plasma')
    PlasmaGrid = imp.load('.plasmagrid', 'PlasmaGrid')
    Point = imp.load('.point', 'Point')
    Select = imp.load('.select', 'Select')
    Wall = imp.load('.wall', 'Wall')
else:
    from nova.biot.field import Field
    from nova.biot.force import Force
    from nova.biot.gap import Gap
    from nova.biot.grid import Grid
    from nova.biot.inductance import Inductance
    from nova.biot.levelset import LevelSet
    from nova.biot.loop import Loop
    from nova.biot.plasma import Plasma
    from nova.biot.plasmagrid import PlasmaGrid
    from nova.biot.point import Point
    from nova.biot.select import Select
    from nova.biot.wall import Wall
