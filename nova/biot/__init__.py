"""Manage deferred import of biot methods."""
__all__ = [
           'BiotFirstWall',
           'BiotGap',
           'BiotGrid',
           'BiotInductance',
           'BiotLoop',
           'BiotPlasmaGrid',
           'BiotPoint',
           'Field',
           'Flux',
           'Force',
           'Plasma',
           ]

from nova import ImportManager

imp = ImportManager(package='nova.biot')

if imp.defer:
    BiotGap = imp.load('.biotgap', 'BiotGap')
    BiotGrid = imp.load('.biotgrid', 'BiotGrid')
    BiotInductance = imp.load('.biotinductance', 'BiotInductance')
    BiotLoop = imp.load('.biotloop', 'BiotLoop')
    BiotFirstWall = imp.load('.biotfirstwall', 'BiotFirstWall')
    BiotPlasmaGrid = imp.load('.biotplasmagrid', 'BiotPlasmaGrid')
    BiotPoint = imp.load('.biotpoint', 'BiotPoint')
    Field = imp.load('.field', 'Field')
    Flux = imp.load('.flux', 'Flux')
    Force = imp.load('.force', 'Force')
    Plasma = imp.load('.plasma', 'Plasma')
else:
    from nova.biot.biotgap import BiotGap
    from nova.biot.biotgrid import BiotGrid
    from nova.biot.biotinductance import BiotInductance
    from nova.biot.biotloop import BiotLoop
    from nova.biot.biotfirstwall import BiotFirstWall
    from nova.biot.biotplasmagrid import BiotPlasmaGrid
    from nova.biot.biotpoint import BiotPoint
    from nova.biot.field import Field
    from nova.biot.flux import Flux
    from nova.biot.force import Force
    from nova.biot.plasma import Plasma
