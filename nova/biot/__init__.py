"""Manage deferred import of biot methods."""
__all__ = ['BiotGrid',
           'BiotInductance',
           'BiotLoop',
           'BiotPlasmaBoundary',
           'BiotPlasmaGrid',
           'BiotPoint',
           'Plasma',
           ]

from nova import ImportManager

imp = ImportManager(package='nova.biot')

if imp.defer:
    BiotGrid = imp.load('.biotgrid', 'BiotGrid')
    BiotInductance = imp.load('.biotinductance', 'BiotInductance')
    BiotLoop = imp.load('.biotloop', 'BiotLoop')
    BiotPlasmaBoundary = imp.load('.biotplasmaboundary', 'BiotPlasmaBoundary')
    BiotPlasmaGrid = imp.load('.biotplasmagrid', 'BiotPlasmaGrid')
    BiotPoint = imp.load('.biotpoint', 'BiotPoint')
    Plasma = imp.load('.plasma', 'Plasma')
else:
    from nova.biot.biotgrid import BiotGrid  # NOQA
    from nova.biot.biotinductance import BiotInductance  # NOQA
    from nova.biot.biotloop import BiotLoop  # NOQA
    from nova.biot.biotplasmaboundary import BiotPlasmaBoundary  # NOQA
    from nova.biot.biotplasmagrid import BiotPlasmaGrid  # NOQA
    from nova.biot.biotpoint import BiotPoint  # NOQA
    from nova.biot.plasma import Plasma  # NOQA
