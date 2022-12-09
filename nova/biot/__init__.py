"""Manage deferred import of biot methods."""
__all__ = ['BiotGrid',
           'BiotInductance',
           'BiotLoop',
           'BiotPlasmaBoundary',
           'BiotPlasmaGrid',
           'BiotPoint',
           'Plasma',
           ]

from nova import DeferredImport as Imp
from nova import ImportManager

if ImportManager().state:
    BiotGrid = Imp('.biot.biotgrid', 'BiotGrid')
    BiotInductance = Imp('.biot.biotinductance', 'BiotInductance')
    BiotLoop = Imp('.biot.biotloop', 'BiotLoop')
    BiotPlasmaBoundary = Imp('.biot.biotplasmaboundary', 'BiotPlasmaBoundary')
    BiotPlasmaGrid = Imp('.biot.biotplasmagrid', 'BiotPlasmaGrid')
    BiotPoint = Imp('.biot.biotpoint', 'BiotPoint')
    Plasma = Imp('.biot.plasma', 'Plasma')
else:
    from nova.biot.biotgrid import BiotGrid  # NOQA
    from nova.biot.biotinductance import BiotInductance  # NOQA
    from nova.biot.biotloop import BiotLoop  # NOQA
    from nova.biot.biotplasmaboundary import BiotPlasmaBoundary  # NOQA
    from nova.biot.biotplasmagrid import BiotPlasmaGrid  # NOQA
    from nova.biot.biotpoint import BiotPoint  # NOQA
    from nova.biot.plasma import Plasma  # NOQA
