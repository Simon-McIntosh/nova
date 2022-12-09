"""Manage deferred import of circuit methods."""
__all__ = ['Circuit']

from nova import DeferredImport as Imp
from nova import ImportManager

if ImportManager().state:
    Circuit = Imp('.control.circuit', 'Circuit')
else:
    from nova.control.circuit import Circuit  # NOQA
