"""Manage deferred import of circuit methods."""
__all__ = ['Circuit']

from nova.utilities.importmanager import ImportManager

imp = ImportManager(package='nova.control')

if imp.defer:
    Circuit = imp.load('.circuit', 'Circuit')
else:
    from nova.control.circuit import Circuit  # NOQA
