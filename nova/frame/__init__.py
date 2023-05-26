"""Manage deferred import of frame methods."""
__all__ = ['Coil',
           'Ferritic',
           'FirstWall',
           'Shell',
           'Turn',
           'Winding'
           ]

from nova.utilities.importmanager import ImportManager

imp = ImportManager(package='nova.frame')

if imp.defer:
    Coil = imp.load('.coil', 'Coil')
    Ferritic = imp.load('.ferritic', 'Ferritic')
    FirstWall = imp.load('.firstwall', 'FirstWall')
    Shell = imp.load('.shell', 'Shell')
    Turn = imp.load('.turn', 'Turn')
    Winding = imp.load('.winding', 'Winding')
else:
    from nova.frame.coil import Coil
    from nova.frame.ferritic import Ferritic
    from nova.frame.firstwall import FirstWall
    from nova.frame.shell import Shell
    from nova.frame.turn import Turn
    from nova.frame.winding import Winding
