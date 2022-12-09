"""Manage deferred import of frame methods."""
__all__ = ['Coil',
           'Ferritic',
           'FirstWall',
           'Shell',
           'Turn',
           'Winding'
           ]

from nova import DeferredImport as Imp
from nova import ImportManager

if ImportManager().state:
    Coil = Imp('.frame.coil', 'Coil')
    Ferritic = Imp('.frame.ferritic', 'Ferritic')
    FirstWall = Imp('.frame.firstwall', 'FirstWall')
    Shell = Imp('.frame.shell', 'Shell')
    Turn = Imp('.frame.turn', 'Turn')
    Winding = Imp('.frame.winding', 'Winding')
else:
    from nova.frame.coil import Coil  # NOQA
    from nova.frame.ferritic import Ferritic  # NOQA
    from nova.frame.firstwall import FirstWall  # NOQA
    from nova.frame.shell import Shell  # NOQA
    from nova.frame.turn import Turn  # NOQA
    from nova.frame.winding import Winding  # NOQA
