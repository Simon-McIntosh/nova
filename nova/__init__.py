
import lazy_loader as lazy

from . import _version
__version__ = _version.get_versions()['version']


__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=['imas'],
    submod_attrs={
        'imas': ['database']
    }
)
