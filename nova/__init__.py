"""Nova electromagnetic analysis package."""

from . import _version
__version__ = _version.get_versions()['version']

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
import os


@contextmanager
def defer_import(defer=True):
    """Manage deferred imports for optional nova frameset methods."""
    previous = os.getenv('NOVA_DEFERRED_IMPORT', None)
    os.environ['NOVA_DEFERRED_IMPORT'] = 'True' if defer else 'False'
    yield
    if previous is not None:
        os.environ['NOVA_DEFERRED_IMPORT'] = previous
    else:
        del os.environ['NOVA_DEFERRED_IMPORT']


@dataclass
class DeferredImport:
    """Manage deferred imports for frame methods."""

    module: str
    method: str
    package: str = 'nova'

    def load(self):
        """Load method."""
        return getattr(import_module(self.module, self.package), self.method)


@dataclass
class ImportManager:
    """Manage deferred import flags."""

    defer_default: str | bool = True
    package: str = 'nova'

    @property
    def defer(self) -> bool:
        """Return NOVA_DEFERRED_IMPORT flag."""
        return os.getenv('NOVA_DEFERRED_IMPORT',
                         str(self.defer_default)) == 'True'

    @defer.setter
    def defer(self, defer: str | bool):
        os.environ['NOVA_DEFERRED_IMPORT'] = str(defer)

    @property
    def unset(self):
        """Clear NOVA_DEFERRED_IMPORT flag."""
        try:
            del os.environ['NOVA_DEFERRED_IMPORT']
        except KeyError:
            pass

    def load(self, module: str, method: str):
        """Return method."""
        if self.defer:
            return DeferredImport(module, method, self.package)
        return getattr(import_module(module, self.package), method)
