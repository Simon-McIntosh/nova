"""Nova electromagnetic analysis package."""

from . import _version
__version__ = _version.get_versions()['version']

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
import os


@dataclass
class ImportManager:
    """Manage deferred import flags."""

    default: str | bool = True

    @property
    def state(self) -> bool:
        """Return NOVA_DEFERRED_IMPORT flag."""
        return os.getenv('NOVA_DEFERRED_IMPORT', str(self.default)) == 'True'

    @state.setter
    def state(self, state: str | bool):
        os.environ['NOVA_DEFERRED_IMPORT'] = str(state)

    @property
    def clear(self):
        """Clear NOVA_DEFERRED_IMPORT flag."""
        del os.environ['NOVA_DEFERRED_IMPORT']

    @staticmethod
    @contextmanager
    def deferred(state=True):
        """Manage deferred imports for optional nova frameset methods."""
        previous_state = os.getenv('NOVA_DEFERRED_IMPORT', None)
        os.environ['NOVA_DEFERRED_IMPORT'] = 'True' if state else 'False'
        yield
        if previous_state is not None:
            os.environ['NOVA_DEFERRED_IMPORT'] = previous_state
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
