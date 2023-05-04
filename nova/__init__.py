"""Nova electromagnetic analysis package."""

from . import _version
__version__ = _version.get_versions()['version']

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
import os
import pytest


def _report(dependencies: tuple[str, ...]):
    """Return module not found error meassage for dependency list."""
    dependency_list = f"{', '.join(dependencies)}"
    return f"Optional dependencies [{dependency_list}] not installed. " \
        f"pip install .['{dependency_list}']"


@contextmanager
def check_import(*dependencies: str):
    """Check module import, raise if not found."""
    try:
        yield
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(_report(dependencies)) from error


@contextmanager
def mark_import(*dependencies: str, skip=False):
    """Return pytest mark, skip if not found."""
    skip = []
    try:
        yield pytest.mark.skipif(skip, reason=_report(dependencies))
    except ModuleNotFoundError:
        skip.append(True)


@contextmanager
def skip_import(*dependencies: str):
    """Check module import, skip test if not found."""
    try:
        yield
    except ModuleNotFoundError:
        pytest.skip(_report(dependencies), allow_module_level=True)


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
