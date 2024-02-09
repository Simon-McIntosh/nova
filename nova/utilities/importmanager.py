"""Manage defered imports."""

from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
import os

THIRDPARTY = ["imas", "codac_uda"]


def _report(dependencies: tuple[str, ...]):
    """Return module not found error meassage for dependency list."""
    dependency_list = f"{', '.join(dependencies)}"
    third_party = ""
    for dependancy in dependencies:
        if dependancy.lower() in THIRDPARTY:
            third_party += f"The external {dependancy.upper()} library is required. "

    return (
        f"Optional dependencies [{dependency_list}] not installed. "
        f"Try pip install .[{dependency_list}]. "
        f"{third_party}"
    )


@contextmanager
def check_import(*dependencies: str):
    """Check module import, raise if not found."""
    try:
        yield
    except (ImportError, ModuleNotFoundError) as error:
        raise ModuleNotFoundError(_report(dependencies)) from error


@contextmanager
def mark_import(*dependencies: str, skip=None):
    """Return pytest mark, skip if not found."""
    if skip is None:
        skip = []
    try:
        import pytest

        yield pytest.mark.skipif(skip, reason=_report(dependencies))
    except (ModuleNotFoundError, ImportError):
        skip.append(True)


@contextmanager
def skip_import(*dependencies: str):
    """Check module import, skip test if not found."""
    try:
        yield
    except ModuleNotFoundError:
        import pytest

        pytest.skip(_report(dependencies), allow_module_level=True)


@contextmanager
def defer_import(defer=True):
    """Manage deferred imports for optional nova frameset methods."""
    previous = os.getenv("NOVA_DEFERRED_IMPORT", None)
    os.environ["NOVA_DEFERRED_IMPORT"] = "True" if defer else "False"
    yield
    if previous is not None:
        os.environ["NOVA_DEFERRED_IMPORT"] = previous
    else:
        del os.environ["NOVA_DEFERRED_IMPORT"]


@dataclass
class DeferredImport:
    """Manage deferred imports for frame methods."""

    module: str
    method: str
    package: str = "nova"

    def load(self):
        """Load method."""
        return getattr(import_module(self.module, self.package), self.method)


@dataclass
class ImportManager:
    """Manage deferred import flags."""

    defer_default: str | bool = True
    package: str = "nova"

    @property
    def defer(self) -> bool:
        """Return NOVA_DEFERRED_IMPORT flag."""
        return os.getenv("NOVA_DEFERRED_IMPORT", str(self.defer_default)) == "True"

    @defer.setter
    def defer(self, defer: str | bool):
        os.environ["NOVA_DEFERRED_IMPORT"] = str(defer)

    @property
    def unset(self):
        """Clear NOVA_DEFERRED_IMPORT flag."""
        try:
            del os.environ["NOVA_DEFERRED_IMPORT"]
        except KeyError:
            pass

    def load(self, module: str, method: str):
        """Return method."""
        if self.defer:
            return DeferredImport(module, method, self.package)
        return getattr(import_module(module, self.package), method)
