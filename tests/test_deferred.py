from contextlib import contextmanager
import itertools
import os
import pytest

from nova.utilities.importmanager import (
    defer_import,
    DeferredImport,
    ImportManager
)


@contextmanager
def deferred_default_unset():
    previous_state = os.getenv('NOVA_DEFERRED_IMPORT', None)
    imp = ImportManager()
    imp.unset
    yield
    if previous_state is not None:
        imp.defer = previous_state


@pytest.mark.parametrize('defer,defer_default',
                         itertools.product([True, False], [True, False]))
def test_with_defer_import(defer, defer_default):
    imp = ImportManager(defer_default)
    previous_state = imp.defer
    with defer_import(defer):
        assert imp.defer == defer
    assert imp.defer == previous_state


@pytest.mark.parametrize('defer', [True, False])
def test_import(defer):
    imp = ImportManager(package='nova.frame')
    with defer_import(defer):
        Coil = imp.load('.coil', 'Coil')
    assert isinstance(Coil, DeferredImport) == defer


@pytest.mark.parametrize('defer_default', [True, False])
def test_imp_clear(defer_default):
    with deferred_default_unset():
        imp = ImportManager(defer_default)
        imp.unset
        with pytest.raises(KeyError):
            _ = os.environ['NOVA_DEFERRED_IMPORT']


@pytest.mark.parametrize('defer_default', [True, False, 'True', 'False'])
def test_imp_defaut_types(defer_default):
    with deferred_default_unset():
        imp = ImportManager(defer_default)
        imp.unset
        assert imp.defer == (str(defer_default) == 'True')


@pytest.mark.parametrize('defer', [True, False])
def test_defered_import_load(defer):
    with defer_import(defer):
        Coil = DeferredImport('nova.frame.coil', 'Coil').load()
        assert not isinstance(Coil, DeferredImport)



if __name__ == '__main__':

    pytest.main([__file__])
