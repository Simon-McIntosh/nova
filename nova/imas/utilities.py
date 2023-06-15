"""Manage test utility funcitons."""
import pytest

from nova.imas.database import Database

try:
    from imas.hli_exception import ALException

    IMPORT_IMAS = True
except ImportError:
    IMPORT_IMAS = False
    ALException = None

ids_attrs = dict(
    pf_active=dict(pulse=111001, run=202, name="pf_active", machine="iter_md"),
    pf_active_iter=dict(pulse=105011, run=9, name="pf_active"),
    equilibrium=dict(pulse=130506, run=403, name="equilibrium"),
    equilibrium_pds=dict(pulse=135013, run=2, name="equilibrium"),
    wall=dict(pulse=116000, run=2, name="wall", machine="iter_md"),
    pf_passive=dict(pulse=115005, run=2, name="pf_passive", machine="iter_md"),
)


def load_ids(*args, **kwargs):
    """Return database instance."""
    if not IMPORT_IMAS:
        return False
    try:
        database = Database(*args, **kwargs)
        database.get_ids()
        return database
    except (ModuleNotFoundError, ALException):
        return False


mark = {"imas": pytest.mark.skipif(not IMPORT_IMAS, reason="imas module not loaded")}
for attr in ids_attrs:
    mark[attr] = pytest.mark.skipif(
        not load_ids(**ids_attrs[attr]), reason=f"{attr} database unavalible"
    )
