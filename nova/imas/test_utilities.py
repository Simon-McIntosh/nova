"""Manage test utility funcitons."""

import pytest

from nova.imas.database import Database
from nova.utilities.importmanager import mark_import

with mark_import("imaspy") as mark_imaspy:
    from imaspy.exception import ALException  # noqa

IMPORT_IMASPY = not any(mark_imaspy.args[0])


ids_attrs = dict(
    pf_active=dict(pulse=111001, run=203, name="pf_active", machine="iter_md"),
    pf_active_iter=dict(pulse=105011, run=9, name="pf_active"),
    equilibrium=dict(pulse=130506, run=403, name="equilibrium"),
    equilibrium_pds=dict(pulse=135013, run=2, name="equilibrium"),
    wall=dict(pulse=116000, run=2, name="wall", machine="iter_md"),
    pf_passive=dict(pulse=115005, run=2, name="pf_passive", machine="iter_md"),
    coils_non_axisymmetric=dict(
        pulse=111003, run=1, name="coils_non_axisymmetric", machine="iter_md"
    ),
)


def load_ids(*args, **kwargs):
    """Return database instance."""
    if not IMPORT_IMASPY:
        return False
    return Database(*args, **kwargs)


mark = {"imas": mark_imaspy}
for attr in ids_attrs:
    if not IMPORT_IMASPY:
        mark[attr] = mark["imas"]
        continue
    mark[attr] = pytest.mark.skipif(
        not load_ids(**ids_attrs[attr]).db_entry.is_valid,
        reason=f"{attr} database unavalible",
    )
