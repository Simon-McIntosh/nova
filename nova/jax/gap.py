"""Develop boundary transition elements."""

import logging
from timer import timer

import jax

from nova.imas.operate import Operate
from nova.imas.pulseschedule import PulseSchedule

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, force=True)

timer.set_level(logging.INFO)
jax.config.update("jax_enable_x64", True)


kwargs = {
    "pulse": 135013,
    "run": 2,
    "machine": "iter",
    "pf_passive": True,
    "pf_active": True,
}

operate = Operate(
    **kwargs,
    tplasma="h",
    nwall=-0.2,
    ngrid=5e3,
    limit=[3.0, 9.0, -6.0, 6.0],
    nlevelset=None,
    ngap=21,
)

schedule = PulseSchedule(**kwargs, dd_version="3.38.0")

schedule.time = 500
schedule.plot_gaps()
