"""Generate coil voltage and current waveforms to suport pulse design."""
from dataclasses import dataclass

from nova.imas.database import Ids
from nova.imas.machine import Machine
from nova.imas.pulse_schedule import PulseSchedule

import rdp

@dataclass
class Waveform(Machine, PulseSchedule):
    """Generated coilset voltage and current waveforms."""

    #name: str = ''
    #name: str = 'pulse_schedule'
    pf_active: Ids | bool | str = 'iter_md'
    pf_passive: Ids | bool | str = 'iter_md'
    wall: Ids | bool | str = 'iter_md'

    def __post_init__(self):
        pass

if __name__ == '__main__':

    pulse, run = 135003, 5

    waveform = Waveform(pulse, run)

    #waveform.plasma.separatrix = dict(e=[6, 0.5, 3, 6])

    #waveform.plot()
    #waveform.firstwall.plot()

    #waveform.loc
