"""Generate coil voltage and current waveforms to suport pulse design."""
from dataclasses import dataclass

from nova.imas.database import Ids
from nova.imas.machine import Machine


@dataclass
class Waveform(Machine):
    """Generated coilset voltage and current waveforms."""

    name: str = ''
    pf_active: Ids | bool | str = 'iter_md'
    pf_passive: Ids | bool | str = 'iter_md'
    wall: Ids | bool | str = 'iter_md'


if __name__ == '__main__':

    waveform = Waveform()

    waveform.plasma.separatrix = dict(e=[6, 0.5, 3, 6])

    waveform.plot()
    waveform.firstwall.plot()

    #waveform.loc
