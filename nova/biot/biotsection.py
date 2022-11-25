"""Section methods for BiotFrame."""
from dataclasses import dataclass, field

import numpy as np

from nova.frame.metamethod import MetaMethod
from nova.frame.framelink import FrameLink


@dataclass
class BiotSection(MetaMethod):
    """
    Section methods for BiotFrame.

    Set cross-section factors used in Biot_Savart calculations.
    """

    name = 'biotsection'

    frame: FrameLink = field(repr=False)
    required: list[str] = field(default_factory=lambda: ['section'])
    additional: list[str] = field(default_factory=lambda: ['turnturn'])
    section_factor: dict[str, float] = field(
        init=False, default_factory=lambda: {
            'disc': np.exp(-0.25),  # disc-disc
            'square': 2*0.447049,  # square-square
            'skin': 1})  # skin-skin
    section_key: dict[str, str] = field(
        init=False, default_factory=lambda: {
            'rectangle': 'square',
            'ellipse': 'disc',
            'polygon': 'square',
            'shell': 'square',
            'hexagon': 'disc'})

    def initialize(self):
        """Calculate section self inductance factors."""
        biot_section = [section if section in self.section_factor
                        else self.section_key.get(section, None)
                        for section in self.frame.section]
        undefined = [section is None for section in biot_section]
        if any(undefined):
            raise KeyError('Biotsection '
                           f'{self.frame.section[undefined].unique()} '
                           'undefined. List in section_factor or section_key.')
        self.frame['turnturn'] = [self.section_factor[section]
                                  for section in biot_section]
