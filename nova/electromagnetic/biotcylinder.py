"""Biot-Savart calculation for rectangular section circular filaments."""
from dataclasses import dataclass, field
from functools import cached_property
from typing import ClassVar

import dask.array as da
import numpy as np
import scipy.special

from nova.electromagnetic.biotframe import BiotFrame
from nova.electromagnetic.biotbase import BiotBase


@dataclass
class Coefficients:
    """Compute Biot Savart intergration coefficients."""

    source: BiotFrame
    target: BiotFrame

    # pylint: disable=W0631


    '''
        coeff = {'rs': offset.source_radius, 'zs': offset.source_height,
                 'r': offset.target_radius, 'z': offset.target_height}
        coeff['b'] = coeff['rs'] + coeff['r']
        coeff['gamma'] = coeff['zs'] - coeff['z']
        coeff['a2'] = coeff['gamma']**2 + (coeff['r'] + coeff['rs'])**2
        coeff['a'] = np.sqrt(coeff['a2'])
        coeff['k2'] = 4 * coeff['r'] * coeff['rs'] / coeff['a2']
        coeff['ck2'] = 1 - coeff['k2']  # complementary modulus
        coeff['K'] = scipy.special.ellipk(coeff['k2'])  # ellip integral - 1st
        coeff['E'] = scipy.special.ellipe(coeff['k2'])  # ellip integral - 2nd
        return coeff
    '''

    @cached_property
    def source_radius(self):
        """Return source radius."""
        #return da.from_array(self.source('x'), chunks=1000)
        return self.source('z')

    @cached_property
    def source_height(self):
        """Return source height."""
        return self.source('z')
        #return self.source('z')

    @cached_property
    def target_radius(self):
        """Return target radius."""
        return self.target('x')

    @cached_property
    def target_height(self):
        """Return source radius."""
        return self.target('z'),
        #return self.target('z')

    def gamma(self):
        """Return gamma coefficient."""
        return self.source_height - self.target_height

    @property
    def a(self):
        """Return gamma coefficent."""
        return 1


if __name__ == '__main__':

    from nova.electromagnetic.biotframe import BiotFrame
    from nova.electromagnetic.framespace import FrameSpace

    radius, height = np.meshgrid(np.linspace(4, 7, 100),
                                 np.linspace(-1, 1, 10))
    frame = FrameSpace(dict(x=radius.flatten(),
                            z=height.flatten(), segment='ring'))
    source = BiotFrame(frame)
    target = BiotFrame(frame)
    source.set_target(len(target))
    target.set_source(len(source))

    coef = Coefficients(source, target)

    print(coef.gamma().shape)
