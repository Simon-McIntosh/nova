import numpy as np
import pandas

from nova.electromagnetic.frameset import FrameSet
from nova.electromagnetic.coilmatrix import CoilMatrix
from nova.utilities.pyplot import plt


class BiotAttributes:
    """Manage attributes to and from Biot derived classes."""

    _biot_attributes = []
    _default_biot_attributes = {}

    def __init__(self, **biot_attributes):
        self._append_biot_attributes(self._biot_attributes)
        self._append_biot_attributes(self._coilmatrix_attributes)
        self._append_biot_attributes(self._default_coilmatrix_attributes)
        self._default_biot_attributes = {
            **self._default_coilmatrix_attributes,
            **self._default_biot_attributes}
        self.biot_attributes = biot_attributes

    def _append_biot_attributes(self, attributes):
        self._biot_attributes += [attr for attr in attributes
                                  if attr not in self._biot_attributes]

    @property
    def biot_attributes(self):
        return {attribute: getattr(self, attribute) for attribute in
                self._biot_attributes}

    @biot_attributes.setter
    def biot_attributes(self, _biot_attributes):
        for attribute in self._biot_attributes:
            default = self._default_biot_attributes.get(attribute, None)
            value = _biot_attributes.get(attribute, None)
            if value is not None:
                if type(value) == BiotFrame:
                    BiotFrame.__init__(getattr(self, attribute), value)
                    self.target.rebuild_coildata()
                else:
                    setattr(self, attribute, value)  # set value
            elif not hasattr(self, attribute):
                setattr(self, attribute, default)  # set default



if __name__ == '__main__':

    from nova.electromagnetic.coilset import CoilSet
    cs = CoilSet(dCoil=0.2, dPlasma=0.05, turn_fraction=0.5)
    cs.add_coil(3.943, 7.564, 0.959, 0.984, nturn=248.64, name='PF1', part='PF')
    cs.add_coil(1.6870, 5.4640, 0.7400, 2.093, nturn=554, name='CS3U', part='CS')
    #cs.add_coil(1.6870, 3.2780, 0.7400, 2.093, nturn=554, name='CS2U', part='CS')
    #cs.add_plasma(3.5, 4.5, 1.5, 2.5, It=-15e6, cross_section='ellipse')

    #cs.add_plasma(3.5, 4.5, 1.5, 2.5, dPlasma=0.5,
    #              It=-15e6, cross_section='circle')

    cs.plot(True)


    source = BiotFrame(cs.subcoil)
