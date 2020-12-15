import pandas as pd
import shapely.geometry
import numpy as np

from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.IO.read_scenario import scenario_data
from nova.electromagnetic.IO.read_scenario import forcefield_data


class CoilClass(CoilSet):
    """
    Extends CoilSet.

    - Implements methods to manage input and output to the CoilSet class.

    - Provides interface to eqdsk files containing coil data.

    - Provides interface to DINA scenaria data.

    """

    def __init__(self, *args, eqdsk=None, filename=None, **kwargs):
        CoilSet.__init__(self, *args, **kwargs)  # inherent from CoilSet
        self.add_eqdsk(eqdsk)
        self.initalize_functions()
        self.initalize_metadata()
        self.filename = filename

    def add_eqdsk(self, eqdsk):
        if eqdsk:
            coil = self.coil.get_coil(
                    eqdsk['xc'], eqdsk['zc'], eqdsk['dxc'], eqdsk['dzc'],
                    It=eqdsk['It'], name='eqdsk', delim='')
            coil = self.categorize_coilset(coil)
            self.coil.concatenate(coil)
            self.add_subcoil(index=coil.index)

    def initalize_functions(self):
        self.t = None  # scenario time instance (d2.to)
        self.d2 = scenario_data()
        self.d3 = forcefield_data()

    def initalize_metadata(self):
        self._scenario_filename = ''

    @property
    def filename(self):
        return self.scenario_filename

    @filename.setter
    def filename(self, filename):
        self.scenario_filename = filename

    @property
    def scenario_filename(self):
        return self._scenario_filename

    @scenario_filename.setter
    def scenario_filename(self, filename):
        '''
        Attributes:
            filename (str) DINA filename
            filename (int) DINA fileindex
        '''
        if filename != self._scenario_filename and filename is not None:
            self.d2.load_file(filename)
            self.d3.load_file(filename)
            self._scenario_filename = self.d2.filename

    @property
    def scenario(self):
        """
        Manage scenario metadata

        Parameters
        ----------
        to : float or str
            Update scenario time input as time or feature_keypoint.

        Returns
        -------
        metadata : pandas.Series
            Metadata of current time instance.

        """
        return pd.Series({'filename': self.scenario_filename,
                          'to': self.d2.to, 'ko': self.d2.ko})

    @scenario.setter
    def scenario(self, to):
        self.to = to  # time or keypoint
        self.d2.to = to  # update scenario data (time or keypoint)
        #self.d3.to = self.d2.to  # update forcefield data
        self.t = self.d2.to  # time instance
        #if self._update_plasma:
        #    self.update_plasma_position()
        self.Ic = self.d2.Ic.to_dict()
        self.Ip = self.d2.Ip

    def update_plasma_position(self, r=1.5):
        rms, z = self.d2.vector['Rcur'], self.d2.vector['Zcur']
        if rms > 0:
            x = np.sqrt(rms**2 - (2*r)**2 / 16)  # rms to x, circle
            polygon = shapely.geometry.Point(x, z).buffer(r)
            self.separatrix = polygon
        else:
            self.Ip = 0
            self.Np = 0
