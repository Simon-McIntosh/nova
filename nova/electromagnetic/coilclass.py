import pandas as pd

from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.IO.read_scenario import scenario_data
from nova.electromagnetic.IO.read_scenario import forcefield_data


class CoilClass(CoilSet):
    '''
    CoilClass:
        - implements methods to manage input and
            output of data to/from the CoilSet class
        - provides interface to eqdsk files containing coil data
        - provides interface to DINA scenaria data
    '''

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
        '''
        return scenario metadata
        '''
        return pd.Series({'filename': self.scenario_filename,
                          'to': self.d2.to, 'ko': self.d2.ko})

    @scenario.setter
    def scenario(self, to):
        '''
        Attributes:
            to (float): input time
            to (str): feature_keypoint
        '''
        self.to = to  # time or keypoint
        self.d2.to = to  # update scenario data (time or keypoint)
        self.d3.to = self.d2.to # update forcefield data
        self.t = self.d2.to  # time instance
        #self.update_plasma()
        self.Ic = self.d2.Ic.to_dict()


