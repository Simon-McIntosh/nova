import numpy as np
import pandas as pd

import amigo.geom
from nova.electromagnetic.coilset import CoilSet
from nova.electromagnetic.IO.read_scenario import scenario_data


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
        self.t = self.d2.to  # time instance
        #self.update_plasma()
        self.Ic = self.d2.Ic.to_dict()


if __name__ == '__main__':

    
    """
    #cc.add_coil(4, 3, 2, 2, name='PF2', dCoil=1)
    #cc.add_coil(6, -1, 2, 2, name='PF3', dCoil=1)

    plt.plot(*cc.coil.at['PF1', 'polygon'].exterior.xy, 'C3')
    #cc.add_plasma(1, [1.5, 2, 2.5], 0.5, 0.2, It=-15e6/3)
    cc.plot()
    # cc.add_plasma(6, [1.5, 2, 2.5], 0.5, 0.2, It=-15e6/3)

    cc.coil.Ic = 5
    #cc.scenario = 100
    #cc.solve_colocation()
    #cc.solve_interaction(plot=True)
    
    #plt.plot(cc.coil.x, cc.coil.z, 'C1o')
    """

    """
    from nova.electromagnetic.coilgeom import PFgeom
    pf = PFgeom(dCoil=0.35)
    cc = CoilClass(**pf.coilset)
    
    
    #cc.add_coil(4, 3, 2, 2, name='PF12', dCoil=-1)

    cc.plot(label=['CS', 'PF'])
    #cc.plot_grid()
       
    cc.scenario_filename = -2
    cc.scenario = 'SOB'
    
    cc.grid.generate_grid()
    #cc.grid.solve_interaction()
    cc.grid.plot_flux()
    
    cc.scenario = 'EOB'
    cc.grid.plot_flux()
    #cc.solve_interaction(plot=False)
    
    #cc.scenario = 'EOB'
    #cc.solve_interaction(plot=True)
    
    #for t in np.arange(1, 100, 1):
    #    cc.scenario = t
    #    #    #cc.solve_interaction()
    '''  
    cc.add_targets(([1.0, 2], [4, 5.5]))
    cc.update_interaction()

    for t in np.arange(120, 130, 1):
        cc.scenario = t
        cc.solve_interaction()

    print(cc.target['psi'])

    #cc.solve_interaction(plot=True)
    '''

    '''
    cc.generate_grid(n=0)
    cc.add_targets(([1.0, 2], [4, 5]))
    print(cc.target['targets'].index)
    cc.update_interaction()

    cc.add_targets(([1, 2, 3], [4, 5, 3]), append=True)
    print(cc.target['targets'].index)

    cc.update_interaction()

    cc.add_targets((1, 4), append=True, update=True)
    print(cc.target['targets'].index)

    cc.add_targets(([1, 2, 3], [4, 5, 3.1]), append=True)
    print(cc.target['targets'].index)
    '''



    '''
    #cc.plot(label=True)
    #cc.update_inductance()

    cc.scenario_filename = -2
    cc.scenario = 'EOF'
    # cc.update_inductance(source_index=['Plasma'])

    #cc.solve_grid(n=2e3, plot=True, update=True, expand=0.25,
    #              nlevels=31, color='k')
    cc.plot(subcoil=False)
    cc.plot(label=True)
    '''
    """






