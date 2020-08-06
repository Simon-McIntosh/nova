import numpy as np
from pandas import DataFrame


class PoloidalLimit:
    'operating limits for poloidal field coils (PF and CS)'
    
    _limit_key = {'I': 'current', 'F': 'force'}
    _limit_unit = {'current': 'kA', 'force': 'MN', 'field': 'T'}
    _limit_bound = 1e16
    
    def __init__(self):
        self.initalise_limits()
        
    def initalise_limits(self):
        self.limit = DataFrame(columns=['name', 'coil', 'lower', 'upper',
                                        'unit'])
        self.limit.set_index(['name', 'coil'], inplace=True)
        
    def _initalise_limit(self, name, coil, unit):
        'add limit to dataframe with bounding values'
        if (name, coil) not in self.limit.index:
            self.limit.loc[(name, coil), ['lower', 'upper']] = \
                [-self._limit_bound, self._limit_bound]
        self.set_unit(name, coil, unit)

    def set_unit(self, name, coil, unit):
        if unit is None:
            unit = self._limit_unit[name]
        self.limit.loc[(name, coil), 'unit'] = unit
        
    def add_limit(self, bound='both', eps=1e-2, unit=None, **limits):
        '''
        Attributes:
            limits (dict): listing of limits name: value
                           set limit name as ICSsum for [I][CSsum] etc...
            bound (str): set bounds [lower, upper, both, equal] 
        '''
        if bound == 'both' or bound == 'equal':
            bounds = ['lower', 'upper']
        else:
            bounds = bound
        for limit in limits:
            name = self._limit_key[limit[0]]
            coil = limit[1:]
            value = limits[limit]                
            self._initalise_limit(name, coil, unit)
            if bound == 'equal':
                value = value + eps*np.array([-1, 1])
            elif bound == 'both':
                value = value * np.array([-1, 1])
            self.limit.loc[(name, coil), bounds] = value

    def load_ITER_limits(self):
        'add default limits for ITER coil-set'
        self.add_limit(ICS=45)  # kA current limits
        self.add_limit(IPF1=48, IPF2=55, IPF3=55, IPF4=55, IPF5=52,
                       IPF6=52)
        self.add_limit(FCSsep=240, bound='upper')  # force limits
        self.add_limit(FCSsum=60, bound='both')
        self.add_limit(FPF1=-150, FPF2=-75, FPF3=-90, FPF4=-40,
                       FPF5=-10, FPF6=-190, bound='lower')
        self.add_limit(FPF1=110, FPF2=15, FPF3=40, FPF4=90,
                       FPF5=160, FPF6=170, bound='upper')
        
    def get_limit(self, index, name, bound, unit=None):
        _index = np.copy(index)
        limit_xs = self.limit.xs(name)
        limit_index = limit_xs.index.to_list()
        for i, coil in enumerate(index):
            if coil in limit_index:  # full label
                _index[i] = coil
            elif coil[:2] in limit_xs.index:  # prefix
                _index[i] = coil[:2]
            else:  # default
                _index[i] = coil
                self._initalise_limit(name, coil, unit)
        return limit_xs.loc[_index, bound]


    def get_PFz_limit(self):
        PFz_limit = np.zeros((self.nPF, 2))
        for i, coil in enumerate(self.PFcoils):
            if coil in self.limit['F']:  # per-coil
                PFz_limit[i] = self.limit['F'][coil]
            elif coil[:2] in self.limit['F']:  # per-set
                PFz_limit[i] = self.limit['F'][coil[:2]]
            else:  # no limit
                PFz_limit[i] = [-self._bound, self._bound]
        return PFz_limit

    def get_CSsep_limit(self):
        CSsep_limit = np.zeros((self.nCS - 1, 2))
        for i in range(self.nCS - 1):  # gaps, bottom-top
            gap = 'CS{}sep'.format(i)
            if gap in self.limit['F']:  # per-gap
                CSsep_limit[i] = self.limit['F'][gap]
            elif 'CSsep' in self.limit['F']:  # per-set
                CSsep_limit[i] = self.limit['F']['CSsep']
            else:  # no limit
                CSsep_limit[i] = [-self._bound, self._bound]
        return CSsep_limit

    def get_CSsum_limit(self):
        CSsum_limit = np.zeros((1, 2))
        if 'CSsum' in self.limit['F']:  # per-set
            CSsum_limit = self.limit['F']['CSsum']
        else:  # no limit
            CSsum_limit = [-self._bound, self._bound]
        return CSsum_limit
    
    def get_CSaxial_limit(self):
        CSaxial_limit = np.zeros((self.nCS + 1, 2))
        for i in range(self.nCS + 1):  # gaps, top-bottom
            gap = 'CS{}axial'.format(i)
            if gap in self.limit['F']:  # per-gap
                CSaxial_limit[i] = self.limit['F'][gap]
            elif 'CSaxial' in self.limit['F']:  # per-set
                CSaxial_limit[i] = self.limit['F']['CSaxial']
            else:  # no limit
                CSaxial_limit[i] = [-self._bound, self._bound]
        return CSaxial_limit
    
if __name__ == '__main__':
    
    pl = PoloidalLimit()
    pl.add_limit(ICS=40, bound='lower')
    
    pl.load_ITER_limits()
    print(pl.limit)

