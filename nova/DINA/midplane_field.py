from nep.DINA.capacitor_discharge import power_supply
from nova.cross_coil import Bpoint
import numpy as np
from amigo.pyplot import plt
from amigo.geom import grid
import nova.cross_coil as cc
from nova.streamfunction import SF
from nep.DINA.read_eqdsk import read_eqdsk
from collections import OrderedDict


class midplane:
    
    def __init__(self, vessel=False, invessel=True):
        self.ps = power_supply(nturn=4, vessel=vessel, scenario=-1, 
                               code='Nova', Ip_scale=15/15, read_txt=False, 
                               vessel_model='full', Io=0, sign=-1, 
                               t_pulse=1.8, origin='start', impulse=True,
                               invessel=invessel)
        self.grid_sf()
        self.load_first_wall()
        
    def grid_sf(self, n=5e3, limit=[2.5, 10.5, -7, 7.5]):
        self.x2d, self.z2d, self.x, self.z = grid(n, limit)[:4]
        
    def load_first_wall(self):
        eqdsk = read_eqdsk(file='burn').eqdsk
        self.xlim, self.zlim = eqdsk['xlim'], eqdsk['zlim']

    def set_vs3_current(self, Ivs3):
        Ivs3 = float(Ivs3)  # store Ivs3 current
        Ic = {'upperVS': -4*Ivs3, 'lowerVS': 4*Ivs3}
        self.ps.vv.pf.update_current(Ic)
            
    def vs3_update(self, t):  # update vs3 coil and structure
        self.Ivs3 = self.Ivs3_fun(t)  # current vector
        self.set_vs3_current(self.Ivs3[0])  # vs3 coil current
        coil_list = list(self.ps.vv.pf.coilset['coil'].keys())
        Ic = {}  # coil jacket
        for i, coil in enumerate(coil_list[2:6]):
            Ic[coil] = self.Ivs3[1]  # lower VS jacket
        for i, coil in enumerate(coil_list[6:10]):
            Ic[coil] = self.Ivs3[2]  # upper VS jacket
        self.ps.vv.pf.update_current(Ic)  # dissable to remove jacket field
        if self.ps.vessel:  # vv and trs currents    
            Ic = {}  # vv and trs
            for i, coil in enumerate(coil_list[10:]):
                Ic[coil] = self.Ivs3[i+3]
            self.ps.vv.pf.update_current(Ic)  # dissable to remove vv field
            
    def solve_field(self, point=(6.1929, 0.5014), n=300,  plot=False,
                    **kwargs):
        self.Ivs3_fun = self.ps.solve(**kwargs)  # solve power supply
        data = np.zeros(n, dtype=[('time', float), ('B', '2float'),
                                  ('Ivs3', float)])
        data['time'] = np.linspace(self.ps.data['t'][0],
                                   self.ps.data['t'][-1], n)
        for i, t in enumerate(data['time']):
            self.vs3_update(t)  # update VS3 and vessel currents
            data[i]['B'] = Bpoint(point, self.ps.vv.pf.coilset)
            data[i]['Ivs3'] = self.Ivs3[0]
        if plot:
            self.plot_contour()
        return data
        
    @staticmethod
    def plot_field(data, tmax=None, ax=None, label=None):
        if ax is None:
            ax = plt.subplots(2, 1, sharex=True)[1]
        Bx_max = np.max(abs(data['B'][:, 0]))
        if tmax:
            index = np.argmin(abs(data['time'] - tmax))
        else:
            index = len(data['B'])
        t10 = data['time'][np.argmin(abs(abs(data['B'][:index, 0])-0.1*Bx_max))]
        t90 = data['time'][np.argmin(abs(abs(data['B'][:index, 0])-0.9*Bx_max))]
        risetime = t90-t10
        if label:
            label += r' $\tau$'
            label += f'={1e3*risetime:1.0f}ms'
            
        ax[0].plot(1e3*data['time'], 1e-3*data['Ivs3'][:])
        ax[1].plot(1e3*data['time'], 1e3*data['B'][:, 0], label=label)
        ax[0].set_ylabel('$I_vs3$ kA')
        ax[1].set_ylabel('$B_x$ mT')
        plt.legend()
        plt.despine()
        plt.detick(ax)
        
    def update_sf(self, t):
        self.vs3_update(t)  # update VS3 and vessel currents
        psi = cc.get_coil_psi(self.x2d, self.z2d,
                              self.ps.vv.pf.coilset['subcoil'],
                              self.ps.vv.pf.coilset['plasma'])
        eqdsk = {'x': self.x, 'z': self.z, 'psi': psi, 'fw_limit': False, 
                 'xlim': self.xlim, 'zlim': self.zlim}
        self.sf = SF(eqdsk=eqdsk)
        
    def plot_contour(self, **kwargs):
        self.update_sf(1.8)
        levels = self.sf.get_levels(Nlevel=81, Nstd=4)
        ax = plt.subplots(2, 3, figsize=(10, 9))[1]
        ax = ax.flatten()
        for ax_, t in zip(ax, [10, 50, 100, 250, 500, 1800]):
            self.update_sf(1e-3*t)
            self.sf.contour(ax=ax_, levels=levels, boundary=False,
                            Xnorm=False)
            self.sf.plot_firstwall(ax=ax_)
            self.ps.vv.plot(ax=ax_, plot_centerlines=True, plot_coils=False)
            ax_.set_title(f't {t:1.0f}ms')

    def plot_timeseries(self, data):
        ax = plt.subplots(2, 1, sharex=True)[1]
        for simulation in data:
            self.plot_field(data[simulation], tmax=1.8,
                            label=simulation.replace('_', ' '), ax=ax)
        ax[-1].set_xlabel('$t$ ms')
        

if __name__ == '__main__':

    mp = midplane(vessel=False, invessel=False)
    data = OrderedDict()
    
    '''
    mp.set_vs3_current(-60e3*1.39)
    B = Bpoint((6.1929, 0.5014), mp.ps.vv.pf.coilset)
    print(f'Bx {1e3*B[0]:1.1f}mT')
    '''
    
    data['ex-vessel'] = mp.solve_field(vessel=True, invessel=False, 
                                       Vo_factor=2.5, Ipulse=83.4e3, plot=True)
    data['in-vessel'] = mp.solve_field(vessel=True, invessel=True,
                                       Vo_factor=1.47, Ipulse=60e3, plot=True)
    data['no-vessel'] = mp.solve_field(vessel=False, invessel=True,
                                       Vo_factor=1, Ipulse=60e3, plot=True)
    mp.plot_timeseries(data)
    




    





