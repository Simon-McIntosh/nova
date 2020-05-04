# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 10:40:20 2018

@author: mcintos
"""

        '''
    def build_coilset(self, include_vessel=True):
        vs_geom = VSgeom()
        self.ind = inductance()
        nvs_o = self.ind.nC
        turns = np.append(np.ones(4), -np.ones(4))
        self.ind.add_pf_coil(vs_geom.pf.sub_coil, turns=turns)
        for i, index in enumerate(nvs_o+np.arange(1, 8)):  # vs3 loops
            self.ind.add_cp([nvs_o, index])  # link VS coils
        dx = dz = 60e-3
        if include_vessel:
            for vv in ['lowerVV', 'upperVV']:
                x, z = self.flux[vv]['x'], self.flux[vv]['z']
                self.ind.add_coil(x, z, dx, dz, 0, R=0, nt=1)
        self.ind.reduce()

    def calculate_background(self):
        print(len(vfun))

        self.build_coilset(include_vessel=True)
        self.ind.Ic[0] = 60e3  # inital current
        self.ind.Rc[0] = 17.66e-3  # total vs3 resistance
        self.ind.M[0, 0] += 0.2e-3  # add busbar inductance
        self.ind.M[0, 0] += 0.3e-4  # adjust self inductance to aggree with LTC

        self.ind.Rc[1] = 2e-3  # lowerVV resistance
        self.ind.Rc[2] = 2e-4  # upperVV resistance
        vs3.get_Rvv()


        self.ind.Ic[0] = -60e3  # inital current
        plt.figure()
        print(self.ind.Rc)
        t = np.linspace(self.t_trip, self.t[-1], 500)
        t = np.linspace(0, 10*self.t[-1], 500)
        Iode = self.ind.solve(t)  # , vfun=vfun
        for i, name in enumerate(['vs3', 'lowerVV', 'upperVV']):
            plt.plot(t, Iode[i], label=name)
        plt.despine()
        plt.legend()

        print(1e-3*np.min((Iode[0])))
        '''
        '''
        self.build_coilset(include_vessel=False)
        self.ind.Ic[0] = -60e3  # inital current
        self.ind.Rc[0] = 17.66e-3  # total vs3 resistance
        self.ind.M[0, 0] += 0.2e-3  # add busbar inductance
        self.ind.M[0, 0] += 0.3e-4  # adjust self inductance to aggree with LTC
        Iode = self.ind.solve(t, vfun=vfun[:1])
        for i, name in enumerate(['vs3 bare']):
            plt.plot(t, Iode[i], label=name)
        plt.despine()
        plt.legend()

        print(1e-3*np.min((Iode[0])))
        '''

        '''
        t = np.linspace(0, self.t[-1], 200)

        Iode = self.ind.solve(t)
        tau = self.ind.M[0, 0] / self.ind.Rc[0]

        self.plot_LTC()

        plt.plot(1e3*t, 1e-3*self.ind.Ic[0]*np.exp(-t/tau), 'C0--')
        plt.plot(1e3*t, 1e-3*Iode[0], 'C1--')
        '''
        '''
    def fit_waveform(self, Rvv):
        Iref = self.LTC['LTC+vessel']['Ic']
        self.update_Rvv(Rvv)
        Iode = self.ind.solve(self.LTC['LTC+vessel']['t'])
        err = np.linalg.norm(Iode[0] - Iref) / np.linalg.norm(Iref)
        return err

    def update_Rvv(self, Rvv):
        self.ind.Rc[1] = np.max([Rvv[0], 1e-4])  # lowerVV resistance
        self.ind.Rc[2] = np.max([Rvv[1], 1e-4])   # upperVV resistance

    def get_Rvv(self):
        xo = [2e-4, 2e-4]  # sead resistance vector
        Rvv = minimize(vs3.fit_waveform, xo, method='Nelder-Mead').x
        self.update_Rvv(Rvv)
        t = self.LTC['LTC+vessel']['t']
        Iode = self.ind.solve(t)

        self.plot_LTC()
        #plt.plot(1e3*t, 1e-3*self.ind.Ic[0]*np.exp(-t/tau), 'C0--')
        plt.plot(1e3*t, 1e-3*Iode[0], 'C1--')
        '''
