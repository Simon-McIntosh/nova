from nep.DINA.read_plasma import read_plasma
from nep.DINA.read_tor import read_tor
from amigo.IO import pythonIO


class toroidal_current(pythonIO):
    def __init__(self, read_txt=False, Iscale=1):
        pythonIO.__init__(self)  # python read/write
        self.pl = read_plasma(
            "disruptions", Iscale=Iscale, read_txt=read_txt
        )  # load plasma
        self.tor = read_tor(
            "disruptions", Iscale=Iscale, read_txt=read_txt
        )  # load currents

        self.tor.load_file(3)

    """
        self.ps = power_supply(vessel=True)
        self.load_coils()

    def load_coils(self):
        pf_geom = PFgeom(VS=False, dCoil=0.25)
        self.pf = pf_geom.pf  # initalise pf opject
        self.tor.load_file(0)  # read toroidal strucutres

        #self.add_highres()  # add high-res coils + remove DINA

        self.add_filament(self.tor.vessel_coil, index='vv_DINA')
        self.add_filament(self.tor.blanket_coil, index='bb_DINA')

    def add_filament(self, filament, sub_coil=None, index=None):
        nCo = self.pf.nC  # start index
        name = []
        for coil in filament:
            name.append(coil)
            self.pf.coil[coil] = filament[coil]
            if sub_coil:
                Nf = sub_coil[coil+'_0']['Nf']
                for i in range(Nf):
                    subname = coil+'_{}'.format(i)
                    self.pf.sub_coil[subname] = sub_coil[subname]
            else:
                self.pf.sub_coil[coil+'_0'] = filament[coil]
                self.pf.sub_coil[coil+'_0']['Nf'] = 1
        if index:
            self.pf.index[index] = {'name': name,
                                    'index': np.arange(nCo, self.pf.nC)}

    def load_file(self, scenario):
        self.tor.load_file(scenario)  # read DINA toroidal currents

        self.ps.solve(self.tor.t[-1], Io=0, sign=-1, nturn=4,
                      scenario=scenario, t_pulse=0.3, impulse=True,
                      pulse=False, plot=True)

    def frame_update(self, frame_index):
        self.frame_index = frame_index
        self.t = self.tor.t[self.frame_index]
        self.set_coil_current(frame_index)
        self.set_filament_current(self.tor.vessel_coil, frame_index)
        self.set_filament_current(self.tor.blanket_coil, frame_index)
        self.load_plasma(frame_index)

    def set_coil_current(self, frame_index):  # PF / Cs
        Ic = dict(zip(self.tor.coil.keys(),
                      self.tor.current['coil'][frame_index]))
        self.pf.update_current(Ic)  # PF / CS coils

    def set_filament_current(self, filament, frame_index):  # vessel / blanket
        Ic = {}  # initalize
        current = self.tor.current['filament'][frame_index]
        for name in filament:
            turn_index = filament[name]['index']
            sign = filament[name]['sign']
            Ic[name] = sign * current[turn_index]
        self.pf.update_current(Ic)

    def load_plasma(self, frame_index):
        self.pf.plasma_coil.clear()  # clear
        for name in self.tor.plasma_coil[frame_index]:
            self.pf.plasma_coil[name] = self.tor.plasma_coil[frame_index][name]

    def plot_coils(self, **kwargs):
        ax = kwargs.get('ax', None)
        if ax is None:
            ax = plt.subplots(1, 1, figsize=(6, 9))[1]
        self.pf.plot(subcoil=True, plasma=True, ax=ax)
    """


if __name__ == "__main__":
    tc = toroidal_current(read_txt=False)

    # tc.frame_update(400)

    tc.tor.plot(400)
    # tc.load_file(-1)
    # tc.plot_coils()
