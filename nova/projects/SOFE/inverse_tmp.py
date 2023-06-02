def snap_PF(self, coil=None, solve=False):
    self.fit_PF(coil=coil)  # fit PF to TF
    self.gap = 0.1
    if solve:
        # coil = deepcopy(self.pf.coil)
        # index = deepcopy(self.pf.index)
        # self.delete_active()  # delete all coils
        Lpf = np.zeros(self.pf.index["PF"]["n"])
        for i, name in enumerate(self.pf.index["PF"]["name"]):
            c = coil[name]
            Lpf[i] = minimize_scalar(
                INV.norm,
                method="bounded",
                args=(self.tf.fun["out"], (c["r"], c["z"])),
                bounds=[0, 1],
            ).x
            # self.add_coil(Lout=Lpf[i], Ctype='PF', norm=self.TFoffset,
            #              dr=c['dr'], dz=c['dz'], I=c['I'])
        Lcs = np.zeros(self.pf.index["CS"]["n"] + 1)
        for i, name in enumerate(self.pf.index["CS"]["name"]):
            c = coil[name]
            # self.add_coil(point=(c['r'], c['z']), Ctype='CS',
            #              dr=c['dr'], dz=c['dz'], I=c['I'])
            if i == 0:
                Lcs[0] = c["z"] - c["dz"] / 2 - self.gap / 2
            Lcs[i + 1] = c["z"] + c["dz"] / 2 + self.gap / 2
        # self.update_coils()
        self.ff.set_force_feild()
        self.set_Lo(np.append(Lpf, Lcs))  # set position bounds
        Lnorm = loops.normalize_variables(self.Lo)
        self.update_position(Lnorm, update_area=True)
        self.plot_coils()


def snap_coils(self, solve=True, **kwargs):
    if "TFoffset" in kwargs:  # option to update TFoffset via kwarg
        self.TFoffset = kwargs["TFoffset"]
    kwargs.get("coil", None)
    # L = self.grid_coils(coil=coil)

    L = self.get_L()
    # L = self.grid_coils()
    # self.update_coils()
    if solve:
        self.set_Lo(L)  # set position bounds
        Lnorm = loops.normalize_variables(self.Lo)
        self.update_position(Lnorm, update_area=True)


# fit PF and CS coils to updated TF coil
def grid_coils(self, coil=None, nCS=None, Zbound=None, gap=0.1):
    self.gap = gap
    index = deepcopy(self.pf.index)
    if coil is None:
        coil = deepcopy(self.pf.coil)  # copy pf coilset
    else:
        coil = deepcopy(coil)  # copy referance
    self.delete_active()
    TFloop = self.tf.fun["out"]
    Lpf = np.zeros(index["PF"]["n"])
    for i, name in enumerate(index["PF"]["name"]):
        c = coil[name]
        Lpf[i] = minimize_scalar(
            INV.norm, method="bounded", args=(TFloop, (c["r"], c["z"])), bounds=[0, 1]
        ).x
        self.add_coil(
            Lout=Lpf[i],
            Ctype="PF",
            norm=self.TFoffset,
            dr=c["dr"],
            dz=c["dz"],
            I=c["I"],
        )
    """
    if Zbound == None:
        zmin = [coil[name]['z']-coil[name]['dz']/2
                for name in self.pf.index['CS']['name']]
        zmax = [coil[name]['z']+coil[name]['dz']/2
                for name in self.pf.index['CS']['name']]
        Zbound = [np.min(zmin)-gap/2,np.max(zmax)+gap/2]
        self.update_limits(LCS=Zbound)
    """

    if nCS is None:
        Lcs = np.zeros(index["CS"]["n"] + 1)
        for i, name in enumerate(index["CS"]["name"]):
            c = coil[name]
            print(name, c["r"], c["z"])
            self.add_coil(
                point=(c["r"], c["z"]), Ctype="CS", dr=c["dr"], dz=c["dz"], I=c["I"]
            )
            if i == 0:
                Lcs[0] = c["z"] - c["dz"] / 2 - gap / 2
            Lcs[i + 1] = c["z"] + c["dz"] / 2 + gap / 2

    else:
        nCS = index["CS"]["n"]
        Lcs = self.grid_CS(nCS=nCS, Zbound=Zbound, gap=gap)

    self.update_coils()
    self.set_force_feild()
    self.fit_PF(TFoffset=self.TFoffset)
    return np.append(Lpf, Lcs)
