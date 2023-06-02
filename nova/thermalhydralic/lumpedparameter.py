def heat_curve(self):
    t = self.lowpassdata.t[self.heat_index].values
    Qdot = self.lowpassdata["Qdot"][self.heat_index].values
    Te = np.mean(
        [
            self.lowpassdata["Tin"][self.heat_index].values,
            self.lowpassdata["Tout"][self.heat_index].values,
        ],
        axis=0,
    )
    # offset time, power
    t -= t[0]
    Qdot -= Qdot[0]
    # squeeze unit dimensions
    t, Te, Qdot = np.squeeze(t), np.squeeze(Te), np.squeeze(Qdot)
    return t, Te, Qdot


def cool_curve(self):
    self.shot[("P. Freq", "Hz")]

    t_dwell = 10  # dwell time after max heating

    t = self.lowpassdata[("t", "s")].values
    Qdot = self.lowpassdata[("Qdot", "W")].values

    # start index
    dt = np.diff(t, axis=0).mean()
    start = Qdot.argmax() + int(t_dwell / dt)
    # stop index
    Qdot_max, Qdot_min = Qdot[start:].max(), Qdot[start:].min()
    dQdot = Qdot_max - Qdot_min
    threshold = Qdot_min + 0.1 * dQdot
    stop = start + np.where(Qdot[start:] < threshold)[0][0]
    index = slice(start, stop)

    # plt.plot(t[index], np.log(Q[index]))

    # plt.plot(t[index], self.lowpassdata['Tout'][index])

    Te = np.mean(
        [self.lowpassdata[("Tout", "K")], self.lowpassdata[("Tout", "K")]], axis=0
    )
    return t[index], Te[index], Qdot[index]


def extract_heating(self):
    data = self.lowpassdata.droplevel(1, axis=1)
    t = data.t
    Qdot = data.Qdot
    Te = np.mean([data.Tin, data.Tout], axis=0)

    hA, C = 0.3, 1
    T = Qdot / hA + Te
    dTdt = np.gradient(T, t)
    S = C * dTdt + Qdot

    dt = np.diff(data["t"], axis=0).mean()
    freq = self.shot[("P. Freq", "Hz")]
    windowlength = int(2.5 / (dt * freq))
    if windowlength % 2 == 0:
        windowlength += 1
    S = scipy.signal.savgol_filter(S, windowlength, polyorder=3)

    plt.plot(t, Qdot)
    plt.plot(t, S)
    self.plot_single("Qdot", lowpass=True)


def extract_thermal_timeconstant(self):
    testcolumns = ["B Sultan", "dm/dt L", "Ipulse", "P. Freq"]
    testdata = spp.testplan.droplevel(1, axis=1).loc[:, testcolumns]
    for s in testdata.index:
        spp.shot = s
        print(s)
        lc = LumpedCapacitance(*spp.cool_curve())
        testdata.loc[s, ["hA", "C", "rms"]] = lc.fit_hA()
        testdata.loc[s, "Te_bar"] = np.mean(spp.cool_curve()[1])
    testdata = testdata[testdata["rms"] < 0.01]

    plt.plot(testdata["dm/dt L"], testdata["C"], "o")
    """

    """
    spp.shot = 2
    lc = LumpedCapacitance(*spp.cool_curve())
    hA, C, err = lc.fit_hA()
    plt.plot(lc.t, lc.Qdot)
    plt.plot(lc.t, lc.solve(hA, C).T)


class LumpedCapacitance:
    def __init__(self, t, Te, Qdot):
        self.t = t
        self.Te = Te
        self.Qdot = Qdot
        self.Qnorm = np.linalg.norm(Qdot)
        self.Te_interp = scipy.interpolate.interp1d(t, Te)

    def dTdt(self, t, T, hA, C):
        Te = self.Te_interp(t)
        tau = C / hA
        return -1 / tau * (T - Te)

    def solve(self, hA, C):
        dTo = -1 / hA * self.Qdot[0]
        sol = scipy.integrate.solve_ivp(
            self.dTdt,
            (self.t[0], self.t[-1]),
            [dTo],
            args=(hA, C),
            t_eval=self.t,
            method="RK45",
        )
        dT = sol.y
        Qdot = -hA * dT
        return Qdot

    def Qdot_err(self, x):
        Qdot = self.solve(*x)
        return np.sqrt(np.mean((Qdot - self.Qdot) ** 2)) / self.Qnorm

    def fit_hA(self):
        xo = [0.1, 1]
        sol = scipy.optimize.minimize(self.Qdot_err, xo, method="COBYLA")
        hA, C = abs(sol.x)
        err = self.Qdot_err((hA, C))
        return hA, C, err
