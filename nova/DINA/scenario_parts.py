   def get_time(self, dt=None):
        if dt is None:
            dt = self.dt  # data-set resolution
        n = int((self.t[-1] - self.t[0]) / dt)
        time = np.linspace(self.t[0], self.t[-1], n)
        return time, n

    def stored_energy(self, M, names, dt=20, plot=False):
        time, n = self.get_time(dt=dt)
        E = np.zeros(n)
        for i, t in enumerate(time):
            Ic = self.get_current(t, names)  # kA
            E[i] = 0.5*M.dot(Ic).dot(Ic)
            E[i] *= 1e-3  # stored energy GJ
        energy = pd.Series()
        index = np.argmax(E)
        energy['t [s]'] = time[index]
        energy['E [GJ]'] = E[index]
        Ic = self.get_current(time[index], names)
        for Io, name in zip(Ic, names):
            energy[f'Io [kA] {name}'] = Io
        if plot:
            ax = plt.subplots(1, 1)[1]
            ax.plot(time, E)
            ax.set_xlabel('$t$ s')
            ax.set_ylabel('$E$ GJ')
            index = np.argmax(E)
            ax.text(time[index], E[index],
                    f'{E[index]:1.1f}GJ, {time[index]:1.1f}s',
                    ha='left', va='bottom')
            plt.despine()
        return energy

    def scenario_fault(self, M, names, dt=None, plot=False):
        time, n = self.get_time(dt=dt)
        Ic = np.zeros((n, len(M)))
        dI = np.zeros((n, len(M)))
        Eo = np.zeros((n, len(M)))
        Efault = np.zeros((n, len(M)))
        for i, t in enumerate(time):
            Ic[i, :], dI[i, :], Eo[i, :], Efault[i, :] = \
                self.fault_current(t, M, names)
        Ic = pd.DataFrame(Ic, columns=names)
        dI = pd.DataFrame(dI, columns=names)
        Eo = pd.DataFrame(Eo, columns=names)
        Efault = pd.DataFrame(Efault, columns=names)
        fault = pd.DataFrame(columns=names)
        for name in names:
            i = Efault.loc[:, name].idxmax()
            fault.loc['t', name] = time[i]
            fault.loc['Io', name] = Ic.loc[i, name]
            fault.loc['dI', name] = dI.loc[i, name]
            fault.loc['Ifault', name] = Ic.loc[i, name] + dI.loc[i, name]
            fault.loc['Eo', name] = Eo.loc[i, name]
            fault.loc['Efault', name] = Efault.loc[i, name]
        if plot:
            ax = plt.subplots(2, 1)[1]
            for i, name in enumerate(names):
                if name != 'Plasma':
                    ls = f'C{i}-' if 'PF' in name else f'C{i-6}--'
                    ax[0].plot(time, Ic.loc[:, name], ls, label=name)
                    ax[1].plot(time, Efault.loc[:, name], ls, label=name)
            ax[0].legend(bbox_to_anchor=(1.1, 1.4), ncol=6)
            plt.despine()
            plt.detick(ax)
            ax[1].set_xlabel('$t$ s')
            ax[0].set_ylabel('$I$ kA')
            ax[1].set_ylabel('$E_{fault}$ GJ')
        return fault

    def fault_current(self, t, Mt, names):
        Ic = self.get_current(t, names)  # kA
        L = np.diag(Mt)
        M = Mt.copy()
        np.fill_diagonal(M, 0)
        Eo = 1e-3 * 0.5 * L * Ic**2  # GJ
        dI = pd.Series(1/L * np.dot(M, Ic), index=Ic.index)
        Efault = 1e-3 * 0.5 * L * (Ic+dI)**2  # GJ
        return Ic, dI, Eo, Efault