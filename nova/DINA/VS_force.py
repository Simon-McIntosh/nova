from nep.DINA.read_dina import dina
from nep.DINA.read_plasma import read_plasma
from nep.DINA.read_psi import read_psi
from amigo.pyplot import plt
from nep.VS.VSgeom import VS
import numpy as np
from amigo.geom import qrotate
from amigo.addtext import linelabel


class Force:

    def __init__(self, database_folder='disruptions'):
        self.dina = dina(database_folder)
        self.psi = read_psi(database_folder)  # load psi
        self.vs = VS()  # load VS object

    def load_scenario(self, folder):
        self.psi.read_file(folder)
        self.initalize_arrays()

    def initalize_arrays(self):
        '''
        if len(self.pl.t) != self.psi.nt or\
                self.pl.t[-1] != self.psi.scalars['t'][-1]:
            raise ValueError('timescale inconsistency between pl and psi ')
        '''
        self.t = self.psi.scalars['t']
        self.nt = self.psi.nt

    def calculate(self, plot=False):
        B, F, = self.psi.get_force_array()  # load point values
        Ftn = np.zeros((self.nt, 3, self.vs.nP))  # rotate force vector
        for i in range(self.vs.nP):
            Ftn[:, :, i] = qrotate(F[:, :, i], theta=-self.vs.theta[i],
                                   dx=[0, 1, 0])

        F_max = np.zeros(self.vs.nP)
        for i in range(self.vs.nP):
            F_max[i] = np.nanmax(np.linalg.norm(F[:, :, i], axis=1))

        if plot:
            ax = plt.subplots(3, 1, sharex=True, sharey=False)[1]
            text = [[] for _ in range(len(ax))]
            for i, a in enumerate(ax):
                text[i] = linelabel(loc='max', postfix='', ax=a)
            ax[0].plot(self.t, )
            for i, coil in enumerate(self.vs.geom):
                ax[i+1].plot(self.t, np.linalg.norm(1e-3*Ftn[:, :, i], axis=1),
                             'C7', label=r'$|\bfF\rm|$')
                text[i+1].add(r'$|\bfF\rm|$')
                ax[i+1].plot(self.t, 1e-3*Ftn[:, 0, i], 'C0', label=r'$F_t$')
                ax[i+1].plot(self.t, 1e-3*Ftn[:, 2, i], 'C3', label=r'$F_n$')

                ax[i+1].set_ylabel('$F_{vs}$ kNm$^{-1}$')
                ax[i+1].text(0.9, 1.05, coil+'-vs',
                             transform=ax[i+1].transAxes, ha='center',
                             bbox=dict(facecolor='grey', alpha=0.25))
            for t in text:
                t.plot()
            ax[1].text(0.15, 1.05, self.psi.name, transform=ax[0].transAxes,
                       weight='bold', ha='center')
            ax[2].set_xlabel('$t$ ms')
            ax[1].legend()
            plt.despine()
            for i in range(2):
                plt.setp(ax[i].get_xticklabels(), visible=False)

        return F_max

    def calculate_max(self):
        folders = self.dina.folders
        F_max = np.zeros((len(folders), self.vs.nP))
        Fo_max = np.zeros((len(folders), self.vs.nP))

        for i, folder in enumerate(folders):
            print(folder)
            f.load_scenario(folder)
            F_max[i, :], Fo_max[i, :] = f.calculate()

        ax = plt.subplots(self.vs.nP, 1, sharex=True, sharey=False)[1]
        x = range(len(folders))
        for i, coil in enumerate(self.vs.geom):
            ax[i].bar(x, Fo_max[:, i], width=0.8,
                      color='C3', label=r'$|\bfF_o\rm|$')
            ax[i].bar(x, F_max[:, i], width=0.6,
                      color='C7', label=r'$|\bfF\rm|$')
            ax[i].set_ylabel('$F_{vs}$ kNm$^{-1}$')
            ax[i].text(0.5, 0.95, coil+'-vs', transform=ax[i].transAxes,
                       ha='center', bbox=dict(facecolor='grey', alpha=0.25))

        ax[0].legend()
        plt.sca(ax[1])
        plt.xticks(x, folders, rotation=70)
        plt.despine()
        plt.setp(ax[0].get_xticklabels(), visible=False)


if __name__ == '__main__':

    f = Force('disruptions')
    f.load_scenario(3)  # MD_UP_exp16
    f.calculate(plot=True)

    # f.calculate_max()
