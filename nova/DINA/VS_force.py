from read_dina import get_folders
import nep
from amigo.IO import class_dir
from os.path import join
from nep.DINA.read_plasma import read_plasma
from nep.DINA.read_psi import read_psi
from amigo.pyplot import plt
from nep.VS.VSgeom import VS
import numpy as np
from amigo.geom import qrotate
from amigo.addtext import linelabel


class Force:

    def __init__(self, directory):
        self.directory = directory
        self.vs = VS()  # load VS object

    def load_scenario(self, folder):
        self.pl = read_plasma(directory, folder=folder)  # load time trace
        self.psi = read_psi(directory, folder)  # load psi
        self.initalize_arrays()

    def initalize_arrays(self):
        if len(self.pl.t) != self.psi.nt or\
                self.pl.t[-1] != self.psi.scalars['t'][-1]:
            raise ValueError('timescale inconsistency between pl and psi ')
        self.t = self.pl.t
        self.nt = self.psi.nt
        self.Ivs = self.pl.Ivs

    def calculate(self, plot=False):
        B, F, Fo = self.psi.get_force_array()  # load point values
        Ftn = np.zeros((self.nt, 3, self.vs.nP))  # rotate force vector
        for i in range(self.vs.nP):
            Ftn[:, :, i] = qrotate(F[:, :, i],
                                   theta=-self.vs.theta[i], dx=[0, 1, 0])

        F_max, Fo_max = np.zeros(self.vs.nP), np.zeros(self.vs.nP)
        for i in range(self.vs.nP):
            F_max[i] = np.nanmax(np.linalg.norm(F[:, :, i], axis=1))
            Fo_max[i] = np.nanmax(np.linalg.norm(Fo[:, :, i], axis=1))

        if plot:
            ax = plt.subplots(3, 1, sharex=True, sharey=False)[1]
            text = [[] for _ in range(2)]
            for i, a in enumerate(ax):
                text[i] = linelabel(loc='max', postfix='', ax=a)
            for i, coil in enumerate(self.vs.geom):
                # ax[i].plot(self.t, np.linalg.norm(Fo[:, :, i], axis=1),
                #            '--C7', label=r'$|\bfF_o\rm|$')
                # text[i].add(r'$|\bfF_o\rm|$')
                ax[i].plot(self.t, np.linalg.norm(Ftn[:, :, i], axis=1),
                           'C7', label=r'$|\bfF\rm|$')
                text[i].add(r'$|\bfF\rm|$')
                ax[i].plot(self.t, Ftn[:, 0, i], 'C0', label=r'$F_t$')
                ax[i].plot(self.t, Ftn[:, 2, i], 'C3', label=r'$F_n$')

                ax[i].set_ylabel('$F_{vs}$ kNm$^{-1}$')
                ax[i].text(0.9, 1.05, coil+'-vs', transform=ax[i].transAxes,
                           ha='center',
                           bbox=dict(facecolor='grey', alpha=0.25))
            for t in text:
                t.plot()
            ax[0].text(0.15, 1.05, self.psi.name, transform=ax[0].transAxes,
                       weight='bold', ha='center')
            ax[1].set_xlabel('$t$ ms')
            ax[0].legend()
            plt.despine()
            plt.setp(ax[0].get_xticklabels(), visible=False)

        return F_max, Fo_max

    def calculate_max(self):
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
    directory = join(class_dir(nep), '../Scenario_database/disruptions')
    folders = get_folders(directory)
    folders = sorted(folders)

    f = Force(directory)
    f.load_scenario(folders[3])  # MD_UP_exp16
    f.calculate(plot=True)

    # f.calculate_max()
