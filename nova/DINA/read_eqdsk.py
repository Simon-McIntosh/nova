from nep.DINA.read_dina import dina
from nova.streamfunction import SF
from amigo.pyplot import plt
from amigo.IO import pythonIO
from os.path import isfile
import numpy as np


class read_eqdsk(pythonIO):

    def __init__(self, database_folder='eqdsk', file='burn', read_txt=False):
        self.read_txt = read_txt
        self.dina = dina(database_folder)
        pythonIO.__init__(self)  # python read/write
        self.load_file(file)

    def load_file(self, file, **kwargs):
        read_txt = kwargs.get('read_txt', self.read_txt)
        filename = self.dina.locate_file(file)
        filename = '.'.join(filename.split('.')[:-1])
        attributes = ['eqdsk']
        if read_txt or not isfile(filename + '.pk'):
            self.read_file(file)
            self.save_pickle(filename, attributes)
        else:
            self.load_pickle(filename)

    def read_file(self, file):
        filename = self.dina.locate_file(file)
        sf = SF(filename=filename)
        self.eqdsk = sf.eqdsk

    def plot(self):
        ax = plt.subplots(1, 1, figsize=(8, 10))[1]
        self.sf.contour(ax=ax)

    #def

    def dPdPsi(self, Psi, I):
        dP = 1
        return dP


    def intergrate(self):
        eqdsk = self.sf.eqdsk
        #self.ode = ode(dIdPsi).set_integrator('dop853')  # 'dopri5'

    def plot_flux_functions(self):
        Xpsi, Mpsi = self.eqdsk['sibdry'], self.eqdsk['simagx']
        psi = self.eqdsk['pnorm'] * (Xpsi - Mpsi) + Mpsi

        ax = plt.subplots(2, 1)[1]
        ax[0].plot(self.eqdsk['pnorm'], 1e-6*self.eqdsk['pressure'])
        ax[1].plot(self.eqdsk['pnorm'], 1e-6*self.eqdsk['pprime'], '-')
        pprime = np.gradient(self.eqdsk['pressure'], psi)
        ax[1].plot(self.eqdsk['pnorm'], 1e-6*pprime, '--')
        ax[0].set_ylabel('$P$ MnTm$^{-2}$')
        ax[1].set_ylabel('$dP/d\Psi$ MnTm$^{-2}\,$rad$\,$Wb$^{-1}$')
        ax[1].set_xlabel('$\Psi_{norm}$')
        plt.despine()
        plt.detick(ax)

        ax = plt.subplots(2, 1)[1]
        ax[0].plot(self.eqdsk['pnorm'], self.eqdsk['fpol'])
        ffprime = self.eqdsk['fpol'] * np.gradient(self.eqdsk['fpol'], psi)
        ax[1].plot(self.eqdsk['pnorm'], self.eqdsk['ffprim'], '-')
        ax[1].plot(self.eqdsk['pnorm'], ffprime, '--')
        ax[1].set_xlabel('$\Psi_{norm}$')
        plt.despine()
        plt.detick(ax)





if __name__ == '__main__':

    eqdsk = read_eqdsk(read_txt=True)
    # eqdsk.plot()
    eqdsk.plot_flux_functions()




