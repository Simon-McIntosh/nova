from coilset import CoilSet
from matplotlib.collections import PatchCollection
from amigo.pyplot import plt
import numpy as np


class CoilClass:
    '''
    Instances of CoilClass implement methods to manage input and
    output of data to/from the CoilSet class
    '''

    def __init__(self, dCoil=0):
        self.dCoil = dCoil
        self.coilset = CoilSet(dCoil=self.dCoil)  # initalise coilset

    def __call__(self, coilset):
        self.coilset = coilset

    def update_metadata(self, coil, **kwargs):
        # update coilset metadata, coil in ['coil', 'subcoil', 'plasma']
        getattr(self.coilset, coil).update_metadata(**kwargs)

    def add_coil(self, *args, **kwargs):
        name = self.coilset.coil.add_coil(*args, **kwargs)  # add primary coil
        self.mesh_coil(name)

    def mesh_coils(self, **kwargs):
        '''
        mesh all coils in self.coilset.coil
        '''
        dCoil = kwargs.pop('dCoil', None)
        self.coilset.subcoil.drop(self.coilset.subcoil.index, inplace=True)
        for name in self.coilset.coil.index:
            self.mesh_coil(name, dCoil=dCoil)

    def mesh_coil(self, name, **kwargs):
        dCoil = kwargs.get('dCoil', None)
        if dCoil is None:
            dCoil = self.coilset.coil.at[name, 'dCoil']
        It = self.coilset.coil.at[name, 'It']
        cross_section = self.coilset.coil.at[name, 'cross_section']
        x, z, dx, dz = self.coilset.coil.loc[name, ['x', 'z', 'dx', 'dz']]
        if dCoil is None or dCoil == 0:
            dCoil = np.max([dx, dz])
        elif dCoil == -1:  # mesh per-turn (inductance calculation)
            Nt = self.coilset.coil.at[name, 'Nt']
            dCoil = (dx * dz / Nt)**0.5
        nx = int(np.ceil(dx / dCoil))
        nz = int(np.ceil(dz / dCoil))
        if nx < 1:
            nx = 1
        if nz < 1:
            nz = 1
        dx_, dz_ = dx / nx, dz / nz  # subcoil dimensions
        x_ = x + np.linspace(dx_ / 2, dx - dx_ / 2, nx) - dx / 2
        z_ = z + np.linspace(dz_ / 2, dz - dz_ / 2, nz) - dz / 2
        xm_, zm_ = np.meshgrid(x_, z_, indexing='ij')
        xm_ = np.reshape(xm_, (-1, 1))[:, 0]
        zm_ = np.reshape(zm_, (-1, 1))[:, 0]
        Nf = len(xm_)  # filament number
        If = It / Nf  # filament current
        self.coilset.coil.at[name, 'Nf'] = Nf
        for i, (x_, z_) in enumerate(zip(xm_, zm_)):
            subname = f'{name}_{i}'
            self.coilset.subcoil.add_coil(
                    x_, z_, dx_, dz_, If=If, name=subname,
                    cross_section=cross_section, label=name)

    def plot(self, subcoil=True, ax=None):
        if ax is None:
            ax = plt.gca()
        if subcoil:
            coil = self.coilset.subcoil
        else:
            coil = self.coilset.coil
        pc = PatchCollection(coil.loc[:, 'patch'], edgecolor='k')
        ax.add_collection(pc)
        ax.axis('equal')
        ax.axis('off')


if __name__ is '__main__':

    coilclass = CoilClass(dCoil=0.05)
    coilclass.update_metadata('coil', additional_columns=['R'])

    coilclass.add_coil(1, 2, 0.1, 0.1, name='PF6', label='CS')
    coilclass.add_coil(1, 1, 0.3, 0.3, name='PF5', label='CS')
    coilclass.add_coil(1.6, 1.5, 0.2, 0.2, name='PF7', label='PF')

    #coilclass.mesh_coils(dCoil=-1)
    coilclass.plot()


    #coilset.coil['c'] = [2,3,4]
    '''
    df = coilset.coil.from_dict({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    #coilset.coil({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})

    coilset.coil.nC = 1
    print(coilset.coil.nC)
    '''







