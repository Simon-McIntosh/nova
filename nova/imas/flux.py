"""Examine equilibrium time slice data."""
from dataclasses import dataclass, field

import numpy as np

from nova.biot.contour import Contour
from nova.biot.fieldnull import FieldNull
from nova.imas.equilibrium import EquilibriumData
from nova.imas.profile import Profile


@dataclass
class Flux(Profile, EquilibriumData):
    """Extract flux profiles from equilibra."""

    levels: int | np.ndarray = 50
    psi2d: np.ndarray = field(init=False, repr=False)
    contour: Contour = field(init=False, repr=False)
    fieldnull: FieldNull = field(init=False, repr=False,
                                 default_factory=FieldNull)

    def __post_init__(self):
        """Initialize contour instance."""
        super().__post_init__()
        x2d, z2d = np.meshgrid(self.data.r, self.data.z, indexing='ij')
        self.psi2d = self['psi2d']
        self.contour = Contour(x2d, z2d, self.psi2d, levels=self.levels)
        self.fieldnull.data.coords['x'] = self.data.r
        self.fieldnull.data.coords['z'] = self.data.z

    def update(self):
        """Extend profile update."""
        super().update()
        self.psi2d[:] = self['psi2d']
        self.contour.generate(self.j_tor_rbs)
        self.fieldnull.update_null(self.psi2d.data)

    def plot(self, axes=None):
        """Plot flux contours."""
        self.set_axes(axes=axes)
        self.contour.plot()
        self.fieldnull.plot()


if __name__ == '__main__':


    #from matplotlib import pyplot as plt
    #import seaborn as sns

    flux = Flux(105028, 1)  # DINA
    flux.itime = 300

    flux.plot()

    #flux = Flux(130506, 403)  # CORSICA
    #flux.itime = 35

    '''
    Jp = flux._rbs('j_tor2d')


    contour.generate(Jp)

    flux.boundary = contour.line(null.x_psi[1])[0].points
    flux.psi_axis = null.o_psi[0]
    flux.psi_boundary = null.x_psi[1]
    flux.update_plasma()

    index = contour.plot_fit(0.5, flux.normalize)

    '''



    '''


    contour = Contour(operate.grid.data.x2d, operate.grid.data.z2d,
                      operate.grid.psi_, levels=50)
    contour.generate(Jp)

    operate.psi_axis = operate.plasmagrid.o_psi[0]
    operate.psi_boundary = operate.plasmagrid.x_psi[0]

    index = contour.plot_fit(0.4, operate.normalize)
    '''


    '''
    plt.figure()
    operate.plot('plasma')

    #operate.grid.plot()
    contour.plot(color='C0')
    contour.lines[index].plot(color='C1')
    #levels = np.unique(contour.loc['psi'])
    #operate.plot_2d(operate.itime, axes=operate.axes,
    #                levels=levels)  # COCOS
    #contour.axes.contour(operate.data.r2d, operate.data.z2d,
    #                     operate['j_tor2d'])
    #plt.tight_layout()
    #plt.savefig('plasma.png')

    contour.plot_contour(null.x_psi[1], color='C3')
    null.plot()


    axes = plt.subplots(2, 1, sharex=True)[1]
    sns.despine()

    operate.plot_1d(operate.itime, 'dpressure_dpsi', axes=axes[0], label='IDS')
    contour.plot_1d(operate.normalize, 0, axes=axes[0], label='fit')

    operate.plot_1d(operate.itime, 'f_df_dpsi', axes=axes[1])
    contour.plot_1d(operate.normalize, 1, axes=axes[1])

    axes[0].set_ylabel(r'$p^\prime$')
    axes[1].set_ylabel(r'$ff^\prime$')
    axes[0].legend()
    '''
