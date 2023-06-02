from nep.DINA.coil_force import coil_force
from amigo.pyplot import plt
from nova.coils import PF

cf = coil_force(t_pulse=0.0)
cf.pf = PF()
cf.add_vv_coils()
cf.load_ps(-1, origin="start")
cf.initalize_sf(n=5e4, limit=[5, 9, -4.5, 6])

cf.t_index = 0.1
cf.vs3_update()

plt.set_context("talk")
ax, xlim, ylim = cf.vv.subplot()
for i in range(2):
    cf.contour(ax=ax[i])
    ax[i].set_xlim(xlim[i])
    ax[i].set_ylim(ylim[i])
