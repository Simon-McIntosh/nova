from nep.DINA.coil_force import coil_force
from amigo.pyplot import plt
from nova.coils import PF

cf = coil_force(t_pulse=0.0)
cf.pf = PF()
cf.add_vv_coils()
cf.load_ps(-1, origin='start')
cf.initalize_sf(n=5e4, limit=[3, 10, -4.5, 6])

cf.t_index = 0.0
cf.vs3_update()

plt.set_context('poster')
ax = plt.subplots(1, 1, figsize=(6, 10))[1]
cf.pf.plot(subcoil=True, ax=ax)
cf.contour(ax=ax)
