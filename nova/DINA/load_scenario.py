from nep import coil_geom
from nep.DINA.read_scenario import read_scenario
from amigo.pyplot import plt
from nova.streamfunction import SF
from nep.DINA.read_eqdsk import read_eqdsk
from nova.elliptic import EQ
from nova.cross_coil import get_coil_psi
from amigo import geom
from nova.force import force_field


vv = coil_geom.VVcoils(model='full')

scn = read_scenario(read_txt=False)
# scn.load_file(folder='15MA DT-DINA2016-01_v1.1')
scn.load_file(folder='15MA DT-DINA2012-02')

scn.update(2)
ff = force_field(scn.pf.index, scn.pf.coil,
                 scn.pf.sub_coil, scn.pf.plasma_coil, multi_filament=True)

ff.set_force_field(state='passive')

scn.update(200)
ff.set_current()
Fcoil = ff.get_force()


x2d, z2d, x, z = geom.grid(1e3, [0.25, 10, -7, 7])[:4]
psi = get_coil_psi(x2d, z2d, scn.pf)
eqdsk = read_eqdsk(file='burn')  # 'burn', 'inductive'
sf = eqdsk.sf
sf.update_plasma(eq={'x': x, 'z': z, 'psi': psi, 'Ipl': 1})


ax = plt.subplots(1, 1, figsize=(8, 10))[1]


sf.contour(boundary=False, Xnorm=False)

sf.get_Xpsi(select='upper')
alpha = 1-1e-4
sf.Spsi = alpha * (sf.Xpsi - sf.Mpsi) + sf.Mpsi
psi_line = sf.get_contour([sf.Spsi], boundary=False)[0]
for line in psi_line[:2]:
    x, z = line[:, 0], line[:, 1]
    plt.plot(x, z, 'C3')

sf.get_Xpsi(select='lower')
alpha = 1-1e-4
sf.Spsi = alpha * (sf.Xpsi - sf.Mpsi) + sf.Mpsi
psi_line = sf.get_contour([sf.Spsi], boundary=False)[0]
for line in psi_line:
    x, z = line[:, 0], line[:, 1]
    plt.plot(x, z, 'C4')


scn.pf.plot(label=True, current=True)
scn.pf.plot(subcoil=True, plasma=True)
# vv.plot()
ax.plot(sf.xlim, sf.zlim, 'gray')

ff.plot()
# sf = SF()

#scn.plot_currents()



