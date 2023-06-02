from nep.DINA.read_scenario import read_scenario
from amigo.pyplot import plt
from amigo import geom
import numpy as np
import nova.cross_coil as cc

scn = read_scenario(read_txt=False, setname="link_f")
scn.load_file(folder="15MA DT-DINA2017-04_v1.2", read_txt=False)
scn.inv.set_limit(ICS=60, side="both")  # elevate ICS limit
scn.update_scenario(600)

eqdsk = scn.update_psi(n=2e3, limit=[2, 8.75, -5, 6])
scn.inv.colocate(eqdsk, targets=False, Xpoint=False, field=False)

ztop = scn.inv.fix["z"][14]

scn.inv.fix["z"][11:16] = ztop
scn.inv.get_weight()
scn.inv.set_background()
scn.inv.set_foreground()

# scn.update_psi(plot=False)
# scn.inv.get_weight()
scn.solve()
scn.update_psi(plot=True, plot_nulls=True)

psi_norm = (scn.sf.Xpsi_array - scn.sf.Mpsi) / (scn.sf.Xpsi - scn.sf.Mpsi)
print(psi_norm)

sf = scn.sf
x, z = sf.get_boundary(alpha=1, reverse=False, xlocate=True)

# x, z = geom.xzInterp(x, z)
xt, zt = geom.tangent(x, z)
that = np.array([xt, zt]) / np.linalg.norm([xt, zt], axis=0)

lnorm = geom.length(x, z)
Bx = sf.Bspline[0].ev(x, z)
Bz = sf.Bspline[1].ev(x, z)

Bt = np.array([np.dot(b, t) for b, t in zip(np.array([Bx, Bz]).T, that.T)])

B = sf.Bspline[2].ev(x, z)

L = geom.length(x, z, norm=False)
Ipl = np.trapz(Bt, L) / cc.mu_o
print(f"Ip {1e-6*Ipl:1.6f}MA")

ax = plt.subplots(2, 1)[1]
ax[0].plot(lnorm, Bt)
ax[0].set_xlabel("$L_{sep}$")
ax[0].set_ylabel("$|B_p|$, T")
ax[1].plot(z, Bt)
ax[1].set_xlabel("$Z_{sep}$")
ax[1].set_ylabel("$|B_p|$, T")
plt.tight_layout()
plt.despine()


# doe.scn.update_psi(plot=True, current='AT')
