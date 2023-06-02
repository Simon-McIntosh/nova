from nep.DINA.VDE_force import VDE_force
from nep.DINA.coil_force import coil_force
from amigo.pyplot import plt
from collections import OrderedDict
from amigo.addtext import linelabel

coil_data = OrderedDict()

folder = "MD_UP_exp16"
folder = "MD_DW_exp22"

# load time-series data
vde = VDE_force()
coil_data["DINA"] = vde.read_data("DINA", nframe=500)[folder]["control"]
force = coil_force(vessel=True, t_pulse=0.0)
coil_data["IO spike"] = force.read_data(nframe=500)[folder]["control"]
force = coil_force(vessel=True, t_pulse=0.3)
coil_data["IO flat-top"] = force.read_data(nframe=500)[folder]["control"]

ax = plt.subplots(2, 1)[1]
text = [linelabel(Ndiv=15, ax=_ax, loc="max") for _ax in ax]
for key in coil_data:
    for iax, k in enumerate([1, 0]):
        ax[iax].plot(1e3 * coil_data[key]["t"], 1e-3 * coil_data[key]["Fmag"][:, k])
        text[iax].add(key.replace("_", " "))
for t in text:
    t.plot(Ralign=True, Roffset=10)
ax[1].set_xlabel("$t$ ms")
for _ax, coil in zip(ax, ["upperVS", "lowerVS"]):
    _ax.set_ylabel("$|F|$ kNm$^{-1}$")
    _ax.text(
        0.95,
        0.95,
        coil,
        transform=_ax.transAxes,
        va="top",
        ha="right",
        bbox=dict(facecolor="w", ec="gray", lw=1, boxstyle="round", pad=0.5),
    )
plt.despine()
plt.detick(ax)
plt.suptitle(folder)

ax = plt.subplots(1, 1)[1]
text = linelabel(Ndiv=30, ax=ax, loc="min")
for key in coil_data:
    ax.plot(1e3 * coil_data[key]["t"], 1e-3 * coil_data[key]["I"])
    text.add(key)
text.plot(Ralign=True, Roffset=20)
ax.set_xlabel("$t$ ms")
ax.set_ylabel("$I$ kA")
plt.despine()
plt.title(folder)


folder = "MD_UP_exp22"
frame_index = 251
vde = VDE_force()
vde.load_file(folder, frame_index=frame_index)

force = coil_force(vessel=True, t_pulse=0.0)
force.load_file(folder)
force.frame_update(frame_index)

ax = plt.subplots(1, 2)[1]
vde.plot(insert=True, contour=True, subcoil=True, plasma=False, ax=ax[0])
force.plot(insert=True, contour=True, subcoil=True, plasma=False, ax=ax[1])
