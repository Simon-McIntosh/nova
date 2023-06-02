from nep.DINA.read_scenario import read_scenario
from amigo.pyplot import plt


scn = read_scenario(read_txt=False)
ax = plt.subplots(sum(index), 1, sharex=True, sharey=True, figsize=(10, 6))[1]
for i, folder in enumerate(folders["name"][index]):
    scn.load_file(folder=folder, read_txt=False)
    for j, name in enumerate(scn.IcCS):
        ax[i].plot(
            scn.t, scn.post["DINA"]["Ic"][name], "-", color=f"C{j%10}", label=name
        )
    scn.get_max_value(scn.t, scn.post["DINA"]["Ic"], ax=ax[i], plot=True, unit="kA")
    scn.get_max_value(
        scn.t, scn.post["DINA"]["Ic"], ax=ax[i], plot=True, unit="kA", sign=-1
    )
    ax[i].text(
        1,
        0.15,
        folder,
        transform=ax[i].transAxes,
        ha="right",
        va="bottom",
        fontsize="x-small",
        bbox=dict(facecolor="w", ec="gray", lw=1, boxstyle="round", pad=0.5),
    )
    ax[i].set_ylabel("$I_c$ kA")
    ax[i].set_ylim([-65, 65])
ax[-1].set_xlabel("$t$ s")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.6), ncol=5)
plt.despine()
plt.detick(ax)
