import numpy as np
import pylab as pl
from nova.finite_element import FE
from nova.config import select
from nova.coils import PF
from nova.inverse import INV
from nova.coils import TF
from nova.loops import Profile
from nova.structure import architect
from nova.streamfunction import SF
import seaborn as sns

rc = {
    "figure.figsize": [8 * 12 / 16, 8],
    "savefig.dpi": 120,
    "savefig.jpeg_quality": 100,
    "savefig.pad_inches": 0.1,
    "lines.linewidth": 2,
}
sns.set(
    context="talk",
    style="white",
    font="sans-serif",
    palette="Set2",
    font_scale=7 / 8,
    rc=rc,
)


class SS:  # structural solver
    """
    Un structural solver incroyable
    Le passé est l'avenir!
    """

    def __init__(self, sf, pf, tf):
        self.sf = sf
        self.pf = pf
        self.tf = tf

        self.inv = INV(pf, tf, dCoil=2.5, offset=0.3)
        self.inv.colocate(sf, n=1e3, expand=0.5, centre=0, width=363 / (2 * np.pi))
        self.inv.wrap_PF(solve=False)

        self.atec = architect(tf, pf, plot=False)  # load sectional properties
        self.fe = FE(frame="3D")  # initalise FE solver

        self.add_mat()

        self.TF_loop()
        self.PF_connect()

    # def update_tf(self):

    def add_mat(self):
        self.fe.add_mat(
            "nose",
            ["wp", "steel_forged"],
            [self.atec.winding_pack(), self.atec.case(0)],
        )
        self.fe.add_mat(
            "loop", ["wp", "steel_cast"], [self.atec.winding_pack(), self.atec.case(1)]
        )
        trans = []  # nose-loop transitions
        for i, l_frac in enumerate(np.linspace(0, 1, 5)):
            trans.append(
                self.fe.add_mat(
                    "trans_{}".format(i),
                    ["wp", "steel_cast"],
                    [self.atec.winding_pack(), self.atec.case(l_frac)],
                )
            )
        self.fe.mat_index["trans_lower"] = trans
        self.fe.mat_index["trans_upper"] = trans[::-1]
        self.fe.add_mat("GS", ["steel_cast"], [self.atec.gravity_support()])
        self.fe.add_mat("OICS", ["wp"], [self.atec.intercoil_support()])
        for name in self.atec.PFsupport:  # PF coil connection sections
            mat_name = "PFS_{}".format(name)
            self.fe.add_mat(
                mat_name,
                ["wp"],
                [self.atec.coil_support(width=self.atec.PFsupport[name]["width"])],
            )

    def TF_loop(self):
        P = np.zeros((len(tf.p["cl"]["x"]), 3))
        P[:, 0], P[:, 2] = tf.p["cl"]["x"], tf.p["cl"]["z"]
        self.fe.add_nodes(P)  # all TF nodes
        TFparts = ["nose", "trans_lower", "loop", "trans_upper"]
        for part in TFparts:  # hookup elements
            self.fe.add_elements(n=tf.p[part]["nd"], part_name=part, nmat=part)

        # constrain TF nose - free translation in z
        self.fe.add_bc("nw", "all", part="nose")

    def PF_connect(self):
        for name in self.atec.PFsupport:  # PF coil connections
            nd_tf = self.atec.PFsupport[name]["nd_tf"]
            p = self.atec.PFsupport[name]["p"]  # connect PF to TF
            # self.fe.add_nodes([p['x'][1], 0, p['z'][1]])
            self.fe.add_nodes([p["x"][0], 0, p["z"][0]])
            self.atec.PFsupport[name]["nd"] = self.fe.nnd - 1  # store node index
            mat_name = "PFS_{}".format(name)
            self.fe.add_elements(
                n=[nd_tf, self.fe.nnd - 1], part_name=name, nmat=mat_name
            )

            # fe.add_elements(
            #         n=[fe.nnd-2, fe.nnd-1], part_name=name, nmat=mat_name)
            # fe.add_cp([nd_tf, fe.nnd-2], dof='fix', rotate=False)

    def PF_load(self):
        self.inv.ff.get_force()
        for name in self.atec.PFsupport:  # PF coil connections
            Fcoil = self.inv.ff.Fcoil[name]
            self.fe.add_nodal_load(
                self.atec.PFsupport[name]["nd"], "fz", 1e6 * Fcoil["fz"]
            )

    def gravity_support(self):
        nGS = 15
        yGS = np.linspace(atec.Gsupport["yfloor"], 0, nGS)
        zGS = np.linspace(atec.Gsupport["zfloor"], atec.Gsupport["zbase"], nGS)
        for ygs, zgs in zip(yGS, zGS):
            fe.add_nodes([atec.Gsupport["Xo"], ygs, zgs])
        yGS = np.linspace(0, -atec.Gsupport["yfloor"], nGS)
        zGS = np.linspace(atec.Gsupport["zbase"], atec.Gsupport["zfloor"], nGS)
        for ygs, zgs in zip(yGS[1:], zGS[1:]):
            fe.add_nodes([atec.Gsupport["Xo"], ygs, zgs])
        fe.add_elements(
            n=range(fe.nnd - 2 * nGS + 1, fe.nnd), part_name="GS", nmat="GS"
        )

        # couple GS to TF loop
        fe.add_cp([atec.Gsupport["nd_tf"], fe.nnd - nGS], dof="nrx", rotate=False)
        fe.add_bc(["nry"], [0], part="GS", ends=0)  # nry - free rotation in y
        fe.add_bc(["nry"], [-1], part="GS", ends=1)

    def plot(self):
        self.fe.plot_nodes()

        pl.axis("off")


if __name__ == "__main__":
    nTF = 16
    base = {"TF": "demo", "eq": "DEMO_SN_SOF"}
    config, setup = select(base, nTF=nTF, update=False)
    profile = Profile(
        config["TF_base"],
        family="S",
        load=True,
        part="TF",
        nTF=nTF,
        obj="L",
        npoints=50,
    )

    sf = SF(setup.filename)
    pf = PF(sf.eqdsk)
    tf = TF(profile=profile, sf=sf)

    ss = structural_solver(sf, pf, tf)

    ss.plot()

    tf.fill()


"""

fe.add_weight()  # add weight to all elements
fe.add_tf_load(sf, inv.ff, tf, inv.eq.Bpoint, parts=TFparts)



fe.solve()
fe.deform(scale=15)

fe.plot_nodes()
fe.plot_F(scale=1e-8)

fe.plot_displacment()
pl.axis('off')

pf.plot(label=True)

fe.plot_3D(pattern=tf.nTF)
"""
