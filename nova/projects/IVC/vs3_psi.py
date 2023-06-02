from nep.coil_geom import PFgeom, VSgeom


class VS3:
    def __init__(self):
        vs_geom = VSgeom()
        self.pf = vs_geom.pf

        pf_geom = PFgeom(VS=True)
        self.pf = pf_geom.pf
        # self.pf.mesh_coils(dCoil=0.25)


if __name__ == "__main__":
    vs3 = VS3()
